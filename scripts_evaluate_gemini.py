#!/usr/bin/env python3
"""
Gemini model evaluation script for OpenXRD.
Supports various Gemini models from Google.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import google.generativeai as genai
from google.generativeai.types import generation_types
from tenacity import retry, stop_after_attempt, wait_exponential
from src.evaluation import OpenXRDEvaluator
from src.utils import validate_api_keys, get_output_path

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'gemini-2.0-flash': {
        'model_id': 'gemini-2.0-flash',
        'description': 'Latest Gemini 2.0 Flash model',
        'temperature': 0.2,
        'max_output_tokens': 1024,
        'supports_temperature': True,
        'uses_max_tokens': True
    },
    'gemini-1.5-pro': {
        'model_id': 'gemini-1.5-pro',
        'description': 'Gemini 1.5 Pro model',
        'temperature': 0.2,
        'max_output_tokens': 1024,
        'supports_temperature': True,
        'uses_max_tokens': True
    },
    'gemini-1.5-flash': {
        'model_id': 'gemini-1.5-flash',
        'description': 'Gemini 1.5 Flash model',
        'temperature': 0.2,
        'max_output_tokens': 1024,
        'supports_temperature': True,
        'uses_max_tokens': True
    }
}


class GeminiModelEvaluator:
    """Evaluator for Google Gemini models."""
    
    def __init__(self):
        """Initialize the Gemini client."""
        api_keys = validate_api_keys()
        if not api_keys['gemini']:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY environment variable.")
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.evaluator = OpenXRDEvaluator()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_gemini_api(self, prompt: str, model_config: dict) -> str:
        """
        Call Gemini API with retry logic.
        
        Args:
            prompt: Input prompt
            model_config: Model configuration
            
        Returns:
            Model response
        """
        generate_kwargs = {
            "model": model_config['model_id'],
            "contents": prompt,
        }
        
        # Configure generation parameters
        generation_config = generation_types.GenerationConfig()
        
        if model_config['supports_temperature'] and model_config['temperature'] is not None:
            generation_config.temperature = model_config['temperature']
        
        if model_config['uses_max_tokens']:
            generation_config.max_output_tokens = 200
        else:
            generation_config.max_output_tokens = 1000
        
        generate_kwargs["generation_config"] = generation_config
        
        response = genai.generate_content(**generate_kwargs)
        return response.text.strip()
    
    def create_model_function(self, model_key: str):
        """
        Create a model function for the evaluator.
        
        Args:
            model_key: Key identifying the model configuration
            
        Returns:
            Function that takes a prompt and returns a response
        """
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_key}. Available models: {list(MODEL_CONFIGS.keys())}")
        
        model_config = MODEL_CONFIGS[model_key]
        
        def model_function(prompt: str) -> str:
            return self._call_gemini_api(prompt, model_config)
        
        return model_function
    
    def evaluate_model(
        self, 
        model_key: str, 
        mode: str = "closedbook",
        material_type: str = "expert_reviewed",
        save_results: bool = True
    ) -> dict:
        """
        Evaluate a specific Gemini model.
        
        Args:
            model_key: Key identifying the model
            mode: Evaluation mode ("closedbook" or "openbook")
            material_type: Type of supporting materials ("generated" or "expert_reviewed")
            save_results: Whether to save results to file
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating Gemini model: {model_key} in {mode} mode")
        
        model_function = self.create_model_function(model_key)
        
        # Generate output path
        output_path = None
        if save_results:
            output_path = get_output_path(model_key, mode)
        
        # Run evaluation
        results = self.evaluator.evaluate_model(
            model_function=model_function,
            model_name=model_key,
            mode=mode,
            material_type=material_type,
            save_path=output_path,
            verbose=True
        )
        
        return results
    
    def evaluate_all_models(
        self, 
        mode: str = "closedbook",
        material_type: str = "expert_reviewed"
    ) -> dict:
        """
        Evaluate all available Gemini models.
        
        Args:
            mode: Evaluation mode
            material_type: Type of supporting materials
            
        Returns:
            Dictionary with results for all models
        """
        all_results = {}
        
        for model_key in MODEL_CONFIGS:
            try:
                logger.info(f"Starting evaluation for {model_key}")
                results = self.evaluate_model(model_key, mode, material_type)
                all_results[model_key] = results
                logger.info(f"Completed evaluation for {model_key}: {results['accuracy']:.2%}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_key}: {e}")
                all_results[model_key] = {"error": str(e)}
        
        return all_results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate Gemini models on OpenXRD benchmark")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="gemini-2.0-flash", help="Model to evaluate")
    parser.add_argument("--mode", type=str, choices=["closedbook", "openbook", "both"],
                        default="closedbook", help="Evaluation mode")
    parser.add_argument("--material_type", type=str, choices=["generated", "expert_reviewed"],
                        default="expert_reviewed", help="Type of supporting materials for openbook mode")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize evaluator
    try:
        evaluator = GeminiModelEvaluator()
    except ValueError as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1
    
    # Run evaluation
    try:
        if args.model == "all":
            if args.mode == "both":
                for mode in ["closedbook", "openbook"]:
                    logger.info(f"Running evaluation in {mode} mode for all models")
                    evaluator.evaluate_all_models(mode, args.material_type)
            else:
                evaluator.evaluate_all_models(args.mode, args.material_type)
        else:
            if args.mode == "both":
                for mode in ["closedbook", "openbook"]:
                    evaluator.evaluate_model(args.model, mode, args.material_type)
            else:
                evaluator.evaluate_model(args.model, args.mode, args.material_type)
        
        logger.info("Evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
