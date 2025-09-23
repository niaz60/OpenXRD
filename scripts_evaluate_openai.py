#!/usr/bin/env python3
"""
OpenAI model evaluation script for OpenXRD.
Supports GPT-4, O1, O3, and other OpenAI models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from src.evaluation import OpenXRDEvaluator
from src.utils import validate_api_keys, get_output_path

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    'gpt-4.5-preview': {
        'model_id': 'gpt-4.5-preview-2025-02-27',
        'description': 'Latest GPT-4.5 preview model',
        'temperature': 0.2,
        'uses_max_tokens': True,
        'supports_temperature': True
    },
    'gpt-4-turbo': {
        'model_id': 'gpt-4-turbo-2024-04-09',
        'description': 'Latest GPT-4 Turbo model',
        'temperature': 0.2,
        'uses_max_tokens': True,
        'supports_temperature': True
    },
    'gpt-4-turbo-preview': {
        'model_id': 'gpt-4-0125-preview',
        'description': 'GPT-4 Turbo preview version',
        'temperature': 0.2,
        'uses_max_tokens': True,
        'supports_temperature': True
    },
    'gpt-4': {
        'model_id': 'gpt-4-0613',
        'description': 'GPT-4 base model',
        'temperature': 0.2,
        'uses_max_tokens': True,
        'supports_temperature': True
    },
    'o3-mini': {
        'model_id': 'o3-mini-2025-01-31',
        'description': 'Latest O3-mini reasoning model',
        'temperature': None,
        'uses_max_tokens': False,
        'supports_temperature': False
    },
    'o1': {
        'model_id': 'o1-2024-12-17',
        'description': 'O1 reasoning model',
        'temperature': None,
        'uses_max_tokens': False,
        'supports_temperature': False
    },
    'o1-pro': {
        'model_id': 'o1-pro-2025-03-19',
        'description': 'O1 Pro reasoning model',
        'temperature': None,
        'uses_max_tokens': False,
        'supports_temperature': False
    }
}


class OpenAIModelEvaluator:
    """Evaluator for OpenAI models."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        api_keys = validate_api_keys()
        if not api_keys['openai']:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI()
        self.evaluator = OpenXRDEvaluator()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_openai_api(self, prompt: str, model_config: dict) -> str:
        """
        Call OpenAI API with retry logic.
        
        Args:
            prompt: Input prompt
            model_config: Model configuration
            
        Returns:
            Model response
        """
        # Base parameters
        params = {
            "model": model_config['model_id'],
            "messages": [
                {"role": "system", "content": "You are a helpful assistant who answers multiple-choice questions about crystallography and X-ray diffraction."},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add temperature if supported
        if model_config['supports_temperature'] and model_config['temperature'] is not None:
            params["temperature"] = model_config['temperature']
        
        # Add token parameter based on model type
        if model_config['uses_max_tokens']:
            params["max_tokens"] = 200
        else:
            # For O-series models, use max_completion_tokens
            params["max_completion_tokens"] = 1000
        
        response = self.client.chat.completions.create(**params)
        return response.choices[0].message.content.strip()
    
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
            return self._call_openai_api(prompt, model_config)
        
        return model_function
    
    def evaluate_model(
        self, 
        model_key: str, 
        mode: str = "closedbook",
        material_type: str = "expert_reviewed",
        save_results: bool = True
    ) -> dict:
        """
        Evaluate a specific OpenAI model.
        
        Args:
            model_key: Key identifying the model
            mode: Evaluation mode ("closedbook" or "openbook")
            material_type: Type of supporting materials ("generated" or "expert_reviewed")
            save_results: Whether to save results to file
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating OpenAI model: {model_key} in {mode} mode")
        
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
        Evaluate all available OpenAI models.
        
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
    parser = argparse.ArgumentParser(description="Evaluate OpenAI models on OpenXRD benchmark")
    parser.add_argument("--model", type=str, choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        default="gpt-4", help="Model to evaluate")
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
        evaluator = OpenAIModelEvaluator()
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
