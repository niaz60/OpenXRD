#!/usr/bin/env python3
"""
LLaVA model evaluation script for OpenXRD.
Supports various LLaVA models including v1.5, v1.6, and OneVision variants.
"""

import os
import sys
import argparse
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import OpenXRDEvaluator
from src.utils import validate_api_keys, get_output_path

logger = logging.getLogger(__name__)

# Check if LLaVA dependencies are available
try:
    from llava.model.builder import load_pretrained_model
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False
    logger.warning("LLaVA dependencies not found. Please install LLaVA to use this script.")

# Model configurations
MODEL_CONFIGS = {
    # LLaVA v1.5 Models
    'llava-v1.5-7b': {
        'model_path': 'liuhaotian/llava-v1.5-7b',
        'model_base': None,
        'model_name': 'llava_v15',
        'description': 'LLaVA v1.5 7B model',
        'base_llm': 'Vicuna-7B-v1.5',
        'vision_encoder': 'CLIP-L-336px'
    },
    'llava-v1.5-13b': {
        'model_path': 'liuhaotian/llava-v1.5-13b',
        'model_base': None,
        'model_name': 'llava_v15',
        'description': 'LLaVA v1.5 13B model',
        'base_llm': 'Vicuna-13B-v1.5',
        'vision_encoder': 'CLIP-L-336px'
    },
    # LLaVA v1.6 Models
    'llava-v1.6-vicuna-7b': {
        'model_path': 'liuhaotian/llava-v1.6-vicuna-7b',
        'model_base': None,
        'model_name': 'llava_v16',
        'description': 'LLaVA v1.6 Vicuna 7B model',
        'base_llm': 'Vicuna-7B',
        'vision_encoder': 'CLIP-L-336px'
    },
    'llava-v1.6-vicuna-13b': {
        'model_path': 'liuhaotian/llava-v1.6-vicuna-13b',
        'model_base': None,
        'model_name': 'llava_v16',
        'description': 'LLaVA v1.6 Vicuna 13B model',
        'base_llm': 'Vicuna-13B',
        'vision_encoder': 'CLIP-L-336px'
    },
    'llava-v1.6-mistral-7b': {
        'model_path': 'liuhaotian/llava-v1.6-mistral-7b',
        'model_base': None,
        'model_name': 'llava_v16',
        'description': 'LLaVA v1.6 Mistral 7B model',
        'base_llm': 'Mistral-7B',
        'vision_encoder': 'CLIP-L-336px'
    },
    'llava-v1.6-34b': {
        'model_path': 'liuhaotian/llava-v1.6-34b',
        'model_base': None,
        'model_name': 'llava_v16',
        'description': 'LLaVA v1.6 34B model',
        'base_llm': 'Hermes-Yi-34B',
        'vision_encoder': 'CLIP-L-336px'
    }
}

# OneVision models (require different handling)
ONEVISION_MODELS = {
    'llava-onevision-qwen2-7b-ov': {
        'model_path': 'lmms-lab/llava-onevision-qwen2-7b-ov',
        'model_name': 'llava_qwen',
        'description': 'LLaVA OneVision QWEN2 7B OV',
    },
    'llava-onevision-qwen2-7b-ov-chat': {
        'model_path': 'lmms-lab/llava-onevision-qwen2-7b-ov-chat',
        'model_name': 'llava_qwen',
        'description': 'LLaVA OneVision QWEN2 7B OV Chat',
    },
    'llava-onevision-qwen2-7b-si': {
        'model_path': 'lmms-lab/llava-onevision-qwen2-7b-si',
        'model_name': 'llava_qwen',
        'description': 'LLaVA OneVision QWEN2 7B SI',
    },
    'llava-onevision-qwen2-0.5b-ov': {
        'model_path': 'lmms-lab/llava-onevision-qwen2-0.5b-ov',
        'model_name': 'llava_qwen',
        'description': 'LLaVA OneVision QWEN2 0.5B OV',
    },
    'llava-onevision-qwen2-0.5b-si': {
        'model_path': 'lmms-lab/llava-onevision-qwen2-0.5b-si',
        'model_name': 'llava_qwen',
        'description': 'LLaVA OneVision QWEN2 0.5B SI',
    }
}


class LLaVAModelEvaluator:
    """Evaluator for LLaVA models."""
    
    def __init__(self):
        """Initialize the LLaVA evaluator."""
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA dependencies not available. Please install LLaVA.")
        
        self.evaluator = OpenXRDEvaluator()
        self.loaded_models = {}  # Cache for loaded models
    
    def _load_llava_model(self, model_key: str):
        """
        Load a LLaVA model.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            Tuple of (tokenizer, model, image_processor)
        """
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        if model_key in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_key]
            logger.info(f"Loading LLaVA model: {model_key}")
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                config['model_path'],
                config['model_base'],
                config['model_name'],
                device_map="auto"
            )
            model.eval()
            
        elif model_key in ONEVISION_MODELS:
            config = ONEVISION_MODELS[model_key]
            logger.info(f"Loading LLaVA OneVision model: {model_key}")
            
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                config['model_path'],
                None,  # No separate base model
                config['model_name'],
                device_map="auto"
            )
            model.eval()
            if hasattr(model, 'tie_weights'):
                model.tie_weights()
        else:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Cache the loaded model
        self.loaded_models[model_key] = (tokenizer, model, image_processor)
        return tokenizer, model, image_processor
    
    def _generate_response(self, prompt: str, tokenizer, model) -> str:
        """
        Generate response from LLaVA model.
        
        Args:
            prompt: Input prompt
            tokenizer: Model tokenizer
            model: Loaded model
            
        Returns:
            Model response
        """
        # Create system + user prompt format
        system_prompt = "You are a helpful assistant trained to answer crystallography questions."
        formatted_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        
        # Encode the prompt
        input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=False,  # Greedy decoding
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        new_tokens = output_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        return response
    
    def create_model_function(self, model_key: str):
        """
        Create a model function for the evaluator.
        
        Args:
            model_key: Key identifying the model
            
        Returns:
            Function that takes a prompt and returns a response
        """
        tokenizer, model, image_processor = self._load_llava_model(model_key)
        
        def model_function(prompt: str) -> str:
            return self._generate_response(prompt, tokenizer, model)
        
        return model_function
    
    def evaluate_model(
        self, 
        model_key: str, 
        mode: str = "closedbook",
        material_type: str = "expert_reviewed",
        save_results: bool = True
    ) -> dict:
        """
        Evaluate a specific LLaVA model.
        
        Args:
            model_key: Key identifying the model
            mode: Evaluation mode ("closedbook" or "openbook")
            material_type: Type of supporting materials ("generated" or "expert_reviewed")
            save_results: Whether to save results to file
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating LLaVA model: {model_key} in {mode} mode")
        
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
        material_type: str = "expert_reviewed",
        include_onevision: bool = True
    ) -> dict:
        """
        Evaluate all available LLaVA models.
        
        Args:
            mode: Evaluation mode
            material_type: Type of supporting materials
            include_onevision: Whether to include OneVision models
            
        Returns:
            Dictionary with results for all models
        """
        all_results = {}
        
        # Evaluate standard LLaVA models
        for model_key in MODEL_CONFIGS:
            try:
                logger.info(f"Starting evaluation for {model_key}")
                results = self.evaluate_model(model_key, mode, material_type)
                all_results[model_key] = results
                logger.info(f"Completed evaluation for {model_key}: {results['accuracy']:.2%}")
            except Exception as e:
                logger.error(f"Failed to evaluate {model_key}: {e}")
                all_results[model_key] = {"error": str(e)}
        
        # Evaluate OneVision models if requested
        if include_onevision:
            for model_key in ONEVISION_MODELS:
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
    parser = argparse.ArgumentParser(description="Evaluate LLaVA models on OpenXRD benchmark")
    
    all_models = list(MODEL_CONFIGS.keys()) + list(ONEVISION_MODELS.keys())
    parser.add_argument("--model", type=str, choices=all_models + ["all", "standard", "onevision"],
                        default="llava-v1.6-34b", help="Model to evaluate")
    parser.add_argument("--mode", type=str, choices=["closedbook", "openbook", "both"],
                        default="closedbook", help="Evaluation mode")
    parser.add_argument("--material_type", type=str, choices=["generated", "expert_reviewed"],
                        default="expert_reviewed", help="Type of supporting materials for openbook mode")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no_onevision", action="store_true", 
                        help="Skip OneVision models when evaluating all models")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. LLaVA models may run slowly on CPU.")
    
    # Initialize evaluator
    try:
        evaluator = LLaVAModelEvaluator()
    except ImportError as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        logger.error("Please install LLaVA dependencies to use this script.")
        return 1
    
    # Run evaluation
    try:
        if args.model == "all":
            if args.mode == "both":
                for mode in ["closedbook", "openbook"]:
                    logger.info(f"Running evaluation in {mode} mode for all models")
                    evaluator.evaluate_all_models(mode, args.material_type, not args.no_onevision)
            else:
                evaluator.evaluate_all_models(args.mode, args.material_type, not args.no_onevision)
        elif args.model == "standard":
            # Evaluate only standard LLaVA models
            for model_key in MODEL_CONFIGS:
                if args.mode == "both":
                    for mode in ["closedbook", "openbook"]:
                        evaluator.evaluate_model(model_key, mode, args.material_type)
                else:
                    evaluator.evaluate_model(model_key, args.mode, args.material_type)
        elif args.model == "onevision":
            # Evaluate only OneVision models
            for model_key in ONEVISION_MODELS:
                if args.mode == "both":
                    for mode in ["closedbook", "openbook"]:
                        evaluator.evaluate_model(model_key, mode, args.material_type)
                else:
                    evaluator.evaluate_model(model_key, args.mode, args.material_type)
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
