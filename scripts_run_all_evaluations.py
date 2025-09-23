#!/usr/bin/env python3
"""
Batch evaluation script for OpenXRD.
Runs evaluations across multiple models and modes.
"""

import os
import sys
import argparse
import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import validate_api_keys, save_results, NumpyEncoder

logger = logging.getLogger(__name__)

# Available evaluation modules
EVALUATION_MODULES = {
    'openai': {
        'script': 'scripts.evaluate_openai',
        'class': 'OpenAIModelEvaluator',
        'models': ['gpt-4.5-preview', 'gpt-4-turbo', 'gpt-4', 'o3-mini', 'o1'],
        'requires_api': 'openai'
    },
    'gemini': {
        'script': 'scripts.evaluate_gemini',
        'class': 'GeminiModelEvaluator',
        'models': ['gemini-2.0-flash', 'gemini-1.5-pro', 'gemini-1.5-flash'],
        'requires_api': 'gemini'
    },
    'llava': {
        'script': 'scripts.evaluate_llava',
        'class': 'LLaVAModelEvaluator',
        'models': ['llava-v1.6-34b', 'llava-v1.6-vicuna-13b', 'llava-v1.5-13b'],
        'requires_api': None
    }
}


class BatchEvaluator:
    """Batch evaluator for running multiple model evaluations."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the batch evaluator.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check available APIs
        self.available_apis = validate_api_keys()
        logger.info(f"Available APIs: {[k for k, v in self.available_apis.items() if v]}")
        
        # Store evaluation results
        self.results = {}
        
    def check_model_availability(self, model_family: str) -> bool:
        """
        Check if a model family can be evaluated.
        
        Args:
            model_family: Name of the model family
            
        Returns:
            True if the model family can be evaluated
        """
        if model_family not in EVALUATION_MODULES:
            return False
        
        config = EVALUATION_MODULES[model_family]
        required_api = config.get('requires_api')
        
        if required_api and not self.available_apis.get(required_api, False):
            logger.warning(f"Cannot evaluate {model_family}: {required_api} API key not available")
            return False
        
        return True
    
    def run_model_evaluation(
        self, 
        model_family: str, 
        model_name: str, 
        mode: str,
        material_type: str = "expert_reviewed"
    ) -> Dict[str, Any]:
        """
        Run evaluation for a specific model.
        
        Args:
            model_family: Family of the model (openai, gemini, llava)
            model_name: Specific model name
            mode: Evaluation mode
            material_type: Type of supporting materials
            
        Returns:
            Evaluation results
        """
        try:
            # Dynamic import of the evaluator
            if model_family == 'openai':
                from scripts.evaluate_openai import OpenAIModelEvaluator
                evaluator = OpenAIModelEvaluator()
            elif model_family == 'gemini':
                from scripts.evaluate_gemini import GeminiModelEvaluator
                evaluator = GeminiModelEvaluator()
            elif model_family == 'llava':
                from scripts.evaluate_llava import LLaVAModelEvaluator
                evaluator = LLaVAModelEvaluator()
            else:
                raise ValueError(f"Unknown model family: {model_family}")
            
            # Run evaluation
            results = evaluator.evaluate_model(
                model_key=model_name,
                mode=mode,
                material_type=material_type,
                save_results=True
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_family}/{model_name} in {mode} mode: {e}")
            return {"error": str(e)}
    
    def run_batch_evaluation(
        self,
        model_families: List[str] = None,
        models_per_family: int = 3,
        modes: List[str] = None,
        material_type: str = "expert_reviewed"
    ) -> Dict[str, Any]:
        """
        Run batch evaluation across multiple models and modes.
        
        Args:
            model_families: List of model families to evaluate
            models_per_family: Number of models to evaluate per family
            modes: List of evaluation modes
            material_type: Type of supporting materials
            
        Returns:
            Summary of all evaluations
        """
        if model_families is None:
            model_families = list(EVALUATION_MODULES.keys())
        
        if modes is None:
            modes = ["closedbook", "openbook"]
        
        logger.info(f"Starting batch evaluation for {len(model_families)} model families")
        
        start_time = time.time()
        total_evaluations = 0
        successful_evaluations = 0
        
        for model_family in model_families:
            if not self.check_model_availability(model_family):
                continue
            
            config = EVALUATION_MODULES[model_family]
            models_to_test = config['models'][:models_per_family]
            
            logger.info(f"Evaluating {model_family} models: {models_to_test}")
            
            for model_name in models_to_test:
                for mode in modes:
                    total_evaluations += 1
                    
                    logger.info(f"Evaluating {model_family}/{model_name} in {mode} mode")
                    
                    result = self.run_model_evaluation(
                        model_family=model_family,
                        model_name=model_name,
                        mode=mode,
                        material_type=material_type
                    )
                    
                    # Store result
                    key = f"{model_family}_{model_name}_{mode}"
                    self.results[key] = result
                    
                    if "error" not in result:
                        successful_evaluations += 1
                        logger.info(f"✓ {key}: {result.get('accuracy', 0):.2%}")
                    else:
                        logger.error(f"✗ {key}: {result['error']}")
        
        elapsed_time = time.time() - start_time
        
        # Generate summary
        summary = {
            "batch_evaluation_summary": {
                "total_evaluations": total_evaluations,
                "successful_evaluations": successful_evaluations,
                "failed_evaluations": total_evaluations - successful_evaluations,
                "success_rate": successful_evaluations / total_evaluations if total_evaluations > 0 else 0,
                "elapsed_time_seconds": elapsed_time,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            },
            "model_families_evaluated": model_families,
            "modes_evaluated": modes,
            "material_type": material_type,
            "results": self.results
        }
        
        # Save summary
        summary_path = self.output_dir / "batch_evaluation_summary.json"
        save_results(summary, str(summary_path))
        
        logger.info(f"Batch evaluation completed in {elapsed_time:.1f}s")
        logger.info(f"Success rate: {successful_evaluations}/{total_evaluations} ({successful_evaluations/total_evaluations:.1%})")
        
        return summary
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate a comparison report from evaluation results.
        
        Returns:
            Comparison report
        """
        if not self.results:
            logger.warning("No results available for comparison")
            return {}
        
        # Extract performance data
        performance_data = []
        
        for key, result in self.results.items():
            if "error" in result:
                continue
            
            parts = key.split('_')
            if len(parts) >= 3:
                model_family = parts[0]
                model_name = '_'.join(parts[1:-1])
                mode = parts[-1]
                
                performance_data.append({
                    'model_family': model_family,
                    'model_name': model_name,
                    'mode': mode,
                    'accuracy': result.get('accuracy', 0),
                    'total_questions': result.get('total_questions', 0),
                    'correct_answers': result.get('correct_answers', 0)
                })
        
        if not performance_data:
            return {}
        
        # Best performers by mode
        best_performers = {}
        for mode in ['closedbook', 'openbook']:
            mode_data = [p for p in performance_data if p['mode'] == mode]
            if mode_data:
                best_performers[mode] = sorted(mode_data, key=lambda x: x['accuracy'], reverse=True)[:5]
        
        # Model family comparison
        family_stats = {}
        for family in set(p['model_family'] for p in performance_data):
            family_data = [p for p in performance_data if p['model_family'] == family]
            if family_data:
                accuracies = [p['accuracy'] for p in family_data]
                family_stats[family] = {
                    'mean_accuracy': sum(accuracies) / len(accuracies),
                    'max_accuracy': max(accuracies),
                    'min_accuracy': min(accuracies),
                    'num_evaluations': len(family_data)
                }
        
        # Mode improvements
        mode_improvements = []
        models_evaluated = set((p['model_family'], p['model_name']) for p in performance_data)
        
        for family, model in models_evaluated:
            closed_result = next((p for p in performance_data 
                                if p['model_family'] == family and p['model_name'] == model and p['mode'] == 'closedbook'), None)
            open_result = next((p for p in performance_data 
                              if p['model_family'] == family and p['model_name'] == model and p['mode'] == 'openbook'), None)
            
            if closed_result and open_result:
                improvement = open_result['accuracy'] - closed_result['accuracy']
                mode_improvements.append({
                    'model_family': family,
                    'model_name': model,
                    'closedbook_accuracy': closed_result['accuracy'],
                    'openbook_accuracy': open_result['accuracy'],
                    'improvement': improvement
                })
        
        mode_improvements.sort(key=lambda x: x['improvement'], reverse=True)
        
        report = {
            'best_performers_by_mode': best_performers,
            'model_family_statistics': family_stats,
            'mode_improvements': mode_improvements[:10],  # Top 10
            'total_models_compared': len(models_evaluated),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save comparison report
        report_path = self.output_dir / "comparison_report.json"
        save_results(report, str(report_path))
        
        return report


def main():
    """Main batch evaluation script."""
    parser = argparse.ArgumentParser(description="Run batch evaluation on OpenXRD benchmark")
    parser.add_argument("--model_families", nargs='+', 
                        choices=list(EVALUATION_MODULES.keys()) + ['all'],
                        default=['all'], help="Model families to evaluate")
    parser.add_argument("--models_per_family", type=int, default=3,
                        help="Number of models to evaluate per family")
    parser.add_argument("--modes", nargs='+', choices=['closedbook', 'openbook', 'both'],
                        default=['both'], help="Evaluation modes")
    parser.add_argument("--material_type", type=str, choices=["generated", "expert_reviewed"],
                        default="expert_reviewed", help="Type of supporting materials")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--comparison_only", action="store_true", 
                        help="Skip evaluation and only generate comparison report")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Process arguments
    if 'all' in args.model_families:
        model_families = list(EVALUATION_MODULES.keys())
    else:
        model_families = args.model_families
    
    if 'both' in args.modes:
        modes = ['closedbook', 'openbook']
    else:
        modes = args.modes
    
    # Initialize batch evaluator
    evaluator = BatchEvaluator(args.output_dir)
    
    try:
        if not args.comparison_only:
            # Run batch evaluation
            logger.info("Starting batch evaluation...")
            summary = evaluator.run_batch_evaluation(
                model_families=model_families,
                models_per_family=args.models_per_family,
                modes=modes,
                material_type=args.material_type
            )
            
            # Print summary
            print("\n" + "="*60)
            print("BATCH EVALUATION SUMMARY")
            print("="*60)
            print(f"Total evaluations: {summary['batch_evaluation_summary']['total_evaluations']}")
            print(f"Successful: {summary['batch_evaluation_summary']['successful_evaluations']}")
            print(f"Failed: {summary['batch_evaluation_summary']['failed_evaluations']}")
            print(f"Success rate: {summary['batch_evaluation_summary']['success_rate']:.1%}")
            print(f"Time elapsed: {summary['batch_evaluation_summary']['elapsed_time_seconds']:.1f}s")
            print("="*60)
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        comparison = evaluator.generate_comparison_report()
        
        if comparison:
            print("\nTOP 3 PERFORMERS BY MODE:")
            for mode in ['closedbook', 'openbook']:
                if mode in comparison.get('best_performers_by_mode', {}):
                    print(f"\n{mode.upper()}:")
                    for i, performer in enumerate(comparison['best_performers_by_mode'][mode][:3], 1):
                        print(f"  {i}. {performer['model_family']}/{performer['model_name']}: {performer['accuracy']:.2%}")
            
            if comparison.get('mode_improvements'):
                print("\nTOP 3 OPEN-BOOK IMPROVEMENTS:")
                for i, improvement in enumerate(comparison['mode_improvements'][:3], 1):
                    print(f"  {i}. {improvement['model_family']}/{improvement['model_name']}: +{improvement['improvement']:.2%}")
        
        logger.info("Batch evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Batch evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
