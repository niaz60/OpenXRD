"""
Main evaluation framework for OpenXRD.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from .utils import (
    load_dataset, load_supporting_materials, calculate_metrics, 
    format_prompt, parse_model_response, save_results,
    print_evaluation_summary
)

logger = logging.getLogger(__name__)


class OpenXRDEvaluator:
    """Main evaluator class for OpenXRD benchmark."""
    
    def __init__(self, dataset_path: str = "datasets/benchmarking_questions.json"):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the questions dataset
        """
        self.questions = load_dataset(dataset_path)
        self.supporting_materials_generated = load_supporting_materials("generated")
        self.supporting_materials_expert = load_supporting_materials("expert_reviewed")
        
    def evaluate_model(
        self,
        model_function: Callable[[str], str],
        model_name: str,
        mode: str = "closedbook",
        material_type: str = "expert_reviewed",
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a model on the XRD benchmark.
        
        Args:
            model_function: Function that takes a prompt and returns a response
            model_name: Name of the model being evaluated
            mode: Either "closedbook" or "openbook"
            material_type: Either "generated" or "expert_reviewed" (for openbook mode)
            save_path: Path to save results (optional)
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Starting evaluation: {model_name} in {mode} mode")
        
        # Select supporting materials for openbook mode
        supporting_materials = None
        if mode == "openbook":
            if material_type == "expert_reviewed":
                supporting_materials = self.supporting_materials_expert
            else:
                supporting_materials = self.supporting_materials_generated
                
            if supporting_materials is None:
                logger.warning(f"No {material_type} supporting materials found. Falling back to closedbook mode.")
                mode = "closedbook"
        
        results = []
        correct_count = 0
        total_questions = len(self.questions)
        
        for i, question in enumerate(self.questions):
            try:
                # Get supporting material if in openbook mode
                supporting_material = None
                if mode == "openbook" and supporting_materials and i < len(supporting_materials):
                    supporting_material = supporting_materials[i].get("helper_text", "")
                
                # Format prompt
                prompt = format_prompt(
                    question["question"],
                    question["options"],
                    supporting_material
                )
                
                # Get model response
                response = model_function(prompt)
                
                # Parse response
                model_answer = parse_model_response(response, len(question["options"]))
                is_correct = model_answer == question["correct_answer"]
                
                if is_correct:
                    correct_count += 1
                
                # Store result
                result = {
                    "question_id": i,
                    "question": question["question"],
                    "options": question["options"],
                    "model_answer": model_answer,
                    "correct_answer": question["correct_answer"],
                    "is_correct": is_correct,
                    "category": question.get("category", "N/A"),
                    "subtask": question.get("subtask", "N/A"),
                    "model_response": response[:200] + "..." if len(response) > 200 else response
                }
                results.append(result)
                
                # Log progress
                if verbose:
                    current_accuracy = correct_count / (i + 1)
                    status = "Correct" if is_correct else "Incorrect"
                    logger.info(
                        f"Question {i+1}/{total_questions} [{mode.upper()}]: "
                        f"{status} | Running Accuracy: {current_accuracy:.2%}"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                # Add a failed result to maintain indexing
                result = {
                    "question_id": i,
                    "question": question["question"],
                    "options": question["options"],
                    "model_answer": 0,  # Default to first option
                    "correct_answer": question["correct_answer"],
                    "is_correct": False,
                    "category": question.get("category", "N/A"),
                    "subtask": question.get("subtask", "N/A"),
                    "error": str(e)
                }
                results.append(result)
                continue
        
        # Calculate metrics
        metrics = calculate_metrics(results)
        
        # Compile final results
        final_results = {
            "model_name": model_name,
            "mode": mode,
            "material_type": material_type if mode == "openbook" else None,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "results": results,
            **metrics
        }
        
        # Save results if path provided
        if save_path:
            save_results(final_results, save_path)
        
        # Print summary if verbose
        if verbose:
            print_evaluation_summary(final_results)
        
        logger.info(f"Evaluation complete: {model_name} achieved {metrics['accuracy']:.2%} accuracy")
        
        return final_results
    
    def compare_modes(
        self,
        model_function: Callable[[str], str],
        model_name: str,
        material_type: str = "expert_reviewed",
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Compare closedbook vs openbook performance for a model.
        
        Args:
            model_function: Function that takes a prompt and returns a response
            model_name: Name of the model being evaluated
            material_type: Type of supporting materials for openbook mode
            save_results: Whether to save individual results
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Running mode comparison for {model_name}")
        
        # Run closedbook evaluation
        closedbook_results = self.evaluate_model(
            model_function=model_function,
            model_name=model_name,
            mode="closedbook",
            save_path=f"results/{model_name.replace('/', '_')}_closedbook_results.json" if save_results else None,
            verbose=False
        )
        
        # Run openbook evaluation
        openbook_results = self.evaluate_model(
            model_function=model_function,
            model_name=model_name,
            mode="openbook",
            material_type=material_type,
            save_path=f"results/{model_name.replace('/', '_')}_openbook_{material_type}_results.json" if save_results else None,
            verbose=False
        )
        
        # Calculate improvement
        improvement = openbook_results["accuracy"] - closedbook_results["accuracy"]
        
        comparison = {
            "model_name": model_name,
            "material_type": material_type,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "closedbook_accuracy": closedbook_results["accuracy"],
            "openbook_accuracy": openbook_results["accuracy"],
            "improvement": improvement,
            "improvement_percentage": improvement * 100,
            "closedbook_results": closedbook_results,
            "openbook_results": openbook_results
        }
        
        # Print comparison summary
        print(f"\n{'='*60}")
        print(f"MODE COMPARISON: {model_name}")
        print(f"{'='*60}")
        print(f"Closed-book accuracy: {closedbook_results['accuracy']:.2%}")
        print(f"Open-book accuracy:   {openbook_results['accuracy']:.2%}")
        print(f"Improvement:          {improvement:+.2%}")
        print(f"{'='*60}\n")
        
        # Save comparison results
        if save_results:
            comparison_path = f"results/{model_name.replace('/', '_')}_mode_comparison.json"
            save_results(comparison, comparison_path)
        
        return comparison
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        total_questions = len(self.questions)
        
        # Count by category
        categories = {}
        subtasks = {}
        
        for q in self.questions:
            cat = q.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            sub = q.get("subtask", "Unknown")
            subtasks[sub] = subtasks.get(sub, 0) + 1
        
        return {
            "total_questions": total_questions,
            "num_categories": len(categories),
            "num_subtasks": len(subtasks),
            "categories": dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)),
            "subtasks": dict(sorted(subtasks.items(), key=lambda x: x[1], reverse=True)),
            "has_generated_materials": self.supporting_materials_generated is not None,
            "has_expert_materials": self.supporting_materials_expert is not None
        }


def evaluate_with_retry(
    model_function: Callable[[str], str],
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> str:
    """
    Evaluate a model with retry logic for robustness.
    
    Args:
        model_function: Function that takes a prompt and returns a response
        prompt: The input prompt
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        
    Returns:
        Model response string
    """
    for attempt in range(max_retries + 1):
        try:
            return model_function(prompt)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Model function failed after {max_retries} retries: {e}")
                raise
            else:
                logger.warning(f"Model function failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(retry_delay)
    
    return ""  # Should never reach here
