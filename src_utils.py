"""
Utility functions for OpenXRD project.
"""

import json
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy objects."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_dataset(dataset_path: str = "datasets/benchmarking_questions.json") -> List[Dict[str, Any]]:
    """
    Load the XRD questions dataset.
    
    Args:
        dataset_path: Path to the JSON file containing questions
        
    Returns:
        List of question dictionaries
    """
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        logger.info(f"Loaded {len(questions)} questions from {dataset_path}")
        return questions
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {dataset_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {dataset_path}: {e}")
        raise


def load_supporting_materials(
    material_type: str = "expert_reviewed"
) -> Optional[List[Dict[str, Any]]]:
    """
    Load supporting textual materials for open-book evaluation.
    
    Args:
        material_type: Either "generated" or "expert_reviewed"
        
    Returns:
        List of supporting material dictionaries or None if not found
    """
    if material_type == "expert_reviewed":
        path = "datasets/supporting_textual_materials_expert_reviewed.json"
    elif material_type == "generated":
        path = "datasets/supporting_textual_materials_generated.json"
    else:
        raise ValueError("material_type must be 'generated' or 'expert_reviewed'")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            materials = json.load(f)
        logger.info(f"Loaded {len(materials)} supporting materials from {path}")
        return materials
    except FileNotFoundError:
        logger.warning(f"Supporting materials file not found: {path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {path}: {e}")
        raise


def save_results(
    results: Dict[str, Any], 
    output_path: str,
    create_dirs: bool = True
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Dictionary containing evaluation results
        output_path: Path where to save the results
        create_dirs: Whether to create directories if they don't exist
    """
    if create_dirs:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=4)
    
    logger.info(f"Results saved to {output_path}")


def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from results.
    
    Args:
        results: List of question results
        
    Returns:
        Dictionary containing metrics
    """
    if not results:
        return {}
    
    correct_answers = sum(1 for r in results if r.get('is_correct', False))
    total_questions = len(results)
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Category-specific metrics
    category_metrics = {}
    if any('category' in r for r in results):
        df = pd.DataFrame(results)
        if 'category' in df.columns and 'is_correct' in df.columns:
            category_stats = df.groupby('category')['is_correct'].agg(['mean', 'count', 'sum'])
            category_metrics = {
                cat: {
                    'accuracy': float(row['mean']),
                    'count': int(row['count']),
                    'correct': int(row['sum'])
                }
                for cat, row in category_stats.iterrows()
            }
    
    # Subtask-specific metrics
    subtask_metrics = {}
    if any('subtask' in r for r in results):
        df = pd.DataFrame(results)
        if 'subtask' in df.columns and 'is_correct' in df.columns:
            subtask_stats = df.groupby('subtask')['is_correct'].agg(['mean', 'count', 'sum'])
            subtask_metrics = {
                subtask: {
                    'accuracy': float(row['mean']),
                    'count': int(row['count']),
                    'correct': int(row['sum'])
                }
                for subtask, row in subtask_stats.iterrows()
            }
    
    return {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy,
        'category_metrics': category_metrics,
        'subtask_metrics': subtask_metrics
    }


def format_prompt(
    question: str,
    options: List[str],
    supporting_material: Optional[str] = None
) -> str:
    """
    Format a prompt for model evaluation.
    
    Args:
        question: The question text
        options: List of answer options
        supporting_material: Optional supporting text for open-book mode
        
    Returns:
        Formatted prompt string
    """
    if supporting_material:
        prompt = f"""You have a reference text (cheatsheet) to help you answer the question:

REFERENCE MATERIAL:
{supporting_material}

Now, please analyze the following question and choose the single correct answer.

Question: {question}

Options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

Respond with ONLY the number (1 to {len(options)}) of the correct answer."""
    else:
        prompt = f"""Question: {question}

Options:
{chr(10).join(f"{i+1}. {opt}" for i, opt in enumerate(options))}

Please analyze the question and options carefully, then respond with ONLY the number (1 to {len(options)}) of the correct answer."""
    
    return prompt


def parse_model_response(response: str, num_options: int) -> int:
    """
    Parse model response to extract the chosen option index.
    
    Args:
        response: Raw model response
        num_options: Number of available options
        
    Returns:
        0-based index of chosen option (0 if parsing fails)
    """
    # Extract the first digit from the response
    digits = [int(d) for d in response if d.isdigit()]
    if digits:
        idx = digits[0] - 1  # Convert to 0-based index
        if 0 <= idx < num_options:
            return idx
    
    # Fallback to first option if parsing fails
    logger.warning(f"Could not parse model response: {response}")
    return 0


def get_output_path(model_name: str, mode: str, base_dir: str = None) -> str:
    """
    Generate standardized output path for results.
    
    Args:
        model_name: Name of the model
        mode: Evaluation mode (closedbook/openbook)
        base_dir: Base directory for results
        
    Returns:
        Path string for saving results
    """
    if base_dir is None:
        base_dir = os.getenv('DEFAULT_OUTPUT_DIR', 'results')
    
    # Clean model name for filename
    clean_name = model_name.replace('/', '_').replace('-', '_')
    filename = f"{clean_name}_{mode}_results.json"
    
    return os.path.join(base_dir, filename)


def validate_api_keys() -> Dict[str, bool]:
    """
    Validate that required API keys are available.
    
    Returns:
        Dictionary indicating which API keys are available
    """
    return {
        'openai': bool(os.getenv('OPENAI_API_KEY')),
        'gemini': bool(os.getenv('GEMINI_API_KEY')),
        'huggingface': bool(os.getenv('HUGGINGFACE_TOKEN'))
    }


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: {results.get('model_name', 'Unknown')}")
    print(f"Mode: {results.get('mode', 'Unknown')}")
    print(f"Accuracy: {results.get('accuracy', 0):.2%}")
    print(f"Correct: {results.get('correct_answers', 0)}/{results.get('total_questions', 0)}")
    
    if 'category_metrics' in results and results['category_metrics']:
        print("\nCategory Performance:")
        for category, metrics in results['category_metrics'].items():
            print(f"  {category}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['count']})")
    
    if 'subtask_metrics' in results and results['subtask_metrics']:
        print("\nTop 5 Subtask Performance:")
        sorted_subtasks = sorted(
            results['subtask_metrics'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )[:5]
        for subtask, metrics in sorted_subtasks:
            print(f"  {subtask}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['count']})")
    
    print("="*50)
