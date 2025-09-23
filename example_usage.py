#!/usr/bin/env python3
"""
Basic usage example for OpenXRD.
This script demonstrates how to use the OpenXRD evaluation framework.
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation import OpenXRDEvaluator
from src.utils import validate_api_keys, print_evaluation_summary


def mock_model_function(prompt: str) -> str:
    """
    Mock model function for demonstration purposes.
    In practice, this would call your actual model.
    
    Args:
        prompt: Input prompt for the model
        
    Returns:
        Mock response (always returns "1" for first option)
    """
    # This is a very simple mock that always returns "1"
    # Replace this with actual model inference
    return "1"


def openai_model_function(prompt: str) -> str:
    """
    Example OpenAI model function.
    Requires OPENAI_API_KEY to be set.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant who answers crystallography questions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.2
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "1"  # Fallback


def main():
    """Main example function."""
    print("OpenXRD Basic Usage Example")
    print("=" * 50)
    
    # Check available APIs
    api_status = validate_api_keys()
    print(f"API Status: {api_status}")
    
    # Initialize the evaluator
    evaluator = OpenXRDEvaluator()
    
    # Get dataset information
    dataset_info = evaluator.get_dataset_info()
    print(f"\nDataset Info:")
    print(f"Total questions: {dataset_info['total_questions']}")
    print(f"Categories: {dataset_info['num_categories']}")
    print(f"Subtasks: {dataset_info['num_subtasks']}")
    print(f"Has expert materials: {dataset_info['has_expert_materials']}")
    
    # Example 1: Evaluate with mock model (closed-book)
    print(f"\n1. Mock Model Evaluation (Closed-book)")
    print("-" * 40)
    
    results_mock = evaluator.evaluate_model(
        model_function=mock_model_function,
        model_name="mock_model",
        mode="closedbook",
        verbose=False  # Set to True to see individual question results
    )
    
    print(f"Mock model accuracy: {results_mock['accuracy']:.2%}")
    
    # Example 2: Evaluate with mock model (open-book)
    print(f"\n2. Mock Model Evaluation (Open-book)")
    print("-" * 40)
    
    results_mock_openbook = evaluator.evaluate_model(
        model_function=mock_model_function,
        model_name="mock_model",
        mode="openbook",
        material_type="expert_reviewed",
        verbose=False
    )
    
    print(f"Mock model accuracy (open-book): {results_mock_openbook['accuracy']:.2%}")
    improvement = results_mock_openbook['accuracy'] - results_mock['accuracy']
    print(f"Improvement: {improvement:+.2%}")
    
    # Example 3: OpenAI model (if API key available)
    if api_status.get('openai', False):
        print(f"\n3. OpenAI Model Evaluation")
        print("-" * 40)
        
        try:
            # Evaluate just a few questions for the example
            # Modify the evaluator to limit questions for demo
            original_questions = evaluator.questions
            evaluator.questions = original_questions[:5]  # Only first 5 questions
            
            results_openai = evaluator.evaluate_model(
                model_function=openai_model_function,
                model_name="gpt-3.5-turbo",
                mode="closedbook",
                verbose=True
            )
            
            print(f"OpenAI model accuracy (5 questions): {results_openai['accuracy']:.2%}")
            
            # Restore original questions
            evaluator.questions = original_questions
            
        except Exception as e:
            print(f"OpenAI evaluation failed: {e}")
    else:
        print(f"\n3. OpenAI Model Evaluation")
        print("-" * 40)
        print("Skipped - OpenAI API key not available")
    
    # Example 4: Mode comparison
    print(f"\n4. Mode Comparison Example")
    print("-" * 40)
    
    comparison = evaluator.compare_modes(
        model_function=mock_model_function,
        model_name="mock_model_comparison",
        material_type="expert_reviewed",
        save_results=False
    )
    
    # Example 5: Show dataset statistics
    print(f"\n5. Dataset Statistics")
    print("-" * 40)
    
    print("Top 5 categories by question count:")
    for i, (category, count) in enumerate(list(dataset_info['categories'].items())[:5], 1):
        print(f"  {i}. {category}: {count} questions")
    
    print("\nTop 5 subtasks by question count:")
    for i, (subtask, count) in enumerate(list(dataset_info['subtasks'].items())[:5], 1):
        print(f"  {i}. {subtask}: {count} questions")
    
    print(f"\n6. Sample Questions")
    print("-" * 40)
    
    # Show first 3 questions as examples
    for i, question in enumerate(evaluator.questions[:3]):
        print(f"\nQuestion {i+1}:")
        print(f"Category: {question.get('category', 'N/A')}")
        print(f"Subtask: {question.get('subtask', 'N/A')}")
        print(f"Q: {question['question']}")
        print("Options:")
        for j, option in enumerate(question['options']):
            marker = ">" if j == question['correct_answer'] else " "
            print(f"  {marker} {j+1}. {option}")
    
    print(f"\nExample completed! Check the 'results' directory for detailed output files.")


if __name__ == "__main__":
    main()
