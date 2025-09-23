#!/usr/bin/env python3
"""
Subtask analysis script for OpenXRD.
Analyzes model performance across different crystallography subtasks.
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_dataset, save_results, NumpyEncoder

logger = logging.getLogger(__name__)


def load_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """
    Load all evaluation result files from a directory.
    
    Args:
        results_dir: Directory containing result JSON files
        
    Returns:
        List of evaluation results
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return results
    
    for file_path in results_path.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
                
            # Add filename for reference
            result['source_file'] = file_path.name
            results.append(result)
            logger.info(f"Loaded results from {file_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    return results


def analyze_subtask_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze performance across subtasks for all models.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing subtask analysis
    """
    # Collect all subtask data
    subtask_data = []
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        mode = result.get('mode', 'unknown')
        
        if 'subtask_metrics' in result and result['subtask_metrics']:
            for subtask, metrics in result['subtask_metrics'].items():
                subtask_data.append({
                    'model_name': model_name,
                    'mode': mode,
                    'subtask': subtask,
                    'accuracy': metrics['accuracy'],
                    'correct': metrics['correct'],
                    'total': metrics['count']
                })
    
    if not subtask_data:
        logger.warning("No subtask data found in results")
        return {}
    
    # Create DataFrame for analysis
    df = pd.DataFrame(subtask_data)
    
    # Overall subtask statistics
    subtask_stats = df.groupby('subtask').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'total': 'first'  # Number of questions per subtask
    }).round(4)
    
    subtask_stats.columns = ['mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy', 'num_questions']
    subtask_stats = subtask_stats.sort_values('mean_accuracy', ascending=False)
    
    # Model performance by subtask
    model_subtask_pivot = df.pivot_table(
        index='subtask', 
        columns=['model_name', 'mode'], 
        values='accuracy',
        aggfunc='first'
    ).round(4)
    
    # Best and worst performing subtasks
    best_subtasks = subtask_stats.head(5).to_dict('index')
    worst_subtasks = subtask_stats.tail(5).to_dict('index')
    
    # Mode comparison (closed-book vs open-book)
    mode_comparison = {}
    if 'closedbook' in df['mode'].values and 'openbook' in df['mode'].values:
        for subtask in df['subtask'].unique():
            subtask_df = df[df['subtask'] == subtask]
            closedbook_acc = subtask_df[subtask_df['mode'] == 'closedbook']['accuracy'].mean()
            openbook_acc = subtask_df[subtask_df['mode'] == 'openbook']['accuracy'].mean()
            
            if pd.notna(closedbook_acc) and pd.notna(openbook_acc):
                mode_comparison[subtask] = {
                    'closedbook_accuracy': float(closedbook_acc),
                    'openbook_accuracy': float(openbook_acc),
                    'improvement': float(openbook_acc - closedbook_acc)
                }
    
    return {
        'subtask_statistics': subtask_stats.to_dict('index'),
        'best_performing_subtasks': best_subtasks,
        'worst_performing_subtasks': worst_subtasks,
        'mode_comparison': mode_comparison,
        'model_subtask_performance': model_subtask_pivot.to_dict(),
        'total_subtasks': len(df['subtask'].unique()),
        'total_models_analyzed': len(df[['model_name', 'mode']].drop_duplicates())
    }


def analyze_category_performance(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze performance across categories for all models.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing category analysis
    """
    # Collect all category data
    category_data = []
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        mode = result.get('mode', 'unknown')
        
        if 'category_metrics' in result and result['category_metrics']:
            for category, metrics in result['category_metrics'].items():
                category_data.append({
                    'model_name': model_name,
                    'mode': mode,
                    'category': category,
                    'accuracy': metrics['accuracy'],
                    'correct': metrics['correct'],
                    'total': metrics['count']
                })
    
    if not category_data:
        logger.warning("No category data found in results")
        return {}
    
    # Create DataFrame for analysis
    df = pd.DataFrame(category_data)
    
    # Overall category statistics
    category_stats = df.groupby('category').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'total': 'first'
    }).round(4)
    
    category_stats.columns = ['mean_accuracy', 'std_accuracy', 'min_accuracy', 'max_accuracy', 'num_questions']
    category_stats = category_stats.sort_values('mean_accuracy', ascending=False)
    
    return {
        'category_statistics': category_stats.to_dict('index'),
        'total_categories': len(df['category'].unique())
    }


def generate_model_comparison(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate model comparison analysis.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary containing model comparison
    """
    model_data = []
    
    for result in results:
        model_data.append({
            'model_name': result.get('model_name', 'unknown'),
            'mode': result.get('mode', 'unknown'),
            'accuracy': result.get('accuracy', 0),
            'total_questions': result.get('total_questions', 0),
            'correct_answers': result.get('correct_answers', 0)
        })
    
    df = pd.DataFrame(model_data)
    
    # Best performing models by mode
    best_models = {}
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode].sort_values('accuracy', ascending=False)
        best_models[mode] = mode_df.head(5).to_dict('records')
    
    # Mode improvement analysis
    mode_improvements = {}
    for model in df['model_name'].unique():
        model_df = df[df['model_name'] == model]
        if len(model_df) >= 2:  # Has both modes
            closedbook = model_df[model_df['mode'] == 'closedbook']['accuracy'].values
            openbook = model_df[model_df['mode'] == 'openbook']['accuracy'].values
            
            if len(closedbook) > 0 and len(openbook) > 0:
                improvement = float(openbook[0] - closedbook[0])
                mode_improvements[model] = {
                    'closedbook_accuracy': float(closedbook[0]),
                    'openbook_accuracy': float(openbook[0]),
                    'improvement': improvement
                }
    
    # Sort by improvement
    mode_improvements = dict(sorted(
        mode_improvements.items(), 
        key=lambda x: x[1]['improvement'], 
        reverse=True
    ))
    
    return {
        'best_models_by_mode': best_models,
        'mode_improvements': mode_improvements,
        'total_models': len(df['model_name'].unique())
    }


def generate_summary_report(
    subtask_analysis: Dict[str, Any],
    category_analysis: Dict[str, Any],
    model_comparison: Dict[str, Any],
    output_file: str
) -> None:
    """
    Generate a comprehensive summary report.
    
    Args:
        subtask_analysis: Subtask analysis results
        category_analysis: Category analysis results
        model_comparison: Model comparison results
        output_file: Path to save the report
    """
    report = {
        'analysis_summary': {
            'total_subtasks_analyzed': subtask_analysis.get('total_subtasks', 0),
            'total_categories_analyzed': category_analysis.get('total_categories', 0),
            'total_models_analyzed': model_comparison.get('total_models', 0)
        },
        'subtask_analysis': subtask_analysis,
        'category_analysis': category_analysis,
        'model_comparison': model_comparison,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    save_results(report, output_file)
    logger.info(f"Summary report saved to {output_file}")


def main():
    """Main analysis script."""
    parser = argparse.ArgumentParser(description="Analyze OpenXRD evaluation results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing evaluation results")
    parser.add_argument("--output_file", type=str, default="results/subtask_analysis.json",
                        help="Output file for analysis results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load evaluation results
    logger.info(f"Loading evaluation results from {args.results_dir}")
    results = load_evaluation_results(args.results_dir)
    
    if not results:
        logger.error("No evaluation results found")
        return 1
    
    logger.info(f"Loaded {len(results)} evaluation results")
    
    # Perform analyses
    logger.info("Analyzing subtask performance...")
    subtask_analysis = analyze_subtask_performance(results)
    
    logger.info("Analyzing category performance...")
    category_analysis = analyze_category_performance(results)
    
    logger.info("Generating model comparison...")
    model_comparison = generate_model_comparison(results)
    
    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(
        subtask_analysis, 
        category_analysis, 
        model_comparison, 
        args.output_file
    )
    
    # Print key findings
    print("\n" + "="*60)
    print("OPENXRD ANALYSIS SUMMARY")
    print("="*60)
    
    if subtask_analysis.get('best_performing_subtasks'):
        print("\nTop 3 Best Performing Subtasks:")
        for i, (subtask, stats) in enumerate(list(subtask_analysis['best_performing_subtasks'].items())[:3], 1):
            print(f"{i}. {subtask}: {stats['mean_accuracy']:.2%} (avg)")
    
    if subtask_analysis.get('worst_performing_subtasks'):
        print("\nTop 3 Most Challenging Subtasks:")
        for i, (subtask, stats) in enumerate(list(subtask_analysis['worst_performing_subtasks'].items())[:3], 1):
            print(f"{i}. {subtask}: {stats['mean_accuracy']:.2%} (avg)")
    
    if model_comparison.get('mode_improvements'):
        print("\nTop 3 Models with Best Open-book Improvement:")
        for i, (model, stats) in enumerate(list(model_comparison['mode_improvements'].items())[:3], 1):
            print(f"{i}. {model}: +{stats['improvement']:.2%}")
    
    print("="*60)
    
    logger.info("Analysis completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())
