#!/usr/bin/env python3
"""
Visualization script for OpenXRD results.
Generates plots and word clouds from evaluation results.
"""

import os
import sys
import json
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_dataset

logger = logging.getLogger(__name__)

# Check if optional dependencies are available
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.warning("WordCloud not available. Install with: pip install wordcloud")


def load_evaluation_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all evaluation result files from a directory."""
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        logger.warning(f"Results directory does not exist: {results_dir}")
        return results
    
    for file_path in results_path.glob("*_results.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            result['source_file'] = file_path.name
            results.append(result)
            logger.info(f"Loaded results from {file_path.name}")
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
    
    return results


def generate_subtask_wordcloud(
    results: List[Dict[str, Any]], 
    model_name: str = None,
    mode: str = None,
    output_path: str = "subtask_wordcloud.png"
) -> None:
    """
    Generate a word cloud of subtasks based on question frequency.
    
    Args:
        results: List of evaluation results
        model_name: Specific model to filter by (optional)
        mode: Specific mode to filter by (optional)
        output_path: Path to save the word cloud
    """
    if not WORDCLOUD_AVAILABLE:
        logger.error("WordCloud not available. Cannot generate word cloud.")
        return
    
    # Filter results if specified
    filtered_results = results
    if model_name:
        filtered_results = [r for r in filtered_results if r.get('model_name') == model_name]
    if mode:
        filtered_results = [r for r in filtered_results if r.get('mode') == mode]
    
    if not filtered_results:
        logger.warning("No matching results found for word cloud generation")
        return
    
    # Collect subtask frequencies
    subtask_counts = {}
    for result in filtered_results:
        if 'subtask_metrics' in result:
            for subtask, metrics in result['subtask_metrics'].items():
                subtask_counts[subtask] = metrics.get('total', 1)
    
    if not subtask_counts:
        logger.warning("No subtask data found for word cloud generation")
        return
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200, 
        height=800, 
        background_color='white',
        max_words=100,
        colormap='viridis'
    ).generate_from_frequencies(subtask_counts)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    title = "XRD Subtasks Word Cloud"
    if model_name:
        title += f" - {model_name}"
    if mode:
        title += f" ({mode})"
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Word cloud saved to {output_path}")


def plot_model_comparison(
    results: List[Dict[str, Any]], 
    output_path: str = "model_comparison.png"
) -> None:
    """
    Create a bar plot comparing model accuracies.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
    """
    # Extract accuracy data
    accuracy_data = []
    for result in results:
        accuracy_data.append({
            'model_name': result.get('model_name', 'unknown'),
            'mode': result.get('mode', 'unknown'),
            'accuracy': result.get('accuracy', 0) * 100  # Convert to percentage
        })
    
    if not accuracy_data:
        logger.warning("No accuracy data found for plotting")
        return
    
    df = pd.DataFrame(accuracy_data)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Create grouped bar plot
    sns.barplot(data=df, x='model_name', y='accuracy', hue='mode', palette='Set2')
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Mode')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    ax = plt.gca()
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', rotation=90, padding=3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model comparison plot saved to {output_path}")


def plot_subtask_performance(
    results: List[Dict[str, Any]], 
    output_path: str = "subtask_performance.png",
    top_n: int = 15
) -> None:
    """
    Create a plot showing performance across different subtasks.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
        top_n: Number of top subtasks to show
    """
    # Collect subtask performance data
    subtask_data = []
    for result in results:
        model_name = result.get('model_name', 'unknown')
        mode = result.get('mode', 'unknown')
        
        if 'subtask_metrics' in result:
            for subtask, metrics in result['subtask_metrics'].items():
                subtask_data.append({
                    'model_name': model_name,
                    'mode': mode,
                    'subtask': subtask,
                    'accuracy': metrics['accuracy'] * 100,
                    'total_questions': metrics['total']
                })
    
    if not subtask_data:
        logger.warning("No subtask data found for plotting")
        return
    
    df = pd.DataFrame(subtask_data)
    
    # Calculate average accuracy per subtask and select top N
    avg_accuracy = df.groupby('subtask')['accuracy'].mean().sort_values(ascending=False)
    top_subtasks = avg_accuracy.head(top_n).index
    
    # Filter data to top subtasks
    df_filtered = df[df['subtask'].isin(top_subtasks)]
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Create box plot or violin plot
    sns.boxplot(data=df_filtered, x='subtask', y='accuracy', palette='Set3')
    
    plt.title(f'Performance Across Top {top_n} Subtasks', fontsize=16, fontweight='bold')
    plt.xlabel('Subtask', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Subtask performance plot saved to {output_path}")


def plot_mode_improvement(
    results: List[Dict[str, Any]], 
    output_path: str = "mode_improvement.png"
) -> None:
    """
    Create a plot showing improvement from closed-book to open-book mode.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
    """
    # Group results by model
    model_results = {}
    for result in results:
        model_name = result.get('model_name', 'unknown')
        mode = result.get('mode', 'unknown')
        accuracy = result.get('accuracy', 0)
        
        if model_name not in model_results:
            model_results[model_name] = {}
        model_results[model_name][mode] = accuracy
    
    # Calculate improvements
    improvements = []
    for model_name, modes in model_results.items():
        if 'closedbook' in modes and 'openbook' in modes:
            improvement = (modes['openbook'] - modes['closedbook']) * 100
            improvements.append({
                'model_name': model_name,
                'closedbook_accuracy': modes['closedbook'] * 100,
                'openbook_accuracy': modes['openbook'] * 100,
                'improvement': improvement
            })
    
    if not improvements:
        logger.warning("No paired data found for mode improvement analysis")
        return
    
    df = pd.DataFrame(improvements)
    df = df.sort_values('improvement', ascending=True)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    colors = ['red' if x < 0 else 'green' for x in df['improvement']]
    bars = plt.barh(df['model_name'], df['improvement'], color=colors, alpha=0.7)
    
    plt.title('Open-book vs Closed-book Performance Improvement', fontsize=16, fontweight='bold')
    plt.xlabel('Improvement (Percentage Points)', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, df['improvement'])):
        plt.text(value + (0.5 if value >= 0 else -0.5), bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', ha='left' if value >= 0 else 'right', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Mode improvement plot saved to {output_path}")


def generate_summary_plot(
    results: List[Dict[str, Any]], 
    output_path: str = "summary_dashboard.png"
) -> None:
    """
    Generate a comprehensive summary dashboard.
    
    Args:
        results: List of evaluation results
        output_path: Path to save the plot
    """
    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Extract data
    accuracy_data = []
    subtask_data = []
    
    for result in results:
        model_name = result.get('model_name', 'unknown')
        mode = result.get('mode', 'unknown')
        accuracy = result.get('accuracy', 0) * 100
        
        accuracy_data.append({
            'model_name': model_name,
            'mode': mode,
            'accuracy': accuracy
        })
        
        if 'subtask_metrics' in result:
            for subtask, metrics in result['subtask_metrics'].items():
                subtask_data.append({
                    'subtask': subtask,
                    'accuracy': metrics['accuracy'] * 100,
                    'total_questions': metrics['total']
                })
    
    # Plot 1: Model accuracy comparison
    if accuracy_data:
        df_acc = pd.DataFrame(accuracy_data)
        sns.barplot(data=df_acc, x='model_name', y='accuracy', hue='mode', ax=ax1)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Accuracy (%)')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Subtask difficulty distribution
    if subtask_data:
        df_sub = pd.DataFrame(subtask_data)
        avg_subtask_acc = df_sub.groupby('subtask')['accuracy'].mean().sort_values()
        avg_subtask_acc.tail(10).plot(kind='barh', ax=ax2)
        ax2.set_title('Top 10 Easiest Subtasks', fontweight='bold')
        ax2.set_xlabel('Average Accuracy (%)')
    
    # Plot 3: Accuracy distribution
    if accuracy_data:
        accuracies = [d['accuracy'] for d in accuracy_data]
        ax3.hist(accuracies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_title('Accuracy Distribution', fontweight='bold')
        ax3.set_xlabel('Accuracy (%)')
        ax3.set_ylabel('Frequency')
    
    # Plot 4: Mode comparison
    if accuracy_data:
        df_acc = pd.DataFrame(accuracy_data)
        mode_avg = df_acc.groupby('mode')['accuracy'].mean()
        if len(mode_avg) > 1:
            mode_avg.plot(kind='bar', ax=ax4, color=['lightcoral', 'lightgreen'])
            ax4.set_title('Average Performance by Mode', fontweight='bold')
            ax4.set_ylabel('Average Accuracy (%)')
            ax4.tick_params(axis='x', rotation=0)
    
    plt.suptitle('OpenXRD Evaluation Summary Dashboard', fontsize=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Summary dashboard saved to {output_path}")


def main():
    """Main visualization script."""
    parser = argparse.ArgumentParser(description="Generate visualizations from OpenXRD results")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing evaluation results")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--model_name", type=str, help="Filter by specific model name")
    parser.add_argument("--mode", type=str, choices=['closedbook', 'openbook'],
                        help="Filter by specific mode")
    parser.add_argument("--plots", nargs='+', 
                        choices=['wordcloud', 'comparison', 'subtasks', 'improvement', 'summary', 'all'],
                        default=['all'], help="Types of plots to generate")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    logger.info(f"Loading results from {args.results_dir}")
    results = load_evaluation_results(args.results_dir)
    
    if not results:
        logger.error("No evaluation results found")
        return 1
    
    logger.info(f"Loaded {len(results)} evaluation results")
    
    # Generate plots
    plots_to_generate = args.plots
    if 'all' in plots_to_generate:
        plots_to_generate = ['wordcloud', 'comparison', 'subtasks', 'improvement', 'summary']
    
    try:
        if 'wordcloud' in plots_to_generate:
            logger.info("Generating subtask word cloud...")
            generate_subtask_wordcloud(
                results, 
                args.model_name, 
                args.mode,
                str(output_dir / "subtask_wordcloud.png")
            )
        
        if 'comparison' in plots_to_generate:
            logger.info("Generating model comparison plot...")
            plot_model_comparison(results, str(output_dir / "model_comparison.png"))
        
        if 'subtasks' in plots_to_generate:
            logger.info("Generating subtask performance plot...")
            plot_subtask_performance(results, str(output_dir / "subtask_performance.png"))
        
        if 'improvement' in plots_to_generate:
            logger.info("Generating mode improvement plot...")
            plot_mode_improvement(results, str(output_dir / "mode_improvement.png"))
        
        if 'summary' in plots_to_generate:
            logger.info("Generating summary dashboard...")
            generate_summary_plot(results, str(output_dir / "summary_dashboard.png"))
        
        logger.info(f"All visualizations saved to {output_dir}")
        print(f"\nVisualizations saved to: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
