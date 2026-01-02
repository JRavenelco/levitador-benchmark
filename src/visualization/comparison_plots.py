"""
Comparison Plot Utilities
==========================

Functions for creating comparison plots (boxplots, runtime comparisons, etc.).
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_boxplot(results_dict: Dict[str, List[float]],
                save_path: Optional[str] = None,
                title: str = "Algorithm Performance Comparison",
                ylabel: str = "Best Fitness (MSE)",
                log_scale: bool = True,
                figsize: tuple = (10, 6),
                dpi: int = 300):
    """
    Create box plots to compare algorithm performance across multiple trials.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping algorithm names to lists of best fitness values
    save_path : str, optional
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    log_scale : bool
        Use logarithmic scale for y-axis
    figsize : tuple
        Figure size
    dpi : int
        Resolution for saved figure
    
    Examples
    --------
    >>> results = {
    ...     'DE': [0.001, 0.0012, 0.0009],
    ...     'GA': [0.002, 0.0018, 0.0021]
    ... }
    >>> plot_boxplot(results, save_path='boxplot.png')
    """
    plt.figure(figsize=figsize)
    
    names = list(results_dict.keys())
    values = [results_dict[name] for name in names]
    
    bp = plt.boxplot(values, labels=names, patch_artist=True,
                     notch=True, showmeans=True)
    
    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Boxplot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_runtime(runtime_dict: Dict[str, List[float]],
                save_path: Optional[str] = None,
                title: str = "Runtime Comparison",
                ylabel: str = "Runtime (seconds)",
                figsize: tuple = (10, 6),
                dpi: int = 300):
    """
    Create bar plot comparing algorithm runtimes.
    
    Parameters
    ----------
    runtime_dict : dict
        Dictionary mapping algorithm names to lists of runtime values
    save_path : str, optional
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    figsize : tuple
        Figure size
    dpi : int
        Resolution for saved figure
    
    Examples
    --------
    >>> runtimes = {
    ...     'DE': [10.5, 11.2, 10.8],
    ...     'GA': [12.3, 11.9, 12.1]
    ... }
    >>> plot_runtime(runtimes, save_path='runtime.png')
    """
    plt.figure(figsize=figsize)
    
    names = list(runtime_dict.keys())
    means = [np.mean(runtime_dict[name]) for name in names]
    stds = [np.std(runtime_dict[name]) for name in names]
    
    x_pos = np.arange(len(names))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
    
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.8,
           color=colors, ecolor='black', capsize=5)
    
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std, f'{mean:.2f}s',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Runtime plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(results_dict: Dict[str, Dict[str, float]],
                   save_path: Optional[str] = None,
                   title: str = "Performance Metrics Comparison",
                   figsize: tuple = (12, 6),
                   dpi: int = 300):
    """
    Create grouped bar plot comparing multiple metrics across algorithms.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary: {algorithm: {metric: value}}
    save_path : str, optional
        Path to save the plot
    title : str
        Plot title
    figsize : tuple
        Figure size
    dpi : int
        Resolution for saved figure
    
    Examples
    --------
    >>> results = {
    ...     'DE': {'Best': 0.001, 'Mean': 0.0012, 'Std': 0.0001},
    ...     'GA': {'Best': 0.002, 'Mean': 0.0021, 'Std': 0.0002}
    ... }
    >>> plot_comparison(results, save_path='comparison.png')
    """
    plt.figure(figsize=figsize)
    
    algorithms = list(results_dict.keys())
    metrics = list(results_dict[algorithms[0]].keys())
    
    x = np.arange(len(algorithms))
    width = 0.8 / len(metrics)
    
    for i, metric in enumerate(metrics):
        values = [results_dict[alg][metric] for alg in algorithms]
        offset = (i - len(metrics)/2) * width + width/2
        plt.bar(x + offset, values, width, label=metric, alpha=0.8)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(x, algorithms, rotation=45, ha='right')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
