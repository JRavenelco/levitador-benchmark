"""
Plot Utilities
=============

Functions for plotting convergence curves and comparison charts.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def plot_convergence(histories: Dict[str, List[float]], 
                     save_path: Optional[str] = None,
                     title: str = "Convergence Curves",
                     log_scale: bool = True):
    """
    Plot convergence curves for multiple optimizers.
    
    Args:
        histories: Dict mapping optimizer names to history lists
        save_path: Path to save figure (None for display only)
        title: Plot title
        log_scale: Use logarithmic y-axis
    """
    plt.figure(figsize=(10, 6))
    
    for name, history in histories.items():
        plt.plot(history, label=name, linewidth=2, alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Best Fitness (MSE)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_boxplot(results: Dict[str, List[float]],
                            save_path: Optional[str] = None,
                            title: str = "Algorithm Comparison",
                            log_scale: bool = True):
    """
    Create box plot comparing multiple optimizers.
    
    Args:
        results: Dict mapping optimizer names to lists of final fitness values
        save_path: Path to save figure
        title: Plot title
        log_scale: Use logarithmic y-axis
    """
    plt.figure(figsize=(12, 6))
    
    names = list(results.keys())
    data = [results[name] for name in names]
    
    bp = plt.boxplot(data, labels=names, patch_artist=True)
    
    # Customize colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Final Fitness (MSE)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Box plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_performance_metrics(metrics: Dict[str, Dict[str, float]],
                             save_path: Optional[str] = None):
    """
    Plot performance metrics (mean, std, best, worst) for each optimizer.
    
    Args:
        metrics: Dict mapping optimizer names to metric dicts
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(metrics.keys())
    metric_names = ['mean', 'std', 'best', 'worst']
    titles = ['Mean Fitness', 'Standard Deviation', 'Best Fitness', 'Worst Fitness']
    
    for ax, metric_name, title in zip(axes.flat, metric_names, titles):
        values = [metrics[name][metric_name] for name in names]
        
        bars = ax.bar(range(len(names)), values, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Fitness (MSE)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance metrics plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_runtime_comparison(runtimes: Dict[str, List[float]],
                           save_path: Optional[str] = None):
    """
    Plot runtime comparison between optimizers.
    
    Args:
        runtimes: Dict mapping optimizer names to lists of runtime values (seconds)
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))
    
    names = list(runtimes.keys())
    mean_times = [np.mean(runtimes[name]) for name in names]
    std_times = [np.std(runtimes[name]) for name in names]
    
    x = np.arange(len(names))
    bars = plt.bar(x, mean_times, yerr=std_times, alpha=0.7, capsize=5)
    
    # Color bars
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Runtime (seconds)', fontsize=12)
    plt.title('Runtime Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Runtime comparison saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
