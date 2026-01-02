"""
Convergence Plot Utilities
===========================

Functions for plotting convergence curves of optimization algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def plot_convergence(history_dict: Dict[str, np.ndarray], 
                    save_path: Optional[str] = None,
                    title: str = "Convergence Comparison",
                    ylabel: str = "Best Fitness (MSE)",
                    xlabel: str = "Iteration",
                    log_scale: bool = True,
                    figsize: tuple = (10, 6),
                    dpi: int = 300):
    """
    Plot convergence curves for multiple algorithms.
    
    Parameters
    ----------
    history_dict : dict
        Dictionary mapping algorithm names to convergence history arrays
    save_path : str, optional
        Path to save the plot (if None, displays instead)
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    log_scale : bool
        Use logarithmic scale for y-axis
    figsize : tuple
        Figure size (width, height)
    dpi : int
        Resolution for saved figure
    
    Examples
    --------
    >>> histories = {
    ...     'DE': de_optimizer.get_convergence_curve(),
    ...     'GA': ga_optimizer.get_convergence_curve()
    ... }
    >>> plot_convergence(histories, save_path='convergence.png')
    """
    plt.figure(figsize=figsize)
    
    for name, history in history_dict.items():
        plt.plot(history, label=name, linewidth=2, alpha=0.8)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Convergence plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multiple_trials(trials_dict: Dict[str, List[np.ndarray]],
                        save_path: Optional[str] = None,
                        title: str = "Convergence with Multiple Trials",
                        ylabel: str = "Best Fitness (MSE)",
                        xlabel: str = "Iteration",
                        log_scale: bool = True,
                        figsize: tuple = (12, 7),
                        dpi: int = 300):
    """
    Plot convergence curves with multiple trials (mean ± std).
    
    Parameters
    ----------
    trials_dict : dict
        Dictionary mapping algorithm names to lists of trial histories
    save_path : str, optional
        Path to save the plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
    xlabel : str
        X-axis label
    log_scale : bool
        Use logarithmic scale for y-axis
    figsize : tuple
        Figure size
    dpi : int
        Resolution for saved figure
    
    Examples
    --------
    >>> trials = {
    ...     'DE': [trial1_history, trial2_history, trial3_history]
    ... }
    >>> plot_multiple_trials(trials, save_path='convergence_trials.png')
    """
    plt.figure(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trials_dict)))
    
    for idx, (name, histories) in enumerate(trials_dict.items()):
        histories_array = np.array(histories)
        mean_history = np.mean(histories_array, axis=0)
        std_history = np.std(histories_array, axis=0)
        
        iterations = np.arange(len(mean_history))
        
        plt.plot(iterations, mean_history, label=name, 
                linewidth=2, color=colors[idx], alpha=0.9)
        plt.fill_between(iterations, 
                        mean_history - std_history,
                        mean_history + std_history,
                        alpha=0.2, color=colors[idx])
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if log_scale:
        plt.yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Multi-trial convergence plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
