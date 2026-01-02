"""
Tianji Horse Racing Strategy
=============================

Implementation of the Tianji Horse Racing Strategy, based on the ancient
Chinese strategy where the competitor wins by using inferior horses against
the opponent's superior ones, superior against medium, and medium against inferior.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class TianjiOptimizer(BaseOptimizer):
    """
    Tianji Horse Racing Strategy Optimizer.
    
    Based on the ancient Chinese military strategy where victory is achieved
    through intelligent resource allocation. In optimization, this translates
    to dividing the population into three tiers with different strategies:
    
    - Superior Group: Fine-grained local exploitation
    - Medium Group: Balance between exploration and exploitation
    - Inferior Group: Broad global exploration
    
    Algorithm Steps:
    1. Initialize population (horses) randomly
    2. For each iteration:
       a. Sort population by fitness
       b. Divide into Superior, Medium, and Inferior groups
       c. Superior group: Local exploitation with small perturbations
       d. Medium group: Balance strategy (move towards best or explore)
       e. Inferior group: Global exploration (large jumps or re-initialization)
       f. Update best solution
    3. Return best horse found
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Population size (default: 30)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import TianjiOptimizer
    >>> problema = LevitadorBenchmark()
    >>> optimizer = TianjiOptimizer(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Tianji Optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size
        max_iter : int
            Maximum number of iterations
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Tianji Horse Racing Strategy.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness)
        """
        # Initialize population
        horses = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(h) for h in horses])
        
        best_idx = np.argmin(fitness)
        best_solution = horses[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Group size (divide into 3 tiers)
        group_size = self.pop_size // 3
        
        for t in range(self.max_iter):
            # Sort by fitness
            sorted_idx = np.argsort(fitness)
            
            # Divide into groups
            superior = sorted_idx[:group_size]
            medio = sorted_idx[group_size:2*group_size]
            inferior = sorted_idx[2*group_size:]
            
            # Adaptation factor (decreases over time)
            sigma = 0.1 * (1 - t / self.max_iter)
            
            # === SUPERIOR GROUP: Local Exploitation ===
            for i in superior:
                perturbation = self._rng.normal(0, sigma, self.dim) * (self.ub - self.lb)
                new_pos = horses[i] + perturbation
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    horses[i] = new_pos
                    fitness[i] = new_fitness
            
            # === MEDIUM GROUP: Balance ===
            for i in medio:
                if self._rng.random() < 0.5:
                    # Move towards best
                    r = self._rng.random()
                    new_pos = horses[i] + r * (best_solution - horses[i])
                else:
                    # Moderate exploration
                    perturbation = self._rng.normal(0, sigma * 2, self.dim) * (self.ub - self.lb)
                    new_pos = horses[i] + perturbation
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    horses[i] = new_pos
                    fitness[i] = new_fitness
            
            # === INFERIOR GROUP: Global Exploration ===
            for i in inferior:
                if self._rng.random() < 0.3:
                    # Random re-initialization
                    new_pos = self._rng.uniform(self.lb, self.ub)
                else:
                    # Large jump (learn from superior)
                    r1, r2 = self._rng.random(2)
                    j = self._rng.choice(superior)
                    new_pos = horses[i] + r1 * (horses[j] - horses[i]) + r2 * self._rng.normal(0, 0.5, self.dim) * (self.ub - self.lb)
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                fitness[i] = self._evaluate(new_pos)
                horses[i] = new_pos
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = horses[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
