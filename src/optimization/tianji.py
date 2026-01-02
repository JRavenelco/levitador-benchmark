"""
Tianji Horse Racing Optimizer
=============================

Tianji Horse Racing Strategy inspired by ancient Chinese strategy.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class TianjiOptimizer(BaseOptimizer):
    """
    Tianji Horse Racing Strategy optimizer.
    
    Based on the ancient Chinese strategy where different quality horses
    are strategically matched. In optimization, the population is divided
    into three groups: Superior, Medium, and Inferior, each with different
    exploration/exploitation strategies.
    """
    
    def __init__(self, problema, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Tianji Optimizer.
        
        Args:
            problema: Instance of LevitadorBenchmark
            pop_size: Population size (must be divisible by 3)
            max_iter: Maximum number of iterations
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute Tianji optimization."""
        # Initialize horse population
        horses = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(h) for h in horses])
        
        best_idx = np.argmin(fitness)
        best_solution = horses[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Group size
        group_size = self.pop_size // 3
        
        for t in range(self.max_iter):
            # Sort by fitness
            sorted_idx = np.argsort(fitness)
            
            # Divide into groups
            superior = sorted_idx[:group_size]
            medium = sorted_idx[group_size:2*group_size]
            inferior = sorted_idx[2*group_size:]
            
            # Adaptation factor (decreases with time)
            sigma = 0.1 * (1 - t / self.max_iter)
            
            # === SUPERIOR GROUP: Local exploitation ===
            for i in superior:
                perturbation = self._rng.normal(0, sigma, self.dim) * (self.ub - self.lb)
                new_pos = horses[i] + perturbation
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    horses[i] = new_pos
                    fitness[i] = new_fitness
            
            # === MEDIUM GROUP: Balance ===
            for i in medium:
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
            
            # === INFERIOR GROUP: Global exploration ===
            for i in inferior:
                if self._rng.random() < 0.3:
                    # Random reinitialization
                    new_pos = self._rng.uniform(self.lb, self.ub)
                else:
                    # Large jump, learning from superior
                    r1, r2 = self._rng.random(2)
                    j = self._rng.choice(superior)
                    new_pos = (horses[i] + r1 * (horses[j] - horses[i]) + 
                              r2 * self._rng.normal(0, 0.5, self.dim) * (self.ub - self.lb))
                
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
                print(f"  Iter {t:3d}: Best = {best_fitness:.6e}")
        
        return best_solution, best_fitness
