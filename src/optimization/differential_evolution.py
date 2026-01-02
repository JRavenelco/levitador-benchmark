"""
Differential Evolution Optimizer
================================

Classic Differential Evolution (DE/rand/1/bin) algorithm.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution (DE/rand/1/bin).
    
    Classic implementation by Storn & Price (1997).
    
    Reference:
        Storn, R., & Price, K. (1997). Differential evolution–a simple and 
        efficient heuristic for global optimization over continuous spaces.
        Journal of global optimization, 11(4), 341-359.
    """
    
    def __init__(self, problema, pop_size: int = 30,
                 max_iter: int = 100, F: float = 0.8, CR: float = 0.9,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Differential Evolution.
        
        Args:
            problema: Instance of LevitadorBenchmark
            pop_size: Population size
            max_iter: Maximum number of iterations
            F: Differential weight (mutation factor)
            CR: Crossover probability
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F    # Mutation scale factor
        self.CR = CR  # Crossover probability
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute Differential Evolution optimization."""
        # Initialize population
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_error = fitness[best_idx]
        
        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                # Select 3 distinct individuals
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = population[self._rng.choice(indices, 3, replace=False)]
                
                # Mutation: v = a + F * (b - c)
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Binomial crossover
                trial = population[i].copy()
                j_rand = self._rng.integers(self.dim)
                for j in range(self.dim):
                    if self._rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selection
                trial_fitness = self._evaluate(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_error:
                        best_solution = trial.copy()
                        best_error = trial_fitness
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Best = {best_error:.6e}")
        
        return best_solution, best_error
