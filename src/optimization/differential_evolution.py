"""
Differential Evolution Optimizer
=================================

Implementation of the classic Differential Evolution algorithm (DE/rand/1/bin).

Reference: Storn, R., & Price, K. (1997). Differential evolution–a simple
and efficient heuristic for global optimization over continuous spaces.
Journal of global optimization, 11(4), 341-359.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class DifferentialEvolution(BaseOptimizer):
    """
    Differential Evolution (DE/rand/1/bin).
    
    A population-based optimization algorithm that uses difference vectors
    between population members to guide the search. Particularly effective
    for continuous optimization problems.
    
    Algorithm Overview:
    1. Initialize population randomly
    2. For each generation:
       - Mutation: Create mutant vector v = a + F * (b - c)
       - Crossover: Mix target and mutant vectors
       - Selection: Keep better solution (greedy)
    3. Return best solution found
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Population size (default: 30)
    max_iter : int, optional
        Maximum number of generations (default: 100)
    F : float, optional
        Mutation scaling factor, typically 0.5-0.9 (default: 0.8)
    CR : float, optional
        Crossover probability, typically 0.7-0.95 (default: 0.9)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import DifferentialEvolution
    >>> problema = LevitadorBenchmark()
    >>> optimizer = DifferentialEvolution(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, F: float = 0.8, CR: float = 0.9,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Differential Evolution optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size
        max_iter : int
            Maximum number of iterations
        F : float
            Mutation scaling factor
        CR : float
            Crossover probability
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F    # Mutation scaling factor
        self.CR = CR  # Crossover probability
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Differential Evolution optimization.
        
        Returns
        -------
        tuple
            (best_solution, best_error)
        """
        # Initialize population uniformly
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_error = fitness[best_idx]
        
        # Main evolution loop
        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                # === MUTATION ===
                # Select 3 random distinct individuals
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = population[self._rng.choice(indices, 3, replace=False)]
                
                # Create mutant: v = a + F * (b - c)
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # === CROSSOVER ===
                # Binomial crossover
                trial = population[i].copy()
                j_rand = self._rng.integers(self.dim)
                for j in range(self.dim):
                    if self._rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # === SELECTION ===
                trial_fitness = self._evaluate(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_error:
                        best_solution = trial.copy()
                        best_error = trial_fitness
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Mejor = {best_error:.6e}")
        
        return best_solution, best_error
