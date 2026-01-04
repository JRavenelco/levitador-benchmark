"""
Grey Wolf Optimizer
===================

Implementation of the Grey Wolf Optimizer algorithm inspired by the
leadership hierarchy and hunting behavior of grey wolves.

Reference: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
Grey Wolf Optimizer. Advances in Engineering Software, 69, 46-61.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO).
    
    Inspired by the social hierarchy and hunting behavior of grey wolves.
    Wolves are divided into four categories:
    - Alpha (α): Best solution
    - Beta (β): Second best solution
    - Delta (δ): Third best solution
    - Omega (ω): Remaining solutions
    
    Algorithm Steps:
    1. Initialize wolf population randomly
    2. Identify Alpha, Beta, and Delta wolves (top 3 solutions)
    3. For each iteration:
       a. Update coefficient 'a' (linearly decreases from 2 to 0)
       b. Update position of each wolf based on Alpha, Beta, Delta
       c. Clip positions to bounds
       d. Evaluate and update hierarchy
    4. Return Alpha (best solution)
    
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
    >>> from src.optimization import GreyWolfOptimizer
    >>> problema = LevitadorBenchmark()
    >>> optimizer = GreyWolfOptimizer(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None, 
                 verbose: bool = True):
        """
        Initialize Grey Wolf Optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Number of wolves (population size)
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
        Execute Grey Wolf Optimizer.
        
        Returns
        -------
        tuple
            (alpha, alpha_score) - Best solution and its fitness
        """
        # Initialize wolf population
        wolves = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        if hasattr(self.problema, 'evaluate_batch'):
            fitness = self.problema.evaluate_batch(wolves)
            self.evaluations += self.pop_size
        else:
            fitness = np.array([self._evaluate(w) for w in wolves])
        
        # Sort and get Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness)
        alpha = wolves[sorted_idx[0]].copy()
        beta = wolves[sorted_idx[1]].copy()
        delta = wolves[sorted_idx[2]].copy()
        alpha_score = fitness[sorted_idx[0]]
        beta_score = fitness[sorted_idx[1]]
        delta_score = fitness[sorted_idx[2]]
        
        # Main loop
        for t in range(self.max_iter):
            # Update 'a' linearly from 2 to 0
            a = 2 - t * (2 / self.max_iter)
            
            # Vectorized update of positions
            # Generate random numbers for all wolves and dimensions at once
            r1 = self._rng.random((self.pop_size, self.dim))
            r2 = self._rng.random((self.pop_size, self.dim))
            
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            
            r1 = self._rng.random((self.pop_size, self.dim))
            r2 = self._rng.random((self.pop_size, self.dim))
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            
            r1 = self._rng.random((self.pop_size, self.dim))
            r2 = self._rng.random((self.pop_size, self.dim))
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            
            # Calculate distances and new positions (broadcasting)
            D_alpha = np.abs(C1 * alpha - wolves)
            D_beta = np.abs(C2 * beta - wolves)
            D_delta = np.abs(C3 * delta - wolves)
            
            X1 = alpha - A1 * D_alpha
            X2 = beta - A2 * D_beta
            X3 = delta - A3 * D_delta
            
            wolves = (X1 + X2 + X3) / 3
            
            # Enforce bounds
            wolves = np.clip(wolves, self.lb, self.ub)
            
            # Evaluate and update hierarchy
            if hasattr(self.problema, 'evaluate_batch'):
                fitness = self.problema.evaluate_batch(wolves)
                self.evaluations += self.pop_size
            else:
                fitness = np.array([self._evaluate(w) for w in wolves])
            
            # Update Alpha, Beta, Delta
            # We need to check against current alpha, beta, delta scores
            # Or just resort everyone including the old alpha, beta, delta?
            # Standard GWO updates hierarchy from the current population
            
            # Simple sorting of current population to find new leaders
            current_sorted_idx = np.argsort(fitness)
            
            # Check if we found better leaders than current ones
            # (Elitism is inherent if we keep bests, but standard GWO replaces them from population)
            # To ensure monotonic convergence, we should compare with previous bests
            # But the standard algorithm updates them every iteration from the new positions.
            # However, if the new positions are worse, we might lose the best solution.
            # Usually GWO doesn't guarantee keeping the previous Alpha if everyone moves away.
            # But we want to return the best found ever.
            
            current_best_idx = current_sorted_idx[0]
            current_best_score = fitness[current_best_idx]
            
            if current_best_score < alpha_score:
                # Update hierarchy
                alpha_score = current_best_score
                alpha = wolves[current_best_idx].copy()
                
                # Update Beta and Delta
                if self.pop_size > 1:
                    beta_score = fitness[current_sorted_idx[1]]
                    beta = wolves[current_sorted_idx[1]].copy()
                if self.pop_size > 2:
                    delta_score = fitness[current_sorted_idx[2]]
                    delta = wolves[current_sorted_idx[2]].copy()
            else:
                # If we didn't improve alpha, we might still have improved beta or delta
                # Or just re-evaluate hierarchy from current set
                # For simplicity and standard behavior, let's update from current pop
                # but keep the global best in 'alpha' for return value
                pass 
                # Note: In standard GWO, Alpha, Beta, Delta are agents that move.
                # Here we implicitly updated them.
                # Let's stick to the logic: Alpha is the best solution found SO FAR.
                
                # Re-check all against alpha, beta, delta logic strictly
                for i in range(self.pop_size):
                    if fitness[i] < alpha_score:
                        delta_score = beta_score
                        delta = beta.copy()
                        beta_score = alpha_score
                        beta = alpha.copy()
                        alpha_score = fitness[i]
                        alpha = wolves[i].copy()
                    elif fitness[i] < beta_score:
                        delta_score = beta_score
                        delta = beta.copy()
                        beta_score = fitness[i]
                        beta = wolves[i].copy()
                    elif fitness[i] < delta_score:
                        delta_score = fitness[i]
                        delta = wolves[i].copy()
            
            self.history.append(alpha_score)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Alpha = {alpha_score:.6e}")
        
        return alpha, alpha_score
