"""
Artificial Bee Colony Optimizer
================================

Implementation of the Artificial Bee Colony algorithm inspired by
the foraging behavior of honey bees.

Reference: Karaboga, D. (2005). An idea based on honey bee swarm for
numerical optimization. Technical Report TR06, Erciyes University.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class ArtificialBeeColony(BaseOptimizer):
    """
    Artificial Bee Colony (ABC) Algorithm.
    
    Inspired by the foraging behavior of honey bees. The algorithm
    uses three types of bees:
    - Employed bees: Exploit food sources
    - Onlooker bees: Select food sources based on quality
    - Scout bees: Search for new food sources when one is exhausted
    
    Algorithm Steps:
    1. Initialize food sources (solutions) randomly
    2. For each iteration:
       a. Employed Bees Phase: Explore neighborhood of each source
       b. Onlooker Bees Phase: Select sources probabilistically
       c. Scout Bees Phase: Abandon exhausted sources (trials > limit)
    3. Return best food source found
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Number of food sources (default: 30)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    limit : int, optional
        Abandonment limit (default: pop_size * dim)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import ArtificialBeeColony
    >>> problema = LevitadorBenchmark()
    >>> optimizer = ArtificialBeeColony(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, limit: int = None,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Artificial Bee Colony optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Number of food sources
        max_iter : int
            Maximum number of iterations
        limit : int, optional
            Abandonment limit (default: pop_size * dim)
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit if limit else pop_size * self.dim
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Artificial Bee Colony optimization.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness)
        """
        # Initialize food sources
        foods = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(f) for f in foods])
        trials = np.zeros(self.pop_size)  # Failed improvement attempts
        
        best_idx = np.argmin(fitness)
        best_solution = foods[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # === EMPLOYED BEES PHASE ===
            for i in range(self.pop_size):
                # Select random neighbor
                k = self._rng.choice([j for j in range(self.pop_size) if j != i])
                j = self._rng.integers(self.dim)  # Dimension to modify
                
                # Generate new solution
                phi = self._rng.uniform(-1, 1)
                new_food = foods[i].copy()
                new_food[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                new_food = np.clip(new_food, self.lb, self.ub)
                
                new_fitness = self._evaluate(new_food)
                
                # Greedy selection
                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # === ONLOOKER BEES PHASE ===
            # Calculate selection probabilities
            fit_inv = 1 / (1 + fitness)
            probs = fit_inv / fit_inv.sum()
            
            for _ in range(self.pop_size):
                i = self._rng.choice(self.pop_size, p=probs)
                k = self._rng.choice([j for j in range(self.pop_size) if j != i])
                j = self._rng.integers(self.dim)
                
                phi = self._rng.uniform(-1, 1)
                new_food = foods[i].copy()
                new_food[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                new_food = np.clip(new_food, self.lb, self.ub)
                
                new_fitness = self._evaluate(new_food)
                
                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # === SCOUT BEES PHASE ===
            for i in range(self.pop_size):
                if trials[i] > self.limit:
                    foods[i] = self._rng.uniform(self.lb, self.ub)
                    fitness[i] = self._evaluate(foods[i])
                    trials[i] = 0
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = foods[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
