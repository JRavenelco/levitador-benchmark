"""
Artificial Bee Colony Optimizer
===============================

Artificial Bee Colony algorithm inspired by the foraging behavior
of honey bees.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class ArtificialBeeColony(BaseOptimizer):
    """
    Artificial Bee Colony (ABC) algorithm.
    
    Inspired by the foraging behavior of honey bees. Three types of bees:
    - Employed bees: exploit food sources
    - Onlooker bees: select sources based on quality
    - Scout bees: search for new sources when exhausted
    
    Reference:
        Karaboga, D. (2005). An idea based on honey bee swarm for numerical 
        optimization. Technical Report TR06, Erciyes University.
    """
    
    def __init__(self, problema, pop_size: int = 30,
                 max_iter: int = 100, limit: int = None,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Artificial Bee Colony.
        
        Args:
            problema: Instance of LevitadorBenchmark
            pop_size: Number of food sources (population size)
            max_iter: Maximum number of iterations
            limit: Abandonment limit (default: pop_size * dim)
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit if limit else pop_size * self.dim
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute Artificial Bee Colony optimization."""
        # Initialize food sources
        foods = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(f) for f in foods])
        trials = np.zeros(self.pop_size)  # Failed trial counter
        
        best_idx = np.argmin(fitness)
        best_solution = foods[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # === EMPLOYED BEE PHASE ===
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
            
            # === ONLOOKER BEE PHASE ===
            # Calculate selection probabilities (inverse fitness for minimization)
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
            
            # === SCOUT BEE PHASE ===
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
                print(f"  Iter {t:3d}: Best = {best_fitness:.6e}")
        
        return best_solution, best_fitness
