"""
Genetic Algorithm Optimizer
===========================

Simple genetic algorithm with tournament selection, BLX-alpha crossover,
and Gaussian mutation.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm with tournament selection, BLX-alpha crossover,
    and Gaussian mutation.
    """
    
    def __init__(self, problema, pop_size: int = 30,
                 generations: int = 50, crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2, alpha: float = 0.5,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problema: Instance of LevitadorBenchmark
            pop_size: Population size
            generations: Number of generations
            crossover_prob: Probability of crossover
            mutation_prob: Probability of mutation
            alpha: BLX-alpha crossover parameter
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.alpha = alpha  # BLX-alpha parameter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Execute Genetic Algorithm optimization."""
        # Initialize population
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        best_solution = None
        best_error = float('inf')
        
        for gen in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_error:
                best_error = fitness[best_idx]
                best_solution = population[best_idx].copy()
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Best = {best_error:.6e}")
            
            # Tournament selection
            parents = []
            for _ in range(self.pop_size):
                i, j = self._rng.choice(self.pop_size, 2, replace=False)
                winner = i if fitness[i] < fitness[j] else j
                parents.append(population[winner].copy())
            
            # BLX-alpha crossover
            children = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[min(i+1, self.pop_size-1)]
                
                if self._rng.random() < self.crossover_prob:
                    for d in range(self.dim):
                        rango = abs(p1[d] - p2[d])
                        minimo = min(p1[d], p2[d]) - self.alpha * rango
                        maximo = max(p1[d], p2[d]) + self.alpha * rango
                        p1[d] = self._rng.uniform(minimo, maximo)
                        p2[d] = self._rng.uniform(minimo, maximo)
                
                children.extend([p1, p2])
            
            # Gaussian mutation
            for ind in children:
                if self._rng.random() < self.mutation_prob:
                    for d in range(self.dim):
                        sigma = (self.ub[d] - self.lb[d]) * 0.1
                        ind[d] += self._rng.normal(0, sigma)
                        ind[d] = np.clip(ind[d], self.lb[d], self.ub[d])
            
            # Elitism
            children[0] = best_solution.copy()
            population = np.array(children[:self.pop_size])
        
        return best_solution, best_error
