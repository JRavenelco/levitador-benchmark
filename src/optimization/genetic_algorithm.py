"""
Genetic Algorithm Optimizer
============================

Implementation of a Genetic Algorithm with tournament selection,
BLX-alpha crossover, and Gaussian mutation.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class GeneticAlgorithm(BaseOptimizer):
    """
    Genetic Algorithm with tournament selection, BLX-alpha crossover,
    and Gaussian mutation.
    
    A population-based evolutionary algorithm inspired by natural selection.
    The algorithm evolves a population of solutions through selection,
    crossover, and mutation operations.
    
    Key Features:
    - Tournament selection for parent selection
    - BLX-alpha crossover for recombination
    - Gaussian mutation for exploration
    - Elitism to preserve best solutions
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Population size (default: 30)
    generations : int, optional
        Number of generations (default: 50)
    crossover_prob : float, optional
        Probability of crossover (default: 0.8)
    mutation_prob : float, optional
        Probability of mutation (default: 0.2)
    alpha : float, optional
        BLX-alpha parameter for crossover (default: 0.5)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import GeneticAlgorithm
    >>> problema = LevitadorBenchmark()
    >>> optimizer = GeneticAlgorithm(problema, pop_size=30, generations=50)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 generations: int = 50, crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2, alpha: float = 0.5,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Genetic Algorithm optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size
        generations : int
            Number of generations to evolve
        crossover_prob : float
            Probability of applying crossover
        mutation_prob : float
            Probability of applying mutation
        alpha : float
            BLX-alpha crossover parameter
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.alpha = alpha  # BLX-alpha parameter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Genetic Algorithm optimization.
        
        Returns
        -------
        tuple
            (best_solution, best_error)
        """
        # Initialize population
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        best_solution = None
        best_error = float('inf')
        
        for gen in range(self.generations):
            # Evaluate fitness
            if hasattr(self.problema, 'evaluate_batch'):
                fitness = self.problema.evaluate_batch(population)
                self.evaluations += len(population)
            else:
                fitness = np.array([self._evaluate(ind) for ind in population])
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_error:
                best_error = fitness[best_idx]
                best_solution = population[best_idx].copy()
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Mejor = {best_error:.6e}")
            
            # === SELECTION (Tournament) ===
            parents = []
            for _ in range(self.pop_size):
                i, j = self._rng.choice(self.pop_size, 2, replace=False)
                winner = i if fitness[i] < fitness[j] else j
                parents.append(population[winner].copy())
            
            # === CROSSOVER (BLX-alpha) ===
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
            
            # === MUTATION (Gaussian) ===
            for ind in children:
                if self._rng.random() < self.mutation_prob:
                    for d in range(self.dim):
                        sigma = (self.ub[d] - self.lb[d]) * 0.1
                        ind[d] += self._rng.normal(0, sigma)
                        ind[d] = np.clip(ind[d], self.lb[d], self.ub[d])
            
            # === ELITISM ===
            children[0] = best_solution.copy()
            population = np.array(children[:self.pop_size])
        
        return best_solution, best_error
