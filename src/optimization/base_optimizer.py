"""
Base Optimizer Class
====================

Abstract base class for all optimization algorithms.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional


class BaseOptimizer(ABC):
    """
    Base class for optimization algorithms.
    
    All optimization algorithms must inherit from this class and implement
    the optimize() method.
    
    Attributes:
        problema: Instance of LevitadorBenchmark
        dim: Problem dimensionality
        bounds: Search space bounds
        lb: Lower bounds array
        ub: Upper bounds array
        evaluations: Number of fitness evaluations performed
        history: History of best fitness values
    """
    
    def __init__(self, problema, random_seed: Optional[int] = None):
        """
        Initialize the optimizer.
        
        Args:
            problema: Instance of LevitadorBenchmark
            random_seed: Random seed for reproducibility
        """
        self.problema = problema
        self.dim = problema.dim
        self.bounds = np.array(problema.bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.evaluations = 0
        self.history = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            self._rng = np.random.default_rng(random_seed)
        else:
            self._rng = np.random.default_rng()
    
    def _evaluate(self, solution: List[float]) -> float:
        """
        Evaluate a solution and register the evaluation.
        
        Args:
            solution: Candidate solution
            
        Returns:
            Fitness value (MSE)
        """
        self.evaluations += 1
        return self.problema.fitness_function(solution)
    
    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute the optimization algorithm.
        
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        pass
