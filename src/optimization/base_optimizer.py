"""
Base Optimizer Class
====================

Abstract base class for all optimization algorithms in the framework.
All optimizers should inherit from this class and implement the optimize() method.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import levitador_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    This class provides a common interface and utilities for all optimization
    algorithms in the framework. It handles:
    - Problem initialization and bounds management
    - Random number generation for reproducibility
    - Fitness evaluation tracking
    - Convergence history recording
    
    All concrete optimizer classes must inherit from this class and implement
    the optimize() method.
    
    Attributes
    ----------
    problema : LevitadorBenchmark
        The optimization problem instance
    dim : int
        Dimensionality of the search space
    bounds : np.ndarray
        Search space bounds as (n_dims, 2) array
    lb : np.ndarray
        Lower bounds for each dimension
    ub : np.ndarray
        Upper bounds for each dimension
    evaluations : int
        Counter for function evaluations
    history : list
        History of best fitness values per iteration
    """
    
    def __init__(self, problema: LevitadorBenchmark, random_seed: Optional[int] = None):
        """
        Initialize the base optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            Instance of the optimization problem
        random_seed : int, optional
            Seed for random number generator (for reproducibility)
        """
        self.problema = problema
        self.dim = problema.dim
        self.bounds = np.array(problema.bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.evaluations = 0
        self.history = []
        
        # Configure random number generator
        if random_seed is not None:
            np.random.seed(random_seed)
            self._rng = np.random.default_rng(random_seed)
        else:
            self._rng = np.random.default_rng()
    
    def _evaluate(self, solution: List[float]) -> float:
        """
        Evaluate a solution and track the evaluation count.
        
        This method should be used instead of calling the fitness function directly
        to ensure proper tracking of function evaluations.
        
        Parameters
        ----------
        solution : array-like
            Solution vector to evaluate
        
        Returns
        -------
        float
            Fitness value (MSE) for the solution
        """
        self.evaluations += 1
        return self.problema.fitness_function(solution)
    
    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute the optimization algorithm.
        
        This method must be implemented by all concrete optimizer classes.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness) where:
            - best_solution: np.ndarray of shape (dim,) with parameter values
            - best_fitness: float with the best fitness value found
        """
        pass
    
    def get_convergence_curve(self) -> np.ndarray:
        """
        Get the convergence curve (best fitness per iteration).
        
        Returns
        -------
        np.ndarray
            Array with best fitness values per iteration
        """
        return np.array(self.history)
    
    def reset(self):
        """
        Reset the optimizer state.
        
        Clears evaluation counter and history. Useful for running
        multiple independent trials.
        """
        self.evaluations = 0
        self.history = []
