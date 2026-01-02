"""
Honey Badger Algorithm
=======================

Implementation of the Honey Badger Algorithm inspired by the intelligent
foraging behavior of honey badgers.

Reference: Hashim, F. A., et al. (2022). Honey Badger Algorithm:
New metaheuristic algorithm for solving optimization problems.
Mathematics and Computers in Simulation, 192, 84-110.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class HoneyBadgerAlgorithm(BaseOptimizer):
    """
    Honey Badger Algorithm (HBA).
    
    Inspired by the foraging behavior of honey badgers (Mellivora capensis),
    known for their intelligence and aggressive hunting style. The algorithm
    alternates between two modes:
    - Digging mode: Excavating to find prey
    - Honey mode: Following the honey guide bird
    
    Algorithm Steps:
    1. Initialize badger population randomly
    2. For each iteration:
       a. Calculate smell intensity based on distance to prey (best)
       b. Digging phase: Movement towards prey with perturbation
       c. Honey phase: Following the best with attraction factor
       d. Update positions with decay factor
    3. Return prey (best solution)
    
    Parameters
    ----------
    problema : LevitadorBenchmark
        The optimization problem
    pop_size : int, optional
        Population size (default: 30)
    max_iter : int, optional
        Maximum number of iterations (default: 100)
    beta : float, optional
        Intensity control factor (default: 6.0)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    verbose : bool, optional
        Print progress information (default: True)
    
    Examples
    --------
    >>> from levitador_benchmark import LevitadorBenchmark
    >>> from src.optimization import HoneyBadgerAlgorithm
    >>> problema = LevitadorBenchmark()
    >>> optimizer = HoneyBadgerAlgorithm(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, beta: float = 6.0,
                 random_seed: Optional[int] = None, verbose: bool = True):
        """
        Initialize Honey Badger Algorithm optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size (number of badgers)
        max_iter : int
            Maximum number of iterations
        beta : float
            Intensity control factor
        random_seed : int, optional
            Random seed for reproducibility
        verbose : bool
            Whether to print progress
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.beta = beta
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Execute Honey Badger Algorithm optimization.
        
        Returns
        -------
        tuple
            (prey, prey_fitness) - Best solution and its fitness
        """
        # Initialize population
        badgers = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(b) for b in badgers])
        
        best_idx = np.argmin(fitness)
        prey = badgers[best_idx].copy()  # Prey = best solution
        prey_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            alpha = self._get_alpha(t)  # Decay factor
            
            for i in range(self.pop_size):
                # Smell intensity
                r = self._rng.random()
                di = prey - badgers[i]  # Distance to prey
                S = (prey - badgers[i]) / (np.abs(di) + 1e-10)  # Direction
                I = r * S  # Intensity
                
                # Choose mode: digging or honey
                if self._rng.random() < 0.5:
                    # Digging mode
                    r3, r4, r5 = self._rng.random(3)
                    F = 1 if r3 < 0.5 else -1  # Direction flag
                    new_pos = prey + F * self.beta * I * prey + F * r4 * alpha * di * np.abs(np.cos(2*np.pi*r5) * (1 - np.cos(2*np.pi*r5)))
                else:
                    # Honey mode
                    r6, r7 = self._rng.random(2)
                    F = 1 if r6 < 0.5 else -1
                    new_pos = prey + F * r7 * alpha * di
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    badgers[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < prey_fitness:
                        prey = new_pos.copy()
                        prey_fitness = new_fitness
            
            self.history.append(prey_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {prey_fitness:.6e}")
        
        return prey, prey_fitness
    
    def _get_alpha(self, t):
        """
        Calculate decay factor alpha.
        
        Parameters
        ----------
        t : int
            Current iteration
        
        Returns
        -------
        float
            Decay factor
        """
        C = 2  # Constant
        return C * np.exp(-t / self.max_iter)
