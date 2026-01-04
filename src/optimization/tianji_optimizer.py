"""
Tianji Horse Racing Strategy
=============================

Implementation of the Tianji Horse Racing Strategy, based on the ancient
Chinese strategy where the competitor wins by using inferior horses against
the opponent's superior ones, superior against medium, and medium against inferior.
"""

import numpy as np
from typing import Tuple, Optional
from .base_optimizer import BaseOptimizer
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from levitador_benchmark import LevitadorBenchmark


class TianjiOptimizer(BaseOptimizer):
    """
    Tianji Horse Racing Strategy Optimizer.
    
    Based on the ancient Chinese military strategy where victory is achieved
    through intelligent resource allocation. In optimization, this translates
    to dividing the population into three tiers with different strategies:
    
    - Superior Group: Fine-grained local exploitation
    - Medium Group: Balance between exploration and exploitation
    - Inferior Group: Broad global exploration
    
    Algorithm Steps:
    1. Initialize population (horses) randomly
    2. For each iteration:
       a. Sort population by fitness
       b. Divide into Superior, Medium, and Inferior groups
       c. Superior group: Local exploitation with small perturbations
       d. Medium group: Balance strategy (move towards best or explore)
       e. Inferior group: Global exploration (large jumps or re-initialization)
       f. Update best solution
    3. Return best horse found
    
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
    >>> from src.optimization import TianjiOptimizer
    >>> problema = LevitadorBenchmark()
    >>> optimizer = TianjiOptimizer(problema, pop_size=30, max_iter=100)
    >>> best_sol, best_fit = optimizer.optimize()
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Initialize Tianji Optimizer.
        
        Parameters
        ----------
        problema : LevitadorBenchmark
            The optimization problem
        pop_size : int
            Population size
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
        Execute Tianji Horse Racing Strategy.
        
        Returns
        -------
        tuple
            (best_solution, best_fitness)
        """
        # Initialize population
        horses = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(h) for h in horses])
        
        best_idx = np.argmin(fitness)
        best_solution = horses[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Group size (divide into 3 tiers)
        group_size = self.pop_size // 3
        
        for t in range(self.max_iter):
            # Sort by fitness
            sorted_idx = np.argsort(fitness)
            
            # Divide into groups indices
            superior_idx = sorted_idx[:group_size]
            medio_idx = sorted_idx[group_size:2*group_size]
            inferior_idx = sorted_idx[2*group_size:]
            
            # Adaptation factor (decreases over time)
            sigma = 0.1 * (1 - t / self.max_iter)
            
            # Prepare new positions array
            new_horses = horses.copy()
            
            # === SUPERIOR GROUP: Local Exploitation ===
            # Generate perturbations for superior group
            pert_sup = self._rng.normal(0, sigma, (len(superior_idx), self.dim)) * (self.ub - self.lb)
            new_horses[superior_idx] = horses[superior_idx] + pert_sup
            
            # === MEDIUM GROUP: Balance ===
            # Decide between moving towards best or exploring
            rand_med = self._rng.random(len(medio_idx))
            move_to_best_mask = rand_med < 0.5
            explore_mask = ~move_to_best_mask
            
            # Move towards best
            if np.any(move_to_best_mask):
                idx_move = medio_idx[move_to_best_mask]
                r_move = self._rng.random((len(idx_move), 1))
                new_horses[idx_move] = horses[idx_move] + r_move * (best_solution - horses[idx_move])
            
            # Moderate exploration
            if np.any(explore_mask):
                idx_exp = medio_idx[explore_mask]
                pert_med = self._rng.normal(0, sigma * 2, (len(idx_exp), self.dim)) * (self.ub - self.lb)
                new_horses[idx_exp] = horses[idx_exp] + pert_med
            
            # === INFERIOR GROUP: Global Exploration ===
            rand_inf = self._rng.random(len(inferior_idx))
            reinit_mask = rand_inf < 0.3
            jump_mask = ~reinit_mask
            
            # Random re-initialization
            if np.any(reinit_mask):
                idx_reinit = inferior_idx[reinit_mask]
                new_horses[idx_reinit] = self._rng.uniform(self.lb, self.ub, (len(idx_reinit), self.dim))
            
            # Large jump (learn from superior)
            if np.any(jump_mask):
                idx_jump = inferior_idx[jump_mask]
                r1 = self._rng.random((len(idx_jump), 1))
                r2 = self._rng.random((len(idx_jump), 1))
                
                # Pick random superior horses to learn from
                sup_choices = self._rng.choice(superior_idx, len(idx_jump))
                
                jump_term = r1 * (horses[sup_choices] - horses[idx_jump])
                pert_term = r2 * self._rng.normal(0, 0.5, (len(idx_jump), self.dim)) * (self.ub - self.lb)
                
                new_horses[idx_jump] = horses[idx_jump] + jump_term + pert_term
            
            # Clip all bounds
            new_horses = np.clip(new_horses, self.lb, self.ub)
            
            # Evaluate batch
            if hasattr(self.problema, 'evaluate_batch'):
                new_fitness = self.problema.evaluate_batch(new_horses)
                self.evaluations += self.pop_size
            else:
                new_fitness = np.array([self._evaluate(h) for h in new_horses])
            
            # Update population (Greedy selection per horse)
            improved = new_fitness < fitness
            horses[improved] = new_horses[improved]
            fitness[improved] = new_fitness[improved]
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = horses[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
