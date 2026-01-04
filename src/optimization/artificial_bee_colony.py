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
            # Generate candidates for all bees
            # Select random neighbor k != i for each i
            k_indices = np.zeros(self.pop_size, dtype=int)
            for i in range(self.pop_size):
                candidates = [x for x in range(self.pop_size) if x != i]
                k_indices[i] = self._rng.choice(candidates)
            
            j_indices = self._rng.integers(0, self.dim, self.pop_size)
            phi = self._rng.uniform(-1, 1, self.pop_size)
            
            new_foods = foods.copy()
            # Vectorized update: new_foods[i, j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
            # We need to apply this per row since j varies
            rows = np.arange(self.pop_size)
            new_foods[rows, j_indices] = foods[rows, j_indices] + phi * (foods[rows, j_indices] - foods[k_indices, j_indices])
            new_foods = np.clip(new_foods, self.lb, self.ub)
            
            # Evaluate batch
            if hasattr(self.problema, 'evaluate_batch'):
                new_fitnesses = self.problema.evaluate_batch(new_foods)
                self.evaluations += self.pop_size
            else:
                new_fitnesses = np.array([self._evaluate(nf) for nf in new_foods])
            
            # Greedy selection
            improved = new_fitnesses < fitness
            foods[improved] = new_foods[improved]
            fitness[improved] = new_fitnesses[improved]
            trials[improved] = 0
            trials[~improved] += 1
            
            # === ONLOOKER BEES PHASE ===
            # Calculate probabilities
            # Add epsilon to avoid division by zero if fitness is -1 (though MSE >= 0)
            # For MSE, lower is better. We convert to fitness where higher is better for prob calc
            # Standard ABC: fit = 1/(1+obj) if obj>=0. Here obj is MSE >= 0.
            fit_val = 1.0 / (1.0 + fitness)
            probs = fit_val / fit_val.sum()
            
            # Select targets probabilistically
            target_indices = self._rng.choice(self.pop_size, self.pop_size, p=probs)
            
            # Generate candidates for onlookers
            k_indices_on = np.zeros(self.pop_size, dtype=int)
            for idx, target_i in enumerate(target_indices):
                candidates = [x for x in range(self.pop_size) if x != target_i]
                k_indices_on[idx] = self._rng.choice(candidates)
                
            j_indices_on = self._rng.integers(0, self.dim, self.pop_size)
            phi_on = self._rng.uniform(-1, 1, self.pop_size)
            
            onlooker_candidates = foods[target_indices].copy()
            rows = np.arange(self.pop_size)
            onlooker_candidates[rows, j_indices_on] = (
                foods[target_indices, j_indices_on] + 
                phi_on * (foods[target_indices, j_indices_on] - foods[k_indices_on, j_indices_on])
            )
            onlooker_candidates = np.clip(onlooker_candidates, self.lb, self.ub)
            
            # Evaluate batch
            if hasattr(self.problema, 'evaluate_batch'):
                onlooker_fitnesses = self.problema.evaluate_batch(onlooker_candidates)
                self.evaluations += self.pop_size
            else:
                onlooker_fitnesses = np.array([self._evaluate(oc) for oc in onlooker_candidates])
            
            # Greedy selection (sequential update to handle collisions on targets)
            for idx, target_i in enumerate(target_indices):
                if onlooker_fitnesses[idx] < fitness[target_i]:
                    foods[target_i] = onlooker_candidates[idx]
                    fitness[target_i] = onlooker_fitnesses[idx]
                    trials[target_i] = 0
                else:
                    trials[target_i] += 1
            
            # === SCOUT BEES PHASE ===
            scout_indices = np.where(trials > self.limit)[0]
            if len(scout_indices) > 0:
                # Generate new random foods
                new_randoms = self._rng.uniform(self.lb, self.ub, (len(scout_indices), self.dim))
                
                # Evaluate
                if hasattr(self.problema, 'evaluate_batch'):
                    scout_fitnesses = self.problema.evaluate_batch(new_randoms)
                    self.evaluations += len(scout_indices)
                else:
                    scout_fitnesses = np.array([self._evaluate(nr) for nr in new_randoms])
                
                # Update
                foods[scout_indices] = new_randoms
                fitness[scout_indices] = scout_fitnesses
                trials[scout_indices] = 0
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = foods[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
