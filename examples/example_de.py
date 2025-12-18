"""
Differential Evolution (DE) - Example
======================================

This script demonstrates how to use Differential Evolution (DE) 
to optimize the Levitador Magnético benchmark.

DE is a population-based optimization algorithm that uses difference vectors
between population members to guide the search. It's particularly effective
for continuous optimization problems.

Algorithm Overview:
1. Initialize a population of solution vectors randomly
2. For each generation:
   a. For each target vector in the population:
      - Select 3 random distinct vectors (a, b, c)
      - Create mutant vector: v = a + F * (b - c)
      - Perform crossover between target and mutant
      - Select better solution (greedy selection)
   b. Update best solution
3. Return the best solution found

Strategy: DE/rand/1/bin
- Mutation: v = x_r1 + F * (x_r2 - x_r3)
- Crossover: Binomial (uniform)
- Selection: Greedy (keep better fitness)

Parameters:
- pop_size: Population size (typically 10-50, or 10*dim)
- max_iter: Maximum number of generations
- F: Mutation factor/scaling factor (typically 0.5-0.9)
- CR: Crossover rate (typically 0.7-0.95)

Usage:
    python examples/example_de.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import levitador_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark


class DifferentialEvolution:
    """
    Differential Evolution (DE/rand/1/bin) implementation.
    
    Reference: Storn, R., & Price, K. (1997). Differential evolution–a simple
    and efficient heuristic for global optimization over continuous spaces.
    Journal of global optimization, 11(4), 341-359.
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, F: float = 0.8, CR: float = 0.9,
                 random_seed: int = None, verbose: bool = True):
        """
        Initialize Differential Evolution optimizer.
        
        Args:
            problema: LevitadorBenchmark instance
            pop_size: Population size (number of solution vectors)
            max_iter: Maximum number of iterations/generations
            F: Mutation factor (differential weight), typically 0.5-0.9
            CR: Crossover rate (probability), typically 0.7-0.95
            random_seed: Random seed for reproducibility
            verbose: Print progress information
        """
        self.problema = problema
        self.max_iter = max_iter
        self.F = F    # Mutation/scaling factor
        self.CR = CR  # Crossover rate
        self.verbose = verbose
        
        # Get problem bounds
        bounds = np.array(problema.bounds)
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]
        self.dim = len(self.lb)
        
        # Validate population size (must be at least 4 for DE/rand/1/bin)
        if pop_size < 4:
            raise ValueError("Population size must be at least 4 for DE/rand/1/bin strategy")
        self.pop_size = pop_size
        
        # Random number generator
        self._rng = np.random.default_rng(random_seed)
        
        # Tracking
        self.evaluations = 0
        self.history = []
    
    def optimize(self):
        """
        Run DE optimization.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        # Initialize population uniformly in the search space
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Evaluate initial population
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        if self.verbose:
            print(f"\nInitial population evaluated")
            print(f"  Initial best fitness: {best_fitness:.6e}")
            print(f"  Initial best solution: k0={best_solution[0]:.4f}, k={best_solution[1]:.4f}, a={best_solution[2]:.4f}")
        
        # Main DE loop
        for gen in range(self.max_iter):
            # For each target vector in population
            for i in range(self.pop_size):
                # === MUTATION ===
                # Select 3 random distinct vectors (different from target i)
                candidates = [j for j in range(self.pop_size) if j != i]
                a_idx, b_idx, c_idx = self._rng.choice(candidates, 3, replace=False)
                
                a = population[a_idx]
                b = population[b_idx]
                c = population[c_idx]
                
                # Create mutant vector: v = a + F * (b - c)
                mutant = a + self.F * (b - c)
                
                # Ensure mutant is within bounds
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # === CROSSOVER (Binomial) ===
                # Create trial vector by mixing target and mutant
                trial = population[i].copy()
                
                # Ensure at least one parameter comes from mutant (j_rand)
                j_rand = self._rng.integers(self.dim)
                
                for j in range(self.dim):
                    # Use mutant's parameter if rand < CR or j == j_rand
                    if self._rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # === SELECTION ===
                # Evaluate trial vector
                trial_fitness = self._evaluate(trial)
                
                # Greedy selection: keep better solution
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    # Update global best
                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness
            
            # Store history
            self.history.append(best_fitness)
            
            # Print progress
            if self.verbose and (gen + 1) % 10 == 0:
                print(f"  Generation {gen + 1:3d}: Best fitness = {best_fitness:.6e}")
        
        if self.verbose:
            print(f"\nOptimization completed!")
            print(f"Total evaluations: {self.evaluations}")
        
        return best_solution, best_fitness
    
    def _evaluate(self, solution):
        """Evaluate fitness and track evaluations."""
        self.evaluations += 1
        return self.problema.fitness_function(solution)


def main():
    """Run DE optimization example."""
    print("=" * 70)
    print("  DIFFERENTIAL EVOLUTION (DE) - LEVITADOR BENCHMARK")
    print("=" * 70)
    
    # Create benchmark problem
    print("\n[1/3] Creating benchmark problem...")
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    print(f"  ✓ Benchmark loaded with {len(problema.t_real)} data points")
    print(f"  ✓ Search space: {problema.bounds}")
    
    # Configure DE
    print("\n[2/3] Configuring Differential Evolution...")
    de = DifferentialEvolution(
        problema=problema,
        pop_size=30,         # Population size
        max_iter=100,        # Number of generations
        F=0.8,               # Mutation factor
        CR=0.9,              # Crossover rate
        random_seed=42,
        verbose=True
    )
    print(f"  ✓ DE configured:")
    print(f"    - Strategy: DE/rand/1/bin")
    print(f"    - Population size: {de.pop_size}")
    print(f"    - Max generations: {de.max_iter}")
    print(f"    - Mutation factor (F): {de.F}")
    print(f"    - Crossover rate (CR): {de.CR}")
    
    # Run optimization
    print("\n[3/3] Running DE optimization...")
    best_solution, best_fitness = de.optimize()
    
    # Display results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n🏆 Best solution found:")
    print(f"  k0 = {best_solution[0]:.6f} H  (Inductancia base)")
    print(f"  k  = {best_solution[1]:.6f} H  (Coeficiente de inductancia)")
    print(f"  a  = {best_solution[2]:.6f} m  (Parámetro geométrico)")
    print(f"\n📊 Performance:")
    print(f"  Best fitness (MSE): {best_fitness:.6e}")
    print(f"  Total evaluations: {de.evaluations}")
    
    # Compare with reference solution
    print(f"\n📚 Reference solution:")
    ref = problema.reference_solution
    print(f"  k0 = {ref[0]:.6f} H")
    print(f"  k  = {ref[1]:.6f} H")
    print(f"  a  = {ref[2]:.6f} m")
    
    ref_error = problema.fitness_function(ref)
    print(f"  Reference MSE: {ref_error:.6e}")
    
    improvement = ((ref_error - best_fitness) / ref_error) * 100
    if improvement > 0:
        print(f"\n✅ DE found a {improvement:.2f}% better solution!")
    else:
        print(f"\n📌 DE solution is {-improvement:.2f}% away from reference")
    
    # Optional: Visualize the solution
    try:
        print("\n[Optional] Generating visualization...")
        problema.visualize_solution(best_solution, save_path="de_result.png")
    except Exception as e:
        print(f"  Note: Visualization skipped ({e})")
    
    print("\n" + "=" * 70)
    print("✅ DE optimization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
