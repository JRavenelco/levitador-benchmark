"""
Genetic Algorithm (GA) - Example
=================================

This script demonstrates how to use a Genetic Algorithm (GA) 
to optimize the Levitador Magnético benchmark.

GA is inspired by the process of natural selection and evolution.
It maintains a population of candidate solutions that evolve over 
generations through selection, crossover (recombination), and mutation.

Algorithm Overview:
1. Initialize a population of random solutions (chromosomes)
2. For each generation:
   a. Evaluate fitness of all individuals
   b. Select parents based on fitness (tournament selection)
   c. Create offspring through crossover (BLX-α)
   d. Apply mutation to introduce diversity
   e. Replace population with offspring (elitism preserves best)
3. Return the best solution found

Parameters:
- pop_size: Population size (typically 30-100)
- generations: Number of generations to evolve
- crossover_prob: Probability of crossover (typically 0.7-0.9)
- mutation_prob: Probability of mutation (typically 0.1-0.3)
- alpha: BLX-α crossover parameter (typically 0.3-0.5)

Usage:
    python examples/example_ga.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import levitador_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark


class GeneticAlgorithm:
    """
    Genetic Algorithm implementation with tournament selection,
    BLX-alpha crossover, and Gaussian mutation.
    
    Reference: Holland, J. H. (1992). Adaptation in natural and artificial systems.
    MIT press.
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 50,
                 generations: int = 100, crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2, alpha: float = 0.5,
                 tournament_size: int = 3, random_seed: int = None,
                 verbose: bool = True):
        """
        Initialize Genetic Algorithm.
        
        Args:
            problema: LevitadorBenchmark instance
            pop_size: Population size (number of individuals)
            generations: Number of generations to evolve
            crossover_prob: Probability of applying crossover
            mutation_prob: Probability of mutation per individual
            alpha: BLX-alpha parameter for crossover
            tournament_size: Number of individuals in tournament selection
            random_seed: Random seed for reproducibility
            verbose: Print progress information
        """
        self.problema = problema
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.alpha = alpha
        self.tournament_size = tournament_size
        self.verbose = verbose
        
        # Get problem bounds
        bounds = np.array(problema.bounds)
        self.lb = bounds[:, 0]
        self.ub = bounds[:, 1]
        self.dim = len(self.lb)
        
        # Random number generator
        self._rng = np.random.default_rng(random_seed)
        
        # Tracking
        self.evaluations = 0
        self.history = []
    
    def optimize(self):
        """
        Run GA optimization.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        # Initialize population
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Track best solution
        best_solution = None
        best_fitness = float('inf')
        
        if self.verbose:
            print(f"\nStarting evolution...")
        
        # Evolution loop
        for gen in range(self.generations):
            # Evaluate fitness of all individuals
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            # Update best solution
            gen_best_idx = np.argmin(fitness)
            if fitness[gen_best_idx] < best_fitness:
                best_fitness = fitness[gen_best_idx]
                best_solution = population[gen_best_idx].copy()
                
                if self.verbose and gen == 0:
                    print(f"  Generation   0: Best fitness = {best_fitness:.6e}")
            
            # Store history
            self.history.append(best_fitness)
            
            # Print progress
            if self.verbose and (gen + 1) % 10 == 0:
                print(f"  Generation {gen + 1:3d}: Best fitness = {best_fitness:.6e}")
            
            # === SELECTION ===
            # Tournament selection to create parent pool
            parents = []
            for _ in range(self.pop_size):
                tournament_idx = self._rng.choice(self.pop_size, self.tournament_size, replace=False)
                tournament_fitness = fitness[tournament_idx]
                winner_idx = tournament_idx[np.argmin(tournament_fitness)]
                parents.append(population[winner_idx].copy())
            
            # === CROSSOVER ===
            # Create offspring through BLX-alpha crossover
            offspring = []
            for i in range(0, self.pop_size - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                if self._rng.random() < self.crossover_prob:
                    child1, child2 = self._blx_alpha_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                offspring.extend([child1, child2])
            
            # Handle odd population size by cloning last parent
            if len(offspring) < self.pop_size:
                offspring.append(parents[-1].copy())
            
            # === MUTATION ===
            # Apply Gaussian mutation
            for i in range(self.pop_size):
                if self._rng.random() < self.mutation_prob:
                    offspring[i] = self._gaussian_mutation(offspring[i])
            
            # === ELITISM ===
            # Preserve the best individual
            offspring[0] = best_solution.copy()
            
            # Replace population
            population = np.array(offspring)
        
        if self.verbose:
            print(f"\nEvolution completed!")
            print(f"Total evaluations: {self.evaluations}")
        
        return best_solution, best_fitness
    
    def _blx_alpha_crossover(self, parent1, parent2):
        """
        BLX-alpha (Blend Crossover) operator.
        
        Creates two offspring by blending parent genes with random
        interpolation and extrapolation.
        """
        child1 = np.zeros(self.dim)
        child2 = np.zeros(self.dim)
        
        for d in range(self.dim):
            # Calculate blend range
            min_val = min(parent1[d], parent2[d])
            max_val = max(parent1[d], parent2[d])
            range_val = max_val - min_val
            
            # Extend range by alpha
            lower = min_val - self.alpha * range_val
            upper = max_val + self.alpha * range_val
            
            # Generate offspring
            child1[d] = self._rng.uniform(lower, upper)
            child2[d] = self._rng.uniform(lower, upper)
            
            # Clip to bounds
            child1[d] = np.clip(child1[d], self.lb[d], self.ub[d])
            child2[d] = np.clip(child2[d], self.lb[d], self.ub[d])
        
        return child1, child2
    
    def _gaussian_mutation(self, individual):
        """
        Gaussian mutation operator.
        
        Adds Gaussian noise to genes with adaptive sigma.
        """
        mutated = individual.copy()
        
        for d in range(self.dim):
            # Adaptive sigma (10% of variable range)
            sigma = (self.ub[d] - self.lb[d]) * 0.1
            
            # Add Gaussian noise
            mutated[d] += self._rng.normal(0, sigma)
            
            # Clip to bounds
            mutated[d] = np.clip(mutated[d], self.lb[d], self.ub[d])
        
        return mutated
    
    def _evaluate(self, solution):
        """Evaluate fitness and track evaluations."""
        self.evaluations += 1
        return self.problema.fitness_function(solution)


def main():
    """Run GA optimization example."""
    print("=" * 70)
    print("  GENETIC ALGORITHM (GA) - LEVITADOR BENCHMARK")
    print("=" * 70)
    
    # Create benchmark problem
    print("\n[1/3] Creating benchmark problem...")
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    print(f"  ✓ Benchmark loaded with {len(problema.t_real)} data points")
    print(f"  ✓ Search space: {problema.bounds}")
    
    # Configure GA
    print("\n[2/3] Configuring Genetic Algorithm...")
    ga = GeneticAlgorithm(
        problema=problema,
        pop_size=50,            # Population size
        generations=100,        # Number of generations
        crossover_prob=0.8,     # Crossover probability
        mutation_prob=0.2,      # Mutation probability
        alpha=0.5,              # BLX-alpha parameter
        tournament_size=3,      # Tournament selection size
        random_seed=42,
        verbose=True
    )
    print(f"  ✓ GA configured:")
    print(f"    - Population size: {ga.pop_size}")
    print(f"    - Generations: {ga.generations}")
    print(f"    - Crossover probability: {ga.crossover_prob}")
    print(f"    - Mutation probability: {ga.mutation_prob}")
    print(f"    - BLX-alpha: {ga.alpha}")
    print(f"    - Tournament size: {ga.tournament_size}")
    
    # Run optimization
    print("\n[3/3] Running GA optimization...")
    best_solution, best_fitness = ga.optimize()
    
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
    print(f"  Total evaluations: {ga.evaluations}")
    
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
        print(f"\n✅ GA found a {improvement:.2f}% better solution!")
    else:
        print(f"\n📌 GA solution is {-improvement:.2f}% away from reference")
    
    # Optional: Visualize the solution
    try:
        print("\n[Optional] Generating visualization...")
        problema.visualize_solution(best_solution, save_path="ga_result.png")
    except Exception as e:
        print(f"  Note: Visualization skipped ({e})")
    
    print("\n" + "=" * 70)
    print("✅ GA optimization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
