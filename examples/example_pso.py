"""
Particle Swarm Optimization (PSO) - Example
============================================

This script demonstrates how to use Particle Swarm Optimization (PSO) 
to optimize the Levitador Magnético benchmark.

PSO is inspired by the social behavior of bird flocking or fish schooling.
Each particle represents a potential solution that moves through the search
space influenced by its own best known position and the swarm's best known position.

Algorithm Overview:
1. Initialize a population of particles with random positions and velocities
2. For each iteration:
   a. Evaluate fitness of each particle
   b. Update personal best (pbest) and global best (gbest)
   c. Update velocities using cognitive and social components
   d. Update positions based on velocities
3. Return the global best solution found

Parameters:
- n_particles: Number of particles in the swarm (typically 20-50)
- max_iter: Maximum number of iterations
- w: Inertia weight (controls exploration vs exploitation, typically 0.4-0.9)
- c1: Cognitive coefficient (personal best influence, typically 1.5-2.0)
- c2: Social coefficient (global best influence, typically 1.5-2.0)

Usage:
    python examples/example_pso.py
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import levitador_benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) implementation.
    
    Reference: Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks, 4, 1942-1948.
    """
    
    def __init__(self, problema: LevitadorBenchmark, n_particles: int = 30,
                 max_iter: int = 100, w: float = 0.7, c1: float = 1.5, 
                 c2: float = 1.5, random_seed: int = None, verbose: bool = True):
        """
        Initialize PSO optimizer.
        
        Args:
            problema: LevitadorBenchmark instance
            n_particles: Number of particles in the swarm
            max_iter: Maximum number of iterations
            w: Inertia weight (controls previous velocity influence)
            c1: Cognitive coefficient (personal best influence)
            c2: Social coefficient (global best influence)
            random_seed: Random seed for reproducibility
            verbose: Print progress information
        """
        self.problema = problema
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        
        # Get problem bounds
        bounds = np.array(problema.bounds)
        self.lb = bounds[:, 0]  # Lower bounds
        self.ub = bounds[:, 1]  # Upper bounds
        self.dim = len(self.lb)
        
        # Random number generator
        if random_seed is not None:
            np.random.seed(random_seed)
            self._rng = np.random.default_rng(random_seed)
        else:
            self._rng = np.random.default_rng()
        
        # Tracking
        self.evaluations = 0
        self.history = []
    
    def optimize(self):
        """
        Run PSO optimization.
        
        Returns:
            tuple: (best_position, best_fitness)
        """
        # Initialize particles positions randomly within bounds
        positions = self._rng.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        
        # Initialize velocities
        v_max = (self.ub - self.lb) * 0.2  # Maximum velocity (20% of search space)
        velocities = self._rng.uniform(-v_max, v_max, (self.n_particles, self.dim))
        
        # Evaluate initial fitness
        fitness = np.array([self._evaluate(p) for p in positions])
        
        # Initialize personal best (pbest)
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        # Initialize global best (gbest)
        gbest_idx = np.argmin(fitness)
        gbest_position = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        if self.verbose:
            print(f"\nInitial best fitness: {gbest_fitness:.6e}")
            print(f"Initial best solution: k0={gbest_position[0]:.4f}, k={gbest_position[1]:.4f}, a={gbest_position[2]:.4f}")
        
        # Main PSO loop
        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                # Generate random coefficients
                r1 = self._rng.random(self.dim)
                r2 = self._rng.random(self.dim)
                
                # Update velocity
                # v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Limit velocity to v_max
                velocities[i] = np.clip(velocities[i], -v_max, v_max)
                
                # Update position
                positions[i] = positions[i] + velocities[i]
                
                # Keep within bounds
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Evaluate new position
                fitness[i] = self._evaluate(positions[i])
                
                # Update personal best
                if fitness[i] < pbest_fitness[i]:
                    pbest_fitness[i] = fitness[i]
                    pbest_positions[i] = positions[i].copy()
                    
                    # Update global best
                    if fitness[i] < gbest_fitness:
                        gbest_fitness = fitness[i]
                        gbest_position = positions[i].copy()
            
            # Store history
            self.history.append(gbest_fitness)
            
            # Print progress
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1:3d}: Best fitness = {gbest_fitness:.6e}")
        
        if self.verbose:
            print(f"\nOptimization completed!")
            print(f"Total evaluations: {self.evaluations}")
        
        return gbest_position, gbest_fitness
    
    def _evaluate(self, solution):
        """Evaluate fitness and track evaluations."""
        self.evaluations += 1
        return self.problema.fitness_function(solution)


def main():
    """Run PSO optimization example."""
    print("=" * 70)
    print("  PARTICLE SWARM OPTIMIZATION (PSO) - LEVITADOR BENCHMARK")
    print("=" * 70)
    
    # Create benchmark problem
    print("\n[1/3] Creating benchmark problem...")
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    print(f"  ✓ Benchmark loaded with {len(problema.t_real)} data points")
    print(f"  ✓ Search space: {problema.bounds}")
    
    # Configure PSO
    print("\n[2/3] Configuring PSO...")
    pso = ParticleSwarmOptimizer(
        problema=problema,
        n_particles=30,      # Swarm size
        max_iter=100,        # Number of iterations
        w=0.7,               # Inertia weight
        c1=1.5,              # Cognitive coefficient
        c2=1.5,              # Social coefficient
        random_seed=42,
        verbose=True
    )
    print(f"  ✓ PSO configured:")
    print(f"    - Particles: {pso.n_particles}")
    print(f"    - Max iterations: {pso.max_iter}")
    print(f"    - Inertia (w): {pso.w}")
    print(f"    - Cognitive (c1): {pso.c1}")
    print(f"    - Social (c2): {pso.c2}")
    
    # Run optimization
    print("\n[3/3] Running PSO optimization...")
    best_solution, best_fitness = pso.optimize()
    
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
    print(f"  Total evaluations: {pso.evaluations}")
    
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
        print(f"\n✅ PSO found a {improvement:.2f}% better solution!")
    else:
        print(f"\n📌 PSO solution is {-improvement:.2f}% away from reference")
    
    # Optional: Visualize the solution
    try:
        print("\n[Optional] Generating visualization...")
        problema.visualize_solution(best_solution, save_path="pso_result.png")
    except Exception as e:
        print(f"  Note: Visualization skipped ({e})")
    
    print("\n" + "=" * 70)
    print("✅ PSO optimization completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
