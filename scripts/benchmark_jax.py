#!/usr/bin/env python3
"""
JAX-Accelerated Benchmark for Magnetic Levitator Parameter Identification
==========================================================================

🚀 "El Camino de Doctorado" - Massive GPU Vectorization 🚀

This script leverages JAX's vmap to evaluate the ENTIRE POPULATION
in a SINGLE GPU CALL, achieving massive speedups on L4/T4/A100 GPUs.

Usage (Colab):
    !python scripts/benchmark_jax.py --data data/datos_levitador.txt --pop_size 200 --generations 500

Key Features:
1. vmap: Vectorize fitness evaluation over population
2. jit: JIT-compile all operations for GPU
3. scan: Efficient ODE integration on GPU
"""

import sys
import argparse
import time
from pathlib import Path
import numpy as np

# JAX imports
import jax
import jax.numpy as jnp
from jax import random

# Force JAX to use GPU if available
print(f"JAX devices: {jax.devices()}")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jax_backend.physics_jax import create_vectorized_fitness
from src.jax_backend.genetic_jax import GeneticAlgorithmJAX, DifferentialEvolutionJAX


def load_data(data_path: str, subsample: int = 10):
    """
    Load experimental data from file.
    
    Format expected: columns [t, y, i, u, ...]
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading data from {data_path}...")
    try:
        data = np.loadtxt(data_path)
    except ValueError:
        print("   Header detected, skipping first row...")
        data = np.loadtxt(data_path, skiprows=1)
    
    # Subsample for faster computation (GPU handles large batches, but ODE is sequential)
    data = data[::subsample]
    
    t = data[:, 0].astype(np.float32)
    y = data[:, 1].astype(np.float32)
    i = data[:, 2].astype(np.float32)
    u = data[:, 3].astype(np.float32)
    
    print(f"Loaded {len(t)} data points (subsampled by {subsample}x)")
    
    return jnp.array(t), jnp.array(y), jnp.array(i), jnp.array(u)


def run_benchmark(args):
    """
    Run the JAX-accelerated benchmark.
    """
    print("\n" + "="*60)
    print("🚀 JAX-Accelerated Parameter Identification Benchmark 🚀")
    print("="*60)
    
    # Load data
    t_data, y_data, i_data, u_data = load_data(args.data, args.subsample)
    
    # Physical constants
    m = 0.009  # kg
    g = 9.81   # m/s²
    dt = float(t_data[1] - t_data[0])
    y0 = float(y_data[0])
    
    print(f"\nPhysical parameters: m={m} kg, g={g} m/s², dt={dt:.4f} s")
    print(f"Initial position: y0={y0*1000:.2f} mm")
    
    # Parameter bounds: [K0, A, R0, alpha]
    # Basado en física del levitador de Valentín:
    # - Bobina: L ~ 0.01-0.1 H típico para electroimán pequeño
    # - Gap: A ~ 1-15mm (geometría del sistema)
    # - Resistencia: R0 ~ 2-25 Ω (nominal 2.72Ω, pero estimación puede ser mayor)
    bounds = jnp.array([
        [0.005, 0.5],     # K0: Inductance numerator [H] - ampliado
        [0.001, 0.020],   # A: Geometric parameter [m] (1-20mm)
        [2.0, 25.0],      # R0: Base resistance [Ω] - ampliado por anomalía detectada
        [-0.05, 0.05]     # alpha: Temperature coefficient
    ])
    
    # Create VECTORIZED fitness function
    print("\n📊 Creating vectorized fitness function...")
    t0 = time.time()
    fitness_fn = create_vectorized_fitness(
        t_data, u_data, i_data, y_data, y0, m, g, dt
    )
    
    # Warmup JIT compilation
    print("⚙️  JIT compiling (first call)...")
    test_pop = jnp.ones((10, 4)) * 0.05
    _ = fitness_fn(test_pop)  # Trigger compilation
    print(f"   JIT compilation done in {time.time() - t0:.2f}s")
    
    # Run algorithms
    results = {}
    
    # 1. Genetic Algorithm
    print(f"\n🧬 Running Genetic Algorithm (pop={args.pop_size}, gen={args.generations})...")
    ga = GeneticAlgorithmJAX(
        fitness_fn=fitness_fn,
        bounds=bounds,
        pop_size=args.pop_size,
        elite_size=max(2, args.pop_size // 20),
        crossover_rate=0.8,
        mutation_rate=0.15,
        mutation_scale=0.1,
        tournament_size=3,
        seed=args.seed
    )
    
    t0 = time.time()
    ga_result = ga.run(n_generations=args.generations, verbose=True)
    ga_time = time.time() - t0
    
    results['GA'] = {
        'params': ga_result['best_params'],
        'fitness': ga_result['best_fitness'],
        'time': ga_time,
        'history': ga_result['history']
    }
    
    print(f"\n✅ GA completed in {ga_time:.2f}s")
    print(f"   Best params: K0={ga_result['best_params'][0]:.5f}, "
          f"A={ga_result['best_params'][1]:.5f}, "
          f"R0={ga_result['best_params'][2]:.3f}, "
          f"α={ga_result['best_params'][3]:.6f}")
    print(f"   Best fitness (MSE): {ga_result['best_fitness']:.6e}")
    
    # 2. Differential Evolution
    print(f"\n🔄 Running Differential Evolution (pop={args.pop_size}, gen={args.generations})...")
    de = DifferentialEvolutionJAX(
        fitness_fn=fitness_fn,
        bounds=bounds,
        pop_size=args.pop_size,
        F=0.8,
        CR=0.9,
        seed=args.seed + 1
    )
    
    t0 = time.time()
    de_result = de.run(n_generations=args.generations, verbose=True)
    de_time = time.time() - t0
    
    results['DE'] = {
        'params': de_result['best_params'],
        'fitness': de_result['best_fitness'],
        'time': de_time,
        'history': de_result['history']
    }
    
    print(f"\n✅ DE completed in {de_time:.2f}s")
    print(f"   Best params: K0={de_result['best_params'][0]:.5f}, "
          f"A={de_result['best_params'][1]:.5f}, "
          f"R0={de_result['best_params'][2]:.3f}, "
          f"α={de_result['best_params'][3]:.6f}")
    print(f"   Best fitness (MSE): {de_result['best_fitness']:.6e}")
    
    # Summary
    print("\n" + "="*60)
    print("📈 BENCHMARK SUMMARY")
    print("="*60)
    
    total_evals = args.pop_size * args.generations * 2  # GA + DE
    total_time = ga_time + de_time
    
    print(f"Total fitness evaluations: {total_evals:,}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_evals/total_time:,.0f} evaluations/second")
    print(f"\nComparison (if sequential at 0.1s/eval): {total_evals * 0.1 / 60:.1f} minutes")
    print(f"JAX Speedup: ~{(total_evals * 0.1) / total_time:.0f}x")
    
    # Winner
    winner = min(results.items(), key=lambda x: x[1]['fitness'])
    print(f"\n🏆 WINNER: {winner[0]} with MSE = {winner[1]['fitness']:.6e}")
    print(f"   K0 = {float(winner[1]['params'][0]):.5f} H")
    print(f"   A  = {float(winner[1]['params'][1]):.5f} m")
    print(f"   R0 = {float(winner[1]['params'][2]):.3f} Ω")
    print(f"   α  = {float(winner[1]['params'][3]):.6f}")
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        import json
        best_params = {
            'K0': float(winner[1]['params'][0]),
            'A': float(winner[1]['params'][1]),
            'R0': float(winner[1]['params'][2]),
            'alpha': float(winner[1]['params'][3]),
            'fitness': float(winner[1]['fitness']),
            'algorithm': winner[0]
        }
        with open(output_path / 'best_params_jax.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"\n💾 Results saved to {output_path}")
        
        # Plot convergence
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 5))
            plt.semilogy(results['GA']['history'], label=f"GA ({ga_time:.1f}s)", linewidth=2)
            plt.semilogy(results['DE']['history'], label=f"DE ({de_time:.1f}s)", linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness (MSE)')
            plt.title('JAX-Accelerated Parameter Identification')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(output_path / 'convergence_jax.png', dpi=150)
            plt.close()
            print(f"📊 Convergence plot saved")
        except Exception as e:
            print(f"Could not save plot: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='JAX-Accelerated Benchmark for Levitator Parameter Identification'
    )
    parser.add_argument('--data', type=str, default='data/datos_levitador.txt',
                        help='Path to experimental data file')
    parser.add_argument('--pop_size', type=int, default=200,
                        help='Population size (larger = more GPU parallelism)')
    parser.add_argument('--generations', type=int, default=500,
                        help='Number of generations')
    parser.add_argument('--subsample', type=int, default=10,
                        help='Subsample factor for data (reduces ODE length)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results/jax_benchmark',
                        help='Output directory for results')
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == '__main__':
    main()
