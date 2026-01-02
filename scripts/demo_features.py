#!/usr/bin/env python3
"""
Complete Feature Demonstration
==============================

Demonstrates all features of the modular optimization framework.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark
from src.optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)
from src.utils.config_loader import load_config


def demo_basic_usage():
    """Demonstrate basic usage."""
    print("\n" + "="*70)
    print("DEMO 1: BASIC USAGE")
    print("="*70)
    
    # Create problem
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    # Run an optimizer
    print("\nRunning Grey Wolf Optimizer...")
    gwo = GreyWolfOptimizer(problema, pop_size=20, max_iter=20, random_seed=42, verbose=False)
    best_solution, best_fitness = gwo.optimize()
    
    print(f"\n✓ Optimization complete")
    print(f"  Best solution: k0={best_solution[0]:.6f}, k={best_solution[1]:.6f}, a={best_solution[2]:.6f}")
    print(f"  Fitness (MSE): {best_fitness:.6e}")
    print(f"  Evaluations: {gwo.evaluations}")


def demo_multiple_algorithms():
    """Demonstrate comparing multiple algorithms."""
    print("\n" + "="*70)
    print("DEMO 2: COMPARING MULTIPLE ALGORITHMS")
    print("="*70)
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    algorithms = [
        ('Random Search', RandomSearch, {'n_iterations': 100}),
        ('Differential Evolution', DifferentialEvolution, {'pop_size': 15, 'max_iter': 10}),
        ('Grey Wolf', GreyWolfOptimizer, {'pop_size': 15, 'max_iter': 10}),
        ('Artificial Bee Colony', ArtificialBeeColony, {'pop_size': 15, 'max_iter': 10}),
    ]
    
    results = {}
    for name, algo_class, config in algorithms:
        print(f"\nRunning {name}...")
        algo = algo_class(problema, random_seed=42, verbose=False, **config)
        _, fitness = algo.optimize()
        results[name] = fitness
        print(f"  Fitness: {fitness:.6e}")
    
    print("\n" + "-"*70)
    print("RANKING:")
    for i, (name, fitness) in enumerate(sorted(results.items(), key=lambda x: x[1]), 1):
        print(f"  {i}. {name:<25} {fitness:.6e}")


def demo_config_loading():
    """Demonstrate configuration loading."""
    print("\n" + "="*70)
    print("DEMO 3: CONFIGURATION LOADING")
    print("="*70)
    
    config_path = 'config/quick_test.yaml'
    print(f"\nLoading configuration from: {config_path}")
    
    try:
        config = load_config(config_path)
        print(f"✓ Configuration loaded successfully")
        print(f"\nBenchmark settings:")
        for key, value in config.get('benchmark', {}).items():
            print(f"  {key}: {value}")
        
        print(f"\nConfigured optimizers: {len(config.get('optimizers', {}))}")
        for opt_name in config.get('optimizers', {}).keys():
            print(f"  - {opt_name}")
    except Exception as e:
        print(f"✗ Error loading config: {e}")


def demo_all_optimizers():
    """Demonstrate all 8 optimizers."""
    print("\n" + "="*70)
    print("DEMO 4: ALL 8 OPTIMIZERS")
    print("="*70)
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    optimizers = [
        ('1. Random Search', RandomSearch, {'n_iterations': 100}),
        ('2. Differential Evolution', DifferentialEvolution, {'pop_size': 15, 'max_iter': 10}),
        ('3. Genetic Algorithm', GeneticAlgorithm, {'pop_size': 15, 'generations': 10}),
        ('4. Grey Wolf Optimizer', GreyWolfOptimizer, {'pop_size': 15, 'max_iter': 10}),
        ('5. Artificial Bee Colony', ArtificialBeeColony, {'pop_size': 15, 'max_iter': 10}),
        ('6. Honey Badger Algorithm', HoneyBadgerAlgorithm, {'pop_size': 15, 'max_iter': 10}),
        ('7. Shrimp Optimizer', ShrimpOptimizer, {'pop_size': 15, 'max_iter': 10}),
        ('8. Tianji Horse Racing', TianjiOptimizer, {'pop_size': 15, 'max_iter': 10}),
    ]
    
    print("\nRunning all optimizers (quick test)...\n")
    
    for name, algo_class, config in optimizers:
        try:
            algo = algo_class(problema, random_seed=42, verbose=False, **config)
            _, fitness = algo.optimize()
            print(f"✓ {name:<30} Fitness: {fitness:.6e}")
        except Exception as e:
            print(f"✗ {name:<30} ERROR: {str(e)}")


def main():
    """Run all demos."""
    print("\n" + "🧲"*35)
    print("   LEVITADOR BENCHMARK - COMPLETE FEATURE DEMO")
    print("🧲"*35)
    
    demo_basic_usage()
    demo_multiple_algorithms()
    demo_config_loading()
    demo_all_optimizers()
    
    print("\n" + "="*70)
    print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run full benchmark: python scripts/run_benchmark.py --config config/default.yaml")
    print("  2. Try the Jupyter notebook: jupyter notebook notebooks/parameter_identification_demo.ipynb")
    print("  3. Explore configurations: config/default.yaml, config/quick_test.yaml")
    print()


if __name__ == '__main__':
    main()
