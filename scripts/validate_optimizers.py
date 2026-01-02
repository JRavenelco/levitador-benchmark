#!/usr/bin/env python3
"""
Validation Script
================

Quick validation that all optimizers work correctly.
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


def validate_optimizer(name, optimizer_class, config):
    """Validate a single optimizer."""
    try:
        problema = LevitadorBenchmark(random_seed=42, verbose=False)
        optimizer = optimizer_class(problema, **config)
        best_solution, best_fitness = optimizer.optimize()
        
        # Basic validation checks
        assert best_solution is not None, f"{name}: No solution returned"
        assert len(best_solution) == 3, f"{name}: Solution has wrong dimension"
        assert best_fitness is not None, f"{name}: No fitness returned"
        assert best_fitness > 0, f"{name}: Invalid fitness value"
        assert optimizer.evaluations > 0, f"{name}: No evaluations performed"
        assert len(optimizer.history) > 0, f"{name}: No history recorded"
        
        print(f"✓ {name:<25} Fitness: {best_fitness:.6e}  Evals: {optimizer.evaluations}")
        return True
    except Exception as e:
        print(f"✗ {name:<25} ERROR: {str(e)}")
        return False


def main():
    """Run validation for all optimizers."""
    print("="*70)
    print("OPTIMIZER VALIDATION")
    print("="*70)
    print()
    
    optimizers = [
        ('RandomSearch', RandomSearch, {'n_iterations': 100, 'random_seed': 42, 'verbose': False}),
        ('DifferentialEvolution', DifferentialEvolution, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
        ('GeneticAlgorithm', GeneticAlgorithm, {'pop_size': 15, 'generations': 10, 'random_seed': 42, 'verbose': False}),
        ('GreyWolfOptimizer', GreyWolfOptimizer, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
        ('ArtificialBeeColony', ArtificialBeeColony, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
        ('HoneyBadgerAlgorithm', HoneyBadgerAlgorithm, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
        ('ShrimpOptimizer', ShrimpOptimizer, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
        ('TianjiOptimizer', TianjiOptimizer, {'pop_size': 15, 'max_iter': 10, 'random_seed': 42, 'verbose': False}),
    ]
    
    results = []
    for name, optimizer_class, config in optimizers:
        success = validate_optimizer(name, optimizer_class, config)
        results.append((name, success))
    
    print()
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All optimizers validated successfully!")
        return 0
    else:
        print("\n❌ Some optimizers failed validation")
        return 1


if __name__ == '__main__':
    sys.exit(main())
