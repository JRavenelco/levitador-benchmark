#!/usr/bin/env python3
"""
Benchmark Runner Script
=======================

Run benchmarks across multiple optimization algorithms and compare results.

Usage:
    python scripts/run_benchmark.py --config config/default.yaml
    python scripts/run_benchmark.py --config config/quick_test.yaml
    python scripts/run_benchmark.py --optimizer GreyWolfOptimizer --trials 10
"""

import sys
import argparse
import json
import time
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark
from src.utils.config_loader import load_config
from src.optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)
from src.visualization.plots import (
    plot_convergence, plot_comparison_boxplot,
    plot_performance_metrics, plot_runtime_comparison
)


# Optimizer registry
OPTIMIZERS = {
    'RandomSearch': RandomSearch,
    'DifferentialEvolution': DifferentialEvolution,
    'GeneticAlgorithm': GeneticAlgorithm,
    'GreyWolfOptimizer': GreyWolfOptimizer,
    'ArtificialBeeColony': ArtificialBeeColony,
    'HoneyBadgerAlgorithm': HoneyBadgerAlgorithm,
    'ShrimpOptimizer': ShrimpOptimizer,
    'TianjiOptimizer': TianjiOptimizer,
}


def run_single_trial(optimizer_class, problema, config):
    """Run a single optimization trial."""
    optimizer = optimizer_class(problema, **config)
    start_time = time.time()
    best_solution, best_fitness = optimizer.optimize()
    runtime = time.time() - start_time
    
    return {
        'solution': best_solution,
        'fitness': best_fitness,
        'runtime': runtime,
        'evaluations': optimizer.evaluations,
        'history': optimizer.history
    }


def run_benchmark(config_path: str, optimizer_name: str = None, n_trials: int = None):
    """
    Run benchmark with given configuration.
    
    Args:
        config_path: Path to YAML configuration file
        optimizer_name: Optional specific optimizer to run
        n_trials: Optional override for number of trials
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup benchmark
    bench_config = config.get('benchmark', {})
    data_path = bench_config.get('data_path')
    random_seed = bench_config.get('random_seed', 42)
    
    print("="*70)
    print("LEVITADOR BENCHMARK - OPTIMIZER COMPARISON")
    print("="*70)
    print(f"\nConfiguration: {config_path}")
    print(f"Data: {data_path if data_path else 'Synthetic'}")
    print(f"Random seed: {random_seed}\n")
    
    # Create benchmark instance
    problema = LevitadorBenchmark(
        datos_reales_path=data_path,
        random_seed=random_seed,
        verbose=bench_config.get('verbose', True)
    )
    
    # Get benchmark settings
    bench_settings = config.get('benchmark_settings', {})
    if n_trials is None:
        n_trials = bench_settings.get('n_trials', 10)
    output_dir = Path(bench_settings.get('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select optimizers to run
    optimizer_configs = config.get('optimizers', {})
    if optimizer_name:
        if optimizer_name not in optimizer_configs:
            print(f"Error: Optimizer '{optimizer_name}' not found in config")
            return
        optimizer_configs = {optimizer_name: optimizer_configs[optimizer_name]}
    
    # Run benchmarks
    all_results = {}
    all_histories = {}
    all_runtimes = {}
    
    for opt_name, opt_config in optimizer_configs.items():
        if opt_name not in OPTIMIZERS:
            print(f"Warning: Optimizer '{opt_name}' not implemented, skipping...")
            continue
        
        print(f"\n{'='*70}")
        print(f"Running: {opt_name}")
        print(f"{'='*70}")
        print(f"Trials: {n_trials}")
        
        optimizer_class = OPTIMIZERS[opt_name]
        trial_results = []
        trial_runtimes = []
        best_history = None
        best_fitness_overall = float('inf')
        
        for trial in range(n_trials):
            print(f"\n  Trial {trial+1}/{n_trials}:")
            
            result = run_single_trial(optimizer_class, problema, opt_config)
            trial_results.append(result['fitness'])
            trial_runtimes.append(result['runtime'])
            
            # Keep best history for plotting
            if result['fitness'] < best_fitness_overall:
                best_fitness_overall = result['fitness']
                best_history = result['history']
            
            print(f"    Fitness: {result['fitness']:.6e}")
            print(f"    Runtime: {result['runtime']:.2f}s")
            print(f"    Evaluations: {result['evaluations']}")
        
        # Store results
        all_results[opt_name] = trial_results
        all_histories[opt_name] = best_history
        all_runtimes[opt_name] = trial_runtimes
        
        # Print summary for this optimizer
        print(f"\n  Summary for {opt_name}:")
        print(f"    Mean fitness: {np.mean(trial_results):.6e} ± {np.std(trial_results):.6e}")
        print(f"    Best fitness: {np.min(trial_results):.6e}")
        print(f"    Worst fitness: {np.max(trial_results):.6e}")
        print(f"    Mean runtime: {np.mean(trial_runtimes):.2f}s ± {np.std(trial_runtimes):.2f}s")
    
    # Generate comparison metrics
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}\n")
    
    metrics = {}
    for opt_name, results in all_results.items():
        metrics[opt_name] = {
            'mean': np.mean(results),
            'std': np.std(results),
            'best': np.min(results),
            'worst': np.max(results),
            'median': np.median(results)
        }
    
    # Print comparison table
    print(f"{'Algorithm':<25} {'Mean':<15} {'Std':<15} {'Best':<15} {'Worst':<15}")
    print("-" * 85)
    for opt_name in sorted(metrics.keys(), key=lambda x: metrics[x]['mean']):
        m = metrics[opt_name]
        print(f"{opt_name:<25} {m['mean']:<15.6e} {m['std']:<15.6e} "
              f"{m['best']:<15.6e} {m['worst']:<15.6e}")
    
    # Save results to JSON
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': str(config_path),
            'n_trials': n_trials,
            'results': {k: [float(v) for v in vals] for k, vals in all_results.items()},
            'runtimes': {k: [float(v) for v in vals] for k, vals in all_runtimes.items()},
            'metrics': {k: {mk: float(mv) for mk, mv in m.items()} 
                       for k, m in metrics.items()}
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Generate plots
    if bench_settings.get('save_history', True) and all_histories:
        print("\nGenerating plots...")
        
        # Convergence curves
        plot_convergence(
            all_histories,
            save_path=output_dir / 'convergence.png',
            title='Convergence Comparison'
        )
        
        # Box plot
        plot_comparison_boxplot(
            all_results,
            save_path=output_dir / 'comparison_boxplot.png',
            title='Algorithm Comparison'
        )
        
        # Performance metrics
        plot_performance_metrics(
            metrics,
            save_path=output_dir / 'performance_metrics.png'
        )
        
        # Runtime comparison
        plot_runtime_comparison(
            all_runtimes,
            save_path=output_dir / 'runtime_comparison.png'
        )
        
        print(f"Plots saved to: {output_dir}/")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run optimization benchmark for levitador system',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to YAML configuration file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--optimizer', '-o',
        type=str,
        default=None,
        help='Run only specific optimizer (optional)'
    )
    
    parser.add_argument(
        '--trials', '-t',
        type=int,
        default=None,
        help='Number of trials per optimizer (overrides config)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available optimizers'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable optimizers:")
        for name in OPTIMIZERS.keys():
            print(f"  - {name}")
        return
    
    # Run benchmark
    run_benchmark(args.config, args.optimizer, args.trials)


if __name__ == '__main__':
    main()
