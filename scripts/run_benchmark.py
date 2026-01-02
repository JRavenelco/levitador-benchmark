#!/usr/bin/env python3
"""
Run Benchmark Script
====================

Main script for running optimization algorithm benchmarks with the
levitador system. Supports configuration files, multiple trials,
and comprehensive result visualization.

Usage:
    python scripts/run_benchmark.py --config config/default.yaml
    python scripts/run_benchmark.py --config config/quick_test.yaml
    python scripts/run_benchmark.py --algorithms DE GA GWO --trials 5
"""

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark
from src.optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)
from src.utils import load_config, validate_config
from src.visualization import plot_convergence, plot_boxplot, plot_runtime, plot_comparison, plot_multiple_trials


# Algorithm registry
ALGORITHM_REGISTRY = {
    'RandomSearch': RandomSearch,
    'DifferentialEvolution': DifferentialEvolution,
    'GeneticAlgorithm': GeneticAlgorithm,
    'GreyWolfOptimizer': GreyWolfOptimizer,
    'ArtificialBeeColony': ArtificialBeeColony,
    'HoneyBadgerAlgorithm': HoneyBadgerAlgorithm,
    'ShrimpOptimizer': ShrimpOptimizer,
    'TianjiOptimizer': TianjiOptimizer,
}


def run_single_trial(algorithm_class, algorithm_config, problema, trial_num):
    """
    Run a single trial of an algorithm.
    
    Parameters
    ----------
    algorithm_class : class
        The optimizer class
    algorithm_config : dict
        Configuration dictionary for the algorithm
    problema : LevitadorBenchmark
        The benchmark problem
    trial_num : int
        Trial number (for seeding)
    
    Returns
    -------
    tuple
        (best_solution, best_fitness, runtime, history, evaluations)
    """
    # Update seed for each trial
    config = algorithm_config.copy()
    if 'random_seed' in config and config['random_seed'] is not None:
        config['random_seed'] = config['random_seed'] + trial_num
    
    # Create optimizer
    optimizer = algorithm_class(problema, **config)
    
    # Run optimization
    start_time = time.time()
    best_solution, best_fitness = optimizer.optimize()
    runtime = time.time() - start_time
    
    return best_solution, best_fitness, runtime, optimizer.history, optimizer.evaluations


def run_benchmark(config_path: str, selected_algorithms: List[str] = None,
                 n_trials: int = None):
    """
    Run the complete benchmark suite.
    
    Parameters
    ----------
    config_path : str
        Path to configuration YAML file
    selected_algorithms : list, optional
        List of algorithm names to run (if None, runs all enabled)
    n_trials : int, optional
        Number of trials (overrides config if provided)
    
    Returns
    -------
    dict
        Results dictionary
    """
    print("="*70)
    print("  LEVITADOR BENCHMARK - OPTIMIZATION ALGORITHM COMPARISON")
    print("="*70)
    
    # Load configuration
    print(f"\n[1/5] Loading configuration from: {config_path}")
    config = load_config(config_path)
    
    # Validate configuration
    try:
        validate_config(config)
        print("  ✓ Configuration validated")
    except ValueError as e:
        print(f"  ✗ Configuration validation failed: {e}")
        raise
    
    # Setup benchmark problem
    print("[2/5] Setting up benchmark problem...")
    benchmark_config = config['benchmark']
    data_path = benchmark_config.get('data_path')
    if data_path and not Path(data_path).exists():
        print(f"  Warning: Data file not found: {data_path}")
        print("  Using synthetic data instead.")
        data_path = None
    
    problema = LevitadorBenchmark(
        datos_reales_path=data_path,
        random_seed=benchmark_config.get('random_seed', 42),
        noise_level=benchmark_config.get('noise_level', 1e-5),
        verbose=False
    )
    print(f"  ✓ Problem initialized with {len(problema.t_real)} data points")
    
    # Determine which algorithms to run
    algorithms_config = config['algorithms']
    if selected_algorithms:
        algorithms_to_run = [alg for alg in selected_algorithms if alg in algorithms_config]
    else:
        algorithms_to_run = [alg for alg, cfg in algorithms_config.items() 
                            if cfg.get('enabled', True)]
    
    print(f"  ✓ Algorithms to run: {', '.join(algorithms_to_run)}")
    
    # Override trials if specified
    opt_config = config['optimization']
    trials = n_trials if n_trials is not None else opt_config.get('n_trials', 5)
    print(f"  ✓ Running {trials} trials per algorithm")
    
    # Run benchmarks
    print(f"\n[3/5] Running optimization benchmarks...")
    results = {
        'fitness': {},
        'runtime': {},
        'histories': {},
        'solutions': {},
        'evaluations': {},
        'statistics': {}
    }
    
    for alg_name in algorithms_to_run:
        print(f"\n  Algorithm: {alg_name}")
        print("  " + "-"*50)
        
        if alg_name not in ALGORITHM_REGISTRY:
            print(f"  ⚠ Algorithm not found in registry: {alg_name}")
            continue
        
        algorithm_class = ALGORITHM_REGISTRY[alg_name]
        alg_config = algorithms_config[alg_name].copy()
        
        # Remove non-parameter keys
        alg_config.pop('enabled', None)
        
        fitness_values = []
        runtime_values = []
        histories = []
        solutions = []
        evaluations_list = []
        
        for trial in range(trials):
            print(f"    Trial {trial+1}/{trials}...", end=" ")
            
            try:
                best_sol, best_fit, runtime, history, evals = run_single_trial(
                    algorithm_class, alg_config, problema, trial
                )
                
                fitness_values.append(best_fit)
                runtime_values.append(runtime)
                histories.append(history)
                solutions.append(best_sol)
                evaluations_list.append(evals)
                
                print(f"Fitness: {best_fit:.6e}, Time: {runtime:.2f}s")
                
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        if fitness_values:
            results['fitness'][alg_name] = fitness_values
            results['runtime'][alg_name] = runtime_values
            results['histories'][alg_name] = histories
            results['solutions'][alg_name] = solutions
            results['evaluations'][alg_name] = evaluations_list
            
            # Calculate statistics
            results['statistics'][alg_name] = {
                'best': np.min(fitness_values),
                'worst': np.max(fitness_values),
                'mean': np.mean(fitness_values),
                'median': np.median(fitness_values),
                'std': np.std(fitness_values),
                'mean_runtime': np.mean(runtime_values),
                'mean_evaluations': np.mean(evaluations_list),
            }
            
            print(f"    Summary: Best={np.min(fitness_values):.6e}, "
                  f"Mean={np.mean(fitness_values):.6e}, "
                  f"Std={np.std(fitness_values):.6e}")
    
    # Display results
    print(f"\n[4/5] Results Summary")
    print("="*70)
    print(f"{'Algorithm':<25} {'Best':<12} {'Mean':<12} {'Std':<12} {'Time(s)':<10}")
    print("-"*70)
    for alg_name in results['statistics']:
        stats = results['statistics'][alg_name]
        print(f"{alg_name:<25} {stats['best']:<12.6e} {stats['mean']:<12.6e} "
              f"{stats['std']:<12.6e} {stats['mean_runtime']:<10.2f}")
    
    # Generate visualizations
    viz_config = config.get('visualization', {})
    if viz_config.get('save_plots', True):
        print(f"\n[5/5] Generating visualizations...")
        plot_dir = Path(viz_config.get('plot_dir', 'plots'))
        plot_dir.mkdir(parents=True, exist_ok=True)
        dpi = viz_config.get('dpi', 300)
        
        # Convergence plot (single trial)
        if viz_config.get('plot_convergence', True) and results['histories']:
            single_histories = {name: histories[0] 
                              for name, histories in results['histories'].items()}
            plot_convergence(single_histories, 
                           save_path=str(plot_dir / 'convergence.png'),
                           dpi=dpi)
        
        # Multi-trial convergence
        if results['histories']:
            plot_multiple_trials(results['histories'],
                               save_path=str(plot_dir / 'convergence_trials.png'),
                               dpi=dpi)
        
        # Boxplot
        if viz_config.get('plot_boxplot', True) and results['fitness']:
            plot_boxplot(results['fitness'],
                       save_path=str(plot_dir / 'boxplot.png'),
                       dpi=dpi)
        
        # Runtime comparison
        if viz_config.get('plot_runtime', True) and results['runtime']:
            plot_runtime(results['runtime'],
                       save_path=str(plot_dir / 'runtime.png'),
                       dpi=dpi)
        
        print(f"  ✓ Plots saved to: {plot_dir}")
    
    # Save results to file
    if opt_config.get('save_results', True):
        output_dir = Path(opt_config.get('output_dir', 'results'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save statistics as JSON
        stats_file = output_dir / 'statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(results['statistics'], f, indent=2)
        print(f"\n  ✓ Statistics saved to: {stats_file}")
        
        # Save raw results as NPZ
        raw_file = output_dir / 'raw_results.npz'
        np.savez(raw_file,
                fitness=results['fitness'],
                runtime=results['runtime'],
                evaluations=results['evaluations'])
        print(f"  ✓ Raw results saved to: {raw_file}")
    
    print("\n" + "="*70)
    print("✅ Benchmark completed successfully!")
    print("="*70)
    
    return results


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description='Run optimization algorithm benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py --config config/default.yaml
  python scripts/run_benchmark.py --config config/quick_test.yaml
  python scripts/run_benchmark.py --algorithms DE GA --trials 10
        """
    )
    
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--algorithms', nargs='+', type=str,
                       help='List of algorithms to run (overrides config)')
    parser.add_argument('--trials', type=int,
                       help='Number of trials per algorithm (overrides config)')
    
    args = parser.parse_args()
    
    try:
        results = run_benchmark(args.config, args.algorithms, args.trials)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
