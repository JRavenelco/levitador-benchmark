#!/usr/bin/env python3
"""
Parameter Optimization Script
==============================

Phase 1: Identify physical parameters [K0, A, R0, α] using metaheuristics.

This script runs optimization algorithms to identify the physical parameters
of the magnetic levitator system:
- K0, A: Inductance L(y) = K0 / (1 + y/A)
- R0, α: Resistance R(t) ≈ R0 * (1 + α*(T(t) - T0))

Usage:
    python scripts/optimize_parameters.py --config config/pipeline_config.yaml
    python scripts/optimize_parameters.py --algorithms DE GWO --trials 10
    python scripts/optimize_parameters.py --data data/datos_levitador.txt --output results/
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import ParameterBenchmark
from src.optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)
from src.utils import load_config
from src.visualization import plot_convergence, plot_boxplot

# Algorithm registry
ALGORITHM_REGISTRY = {
    'RandomSearch': RandomSearch,
    'DE': DifferentialEvolution,
    'DifferentialEvolution': DifferentialEvolution,
    'GA': GeneticAlgorithm,
    'GeneticAlgorithm': GeneticAlgorithm,
    'GWO': GreyWolfOptimizer,
    'GreyWolfOptimizer': GreyWolfOptimizer,
    'ABC': ArtificialBeeColony,
    'ArtificialBeeColony': ArtificialBeeColony,
    'HBA': HoneyBadgerAlgorithm,
    'HoneyBadgerAlgorithm': HoneyBadgerAlgorithm,
    'SOA': ShrimpOptimizer,
    'ShrimpOptimizer': ShrimpOptimizer,
    'Tianji': TianjiOptimizer,
    'TianjiOptimizer': TianjiOptimizer,
}


def run_single_trial(algorithm_class, algorithm_config, problema, trial_num):
    """Run a single trial of an algorithm."""
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


def run_optimization(
    data_path: str,
    algorithms: List[str],
    n_trials: int = 5,
    output_dir: str = 'results',
    config_dict: Dict = None
):
    """
    Run parameter optimization with specified algorithms.
    
    Parameters
    ----------
    data_path : str
        Path to experimental data
    algorithms : list of str
        List of algorithm names to run
    n_trials : int
        Number of trials per algorithm
    output_dir : str
        Directory to save results
    config_dict : dict
        Configuration dictionary
    
    Returns
    -------
    results : dict
        Results dictionary with optimal parameters and metrics
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create problem instance
    print(f"\n{'='*70}")
    print(f"  Phase 1: Physical Parameter Identification")
    print(f"{'='*70}")
    print(f"Data: {data_path}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Trials: {n_trials}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")
    
    problema = ParameterBenchmark(data_path, verbose=True)
    
    # Store results
    all_results = {}
    best_overall = None
    best_overall_fitness = float('inf')
    
    # Run each algorithm
    for algo_name in algorithms:
        if algo_name not in ALGORITHM_REGISTRY:
            print(f"Warning: Algorithm '{algo_name}' not found. Skipping.")
            continue
        
        algo_class = ALGORITHM_REGISTRY[algo_name]
        print(f"\n--- Running {algo_name} ---")
        
        # Get algorithm configuration
        if config_dict and 'algorithms' in config_dict and algo_name in config_dict['algorithms']:
            algo_config = config_dict['algorithms'][algo_name].copy()
        else:
            # Default configuration based on algorithm type
            if algo_name in ['RandomSearch']:
                algo_config = {
                    'n_iterations': 1000,
                    'random_seed': 42,
                    'verbose': False
                }
            elif algo_name in ['GeneticAlgorithm', 'GA']:
                algo_config = {
                    'pop_size': 30,
                    'generations': 100,
                    'crossover_prob': 0.8,
                    'mutation_prob': 0.2,
                    'random_seed': 42,
                    'verbose': False
                }
            elif algo_name in ['DifferentialEvolution', 'DE']:
                algo_config = {
                    'pop_size': 30,
                    'max_iter': 100,
                    'F': 0.8,
                    'CR': 0.9,
                    'random_seed': 42,
                    'verbose': False
                }
            else:
                # Generic population-based algorithm
                algo_config = {
                    'pop_size': 30,
                    'max_iter': 100,
                    'random_seed': 42,
                    'verbose': False
                }
        
        # Remove 'enabled' key if present
        algo_config.pop('enabled', None)
        
        # Run trials
        trial_results = []
        trial_solutions = []
        trial_histories = []
        trial_runtimes = []
        
        for trial in range(n_trials):
            print(f"  Trial {trial+1}/{n_trials}... ", end='', flush=True)
            
            try:
                solution, fitness, runtime, history, evals = run_single_trial(
                    algo_class, algo_config, problema, trial
                )
                
                trial_results.append(fitness)
                trial_solutions.append(solution)
                trial_histories.append(history)
                trial_runtimes.append(runtime)
                
                print(f"Fitness: {fitness:.6e}, Time: {runtime:.2f}s")
                
                # Track best overall
                if fitness < best_overall_fitness:
                    best_overall_fitness = fitness
                    best_overall = solution.copy()
                
            except Exception as e:
                print(f"ERROR: {e}")
                trial_results.append(float('inf'))
                trial_solutions.append(None)
                trial_histories.append([])
                trial_runtimes.append(0.0)
        
        # Compute statistics
        valid_results = [r for r in trial_results if r < float('inf')]
        if valid_results:
            all_results[algo_name] = {
                'mean': float(np.mean(valid_results)),
                'std': float(np.std(valid_results)),
                'min': float(np.min(valid_results)),
                'max': float(np.max(valid_results)),
                'median': float(np.median(valid_results)),
                'all_fitness': [float(f) for f in trial_results],
                'all_solutions': trial_solutions,
                'histories': trial_histories,
                'runtimes': trial_runtimes,
                'n_success': len(valid_results),
                'n_trials': n_trials
            }
            
            print(f"  Summary: Mean={np.mean(valid_results):.6e}, "
                  f"Std={np.std(valid_results):.6e}, "
                  f"Best={np.min(valid_results):.6e}")
        else:
            print(f"  All trials failed for {algo_name}")
    
    # Save best parameters
    if best_overall is not None:
        params_dict = {
            'K0': float(best_overall[0]),
            'A': float(best_overall[1]),
            'R0': float(best_overall[2]),
            'alpha': float(best_overall[3]),
            'fitness': float(best_overall_fitness),
            'variable_names': problema.variable_names
        }
        
        params_path = output_path / 'parametros_optimos.json'
        with open(params_path, 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"  Best Parameters Found:")
        print(f"{'='*70}")
        print(f"  K0    = {best_overall[0]:.6f} H")
        print(f"  A     = {best_overall[1]:.6f} m")
        print(f"  R0    = {best_overall[2]:.4f} Ω")
        print(f"  alpha = {best_overall[3]:.6f} 1/°C")
        print(f"  MSE   = {best_overall_fitness:.6e}")
        print(f"{'='*70}")
        print(f"Saved to: {params_path}")
        
        # Visualize best solution
        try:
            vis_path = output_path / 'best_solution.png'
            problema.visualize_solution(best_overall, save_path=str(vis_path))
            print(f"Visualization saved to: {vis_path}")
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    # Save full results
    results_path = output_path / 'optimization_results.json'
    with open(results_path, 'w') as f:
        # Remove non-serializable parts
        save_results = {}
        for algo, data in all_results.items():
            save_results[algo] = {
                'mean': data['mean'],
                'std': data['std'],
                'min': data['min'],
                'max': data['max'],
                'median': data['median'],
                'all_fitness': data['all_fitness'],
                'n_success': data['n_success'],
                'n_trials': data['n_trials']
            }
        json.dump(save_results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    try:
        # Convergence plots
        for algo_name, data in all_results.items():
            if data['histories']:
                histories_dict = {f"{algo_name}_T{i+1}": h 
                                 for i, h in enumerate(data['histories']) if h}
                if histories_dict:
                    plot_path = output_path / f'convergence_{algo_name}.png'
                    plot_convergence(histories_dict, save_path=str(plot_path))
        
        # Boxplot comparison
        boxplot_data = {algo: data['all_fitness'] 
                       for algo, data in all_results.items() 
                       if data['all_fitness']}
        if boxplot_data:
            boxplot_path = output_path / 'comparison_boxplot.png'
            plot_boxplot(boxplot_data, save_path=str(boxplot_path))
        
        print("Visualizations created successfully.")
    except Exception as e:
        print(f"Warning: Could not create all visualizations: {e}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Physical Parameter Identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/optimize_parameters.py --config config/pipeline_config.yaml
  python scripts/optimize_parameters.py --algorithms DE GWO ABC --trials 10
  python scripts/optimize_parameters.py --data data/datos_levitador.txt --output results/
        """
    )
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--data', type=str, default='data/datos_levitador.txt',
                       help='Path to experimental data file')
    parser.add_argument('--algorithms', nargs='+', default=['DE', 'GWO', 'ABC'],
                       help='List of algorithms to run (DE, GA, GWO, ABC, HBA, SOA, Tianji, RandomSearch)')
    parser.add_argument('--trials', type=int, default=5,
                       help='Number of trials per algorithm')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config_dict = None
    if args.config:
        try:
            config_dict = load_config(args.config)
            if 'optimization' in config_dict:
                opt_config = config_dict['optimization']
                args.data = opt_config.get('data_path', args.data)
                args.trials = opt_config.get('n_trials', args.trials)
                args.output = opt_config.get('output_dir', args.output)
                if 'algorithms' in opt_config:
                    args.algorithms = opt_config['algorithms']
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            print("Using command-line arguments.")
    
    # Run optimization
    results = run_optimization(
        data_path=args.data,
        algorithms=args.algorithms,
        n_trials=args.trials,
        output_dir=args.output,
        config_dict=config_dict
    )
    
    print(f"\n{'='*70}")
    print("  Phase 1 Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
