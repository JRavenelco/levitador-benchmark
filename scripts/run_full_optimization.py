#!/usr/bin/env python3
"""
Full Optimization Benchmark Script
===================================

Executes a comprehensive benchmark of all metaheuristic algorithms for
optimizing magnetic levitator parameters and compares results with
theoretical reference values.

This script:
- Loads experimental data from data/datos_levitador.txt
- Runs all 8 available algorithms: DE, GWO, ABC, HBA, SOA, Tianji, GA, RandomSearch
- Executes multiple trials (default: 5) for statistical analysis
- Compares results with theoretical values (k₀=0.0363, k=0.0035, a=0.0052)
- Generates detailed reports and visualizations

Usage:
    python scripts/run_full_optimization.py
    python scripts/run_full_optimization.py --config config/full_optimization.yaml
    python scripts/run_full_optimization.py --trials 10 --seed 123
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from levitador_benchmark import LevitadorBenchmark
from src.optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm,
    ShrimpOptimizer, TianjiOptimizer
)
from src.utils import load_config

# Algorithm registry with short names
ALGORITHM_REGISTRY = {
    'DE': DifferentialEvolution,
    'DifferentialEvolution': DifferentialEvolution,
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
    'GA': GeneticAlgorithm,
    'GeneticAlgorithm': GeneticAlgorithm,
    'RandomSearch': RandomSearch,
}

# Algorithm display names
ALGORITHM_NAMES = {
    'DifferentialEvolution': 'DE',
    'GreyWolfOptimizer': 'GWO',
    'ArtificialBeeColony': 'ABC',
    'HoneyBadgerAlgorithm': 'HBA',
    'ShrimpOptimizer': 'SOA',
    'TianjiOptimizer': 'Tianji',
    'GeneticAlgorithm': 'GA',
    'RandomSearch': 'Random',
}


def calculate_percentage_error(found_value: float, theoretical_value: float) -> float:
    """Calculate percentage error."""
    return abs((found_value - theoretical_value) / theoretical_value) * 100


def run_single_trial(algorithm_class, algorithm_config, problema, trial_num: int) -> Tuple:
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


def generate_convergence_plot(results: Dict, output_dir: Path, dpi: int = 300):
    """Generate comparative convergence curves."""
    print("  Generating convergence plot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot convergence for each algorithm (first trial)
    for algo_name, data in results.items():
        if data['histories']:
            history = data['histories'][0]
            if history:
                display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
                ax.plot(history, label=display_name, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Convergence Curves Comparison - All Algorithms', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_dir / 'convergence_curves.png'}")


def generate_boxplot(results: Dict, output_dir: Path, dpi: int = 300):
    """Generate boxplot for performance comparison."""
    print("  Generating boxplot...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    data_to_plot = []
    labels = []
    for algo_name, data in sorted(results.items(), key=lambda x: x[1]['statistics']['best']):
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        labels.append(display_name)
        data_to_plot.append(data['all_fitness'])
    
    # Create boxplot
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('MSE (log scale)', fontsize=12)
    ax.set_title('Performance Comparison - All Algorithms (5 trials)', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_boxplot.png', dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_dir / 'performance_boxplot.png'}")


def generate_comparison_table(results: Dict, theoretical_values: Dict, output_dir: Path):
    """Generate comparison table with theoretical values."""
    print("  Generating comparison table...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    headers = ['Algorithm', 'Best MSE', 'k₀ (H)', 'Δk₀ (%)', 'k (H)', 'Δk (%)', 'a (m)', 'Δa (%)']
    table_data = []
    
    # Theoretical values
    k0_theo = theoretical_values['k0']
    k_theo = theoretical_values['k']
    a_theo = theoretical_values['a']
    
    for algo_name, data in sorted(results.items(), key=lambda x: x[1]['statistics']['best']):
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        best_sol = data['best_solution']
        best_mse = data['statistics']['best']
        
        # Calculate percentage errors
        k0_error = calculate_percentage_error(best_sol[0], k0_theo)
        k_error = calculate_percentage_error(best_sol[1], k_theo)
        a_error = calculate_percentage_error(best_sol[2], a_theo)
        
        row = [
            display_name,
            f"{best_mse:.2e}",
            f"{best_sol[0]:.4f}",
            f"{k0_error:.1f}%",
            f"{best_sol[1]:.4f}",
            f"{k_error:.1f}%",
            f"{best_sol[2]:.4f}",
            f"{a_error:.1f}%"
        ]
        table_data.append(row)
    
    # Add theoretical reference row
    table_data.insert(0, [
        'Theoretical',
        'Reference',
        f"{k0_theo:.4f}",
        '0.0%',
        f"{k_theo:.4f}",
        '0.0%',
        f"{a_theo:.4f}",
        '0.0%'
    ])
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     colWidths=[0.12, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style the theoretical row
    for i in range(len(headers)):
        table[(1, i)].set_facecolor('#FFD966')
        table[(1, i)].set_text_props(weight='bold')
    
    # Alternate row colors
    for i in range(2, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F2F2F2')
    
    plt.title('Comparison with Theoretical Reference Values', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'comparison_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved: {output_dir / 'comparison_table.png'}")


def generate_markdown_report(results: Dict, theoretical_values: Dict, 
                            config: Dict, output_dir: Path):
    """Generate detailed markdown report."""
    print("  Generating markdown report...")
    
    report_lines = []
    report_lines.append("# Full Optimization Benchmark Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Configuration
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append(f"- **Data source:** `{config['benchmark']['data_path']}`")
    report_lines.append(f"- **Trials per algorithm:** {config['benchmark']['n_trials']}")
    report_lines.append(f"- **Random seed:** {config['benchmark']['random_seed']}")
    report_lines.append(f"- **Population size:** {config['optimization']['pop_size']}")
    report_lines.append(f"- **Max iterations:** {config['optimization']['max_iter']}")
    report_lines.append("")
    
    # Theoretical reference
    report_lines.append("## Theoretical Reference Values")
    report_lines.append("")
    report_lines.append("| Parameter | Value | Unit |")
    report_lines.append("|-----------|-------|------|")
    report_lines.append(f"| k₀ | {theoretical_values['k0']:.4f} | H |")
    report_lines.append(f"| k | {theoretical_values['k']:.4f} | H |")
    report_lines.append(f"| a | {theoretical_values['a']:.4f} | m |")
    report_lines.append(f"| MSE threshold | {theoretical_values['mse_threshold']:.2e} | - |")
    report_lines.append("")
    
    # Rankings
    report_lines.append("## Algorithm Rankings")
    report_lines.append("")
    report_lines.append("### By Best MSE")
    report_lines.append("")
    report_lines.append("| Rank | Algorithm | Best MSE | Mean MSE | Std Dev | Mean Time (s) |")
    report_lines.append("|------|-----------|----------|----------|---------|---------------|")
    
    sorted_algos = sorted(results.items(), key=lambda x: x[1]['statistics']['best'])
    for rank, (algo_name, data) in enumerate(sorted_algos, 1):
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        stats = data['statistics']
        report_lines.append(
            f"| {rank} | {display_name} | {stats['best']:.6e} | "
            f"{stats['mean']:.6e} | {stats['std']:.6e} | {stats['mean_runtime']:.2f} |"
        )
    report_lines.append("")
    
    # Detailed comparison with theoretical values
    report_lines.append("## Comparison with Theoretical Values")
    report_lines.append("")
    report_lines.append("| Algorithm | Best MSE | k₀ | Δk₀ (%) | k | Δk (%) | a | Δa (%) | Within 10%? |")
    report_lines.append("|-----------|----------|-----|---------|---|--------|---|--------|-------------|")
    
    for algo_name, data in sorted_algos:
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        best_sol = data['best_solution']
        stats = data['statistics']
        
        k0_error = calculate_percentage_error(best_sol[0], theoretical_values['k0'])
        k_error = calculate_percentage_error(best_sol[1], theoretical_values['k'])
        a_error = calculate_percentage_error(best_sol[2], theoretical_values['a'])
        
        within_10pct = "✅" if max(k0_error, k_error, a_error) <= 10 else "❌"
        
        report_lines.append(
            f"| {display_name} | {stats['best']:.6e} | "
            f"{best_sol[0]:.4f} | {k0_error:.1f} | "
            f"{best_sol[1]:.4f} | {k_error:.1f} | "
            f"{best_sol[2]:.4f} | {a_error:.1f} | "
            f"{within_10pct} |"
        )
    report_lines.append("")
    
    # Statistical summary
    report_lines.append("## Statistical Summary")
    report_lines.append("")
    
    for algo_name, data in sorted_algos:
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        stats = data['statistics']
        
        report_lines.append(f"### {display_name}")
        report_lines.append("")
        report_lines.append(f"- **Best fitness:** {stats['best']:.6e}")
        report_lines.append(f"- **Worst fitness:** {stats['worst']:.6e}")
        report_lines.append(f"- **Mean fitness:** {stats['mean']:.6e}")
        report_lines.append(f"- **Median fitness:** {stats['median']:.6e}")
        report_lines.append(f"- **Std deviation:** {stats['std']:.6e}")
        report_lines.append(f"- **Mean runtime:** {stats['mean_runtime']:.2f} s")
        report_lines.append(f"- **Mean evaluations:** {stats['mean_evaluations']:.0f}")
        report_lines.append("")
        report_lines.append(f"**Best solution:** k₀={data['best_solution'][0]:.4f}, "
                          f"k={data['best_solution'][1]:.4f}, a={data['best_solution'][2]:.4f}")
        report_lines.append("")
    
    # Success criteria
    report_lines.append("## Success Criteria")
    report_lines.append("")
    
    best_algo, best_data = sorted_algos[0]
    best_mse = best_data['statistics']['best']
    best_sol = best_data['best_solution']
    
    k0_error = calculate_percentage_error(best_sol[0], theoretical_values['k0'])
    k_error = calculate_percentage_error(best_sol[1], theoretical_values['k'])
    a_error = calculate_percentage_error(best_sol[2], theoretical_values['a'])
    max_error = max(k0_error, k_error, a_error)
    
    target_mse = theoretical_values.get('mse_threshold', 1e-7)
    
    report_lines.append(f"- **MSE < {target_mse:.0e}:** {'✅ PASS' if best_mse < target_mse else f'❌ FAIL ({best_mse:.2e})'}")
    report_lines.append(f"- **Parameters within 10% of theoretical:** {'✅ PASS' if max_error <= 10 else f'❌ FAIL (max error: {max_error:.1f}%)'}")
    report_lines.append("")
    
    # Write report
    report_path = output_dir / 'BENCHMARK_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"    ✓ Saved: {report_path}")


def run_full_optimization(config_path: str = None):
    """
    Run the complete optimization benchmark.
    
    Parameters
    ----------
    config_path : str, optional
        Path to configuration YAML file
    
    Returns
    -------
    dict
        Complete results dictionary
    """
    print("=" * 80)
    print("  FULL OPTIMIZATION BENCHMARK - MAGNETIC LEVITATOR")
    print("=" * 80)
    print()
    
    # Load configuration
    if config_path is None:
        config_path = 'config/full_optimization.yaml'
    
    print(f"[1/6] Loading configuration from: {config_path}")
    config = load_config(config_path)
    print("  ✓ Configuration loaded")
    print()
    
    # Extract settings
    benchmark_config = config['benchmark']
    theoretical_values = config['theoretical_values']
    algorithms_config = config['algorithms']
    optimization_config = config['optimization']
    
    data_path = benchmark_config['data_path']
    n_trials = benchmark_config['n_trials']
    random_seed = benchmark_config['random_seed']
    output_dir = Path(benchmark_config['output_dir'])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  ✓ Output directory: {output_dir}")
    print()
    
    # Setup benchmark problem
    print("[2/6] Setting up benchmark problem...")
    if data_path and not Path(data_path).exists():
        print(f"  ✗ Error: Data file not found: {data_path}")
        return None
    
    if not data_path:
        print("  ⚠ No data file specified, using synthetic data")
        data_path = None
    
    problema = LevitadorBenchmark(
        datos_reales_path=data_path,
        random_seed=random_seed,
        verbose=False
    )
    print(f"  ✓ Problem initialized with {len(problema.t_real)} data points")
    print(f"  ✓ Parameters to optimize: {problema.variable_names}")
    print()
    
    # Run algorithms
    print(f"[3/6] Running optimization algorithms ({n_trials} trials each)...")
    print()
    
    results = {}
    
    # Get enabled algorithms
    enabled_algorithms = [name for name, cfg in algorithms_config.items() 
                         if cfg.get('enabled', True)]
    
    for algo_name in enabled_algorithms:
        if algo_name not in ALGORITHM_REGISTRY:
            print(f"  ⚠ Warning: Algorithm '{algo_name}' not found in registry. Skipping.")
            continue
        
        print(f"  Algorithm: {ALGORITHM_NAMES.get(algo_name, algo_name)} ({algo_name})")
        print("  " + "-" * 60)
        
        algorithm_class = ALGORITHM_REGISTRY[algo_name]
        algo_config = algorithms_config[algo_name].copy()
        algo_config.pop('enabled', None)
        
        all_fitness = []
        all_solutions = []
        all_histories = []
        all_runtimes = []
        all_evaluations = []
        
        for trial in range(n_trials):
            print(f"    Trial {trial + 1}/{n_trials}... ", end='', flush=True)
            
            try:
                solution, fitness, runtime, history, evals = run_single_trial(
                    algorithm_class, algo_config, problema, trial
                )
                
                all_fitness.append(fitness)
                all_solutions.append(solution)
                all_histories.append(history)
                all_runtimes.append(runtime)
                all_evaluations.append(evals)
                
                print(f"MSE: {fitness:.6e}, Time: {runtime:.2f}s")
                
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_fitness:
            # Find best solution
            best_idx = np.argmin(all_fitness)
            best_solution = all_solutions[best_idx]
            
            # Calculate statistics
            results[algo_name] = {
                'all_fitness': all_fitness,
                'all_solutions': all_solutions,
                'histories': all_histories,
                'runtimes': all_runtimes,
                'evaluations': all_evaluations,
                'best_solution': best_solution,
                'statistics': {
                    'best': np.min(all_fitness),
                    'worst': np.max(all_fitness),
                    'mean': np.mean(all_fitness),
                    'median': np.median(all_fitness),
                    'std': np.std(all_fitness),
                    'mean_runtime': np.mean(all_runtimes),
                    'mean_evaluations': np.mean(all_evaluations),
                }
            }
            
            stats = results[algo_name]['statistics']
            print(f"    Summary: Best={stats['best']:.6e}, Mean={stats['mean']:.6e}, "
                  f"Std={stats['std']:.6e}")
        
        print()
    
    # Display summary
    print("[4/6] Results Summary")
    print("=" * 80)
    print(f"{'Algorithm':<20} {'Best MSE':<15} {'Mean MSE':<15} {'Std Dev':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['statistics']['best'])
    for algo_name, data in sorted_results:
        display_name = ALGORITHM_NAMES.get(algo_name, algo_name)
        stats = data['statistics']
        print(f"{display_name:<20} {stats['best']:<15.6e} {stats['mean']:<15.6e} "
              f"{stats['std']:<15.6e} {stats['mean_runtime']:<10.2f}")
    
    print()
    
    # Save results
    print("[5/6] Saving results...")
    
    # Save JSON results
    results_json = {}
    for algo_name, data in results.items():
        results_json[algo_name] = {
            'statistics': data['statistics'],
            'best_solution': {
                'k0': float(data['best_solution'][0]),
                'k': float(data['best_solution'][1]),
                'a': float(data['best_solution'][2]),
            },
            'all_fitness': [float(f) for f in data['all_fitness']],
        }
    
    results_path = output_dir / 'optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"  ✓ Saved: {results_path}")
    
    # Generate visualizations
    print()
    print("[6/6] Generating visualizations...")
    
    if config.get('visualization', {}).get('plot_convergence', True):
        generate_convergence_plot(results, output_dir, 
                                 dpi=config.get('visualization', {}).get('dpi', 300))
    
    if config.get('visualization', {}).get('plot_boxplot', True):
        generate_boxplot(results, output_dir,
                        dpi=config.get('visualization', {}).get('dpi', 300))
    
    if config.get('visualization', {}).get('plot_comparison_table', True):
        generate_comparison_table(results, theoretical_values, output_dir)
    
    # Generate markdown report
    if config.get('report', {}).get('generate_markdown', True):
        generate_markdown_report(results, theoretical_values, config, output_dir)
    
    print()
    print("=" * 80)
    print("✅ FULL OPTIMIZATION BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print()
    
    # Display best result
    best_algo_name, best_data = sorted_results[0]
    best_display_name = ALGORITHM_NAMES.get(best_algo_name, best_algo_name)
    best_sol = best_data['best_solution']
    best_mse = best_data['statistics']['best']
    
    print("🏆 BEST ALGORITHM:")
    print(f"   {best_display_name} ({best_algo_name})")
    print(f"   MSE: {best_mse:.6e}")
    print(f"   k₀ = {best_sol[0]:.6f} H  (theoretical: {theoretical_values['k0']:.4f})")
    print(f"   k  = {best_sol[1]:.6f} H  (theoretical: {theoretical_values['k']:.4f})")
    print(f"   a  = {best_sol[2]:.6f} m  (theoretical: {theoretical_values['a']:.4f})")
    print()
    
    # Check success criteria
    k0_error = calculate_percentage_error(best_sol[0], theoretical_values['k0'])
    k_error = calculate_percentage_error(best_sol[1], theoretical_values['k'])
    a_error = calculate_percentage_error(best_sol[2], theoretical_values['a'])
    max_error = max(k0_error, k_error, a_error)
    
    print("✓ SUCCESS CRITERIA:")
    target_mse = theoretical_values.get('mse_threshold', 1e-7)
    print(f"   MSE < {target_mse:.0e}: {'✅ PASS' if best_mse < target_mse else f'❌ FAIL ({best_mse:.2e})'}")
    print(f"   Parameters within 10%: {'✅ PASS' if max_error <= 10 else f'❌ FAIL (max error: {max_error:.1f}%)'}")
    print()
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Full Optimization Benchmark for Magnetic Levitator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_full_optimization.py
  python scripts/run_full_optimization.py --config config/full_optimization.yaml
  python scripts/run_full_optimization.py --trials 10 --seed 123
        """
    )
    
    parser.add_argument('--config', type=str, 
                       default='config/full_optimization.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--trials', type=int,
                       help='Number of trials per algorithm (overrides config)')
    parser.add_argument('--seed', type=int,
                       help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load and modify config if needed
    if args.trials or args.seed:
        config = load_config(args.config)
        if args.trials:
            config['benchmark']['n_trials'] = args.trials
        if args.seed:
            config['benchmark']['random_seed'] = args.seed
            config['optimization']['random_seed'] = args.seed
            for algo_config in config['algorithms'].values():
                if 'random_seed' in algo_config:
                    algo_config['random_seed'] = args.seed
        
        # Save temporary config
        import tempfile
        import yaml
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_config_path = f.name
        
        try:
            results = run_full_optimization(temp_config_path)
        finally:
            Path(temp_config_path).unlink()
    else:
        results = run_full_optimization(args.config)
    
    return 0 if results is not None else 1


if __name__ == '__main__':
    sys.exit(main())
