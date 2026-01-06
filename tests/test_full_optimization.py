#!/usr/bin/env python3
"""
Integration test for run_full_optimization.py

Tests the script with minimal configuration using synthetic data
for speed (no need for full dataset integration).
"""

import sys
import os
from pathlib import Path
import tempfile
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_full_optimization_script():
    """Test the full optimization script with minimal config."""
    print("Testing run_full_optimization.py script...")
    
    # Create minimal test config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test config
        config = {
            'theoretical_values': {
                'k0': 0.0363,
                'k': 0.0035,
                'a': 0.0052,
                'mse_threshold': 1.0e-7
            },
            'benchmark': {
                'data_path': None,  # Use synthetic data
                'random_seed': 42,
                'n_trials': 1,
                'output_dir': str(tmpdir / 'results')
            },
            'optimization': {
                'pop_size': 5,
                'max_iter': 5,
                'random_seed': 42,
                'verbose': False
            },
            'algorithms': {
                'DifferentialEvolution': {
                    'enabled': True,
                    'pop_size': 5,
                    'max_iter': 5,
                    'F': 0.8,
                    'CR': 0.9,
                    'random_seed': 42,
                    'verbose': False
                },
                'RandomSearch': {
                    'enabled': True,
                    'n_iterations': 25,
                    'random_seed': 42,
                    'verbose': False
                },
                # Disable all others
                'GreyWolfOptimizer': {'enabled': False},
                'ArtificialBeeColony': {'enabled': False},
                'HoneyBadgerAlgorithm': {'enabled': False},
                'ShrimpOptimizer': {'enabled': False},
                'TianjiOptimizer': {'enabled': False},
                'GeneticAlgorithm': {'enabled': False}
            },
            'visualization': {
                'plot_convergence': True,
                'plot_boxplot': True,
                'plot_comparison_table': True,
                'save_plots': True,
                'dpi': 100
            },
            'report': {
                'generate_markdown': True,
                'include_statistics': True,
                'include_ranking': True,
                'compare_with_theoretical': True
            }
        }
        
        # Save config
        config_path = tmpdir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"  ✓ Created test config: {config_path}")
        
        # Import and run the optimization
        sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
        from run_full_optimization import run_full_optimization
        
        print("  ✓ Script imported successfully")
        
        # Run optimization
        print("  Running optimization (synthetic data, minimal iterations)...")
        results = run_full_optimization(str(config_path))
        
        assert results is not None, "Optimization should return results"
        print("  ✓ Optimization completed")
        
        # Check results structure
        assert 'DifferentialEvolution' in results, "DE should be in results"
        assert 'RandomSearch' in results, "RandomSearch should be in results"
        print("  ✓ Results contain expected algorithms")
        
        # Check output files
        output_dir = tmpdir / 'results'
        assert output_dir.exists(), "Output directory should exist"
        
        results_json = output_dir / 'optimization_results.json'
        assert results_json.exists(), "Results JSON should exist"
        print(f"  ✓ Results JSON created: {results_json}")
        
        report_md = output_dir / 'BENCHMARK_REPORT.md'
        assert report_md.exists(), "Markdown report should exist"
        print(f"  ✓ Markdown report created: {report_md}")
        
        # Check visualizations
        convergence_plot = output_dir / 'convergence_curves.png'
        assert convergence_plot.exists(), "Convergence plot should exist"
        print(f"  ✓ Convergence plot created: {convergence_plot}")
        
        boxplot = output_dir / 'performance_boxplot.png'
        assert boxplot.exists(), "Boxplot should exist"
        print(f"  ✓ Boxplot created: {boxplot}")
        
        comparison_table = output_dir / 'comparison_table.png'
        assert comparison_table.exists(), "Comparison table should exist"
        print(f"  ✓ Comparison table created: {comparison_table}")
        
        # Verify JSON structure
        with open(results_json) as f:
            json_data = json.load(f)
        
        assert 'DifferentialEvolution' in json_data, "JSON should contain DE results"
        assert 'statistics' in json_data['DifferentialEvolution'], "Should have statistics"
        assert 'best_solution' in json_data['DifferentialEvolution'], "Should have best solution"
        print("  ✓ JSON structure is correct")
        
        # Verify markdown report content
        with open(report_md) as f:
            report_content = f.read()
        
        assert '# Full Optimization Benchmark Report' in report_content, "Report should have title"
        assert '## Algorithm Rankings' in report_content, "Report should have rankings"
        assert '## Comparison with Theoretical Values' in report_content, "Report should have comparison"
        print("  ✓ Markdown report structure is correct")
        
        print("\n✅ All tests passed!")
        return True

if __name__ == '__main__':
    try:
        test_full_optimization_script()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
