"""
Tests for the benchmark CLI.
Ejecutar con: pytest tests/test_cli.py -v
"""

import pytest
import subprocess
import sys
import os
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCLI:
    """Tests for command-line interface."""
    
    @pytest.fixture
    def cli_path(self):
        """Path to the CLI script."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'benchmark_cli.py'
        )
    
    def run_cli(self, cli_path, args):
        """Helper to run CLI commands."""
        result = subprocess.run(
            [sys.executable, cli_path] + args,
            capture_output=True,
            text=True
        )
        return result
    
    def test_help_message(self, cli_path):
        """Test that help message is displayed."""
        result = self.run_cli(cli_path, ['--help'])
        assert result.returncode == 0
        assert 'Levitador Magnético Benchmark' in result.stdout
        assert 'list-algorithms' in result.stdout
        assert 'test' in result.stdout
        assert 'run' in result.stdout
    
    def test_list_algorithms(self, cli_path):
        """Test list-algorithms command."""
        result = self.run_cli(cli_path, ['list-algorithms'])
        assert result.returncode == 0
        assert 'ALGORITMOS DISPONIBLES' in result.stdout
        assert 'Random Search' in result.stdout
        assert 'Differential Evolution' in result.stdout
        assert 'Genetic Algorithm' in result.stdout
        assert 'Grey Wolf Optimizer' in result.stdout
    
    def test_list_algorithms_alias(self, cli_path):
        """Test list command alias."""
        result = self.run_cli(cli_path, ['list'])
        assert result.returncode == 0
        assert 'ALGORITMOS DISPONIBLES' in result.stdout
    
    def test_test_command(self, cli_path):
        """Test the test command."""
        result = self.run_cli(cli_path, ['test', '--seed', '42'])
        assert result.returncode == 0
        assert 'TEST RÁPIDO' in result.stdout
        assert 'Solución de referencia' in result.stdout
        assert 'Solución aleatoria' in result.stdout
        assert 'Test completado exitosamente' in result.stdout
    
    def test_run_random_search(self, cli_path, tmp_path):
        """Test running random search algorithm."""
        output_file = tmp_path / "results.json"
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'random',
            '--n-iterations', '50',
            '--seed', '42',
            '--output', str(output_file),
            '--quiet'
        ])
        assert result.returncode == 0
        assert 'MEJOR SOLUCIÓN ENCONTRADA' in result.stdout
        assert 'Error (MSE)' in result.stdout
        assert output_file.exists()
        
        # Verify JSON content
        with open(output_file) as f:
            data = json.load(f)
        assert 'algorithm' in data
        assert data['algorithm'] == 'Random Search'
        assert 'solution' in data
        assert 'k0' in data['solution']
        assert 'k' in data['solution']
        assert 'a' in data['solution']
        assert 'error_mse' in data
        assert 'evaluations' in data
        assert data['evaluations'] == 50
    
    def test_run_differential_evolution(self, cli_path):
        """Test running differential evolution."""
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'DE',
            '--pop-size', '10',
            '--max-iter', '5',
            '--seed', '42',
            '--quiet'
        ])
        assert result.returncode == 0
        assert 'Differential Evolution' in result.stdout
        assert 'MEJOR SOLUCIÓN ENCONTRADA' in result.stdout
    
    def test_run_genetic_algorithm(self, cli_path):
        """Test running genetic algorithm."""
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'GA',
            '--generations', '5',
            '--seed', '42',
            '--quiet'
        ])
        assert result.returncode == 0
        assert 'Genetic Algorithm' in result.stdout
        assert 'MEJOR SOLUCIÓN ENCONTRADA' in result.stdout
    
    def test_run_with_visualization(self, cli_path, tmp_path):
        """Test running with visualization."""
        output_file = tmp_path / "results.json"
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'random',
            '--n-iterations', '20',
            '--seed', '42',
            '--output', str(output_file),
            '--visualize',
            '--quiet'
        ])
        assert result.returncode == 0
        
        # Check that visualization file was created
        viz_file = tmp_path / "results.png"
        assert viz_file.exists()
    
    def test_invalid_algorithm(self, cli_path):
        """Test with invalid algorithm name."""
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'INVALID_ALGO'
        ])
        assert result.returncode == 1
        assert 'no encontrado' in result.stdout
    
    def test_run_with_custom_parameters(self, cli_path):
        """Test running with custom algorithm parameters."""
        result = self.run_cli(cli_path, [
            'run',
            '--algorithm', 'DE',
            '--pop-size', '20',
            '--max-iter', '3',
            '--F', '0.7',
            '--CR', '0.85',
            '--seed', '42',
            '--quiet'
        ])
        assert result.returncode == 0
        assert 'pop_size: 20' in result.stdout
        assert 'max_iter: 3' in result.stdout
        assert 'F: 0.7' in result.stdout
        assert 'CR: 0.85' in result.stdout
    
    def test_reproducibility_with_seed(self, cli_path, tmp_path):
        """Test that results are reproducible with the same seed."""
        output1 = tmp_path / "results1.json"
        output2 = tmp_path / "results2.json"
        
        # Run twice with same seed
        self.run_cli(cli_path, [
            'run', '--algorithm', 'random', '--n-iterations', '30',
            '--seed', '123', '--output', str(output1), '--quiet'
        ])
        self.run_cli(cli_path, [
            'run', '--algorithm', 'random', '--n-iterations', '30',
            '--seed', '123', '--output', str(output2), '--quiet'
        ])
        
        with open(output1) as f1, open(output2) as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        
        # Solutions should be identical with same seed
        assert data1['error_mse'] == data2['error_mse']
        assert data1['solution'] == data2['solution']


class TestCLIAlgorithms:
    """Test all available algorithms via CLI."""
    
    @pytest.fixture
    def cli_path(self):
        """Path to the CLI script."""
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'benchmark_cli.py'
        )
    
    def run_cli(self, cli_path, args):
        """Helper to run CLI commands."""
        result = subprocess.run(
            [sys.executable, cli_path] + args,
            capture_output=True,
            text=True
        )
        return result
    
    @pytest.mark.parametrize('algorithm', ['random', 'DE', 'GA', 'GWO', 'ABC', 'HBA', 'Shrimp', 'Tianji'])
    def test_all_algorithms_run(self, cli_path, algorithm):
        """Test that all algorithms can be executed."""
        # Use minimal iterations for speed
        args = ['run', '--algorithm', algorithm, '--seed', '42', '--quiet']
        
        if algorithm == 'random':
            args.extend(['--n-iterations', '10'])
        elif algorithm in ['DE', 'GA']:
            args.extend(['--max-iter', '2'] if algorithm == 'DE' else ['--generations', '2'])
            args.extend(['--pop-size', '5'])
        else:
            args.extend(['--pop-size', '5', '--max-iter', '2'])
        
        result = self.run_cli(cli_path, args)
        assert result.returncode == 0, f"Algorithm {algorithm} failed: {result.stderr}"
        assert 'MEJOR SOLUCIÓN ENCONTRADA' in result.stdout


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
