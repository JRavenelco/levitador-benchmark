"""
Tests para la evaluación en lote (batch evaluation) del benchmark.
Ejecutar con: pytest tests/test_batch_evaluation.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from levitador_benchmark import LevitadorBenchmark, PENALTY_VALUE


class TestBatchEvaluation:
    """Tests para evaluación en lote."""
    
    @pytest.fixture
    def benchmark(self):
        """Crea una instancia del benchmark con datos sintéticos."""
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_evaluate_batch_single_individual(self, benchmark):
        """Verifica que evaluate_batch funciona con un solo individuo."""
        individual = np.array([[0.036, 0.0035, 0.005]])
        result = benchmark.evaluate_batch(individual)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert not np.isinf(result[0])
    
    def test_evaluate_batch_multiple_individuals(self, benchmark):
        """Verifica que evaluate_batch funciona con múltiples individuos."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004]
        ])
        
        results = benchmark.evaluate_batch(population)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert all(not np.isnan(r) for r in results)
        assert all(not np.isinf(r) for r in results)
    
    def test_evaluate_batch_consistency_with_fitness_function(self, benchmark):
        """Verifica que evaluate_batch da los mismos resultados que fitness_function."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004]
        ])
        
        # Evaluar con evaluate_batch
        batch_results = benchmark.evaluate_batch(population)
        
        # Evaluar individualmente con fitness_function
        individual_results = [benchmark.fitness_function(ind.tolist()) for ind in population]
        
        # Comparar resultados
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-10)
    
    def test_evaluate_batch_penalizes_invalid(self, benchmark):
        """Verifica que evaluate_batch penaliza individuos inválidos."""
        population = np.array([
            [0.036, 0.0035, 0.005],  # válido
            [-0.01, 0.003, 0.004],    # inválido (negativo)
            [0.5, 0.5, 0.5],          # inválido (fuera de límites)
        ])
        
        results = benchmark.evaluate_batch(population)
        
        assert results[0] < PENALTY_VALUE  # El válido debe tener error razonable
        assert results[1] == PENALTY_VALUE  # Penalización para negativo
        assert results[2] == PENALTY_VALUE  # Penalización para fuera de límites
    
    def test_evaluate_batch_vectorized_single_individual(self, benchmark):
        """Verifica que evaluate_batch_vectorized funciona con un solo individuo."""
        individual = np.array([[0.036, 0.0035, 0.005]])
        result = benchmark.evaluate_batch_vectorized(individual)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert not np.isnan(result[0])
        assert result[0] < PENALTY_VALUE  # No debe estar penalizado
    
    def test_evaluate_batch_vectorized_multiple_individuals(self, benchmark):
        """Verifica que evaluate_batch_vectorized funciona con múltiples individuos."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004]
        ])
        
        results = benchmark.evaluate_batch_vectorized(population)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == 3
        assert all(r < PENALTY_VALUE for r in results)  # Todos deben ser válidos
    
    def test_evaluate_batch_vectorized_consistency(self, benchmark):
        """Verifica que evaluate_batch_vectorized da los mismos resultados que evaluate_batch."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004]
        ])
        
        # Evaluar con ambos métodos
        batch_results = benchmark.evaluate_batch(population)
        vectorized_results = benchmark.evaluate_batch_vectorized(population)
        
        # Comparar resultados
        np.testing.assert_allclose(batch_results, vectorized_results, rtol=1e-10)
    
    def test_evaluate_batch_vectorized_handles_invalid(self, benchmark):
        """Verifica que evaluate_batch_vectorized maneja individuos inválidos correctamente."""
        population = np.array([
            [0.036, 0.0035, 0.005],  # válido
            [-0.01, 0.003, 0.004],    # inválido (negativo)
            [0.5, 0.5, 0.5],          # inválido (fuera de límites)
            [0.04, 0.004, 0.006],     # válido
        ])
        
        results = benchmark.evaluate_batch_vectorized(population)
        
        assert results[0] < PENALTY_VALUE  # válido
        assert results[1] == PENALTY_VALUE  # inválido
        assert results[2] == PENALTY_VALUE  # inválido
        assert results[3] < PENALTY_VALUE  # válido
    
    def test_evaluate_batch_with_large_population(self, benchmark):
        """Verifica que evaluate_batch funciona con poblaciones grandes."""
        np.random.seed(42)
        n_individuals = 50
        
        # Generar población aleatoria dentro de los límites
        lb, ub = benchmark.get_bounds_array()
        population = np.random.uniform(lb, ub, (n_individuals, 3))
        
        results = benchmark.evaluate_batch(population)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == n_individuals
        assert all(not np.isnan(r) for r in results)
    
    def test_evaluate_batch_parallel(self, benchmark):
        """Verifica que evaluate_batch con n_jobs > 1 funciona correctamente."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004],
            [0.035, 0.0033, 0.0048]
        ])
        
        # Evaluar de forma secuencial
        sequential_results = benchmark.evaluate_batch(population, n_jobs=1)
        
        # Evaluar en paralelo
        parallel_results = benchmark.evaluate_batch(population, n_jobs=2)
        
        # Los resultados deben ser idénticos
        np.testing.assert_allclose(sequential_results, parallel_results, rtol=1e-10)
    
    def test_evaluate_batch_empty_population(self, benchmark):
        """Verifica que evaluate_batch maneja poblaciones vacías."""
        population = np.array([]).reshape(0, 3)
        results = benchmark.evaluate_batch(population)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == 0
    
    def test_batch_evaluation_preserves_order(self, benchmark):
        """Verifica que el orden de evaluación se preserva."""
        # Crear población con valores distintivos
        population = np.array([
            [0.05, 0.005, 0.01],   # Alto
            [0.036, 0.0035, 0.005], # Medio (cerca de referencia)
            [0.02, 0.002, 0.003],   # Bajo
        ])
        
        batch_results = benchmark.evaluate_batch(population)
        
        # Verificar que cada resultado corresponde a su individuo
        for i, ind in enumerate(population):
            individual_result = benchmark.fitness_function(ind.tolist())
            assert abs(batch_results[i] - individual_result) < 1e-10


class TestBatchEvaluationPerformance:
    """Tests de rendimiento para evaluación en lote."""
    
    @pytest.fixture
    def benchmark(self):
        """Crea una instancia del benchmark."""
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_batch_evaluation_is_correct(self, benchmark):
        """Verifica que la evaluación en lote es correcta comparada con evaluación individual."""
        np.random.seed(42)
        
        # Generar población pequeña
        lb, ub = benchmark.get_bounds_array()
        population = np.random.uniform(lb, ub, (10, 3))
        
        # Evaluar con batch
        batch_results = benchmark.evaluate_batch(population)
        
        # Evaluar individualmente
        individual_results = np.array([
            benchmark.fitness_function(ind.tolist()) 
            for ind in population
        ])
        
        # Verificar que son idénticos
        np.testing.assert_allclose(batch_results, individual_results, rtol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
