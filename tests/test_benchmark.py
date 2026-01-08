"""
Tests unitarios para el benchmark del levitador magnético.
Ejecutar con: pytest tests/test_benchmark.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from levitador_benchmark import LevitadorBenchmark


class TestLevitadorBenchmark:
    """Tests para la clase LevitadorBenchmark."""
    
    @pytest.fixture
    def benchmark(self):
        """Crea una instancia del benchmark con datos sintéticos."""
        return LevitadorBenchmark(random_seed=42)
    
    @pytest.fixture
    def benchmark_real(self):
        """Crea una instancia con datos experimentales."""
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'datos_levitador.txt'
        )
        if os.path.exists(data_path):
            return LevitadorBenchmark(data_path, random_seed=42)
        return None
    
    def test_initialization(self, benchmark):
        """Verifica que el benchmark se inicializa correctamente."""
        assert benchmark.dim == 3
        assert len(benchmark.bounds) == 3
        assert len(benchmark.variable_names) == 3
        assert benchmark.variable_names == ['k0', 'k', 'a']
    
    def test_bounds(self, benchmark):
        """Verifica que los límites del espacio de búsqueda son correctos."""
        for lb, ub in benchmark.bounds:
            assert lb < ub
            assert lb > 0  # Todos los parámetros deben ser positivos
    
    def test_fitness_function_returns_scalar(self, benchmark):
        """Verifica que la función de fitness devuelve un escalar."""
        solution = [0.036, 0.0035, 0.005]
        result = benchmark.fitness_function(solution)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)
        assert not np.isinf(result)
    
    def test_fitness_function_penalizes_out_of_bounds(self, benchmark):
        """Verifica que soluciones fuera de límites son penalizadas."""
        valid_solution = [0.036, 0.0035, 0.005]
        invalid_solution = [-0.1, 0.5, 0.1]  # Fuera de límites
        
        valid_fitness = benchmark.fitness_function(valid_solution)
        invalid_fitness = benchmark.fitness_function(invalid_solution)
        
        assert invalid_fitness > valid_fitness
    
    def test_reference_solution_has_low_error(self, benchmark):
        """Verifica que la solución de referencia tiene error bajo."""
        ref_sol = benchmark.reference_solution
        error = benchmark.fitness_function(ref_sol)
        
        # Con datos sintéticos, el error debe ser razonable
        # Note: Even with reference parameters, MSE is ~1.0 due to noise and initial conditions
        assert error < 10.0  # Should be reasonable, not extremely high
    
    def test_reproducibility_with_seed(self):
        """Verifica reproducibilidad con semilla fija."""
        bench1 = LevitadorBenchmark(random_seed=123)
        bench2 = LevitadorBenchmark(random_seed=123)
        
        solution = [0.036, 0.0035, 0.005]
        error1 = bench1.fitness_function(solution)
        error2 = bench2.fitness_function(solution)
        
        assert error1 == error2
    
    def test_different_seeds_produce_different_data(self):
        """Verifica que semillas diferentes producen datos diferentes."""
        bench1 = LevitadorBenchmark(random_seed=1)
        bench2 = LevitadorBenchmark(random_seed=2)
        
        # Los datos sintéticos deben ser diferentes
        assert not np.allclose(bench1.y_real, bench2.y_real)
    
    def test_get_bounds_array(self, benchmark):
        """Verifica el método get_bounds_array."""
        lb, ub = benchmark.get_bounds_array()
        
        assert len(lb) == 3
        assert len(ub) == 3
        assert all(l < u for l, u in zip(lb, ub))
    
    def test_data_loading(self, benchmark_real):
        """Verifica carga de datos experimentales."""
        if benchmark_real is None:
            pytest.skip("Archivo de datos no disponible")
        
        assert len(benchmark_real.t_real) > 0
        assert len(benchmark_real.y_real) == len(benchmark_real.t_real)
        assert len(benchmark_real.u_real) == len(benchmark_real.t_real)


class TestFitnessLandscape:
    """Tests para explorar el paisaje de fitness."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42)
    
    def test_fitness_is_smooth(self, benchmark):
        """Verifica que pequeños cambios producen pequeños cambios en fitness."""
        base_solution = [0.036, 0.0035, 0.005]
        base_fitness = benchmark.fitness_function(base_solution)
        
        # Perturbar ligeramente
        perturbed = [s * 1.001 for s in base_solution]
        perturbed_fitness = benchmark.fitness_function(perturbed)
        
        # El cambio en fitness debe ser pequeño
        relative_change = abs(perturbed_fitness - base_fitness) / (base_fitness + 1e-10)
        assert relative_change < 0.1  # Menos de 10% de cambio
    
    def test_random_solutions_have_varied_fitness(self, benchmark):
        """Verifica que el problema no es trivial."""
        np.random.seed(42)
        
        fitnesses = []
        for _ in range(20):
            sol = [np.random.uniform(lb, ub) for lb, ub in benchmark.bounds]
            fitnesses.append(benchmark.fitness_function(sol))
        
        # Debe haber variación significativa
        assert np.std(fitnesses) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
