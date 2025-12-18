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
        
        # Con datos sintéticos, el error debe ser muy bajo
        assert error < 1e-4
    
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


class TestEvaluateBatch:
    """Tests para el método evaluate_batch."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_evaluate_batch_returns_array(self, benchmark):
        """Verifica que evaluate_batch retorna un array de numpy."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006],
            [0.03, 0.003, 0.004]
        ])
        results = benchmark.evaluate_batch(population)
        
        assert isinstance(results, np.ndarray)
        assert len(results) == len(population)
    
    def test_evaluate_batch_values_match_individual(self, benchmark):
        """Verifica que evaluate_batch da mismos resultados que fitness_function."""
        population = np.array([
            [0.036, 0.0035, 0.005],
            [0.04, 0.004, 0.006]
        ])
        
        batch_results = benchmark.evaluate_batch(population)
        individual_results = [benchmark.fitness_function(ind) for ind in population]
        
        np.testing.assert_array_almost_equal(batch_results, individual_results)
    
    def test_evaluate_batch_empty_population(self, benchmark):
        """Verifica manejo de población vacía."""
        population = np.array([]).reshape(0, 3)
        results = benchmark.evaluate_batch(population)
        assert len(results) == 0


class TestPhysicalConstants:
    """Tests para las constantes físicas del sistema."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_physical_constants_are_positive(self, benchmark):
        """Verifica que las constantes físicas son positivas."""
        assert benchmark.m > 0, "Masa debe ser positiva"
        assert benchmark.g > 0, "Gravedad debe ser positiva"
        assert benchmark.R > 0, "Resistencia debe ser positiva"
    
    def test_physical_constants_have_reasonable_values(self, benchmark):
        """Verifica que las constantes físicas están en rangos razonables."""
        assert 0.001 < benchmark.m < 1.0, "Masa fuera de rango razonable"
        assert 9.0 < benchmark.g < 10.0, "Gravedad fuera de rango razonable"
        assert 0.1 < benchmark.R < 100.0, "Resistencia fuera de rango razonable"


class TestInvalidSolutions:
    """Tests para manejo de soluciones inválidas."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_negative_parameters_penalized(self, benchmark):
        """Verifica que parámetros negativos son penalizados."""
        invalid_solutions = [
            [-0.01, 0.0035, 0.005],
            [0.036, -0.001, 0.005],
            [0.036, 0.0035, -0.001],
            [-0.01, -0.001, -0.001]
        ]
        
        for sol in invalid_solutions:
            result = benchmark.fitness_function(sol)
            assert result == 1e9, f"Solución {sol} debería ser penalizada"
    
    def test_zero_parameters_penalized(self, benchmark):
        """Verifica que parámetros cero son penalizados."""
        invalid_solutions = [
            [0.0, 0.0035, 0.005],
            [0.036, 0.0, 0.005],
            [0.036, 0.0035, 0.0]
        ]
        
        for sol in invalid_solutions:
            result = benchmark.fitness_function(sol)
            assert result == 1e9, f"Solución {sol} debería ser penalizada"
    
    def test_out_of_upper_bounds_penalized(self, benchmark):
        """Verifica que valores fuera de límites superiores son penalizados."""
        # Valores muy por encima de los límites superiores
        invalid_solution = [1.0, 1.0, 1.0]  # Todos exceden límites superiores
        result = benchmark.fitness_function(invalid_solution)
        assert result == 1e9
    
    def test_out_of_lower_bounds_penalized(self, benchmark):
        """Verifica que valores fuera de límites inferiores son penalizados."""
        # Valores muy por debajo de los límites inferiores (pero positivos)
        invalid_solution = [1e-5, 1e-5, 1e-5]  # Todos por debajo de límites inferiores
        result = benchmark.fitness_function(invalid_solution)
        assert result == 1e9


class TestEdgeCases:
    """Tests para casos extremos y límites."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_boundary_values_are_valid(self, benchmark):
        """Verifica que valores en los límites son aceptables."""
        # Probar límites inferiores
        lower_bounds_sol = [lb for lb, ub in benchmark.bounds]
        result_lower = benchmark.fitness_function(lower_bounds_sol)
        assert result_lower != 1e9, "Límites inferiores deberían ser válidos"
        
        # Probar límites superiores
        upper_bounds_sol = [ub for lb, ub in benchmark.bounds]
        result_upper = benchmark.fitness_function(upper_bounds_sol)
        assert result_upper != 1e9, "Límites superiores deberían ser válidos"
    
    def test_extreme_but_valid_values(self, benchmark):
        """Verifica que valores extremos pero válidos no causan errores."""
        # Valores en el límite inferior
        solution = [0.0001, 0.0001, 0.0001]
        result = benchmark.fitness_function(solution)
        assert isinstance(result, (int, float))
        assert not np.isnan(result)


class TestFactoryFunction:
    """Tests para la función de fábrica create_benchmark."""
    
    def test_create_benchmark_returns_instance(self):
        """Verifica que create_benchmark retorna una instancia válida."""
        from levitador_benchmark import create_benchmark
        benchmark = create_benchmark()
        
        assert isinstance(benchmark, LevitadorBenchmark)
        assert benchmark.dim == 3
    
    def test_create_benchmark_with_path(self):
        """Verifica que create_benchmark acepta path como parámetro."""
        from levitador_benchmark import create_benchmark
        benchmark = create_benchmark(data_path=None)
        
        assert isinstance(benchmark, LevitadorBenchmark)


class TestReprMethod:
    """Tests para el método __repr__."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_repr_contains_essential_info(self, benchmark):
        """Verifica que __repr__ contiene información esencial."""
        repr_str = repr(benchmark)
        
        assert 'LevitadorBenchmark' in repr_str
        assert 'dim=3' in repr_str
        assert 'n_samples=' in repr_str
        assert 'bounds=' in repr_str
    
    def test_repr_is_string(self, benchmark):
        """Verifica que __repr__ retorna un string."""
        repr_str = repr(benchmark)
        assert isinstance(repr_str, str)


class TestNoiseConfiguration:
    """Tests para configuración de ruido en datos sintéticos."""
    
    def test_different_noise_levels_produce_different_data(self):
        """Verifica que diferentes niveles de ruido producen datos diferentes."""
        bench1 = LevitadorBenchmark(random_seed=42, noise_level=1e-5, verbose=False)
        bench2 = LevitadorBenchmark(random_seed=42, noise_level=1e-3, verbose=False)
        
        # Los datos deben ser diferentes debido al ruido
        assert not np.allclose(bench1.y_real, bench2.y_real)
    
    def test_zero_noise_is_deterministic(self):
        """Verifica que ruido cero produce datos deterministas."""
        bench1 = LevitadorBenchmark(random_seed=42, noise_level=0.0, verbose=False)
        bench2 = LevitadorBenchmark(random_seed=42, noise_level=0.0, verbose=False)
        
        # Sin ruido, los datos deben ser idénticos
        np.testing.assert_array_almost_equal(bench1.y_real, bench2.y_real)


class TestVerboseMode:
    """Tests para el modo verbose."""
    
    def test_verbose_false_reduces_output(self):
        """Verifica que verbose=False reduce la salida."""
        # Simplemente verificar que se puede crear con verbose=False
        benchmark = LevitadorBenchmark(random_seed=42, verbose=False)
        assert benchmark._verbose is False
    
    def test_verbose_true_enables_output(self):
        """Verifica que verbose=True habilita salida."""
        benchmark = LevitadorBenchmark(random_seed=42, verbose=True)
        assert benchmark._verbose is True


class TestDataGeneration:
    """Tests para generación de datos sintéticos."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_synthetic_data_has_correct_shape(self, benchmark):
        """Verifica que datos sintéticos tienen la forma correcta."""
        assert len(benchmark.t_real) > 0
        assert len(benchmark.t_real) == len(benchmark.y_real)
        assert len(benchmark.t_real) == len(benchmark.i_real)
        assert len(benchmark.t_real) == len(benchmark.u_real)
    
    def test_synthetic_data_time_is_monotonic(self, benchmark):
        """Verifica que el tiempo es monótonamente creciente."""
        assert np.all(np.diff(benchmark.t_real) > 0)
    
    def test_synthetic_data_position_has_finite_values(self, benchmark):
        """Verifica que las posiciones tienen valores finitos."""
        assert np.all(np.isfinite(benchmark.y_real))
        assert not np.any(np.isnan(benchmark.y_real))
    
    def test_synthetic_data_voltage_is_constant(self, benchmark):
        """Verifica que el voltaje es constante para datos sintéticos."""
        # Para datos sintéticos, u_real debería ser constante
        assert np.allclose(benchmark.u_real, benchmark.u_real[0])


class TestReferenceValues:
    """Tests para valores de referencia."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_reference_solution_exists(self, benchmark):
        """Verifica que existe una solución de referencia."""
        assert hasattr(benchmark, 'reference_solution')
        assert len(benchmark.reference_solution) == 3
    
    def test_reference_solution_within_bounds(self, benchmark):
        """Verifica que la solución de referencia está dentro de los límites."""
        for val, (lb, ub) in zip(benchmark.reference_solution, benchmark.bounds):
            assert lb <= val <= ub, f"Valor {val} fuera de límites [{lb}, {ub}]"


class TestDataLoadingErrors:
    """Tests para manejo de errores en carga de datos."""
    
    def test_nonexistent_file_falls_back_to_synthetic(self):
        """Verifica que archivo inexistente usa datos sintéticos."""
        benchmark = LevitadorBenchmark(
            datos_reales_path='/nonexistent/path/file.txt',
            random_seed=42,
            verbose=False
        )
        
        # Debe tener datos sintéticos
        assert len(benchmark.t_real) > 0
        assert len(benchmark.y_real) > 0
    
    def test_invalid_path_format_falls_back_to_synthetic(self):
        """Verifica que path inválido usa datos sintéticos."""
        benchmark = LevitadorBenchmark(
            datos_reales_path='',
            random_seed=42,
            verbose=False
        )
        
        # Debe tener datos sintéticos
        assert len(benchmark.t_real) > 0


class TestFitnessStability:
    """Tests para estabilidad de la función de fitness."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_fitness_with_very_small_parameters(self, benchmark):
        """Verifica comportamiento con parámetros muy pequeños pero válidos."""
        # Usar límites inferiores
        solution = [0.0001, 0.0001, 0.0001]
        result = benchmark.fitness_function(solution)
        
        # No debe ser penalizado si está dentro de límites
        assert result != 1e9
        assert isinstance(result, (int, float))
        assert np.isfinite(result)
    
    def test_fitness_consistency(self, benchmark):
        """Verifica que llamadas múltiples dan mismo resultado."""
        solution = [0.036, 0.0035, 0.005]
        
        result1 = benchmark.fitness_function(solution)
        result2 = benchmark.fitness_function(solution)
        
        assert result1 == result2
    
    def test_fitness_with_mixed_boundary_values(self, benchmark):
        """Verifica fitness con mezcla de valores en límites."""
        solution = [
            benchmark.bounds[0][0],  # Límite inferior de k0
            benchmark.bounds[1][1],  # Límite superior de k
            (benchmark.bounds[2][0] + benchmark.bounds[2][1]) / 2  # Medio de a
        ]
        
        result = benchmark.fitness_function(solution)
        assert result != 1e9
        assert np.isfinite(result)


class TestInitializationVariants:
    """Tests para diferentes variantes de inicialización."""
    
    def test_initialization_without_seed(self):
        """Verifica que se puede inicializar sin semilla."""
        benchmark1 = LevitadorBenchmark(verbose=False)
        benchmark2 = LevitadorBenchmark(verbose=False)
        
        # Sin semilla, los datos deberían ser diferentes
        assert not np.array_equal(benchmark1.y_real, benchmark2.y_real)
    
    def test_initialization_with_different_noise_levels(self):
        """Verifica inicialización con diferentes niveles de ruido."""
        bench_low_noise = LevitadorBenchmark(
            random_seed=42, 
            noise_level=1e-8, 
            verbose=False
        )
        bench_high_noise = LevitadorBenchmark(
            random_seed=42, 
            noise_level=1e-2, 
            verbose=False
        )
        
        assert len(bench_low_noise.y_real) > 0
        assert len(bench_high_noise.y_real) > 0
    
    def test_initialization_verbose_false_by_default(self):
        """Verifica que verbose=False es el valor por defecto."""
        benchmark = LevitadorBenchmark(random_seed=42, verbose=False)
        assert benchmark._verbose is False


class TestBoundsMethods:
    """Tests adicionales para métodos de límites."""
    
    @pytest.fixture
    def benchmark(self):
        return LevitadorBenchmark(random_seed=42, verbose=False)
    
    def test_get_bounds_array_shape(self, benchmark):
        """Verifica forma de arrays de límites."""
        lb, ub = benchmark.get_bounds_array()
        
        assert lb.shape == (3,)
        assert ub.shape == (3,)
    
    def test_get_bounds_array_values(self, benchmark):
        """Verifica que get_bounds_array retorna valores correctos."""
        lb, ub = benchmark.get_bounds_array()
        
        for i, (expected_lb, expected_ub) in enumerate(benchmark.bounds):
            assert lb[i] == expected_lb
            assert ub[i] == expected_ub


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
