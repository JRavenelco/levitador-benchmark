"""
Script para medir el rendimiento de la evaluación en lote.

Compara el tiempo de ejecución de:
1. Evaluación individual con list comprehension
2. Evaluación en lote secuencial (evaluate_batch)
3. Evaluación en lote vectorizada (evaluate_batch_vectorized)
4. Evaluación en lote paralela (evaluate_batch con n_jobs=-1)

Ejecutar: python benchmark_batch_performance.py
"""

import numpy as np
import time
from levitador_benchmark import LevitadorBenchmark
from typing import Callable, Tuple


def benchmark_evaluation_method(
    benchmark: LevitadorBenchmark,
    population: np.ndarray,
    method_name: str,
    evaluation_func: Callable,
    n_runs: int = 3
) -> Tuple[float, float]:
    """
    Mide el tiempo de ejecución de un método de evaluación.
    
    Args:
        benchmark: Instancia del benchmark
        population: Población a evaluar
        method_name: Nombre del método (para display)
        evaluation_func: Función de evaluación a testear
        n_runs: Número de ejecuciones para promediar
    
    Returns:
        (tiempo_promedio, desviación_estándar)
    """
    times = []
    
    for _ in range(n_runs):
        start_time = time.time()
        results = evaluation_func(population)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time


def run_benchmark(population_sizes=[10, 30, 50, 100], n_runs=3):
    """
    Ejecuta el benchmark completo para diferentes tamaños de población.
    
    Args:
        population_sizes: Lista de tamaños de población a probar
        n_runs: Número de ejecuciones por tamaño para promediar
    """
    print("=" * 80)
    print("BENCHMARK: EVALUACIÓN EN LOTE")
    print("=" * 80)
    print()
    
    # Crear benchmark con datos sintéticos y semilla fija
    benchmark = LevitadorBenchmark(random_seed=42, verbose=False)
    lb, ub = benchmark.get_bounds_array()
    
    print(f"Configuración:")
    print(f"  - Espacio de búsqueda: {benchmark.dim}D")
    print(f"  - Puntos temporales: {len(benchmark.t_real)}")
    print(f"  - Ejecuciones por test: {n_runs}")
    print()
    
    # Resultados
    results = {
        'Individual (list comp)': [],
        'Batch Sequential': [],
        'Batch Vectorized': [],
        'Batch Parallel': []
    }
    
    for pop_size in population_sizes:
        print(f"{'=' * 80}")
        print(f"Tamaño de población: {pop_size}")
        print(f"{'-' * 80}")
        
        # Generar población aleatoria
        np.random.seed(42)
        population = np.random.uniform(lb, ub, (pop_size, 3))
        
        # 1. Evaluación individual con list comprehension
        def eval_individual(pop):
            return np.array([benchmark.fitness_function(ind.tolist()) for ind in pop])
        
        avg_time, std_time = benchmark_evaluation_method(
            benchmark, population, "Individual (list comp)", eval_individual, n_runs
        )
        results['Individual (list comp)'].append(avg_time)
        print(f"  Individual (list comp): {avg_time:.4f} ± {std_time:.4f} s")
        
        # 2. Evaluación en lote secuencial
        def eval_batch_seq(pop):
            return benchmark.evaluate_batch(pop, n_jobs=1)
        
        avg_time, std_time = benchmark_evaluation_method(
            benchmark, population, "Batch Sequential", eval_batch_seq, n_runs
        )
        results['Batch Sequential'].append(avg_time)
        speedup = results['Individual (list comp)'][-1] / avg_time
        print(f"  Batch Sequential:       {avg_time:.4f} ± {std_time:.4f} s  (speedup: {speedup:.2f}x)")
        
        # 3. Evaluación en lote vectorizada
        def eval_batch_vec(pop):
            return benchmark.evaluate_batch_vectorized(pop)
        
        avg_time, std_time = benchmark_evaluation_method(
            benchmark, population, "Batch Vectorized", eval_batch_vec, n_runs
        )
        results['Batch Vectorized'].append(avg_time)
        speedup = results['Individual (list comp)'][-1] / avg_time
        print(f"  Batch Vectorized:       {avg_time:.4f} ± {std_time:.4f} s  (speedup: {speedup:.2f}x)")
        
        # 4. Evaluación en lote paralela
        def eval_batch_par(pop):
            return benchmark.evaluate_batch(pop, n_jobs=-1)
        
        avg_time, std_time = benchmark_evaluation_method(
            benchmark, population, "Batch Parallel", eval_batch_par, n_runs
        )
        results['Batch Parallel'].append(avg_time)
        speedup = results['Individual (list comp)'][-1] / avg_time
        print(f"  Batch Parallel:         {avg_time:.4f} ± {std_time:.4f} s  (speedup: {speedup:.2f}x)")
        print()
    
    # Resumen final
    print(f"{'=' * 80}")
    print("RESUMEN DE SPEEDUPS")
    print(f"{'-' * 80}")
    print(f"{'Método':<25} | {'Pop=10':<10} | {'Pop=30':<10} | {'Pop=50':<10} | {'Pop=100':<10}")
    print(f"{'-' * 80}")
    
    for method in ['Batch Sequential', 'Batch Vectorized', 'Batch Parallel']:
        speedups = [results['Individual (list comp)'][i] / results[method][i] 
                   for i in range(len(population_sizes))]
        speedup_str = ' | '.join([f"{s:.2f}x" for s in speedups])
        print(f"{method:<25} | {speedup_str}")
    
    print(f"{'=' * 80}")
    print()
    
    # Recomendaciones
    print("RECOMENDACIONES:")
    print("  - Para poblaciones pequeñas (<30): Usar evaluate_batch_vectorized()")
    print("  - Para poblaciones medianas (30-100): Usar evaluate_batch_vectorized()")
    print("  - Para poblaciones grandes (>100): Usar evaluate_batch(n_jobs=-1)")
    print()


def verify_correctness():
    """Verifica que todos los métodos dan los mismos resultados."""
    print("=" * 80)
    print("VERIFICACIÓN DE CORRECTITUD")
    print("=" * 80)
    print()
    
    benchmark = LevitadorBenchmark(random_seed=42, verbose=False)
    lb, ub = benchmark.get_bounds_array()
    
    # Generar población pequeña
    np.random.seed(42)
    population = np.random.uniform(lb, ub, (5, 3))
    
    # Evaluar con todos los métodos
    results_individual = np.array([benchmark.fitness_function(ind.tolist()) for ind in population])
    results_batch_seq = benchmark.evaluate_batch(population, n_jobs=1)
    results_batch_vec = benchmark.evaluate_batch_vectorized(population)
    results_batch_par = benchmark.evaluate_batch(population, n_jobs=2)
    
    # Verificar que todos son iguales
    tol = 1e-10
    
    checks = [
        ("Batch Sequential vs Individual", np.allclose(results_batch_seq, results_individual, rtol=tol)),
        ("Batch Vectorized vs Individual", np.allclose(results_batch_vec, results_individual, rtol=tol)),
        ("Batch Parallel vs Individual", np.allclose(results_batch_par, results_individual, rtol=tol)),
    ]
    
    all_correct = True
    for name, is_correct in checks:
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"  {name:<40}: {status}")
        if not is_correct:
            all_correct = False
    
    print()
    if all_correct:
        print("✓ Todos los métodos producen resultados idénticos")
    else:
        print("✗ ADVERTENCIA: Algunos métodos producen resultados diferentes")
    
    print()


if __name__ == "__main__":
    # Primero verificar correctitud
    verify_correctness()
    
    # Luego ejecutar benchmark de rendimiento
    run_benchmark(population_sizes=[10, 30, 50, 100], n_runs=3)
