"""
Ejemplo de uso de evaluación en lote para optimización rápida.

Este script demuestra cómo usar las nuevas funciones de evaluación en lote
para acelerar algoritmos de optimización.

Ejecutar: python example_batch_usage.py
"""

import numpy as np
from levitador_benchmark import LevitadorBenchmark
from example_optimization import DifferentialEvolution, GeneticAlgorithm
import time


def example_basic_batch_evaluation():
    """Ejemplo básico de evaluación en lote."""
    print("=" * 70)
    print("EJEMPLO 1: Evaluación en Lote Básica")
    print("=" * 70)
    print()
    
    # Crear el problema
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    # Generar una población de soluciones candidatas
    lb, ub = problema.get_bounds_array()
    poblacion = np.random.uniform(lb, ub, (10, 3))
    
    print(f"Población generada: {poblacion.shape[0]} individuos")
    print()
    
    # Método 1: Evaluación individual (tradicional)
    print("Método 1: Evaluación individual")
    start = time.time()
    fitness_individual = [problema.fitness_function(ind.tolist()) for ind in poblacion]
    tiempo_individual = time.time() - start
    print(f"  Tiempo: {tiempo_individual:.4f} s")
    print(f"  Mejor fitness: {min(fitness_individual):.6e}")
    print()
    
    # Método 2: Evaluación en lote vectorizada
    print("Método 2: Evaluación en lote vectorizada")
    start = time.time()
    fitness_batch = problema.evaluate_batch_vectorized(poblacion)
    tiempo_batch = time.time() - start
    print(f"  Tiempo: {tiempo_batch:.4f} s")
    print(f"  Mejor fitness: {min(fitness_batch):.6e}")
    print(f"  Speedup: {tiempo_individual/tiempo_batch:.2f}x")
    print()
    
    # Método 3: Evaluación en lote paralela
    print("Método 3: Evaluación en lote paralela")
    start = time.time()
    fitness_parallel = problema.evaluate_batch(poblacion, n_jobs=-1)
    tiempo_parallel = time.time() - start
    print(f"  Tiempo: {tiempo_parallel:.4f} s")
    print(f"  Mejor fitness: {min(fitness_parallel):.6e}")
    print(f"  Speedup: {tiempo_individual/tiempo_parallel:.2f}x")
    print()


def example_optimization_with_batch():
    """Ejemplo de uso con algoritmos de optimización."""
    print("=" * 70)
    print("EJEMPLO 2: Optimización con Evaluación en Lote")
    print("=" * 70)
    print()
    
    # Crear el problema
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    print("Algoritmo: Evolución Diferencial")
    print(f"Configuración: población=30, iteraciones=20")
    print()
    
    # Evolución Diferencial con evaluación en lote
    de = DifferentialEvolution(
        problema, 
        pop_size=30, 
        max_iter=20,
        random_seed=42,
        verbose=False
    )
    
    start = time.time()
    mejor_solucion, mejor_error = de.optimize()
    tiempo_de = time.time() - start
    
    print(f"Resultados:")
    print(f"  Mejor solución: k0={mejor_solucion[0]:.4f}, k={mejor_solucion[1]:.4f}, a={mejor_solucion[2]:.4f}")
    print(f"  Mejor error (MSE): {mejor_error:.6e}")
    print(f"  Evaluaciones: {de.evaluations}")
    print(f"  Tiempo total: {tiempo_de:.2f} s")
    print(f"  Tiempo por evaluación: {tiempo_de/de.evaluations*1000:.2f} ms")
    print()


def example_parallel_optimization():
    """Ejemplo de optimización con procesamiento paralelo."""
    print("=" * 70)
    print("EJEMPLO 3: Optimización Paralela para Grandes Poblaciones")
    print("=" * 70)
    print()
    
    # Crear el problema
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    print("Comparación: Población Grande (100 individuos)")
    print()
    
    # Generar población grande
    lb, ub = problema.get_bounds_array()
    poblacion_grande = np.random.uniform(lb, ub, (100, 3))
    
    # Evaluación secuencial
    print("Evaluación secuencial:")
    start = time.time()
    fitness_seq = problema.evaluate_batch_vectorized(poblacion_grande)
    tiempo_seq = time.time() - start
    print(f"  Tiempo: {tiempo_seq:.4f} s")
    print()
    
    # Evaluación paralela
    print("Evaluación paralela (todos los CPUs):")
    start = time.time()
    fitness_par = problema.evaluate_batch(poblacion_grande, n_jobs=-1)
    tiempo_par = time.time() - start
    print(f"  Tiempo: {tiempo_par:.4f} s")
    print(f"  Speedup: {tiempo_seq/tiempo_par:.2f}x")
    print()
    
    # Verificar que los resultados son idénticos
    if np.allclose(fitness_seq, fitness_par):
        print("✓ Verificación: Ambos métodos producen resultados idénticos")
    else:
        print("✗ Advertencia: Los resultados difieren")
    print()


def example_custom_algorithm():
    """Ejemplo de cómo implementar un algoritmo personalizado con batch evaluation."""
    print("=" * 70)
    print("EJEMPLO 4: Algoritmo Personalizado con Batch Evaluation")
    print("=" * 70)
    print()
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    lb, ub = problema.get_bounds_array()
    
    # Parámetros del algoritmo
    pop_size = 20
    n_iterations = 10
    
    print(f"Algoritmo de Búsqueda Aleatoria Simple")
    print(f"Población: {pop_size}, Iteraciones: {n_iterations}")
    print()
    
    # Inicializar población
    mejor_solucion = None
    mejor_error = float('inf')
    
    for i in range(n_iterations):
        # Generar población aleatoria
        poblacion = np.random.uniform(lb, ub, (pop_size, 3))
        
        # Evaluar en lote (más rápido que evaluar individualmente)
        fitness = problema.evaluate_batch_vectorized(poblacion)
        
        # Encontrar el mejor
        idx_mejor = np.argmin(fitness)
        if fitness[idx_mejor] < mejor_error:
            mejor_error = fitness[idx_mejor]
            mejor_solucion = poblacion[idx_mejor].copy()
            print(f"  Iteración {i+1}: Nuevo mejor = {mejor_error:.6e}")
    
    print()
    print(f"Solución final:")
    print(f"  k0={mejor_solucion[0]:.4f}, k={mejor_solucion[1]:.4f}, a={mejor_solucion[2]:.4f}")
    print(f"  Error (MSE): {mejor_error:.6e}")
    print()


if __name__ == "__main__":
    # Ejecutar todos los ejemplos
    example_basic_batch_evaluation()
    print()
    
    example_optimization_with_batch()
    print()
    
    example_parallel_optimization()
    print()
    
    example_custom_algorithm()
    
    print("=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print()
    print("Métodos de evaluación disponibles:")
    print()
    print("1. fitness_function(solucion)")
    print("   - Evalúa una solución individual")
    print("   - Uso: Para evaluaciones puntuales")
    print()
    print("2. evaluate_batch_vectorized(poblacion)")
    print("   - Evaluación en lote con validación vectorizada")
    print("   - Uso: Poblaciones pequeñas/medianas (<100)")
    print("   - Speedup: ~3-4%")
    print()
    print("3. evaluate_batch(poblacion, n_jobs=-1)")
    print("   - Evaluación en lote con procesamiento paralelo")
    print("   - Uso: Poblaciones grandes (>100)")
    print("   - Speedup: ~2x (depende del número de CPUs)")
    print()
