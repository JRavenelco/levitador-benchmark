"""
Ejemplo de Uso del Levitador Benchmark
=======================================

Este script demuestra cómo usar el benchmark con diferentes 
algoritmos de optimización bio-inspirados.

Ejecutar: python example_optimization.py
"""

import numpy as np
from levitador_benchmark import LevitadorBenchmark

# =============================================================================
# EJEMPLO 1: Evaluación básica
# =============================================================================

def ejemplo_basico():
    """Muestra cómo evaluar soluciones individuales."""
    print("\n" + "="*60)
    print("EJEMPLO 1: Evaluación Básica")
    print("="*60)
    
    # Crear benchmark (usa datos sintéticos si no hay archivo)
    problema = LevitadorBenchmark()
    
    # Evaluar diferentes soluciones
    soluciones = [
        [0.0363, 0.0035, 0.0052],  # Solución de referencia
        [0.05, 0.01, 0.01],        # Solución aleatoria 1
        [0.02, 0.002, 0.003],      # Solución aleatoria 2
    ]
    
    print("\nEvaluando soluciones:")
    for sol in soluciones:
        error = problema.fitness_function(sol)
        print(f"  {sol} → MSE = {error:.6e}")


# =============================================================================
# EJEMPLO 2: Evolución Diferencial (SciPy)
# =============================================================================

def ejemplo_differential_evolution():
    """Optimización con Evolución Diferencial."""
    print("\n" + "="*60)
    print("EJEMPLO 2: Evolución Diferencial (SciPy)")
    print("="*60)
    
    from scipy.optimize import differential_evolution
    
    problema = LevitadorBenchmark()
    
    print(f"\nEspacio de búsqueda: {problema.bounds}")
    print("Iniciando optimización...\n")
    
    resultado = differential_evolution(
        problema.fitness_function,
        problema.bounds,
        strategy='best1bin',
        maxiter=50,
        popsize=15,
        tol=1e-10,
        seed=42,
        disp=True,
        polish=True
    )
    
    print("\n🏆 Resultado:")
    print(f"  k0 = {resultado.x[0]:.6f} H")
    print(f"  k  = {resultado.x[1]:.6f} H")
    print(f"  a  = {resultado.x[2]:.6f} m")
    print(f"  Error final (MSE): {resultado.fun:.6e}")
    print(f"  Evaluaciones: {resultado.nfev}")
    
    return resultado.x


# =============================================================================
# EJEMPLO 3: Búsqueda Aleatoria (Baseline)
# =============================================================================

def ejemplo_random_search(n_iter=1000):
    """Búsqueda aleatoria como baseline simple."""
    print("\n" + "="*60)
    print("EJEMPLO 3: Búsqueda Aleatoria (Baseline)")
    print("="*60)
    
    problema = LevitadorBenchmark()
    
    mejor_error = float('inf')
    mejor_sol = None
    
    print(f"\nEjecutando {n_iter} evaluaciones aleatorias...")
    
    for i in range(n_iter):
        # Generar solución aleatoria dentro de los límites
        sol = [np.random.uniform(lb, ub) for lb, ub in problema.bounds]
        error = problema.fitness_function(sol)
        
        if error < mejor_error:
            mejor_error = error
            mejor_sol = sol.copy()
            print(f"  Iter {i+1}: Nuevo mejor = {error:.6e}")
    
    print("\n🎯 Mejor solución encontrada:")
    print(f"  k0 = {mejor_sol[0]:.6f}")
    print(f"  k  = {mejor_sol[1]:.6f}")
    print(f"  a  = {mejor_sol[2]:.6f}")
    print(f"  Error: {mejor_error:.6e}")
    
    return mejor_sol


# =============================================================================
# EJEMPLO 4: Algoritmo Genético Simple
# =============================================================================

def ejemplo_genetic_algorithm(pop_size=30, generations=50):
    """Algoritmo genético básico implementado desde cero."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Algoritmo Genético Simple")
    print("="*60)
    
    problema = LevitadorBenchmark()
    bounds = np.array(problema.bounds)
    
    # Inicializar población
    population = np.random.uniform(
        bounds[:, 0], bounds[:, 1], 
        size=(pop_size, problema.dim)
    )
    
    print(f"\nPoblación inicial: {pop_size} individuos")
    print(f"Generaciones: {generations}")
    
    mejor_historico = []
    
    for gen in range(generations):
        # Evaluar fitness
        fitness = np.array([problema.fitness_function(ind) for ind in population])
        
        # Guardar mejor
        mejor_idx = np.argmin(fitness)
        mejor_historico.append(fitness[mejor_idx])
        
        if gen % 10 == 0:
            print(f"  Gen {gen:3d}: Mejor = {fitness[mejor_idx]:.6e}")
        
        # Selección por torneo
        nuevos_padres = []
        for _ in range(pop_size):
            i, j = np.random.choice(pop_size, 2, replace=False)
            ganador = i if fitness[i] < fitness[j] else j
            nuevos_padres.append(population[ganador].copy())
        
        # Cruce (BLX-alpha)
        alpha = 0.5
        hijos = []
        for i in range(0, pop_size, 2):
            p1, p2 = nuevos_padres[i], nuevos_padres[min(i+1, pop_size-1)]
            
            # Cruce con probabilidad 0.8
            if np.random.random() < 0.8:
                for d in range(problema.dim):
                    rango = abs(p1[d] - p2[d])
                    minimo = min(p1[d], p2[d]) - alpha * rango
                    maximo = max(p1[d], p2[d]) + alpha * rango
                    p1[d] = np.random.uniform(minimo, maximo)
                    p2[d] = np.random.uniform(minimo, maximo)
            
            hijos.extend([p1, p2])
        
        # Mutación gaussiana
        for ind in hijos:
            if np.random.random() < 0.2:
                for d in range(problema.dim):
                    sigma = (bounds[d, 1] - bounds[d, 0]) * 0.1
                    ind[d] += np.random.normal(0, sigma)
                    ind[d] = np.clip(ind[d], bounds[d, 0], bounds[d, 1])
        
        # Elitismo: mantener el mejor
        hijos[0] = population[mejor_idx].copy()
        population = np.array(hijos[:pop_size])
    
    # Resultado final
    fitness_final = np.array([problema.fitness_function(ind) for ind in population])
    mejor_idx = np.argmin(fitness_final)
    mejor = population[mejor_idx]
    
    print("\n🧬 Mejor individuo:")
    print(f"  k0 = {mejor[0]:.6f}")
    print(f"  k  = {mejor[1]:.6f}")
    print(f"  a  = {mejor[2]:.6f}")
    print(f"  Error: {fitness_final[mejor_idx]:.6e}")
    
    return mejor


# =============================================================================
# EJEMPLO 5: Comparación de Algoritmos
# =============================================================================

def ejemplo_comparacion():
    """Compara diferentes algoritmos en el mismo problema."""
    print("\n" + "="*60)
    print("EJEMPLO 5: Comparación de Algoritmos")
    print("="*60)
    
    problema = LevitadorBenchmark()
    
    resultados = {}
    
    # 1. Random Search
    print("\n[1/3] Búsqueda Aleatoria...")
    mejor_rand = ejemplo_random_search(500)
    resultados['Random'] = problema.fitness_function(mejor_rand)
    
    # 2. Differential Evolution
    print("\n[2/3] Evolución Diferencial...")
    mejor_de = ejemplo_differential_evolution()
    resultados['DE'] = problema.fitness_function(mejor_de)
    
    # 3. Genetic Algorithm
    print("\n[3/3] Algoritmo Genético...")
    mejor_ga = ejemplo_genetic_algorithm(30, 30)
    resultados['GA'] = problema.fitness_function(mejor_ga)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE COMPARACIÓN")
    print("="*60)
    print(f"\n{'Algoritmo':<20} {'MSE':>15}")
    print("-" * 35)
    for alg, mse in sorted(resultados.items(), key=lambda x: x[1]):
        print(f"{alg:<20} {mse:>15.6e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🧲"*30)
    print("   LEVITADOR MAGNÉTICO BENCHMARK")
    print("   Ejemplos de Optimización Bio-Inspirada")
    print("🧲"*30)
    
    # Ejecutar ejemplos
    ejemplo_basico()
    
    # Descomentar para ejecutar otros ejemplos:
    # ejemplo_differential_evolution()
    # ejemplo_random_search()
    # ejemplo_genetic_algorithm()
    # ejemplo_comparacion()
    
    print("\n✅ Ejemplos completados.")
    print("Descomentar otras funciones en __main__ para más demos.")
