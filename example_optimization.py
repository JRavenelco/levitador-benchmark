"""
Ejemplo de Uso del Levitador Benchmark
=======================================

Este script demuestra cómo usar el benchmark con diferentes 
algoritmos de optimización bio-inspirados.

Ejecutar: python example_optimization.py

Clases disponibles para importar:
    - RandomSearch: Búsqueda aleatoria (baseline)
    - DifferentialEvolution: Evolución Diferencial
    - GeneticAlgorithm: Algoritmo Genético

Uso:
    from example_optimization import RandomSearch, GeneticAlgorithm
    algo = GeneticAlgorithm(problema, pop_size=30, generations=50, random_seed=42)
    mejor_sol, mejor_error = algo.optimize()
"""

import numpy as np
from levitador_benchmark import LevitadorBenchmark
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

# =============================================================================
# CLASE BASE PARA ALGORITMOS
# =============================================================================

class BaseOptimizer(ABC):
    """
    Clase base abstracta para algoritmos de optimización.
    
    Todos los algoritmos deben heredar de esta clase e implementar
    el método optimize().
    """
    
    def __init__(self, problema: LevitadorBenchmark, random_seed: Optional[int] = None):
        """
        Args:
            problema: Instancia de LevitadorBenchmark
            random_seed: Semilla para reproducibilidad
        """
        self.problema = problema
        self.dim = problema.dim
        self.bounds = np.array(problema.bounds)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.evaluations = 0
        self.history = []
        
        if random_seed is not None:
            np.random.seed(random_seed)
            self._rng = np.random.default_rng(random_seed)
        else:
            self._rng = np.random.default_rng()
    
    def _evaluate(self, solution: List[float]) -> float:
        """Evalúa una solución y registra la evaluación."""
        self.evaluations += 1
        return self.problema.fitness_function(solution)
    
    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el algoritmo de optimización.
        
        Returns:
            Tuple[mejor_solucion, mejor_error]
        """
        pass


# =============================================================================
# BUSQUEDA ALEATORIA
# =============================================================================

class RandomSearch(BaseOptimizer):
    """
    Búsqueda Aleatoria (Random Search).
    
    Algoritmo baseline que genera soluciones aleatorias uniformemente
    distribuidas en el espacio de búsqueda.
    """
    
    def __init__(self, problema: LevitadorBenchmark, n_iterations: int = 1000,
                 random_seed: Optional[int] = None, verbose: bool = True):
        super().__init__(problema, random_seed)
        self.n_iterations = n_iterations
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        best_error = float('inf')
        best_solution = None
        
        for i in range(self.n_iterations):
            solution = self._rng.uniform(self.lb, self.ub)
            error = self._evaluate(solution)
            
            if error < best_error:
                best_error = error
                best_solution = solution.copy()
                if self.verbose:
                    print(f"  Iter {i+1}: Nuevo mejor = {error:.6e}")
            
            self.history.append(best_error)
        
        return best_solution, best_error


# =============================================================================
# EVOLUCION DIFERENCIAL
# =============================================================================

class DifferentialEvolution(BaseOptimizer):
    """
    Evolución Diferencial (DE/rand/1/bin).
    
    Implementación clásica del algoritmo de Storn & Price (1997).
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, F: float = 0.8, CR: float = 0.9,
                 random_seed: Optional[int] = None, verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F    # Factor de escala de mutación
        self.CR = CR  # Probabilidad de cruce
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_error = fitness[best_idx]
        
        for gen in range(self.max_iter):
            for i in range(self.pop_size):
                # Seleccionar 3 individuos distintos
                indices = [j for j in range(self.pop_size) if j != i]
                a, b, c = population[self._rng.choice(indices, 3, replace=False)]
                
                # Mutación: v = a + F * (b - c)
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                
                # Cruce binomial
                trial = population[i].copy()
                j_rand = self._rng.integers(self.dim)
                for j in range(self.dim):
                    if self._rng.random() < self.CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Selección
                trial_fitness = self._evaluate(trial)
                
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    
                    if trial_fitness < best_error:
                        best_solution = trial.copy()
                        best_error = trial_fitness
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Mejor = {best_error:.6e}")
        
        return best_solution, best_error


# =============================================================================
# ALGORITMO GENETICO
# =============================================================================

class GeneticAlgorithm(BaseOptimizer):
    """
    Algoritmo Genético con selección por torneo, cruce BLX-alpha y mutación gaussiana.
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 generations: int = 50, crossover_prob: float = 0.8,
                 mutation_prob: float = 0.2, alpha: float = 0.5,
                 random_seed: Optional[int] = None, verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.alpha = alpha  # Parámetro para cruce BLX-alpha
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        best_solution = None
        best_error = float('inf')
        
        for gen in range(self.generations):
            # Evaluar fitness
            fitness = np.array([self._evaluate(ind) for ind in population])
            
            # Actualizar mejor
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_error:
                best_error = fitness[best_idx]
                best_solution = population[best_idx].copy()
            
            self.history.append(best_error)
            
            if self.verbose and gen % 10 == 0:
                print(f"  Gen {gen:3d}: Mejor = {best_error:.6e}")
            
            # Selección por torneo
            parents = []
            for _ in range(self.pop_size):
                i, j = self._rng.choice(self.pop_size, 2, replace=False)
                winner = i if fitness[i] < fitness[j] else j
                parents.append(population[winner].copy())
            
            # Cruce BLX-alpha
            children = []
            for i in range(0, self.pop_size, 2):
                p1, p2 = parents[i], parents[min(i+1, self.pop_size-1)]
                
                if self._rng.random() < self.crossover_prob:
                    for d in range(self.dim):
                        rango = abs(p1[d] - p2[d])
                        minimo = min(p1[d], p2[d]) - self.alpha * rango
                        maximo = max(p1[d], p2[d]) + self.alpha * rango
                        p1[d] = self._rng.uniform(minimo, maximo)
                        p2[d] = self._rng.uniform(minimo, maximo)
                
                children.extend([p1, p2])
            
            # Mutación gaussiana
            for ind in children:
                if self._rng.random() < self.mutation_prob:
                    for d in range(self.dim):
                        sigma = (self.ub[d] - self.lb[d]) * 0.1
                        ind[d] += self._rng.normal(0, sigma)
                        ind[d] = np.clip(ind[d], self.lb[d], self.ub[d])
            
            # Elitismo
            children[0] = best_solution.copy()
            population = np.array(children[:self.pop_size])
        
        return best_solution, best_error


# =============================================================================
# FUNCIONES DE EJEMPLO (compatibilidad hacia atrás)
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
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    algo = RandomSearch(problema, n_iterations=n_iter, random_seed=42)
    
    print(f"\nEjecutando {n_iter} evaluaciones aleatorias...")
    mejor_sol, mejor_error = algo.optimize()
    
    print("\n🎯 Mejor solución encontrada:")
    print(f"  k0 = {mejor_sol[0]:.6f}")
    print(f"  k  = {mejor_sol[1]:.6f}")
    print(f"  a  = {mejor_sol[2]:.6f}")
    print(f"  Error: {mejor_error:.6e}")
    print(f"  Evaluaciones: {algo.evaluations}")
    
    return mejor_sol


# =============================================================================
# EJEMPLO 4: Algoritmo Genético Simple
# =============================================================================

def ejemplo_genetic_algorithm(pop_size=30, generations=50):
    """Algoritmo genético básico."""
    print("\n" + "="*60)
    print("EJEMPLO 4: Algoritmo Genético Simple")
    print("="*60)
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    algo = GeneticAlgorithm(problema, pop_size=pop_size, generations=generations, random_seed=42)
    
    print(f"\nPoblación: {pop_size} individuos")
    print(f"Generaciones: {generations}")
    
    mejor, mejor_error = algo.optimize()
    
    print("\n🧬 Mejor individuo:")
    print(f"  k0 = {mejor[0]:.6f}")
    print(f"  k  = {mejor[1]:.6f}")
    print(f"  a  = {mejor[2]:.6f}")
    print(f"  Error: {mejor_error:.6e}")
    print(f"  Evaluaciones: {algo.evaluations}")
    
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
