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
# GREY WOLF OPTIMIZER (GWO) - Algoritmo de Lobos
# =============================================================================

class GreyWolfOptimizer(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) - Optimizador de Lobos Grises.
    
    Inspirado en la jerarquía social y comportamiento de caza de los lobos grises.
    Los lobos se dividen en: Alpha (mejor), Beta (segundo), Delta (tercero) y Omega (resto).
    
    Referencia: Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
    "Grey Wolf Optimizer." Advances in Engineering Software, 69, 46-61.
    
    Pseudocódigo:
    1. Inicializar población de lobos
    2. Identificar Alpha, Beta, Delta (3 mejores soluciones)
    3. Para cada iteración:
       a. Actualizar a (decrece linealmente de 2 a 0)
       b. Para cada lobo:
          - Calcular distancia a Alpha, Beta, Delta
          - Actualizar posición como promedio ponderado
    4. Retornar Alpha (mejor solución)
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None, 
                 verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        wolves = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(w) for w in wolves])
        
        # Ordenar y obtener Alpha, Beta, Delta
        sorted_idx = np.argsort(fitness)
        alpha, beta, delta = wolves[sorted_idx[0]].copy(), wolves[sorted_idx[1]].copy(), wolves[sorted_idx[2]].copy()
        alpha_score, beta_score, delta_score = fitness[sorted_idx[0]], fitness[sorted_idx[1]], fitness[sorted_idx[2]]
        
        for t in range(self.max_iter):
            # a decrece linealmente de 2 a 0
            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.pop_size):
                for d in range(self.dim):
                    # Coeficientes aleatorios
                    r1, r2 = self._rng.random(), self._rng.random()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    
                    r1, r2 = self._rng.random(), self._rng.random()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    
                    r1, r2 = self._rng.random(), self._rng.random()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    
                    # Distancia a Alpha, Beta, Delta
                    D_alpha = abs(C1 * alpha[d] - wolves[i, d])
                    D_beta = abs(C2 * beta[d] - wolves[i, d])
                    D_delta = abs(C3 * delta[d] - wolves[i, d])
                    
                    # Posiciones candidatas
                    X1 = alpha[d] - A1 * D_alpha
                    X2 = beta[d] - A2 * D_beta
                    X3 = delta[d] - A3 * D_delta
                    
                    # Nueva posición (promedio)
                    wolves[i, d] = (X1 + X2 + X3) / 3
                
                # Limitar a bounds
                wolves[i] = np.clip(wolves[i], self.lb, self.ub)
            
            # Evaluar y actualizar jerarquía
            fitness = np.array([self._evaluate(w) for w in wolves])
            
            for i in range(self.pop_size):
                if fitness[i] < alpha_score:
                    delta, delta_score = beta.copy(), beta_score
                    beta, beta_score = alpha.copy(), alpha_score
                    alpha, alpha_score = wolves[i].copy(), fitness[i]
                elif fitness[i] < beta_score:
                    delta, delta_score = beta.copy(), beta_score
                    beta, beta_score = wolves[i].copy(), fitness[i]
                elif fitness[i] < delta_score:
                    delta, delta_score = wolves[i].copy(), fitness[i]
            
            self.history.append(alpha_score)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Alpha = {alpha_score:.6e}")
        
        return alpha, alpha_score


# =============================================================================
# ARTIFICIAL BEE COLONY (ABC) - Colonia de Abejas
# =============================================================================

class ArtificialBeeColony(BaseOptimizer):
    """
    Artificial Bee Colony (ABC) - Colonia Artificial de Abejas.
    
    Inspirado en el comportamiento de forrajeo de las abejas melíferas.
    Tres tipos de abejas: Empleadas (explotan fuentes), Observadoras (eligen por calidad),
    y Exploradoras (buscan nuevas fuentes cuando una se agota).
    
    Referencia: Karaboga, D. (2005). "An idea based on honey bee swarm for 
    numerical optimization." Technical Report TR06, Erciyes University.
    
    Pseudocódigo:
    1. Inicializar fuentes de alimento (soluciones)
    2. Para cada iteración:
       a. Fase Empleadas: explorar vecindario de cada fuente
       b. Fase Observadoras: seleccionar fuentes por ruleta (probabilidad)
       c. Fase Exploradoras: abandonar fuentes estancadas (limit)
    3. Retornar mejor fuente encontrada
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, limit: int = None,
                 random_seed: Optional[int] = None, verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size  # Número de fuentes de alimento
        self.max_iter = max_iter
        self.limit = limit if limit else pop_size * self.dim  # Límite de estancamiento
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar fuentes de alimento
        foods = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(f) for f in foods])
        trials = np.zeros(self.pop_size)  # Contador de intentos fallidos
        
        best_idx = np.argmin(fitness)
        best_solution = foods[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # === FASE EMPLEADAS ===
            for i in range(self.pop_size):
                # Seleccionar vecino aleatorio diferente
                k = self._rng.choice([j for j in range(self.pop_size) if j != i])
                j = self._rng.integers(self.dim)  # Dimensión a modificar
                
                # Generar nueva solución
                phi = self._rng.uniform(-1, 1)
                new_food = foods[i].copy()
                new_food[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                new_food = np.clip(new_food, self.lb, self.ub)
                
                new_fitness = self._evaluate(new_food)
                
                # Selección voraz
                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # === FASE OBSERVADORAS ===
            # Calcular probabilidades (inversas al fitness para minimización)
            fit_inv = 1 / (1 + fitness)
            probs = fit_inv / fit_inv.sum()
            
            for _ in range(self.pop_size):
                i = self._rng.choice(self.pop_size, p=probs)
                k = self._rng.choice([j for j in range(self.pop_size) if j != i])
                j = self._rng.integers(self.dim)
                
                phi = self._rng.uniform(-1, 1)
                new_food = foods[i].copy()
                new_food[j] = foods[i, j] + phi * (foods[i, j] - foods[k, j])
                new_food = np.clip(new_food, self.lb, self.ub)
                
                new_fitness = self._evaluate(new_food)
                
                if new_fitness < fitness[i]:
                    foods[i] = new_food
                    fitness[i] = new_fitness
                    trials[i] = 0
                else:
                    trials[i] += 1
            
            # === FASE EXPLORADORAS ===
            for i in range(self.pop_size):
                if trials[i] > self.limit:
                    foods[i] = self._rng.uniform(self.lb, self.ub)
                    fitness[i] = self._evaluate(foods[i])
                    trials[i] = 0
            
            # Actualizar mejor
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = foods[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness


# =============================================================================
# HONEY BADGER ALGORITHM (HBA) - Tejón de Miel
# =============================================================================

class HoneyBadgerAlgorithm(BaseOptimizer):
    """
    Honey Badger Algorithm (HBA) - Algoritmo del Tejón de Miel.
    
    Inspirado en el comportamiento de forrajeo del tejón de miel (Mellivora capensis),
    conocido por su inteligencia y agresividad. Alterna entre modos de excavación
    (digging) y seguimiento de miel (honey).
    
    Referencia: Hashim, F. A., et al. (2022). "Honey Badger Algorithm: 
    New metaheuristic algorithm for solving optimization problems."
    Mathematics and Computers in Simulation, 192, 84-110.
    
    Pseudocódigo:
    1. Inicializar población
    2. Para cada iteración:
       a. Calcular intensidad de olor (I) basada en distancia a presa
       b. Fase de excavación: movimiento hacia la presa con perturbación
       c. Fase de miel: seguir al mejor con factor de atracción
       d. Actualizar posiciones con factor de decaimiento
    3. Retornar mejor solución
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, beta: float = 6.0,
                 random_seed: Optional[int] = None, verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.beta = beta  # Factor de control de intensidad
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        badgers = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(b) for b in badgers])
        
        best_idx = np.argmin(fitness)
        prey = badgers[best_idx].copy()  # Presa = mejor solución
        prey_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            alpha = self._get_alpha(t)  # Factor de decaimiento
            
            for i in range(self.pop_size):
                # Intensidad de olor
                r = self._rng.random()
                di = prey - badgers[i]  # Distancia a la presa
                S = (prey - badgers[i]) / (np.abs(di) + 1e-10)  # Dirección
                I = r * S  # Intensidad
                
                # Elegir modo: excavación o miel
                if self._rng.random() < 0.5:
                    # Modo excavación (digging)
                    r3, r4, r5 = self._rng.random(3)
                    F = 1 if r3 < 0.5 else -1  # Flag de dirección
                    new_pos = prey + F * self.beta * I * prey + F * r4 * alpha * di * np.abs(np.cos(2*np.pi*r5) * (1 - np.cos(2*np.pi*r5)))
                else:
                    # Modo miel (honey)
                    r6, r7 = self._rng.random(2)
                    F = 1 if r6 < 0.5 else -1
                    new_pos = prey + F * r7 * alpha * di
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    badgers[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < prey_fitness:
                        prey = new_pos.copy()
                        prey_fitness = new_fitness
            
            self.history.append(prey_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {prey_fitness:.6e}")
        
        return prey, prey_fitness
    
    def _get_alpha(self, t):
        """Factor de decaimiento que decrece con las iteraciones."""
        C = 2  # Constante
        return C * np.exp(-t / self.max_iter)


# =============================================================================
# SHRIMP OPTIMIZATION ALGORITHM - Algoritmo del Camarón
# =============================================================================

class ShrimpOptimizer(BaseOptimizer):
    """
    Shrimp Optimization Algorithm (SOA) - Optimizador del Camarón.
    
    Inspirado en el comportamiento social de los camarones mantis.
    Combina exploración (movimiento aleatorio) con explotación (seguir al líder).
    
    Pseudocódigo:
    1. Inicializar población de camarones
    2. Para cada iteración:
       a. Fase de exploración: movimiento browniano
       b. Fase de ataque: movimiento hacia la presa (mejor)
       c. Fase de defensa: alejarse de amenazas
       d. Fase social: seguir al grupo
    3. Retornar mejor solución
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        shrimps = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(s) for s in shrimps])
        
        best_idx = np.argmin(fitness)
        best_solution = shrimps[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        for t in range(self.max_iter):
            # Factor de exploración/explotación
            w = 1 - t / self.max_iter  # Decrece de 1 a 0
            
            for i in range(self.pop_size):
                r = self._rng.random()
                
                if r < 0.5:
                    # Fase de exploración: Lévy flight simplificado
                    levy = self._levy_flight()
                    new_pos = shrimps[i] + w * levy * (self.ub - self.lb)
                else:
                    # Fase de explotación: movimiento hacia el mejor
                    r1, r2 = self._rng.random(2)
                    new_pos = shrimps[i] + r1 * (best_solution - shrimps[i]) + r2 * (shrimps[self._rng.integers(self.pop_size)] - shrimps[i])
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    shrimps[i] = new_pos
                    fitness[i] = new_fitness
                    
                    if new_fitness < best_fitness:
                        best_solution = new_pos.copy()
                        best_fitness = new_fitness
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
    
    def _levy_flight(self, beta=1.5):
        """Genera un paso de Lévy flight."""
        from math import gamma
        sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = self._rng.normal(0, sigma, self.dim)
        v = self._rng.normal(0, 1, self.dim)
        return u / (np.abs(v)**(1 / beta))


# =============================================================================
# TIANJI HORSE RACING - Estrategia Tianji
# =============================================================================

class TianjiOptimizer(BaseOptimizer):
    """
    Tianji Horse Racing Strategy - Estrategia de Carreras de Caballos Tianji.
    
    Basado en la antigua estrategia china donde se gana al oponente usando
    el caballo inferior contra el superior del rival, el superior contra el medio,
    y el medio contra el inferior. En optimización, esto se traduce en una
    estrategia de reemplazo inteligente.
    
    Pseudocódigo:
    1. Dividir población en 3 grupos: Superior, Medio, Inferior
    2. Para cada iteración:
       a. Grupo Superior explota (movimiento local fino)
       b. Grupo Medio balancea exploración/explotación
       c. Grupo Inferior explora (movimiento global amplio)
       d. Intercambio estratégico entre grupos
    3. Retornar mejor solución
    """
    
    def __init__(self, problema: LevitadorBenchmark, pop_size: int = 30,
                 max_iter: int = 100, random_seed: Optional[int] = None,
                 verbose: bool = True):
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        # Inicializar población
        horses = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(h) for h in horses])
        
        best_idx = np.argmin(fitness)
        best_solution = horses[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Tamaño de cada grupo
        group_size = self.pop_size // 3
        
        for t in range(self.max_iter):
            # Ordenar por fitness
            sorted_idx = np.argsort(fitness)
            
            # Dividir en grupos
            superior = sorted_idx[:group_size]
            medio = sorted_idx[group_size:2*group_size]
            inferior = sorted_idx[2*group_size:]
            
            # Factor de adaptación
            sigma = 0.1 * (1 - t / self.max_iter)  # Decrece con el tiempo
            
            # === GRUPO SUPERIOR: Explotación local ===
            for i in superior:
                perturbation = self._rng.normal(0, sigma, self.dim) * (self.ub - self.lb)
                new_pos = horses[i] + perturbation
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    horses[i] = new_pos
                    fitness[i] = new_fitness
            
            # === GRUPO MEDIO: Balance ===
            for i in medio:
                if self._rng.random() < 0.5:
                    # Moverse hacia el mejor
                    r = self._rng.random()
                    new_pos = horses[i] + r * (best_solution - horses[i])
                else:
                    # Exploración moderada
                    perturbation = self._rng.normal(0, sigma * 2, self.dim) * (self.ub - self.lb)
                    new_pos = horses[i] + perturbation
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fitness = self._evaluate(new_pos)
                
                if new_fitness < fitness[i]:
                    horses[i] = new_pos
                    fitness[i] = new_fitness
            
            # === GRUPO INFERIOR: Exploración global ===
            for i in inferior:
                if self._rng.random() < 0.3:
                    # Reinicialización aleatoria
                    new_pos = self._rng.uniform(self.lb, self.ub)
                else:
                    # Salto grande
                    r1, r2 = self._rng.random(2)
                    j = self._rng.choice(superior)  # Aprender del superior
                    new_pos = horses[i] + r1 * (horses[j] - horses[i]) + r2 * self._rng.normal(0, 0.5, self.dim) * (self.ub - self.lb)
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                fitness[i] = self._evaluate(new_pos)
                horses[i] = new_pos
            
            # Actualizar mejor
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_solution = horses[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            self.history.append(best_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness


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
