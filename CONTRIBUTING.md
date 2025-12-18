# 🤝 Guía de Contribución

¡Gracias por tu interés en contribuir al **Levitador Magnético Benchmark**! Este documento proporciona directrices para contribuir implementaciones de metaheurísticas y mejoras al proyecto.

---

## 📋 Tabla de Contenidos

1. [Código de Conducta](#código-de-conducta)
2. [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
3. [Contribuir Metaheurísticas](#contribuir-metaheurísticas)
4. [Estándares de Código](#estándares-de-código)
5. [Proceso de Pull Request](#proceso-de-pull-request)
6. [Reportar Bugs](#reportar-bugs)
7. [Sugerencias de Mejoras](#sugerencias-de-mejoras)

---

## Código de Conducta

Este proyecto se compromete a proporcionar un ambiente acogedor y respetuoso para todos los contribuyentes. Esperamos que todos los participantes:

- Usen lenguaje acogedor e inclusivo
- Respeten diferentes puntos de vista y experiencias
- Acepten críticas constructivas con gracia
- Se enfoquen en lo que es mejor para la comunidad

---

## ¿Cómo Puedo Contribuir?

Hay varias formas de contribuir a este proyecto:

### 1. 🧬 Implementar Nuevas Metaheurísticas

La contribución más valiosa es agregar implementaciones de nuevos algoritmos bio-inspirados y metaheurísticas. Ver la sección [Contribuir Metaheurísticas](#contribuir-metaheurísticas) más abajo.

### 2. 📊 Compartir Resultados

- Reporta resultados de tus experimentos con diferentes algoritmos
- Comparte configuraciones de parámetros que funcionan bien
- Documenta casos de uso interesantes del benchmark

### 3. 🐛 Reportar Bugs

- Revisa primero los issues existentes para evitar duplicados
- Proporciona pasos claros para reproducir el problema
- Incluye información del sistema (Python version, OS, dependencias)

### 4. 📚 Mejorar Documentación

- Corregir errores tipográficos o gramaticales
- Mejorar explicaciones existentes
- Agregar ejemplos adicionales
- Traducir documentación a otros idiomas

### 5. ✨ Proponer Mejoras

- Nuevas características para el benchmark
- Mejoras en la API
- Herramientas de visualización
- Utilidades de análisis

---

## Contribuir Metaheurísticas

Esta sección proporciona una guía completa para contribuir implementaciones de algoritmos metaheurísticos.

### Estructura de un Algoritmo

Todos los algoritmos deben heredar de la clase `BaseOptimizer` y seguir esta estructura:

```python
from example_optimization import BaseOptimizer
from levitador_benchmark import LevitadorBenchmark
from typing import Tuple, Optional
import numpy as np

class MiAlgoritmo(BaseOptimizer):
    """
    Nombre del Algoritmo (Acrónimo).
    
    Descripción breve del algoritmo y su inspiración biológica o física.
    
    Referencia: Autor, A. et al. (año). "Título del paper."
    Nombre de la revista/conferencia, volumen, páginas.
    
    Pseudocódigo:
    1. Inicializar población
    2. Para cada iteración:
       a. Paso de exploración
       b. Paso de explotación
       c. Actualizar mejor solución
    3. Retornar mejor solución encontrada
    """
    
    def __init__(self, problema: LevitadorBenchmark, 
                 pop_size: int = 30,
                 max_iter: int = 100,
                 random_seed: Optional[int] = None,
                 verbose: bool = True,
                 **kwargs):
        """
        Inicializa el algoritmo.
        
        Args:
            problema: Instancia de LevitadorBenchmark
            pop_size: Tamaño de la población
            max_iter: Número máximo de iteraciones
            random_seed: Semilla para reproducibilidad
            verbose: Si True, muestra mensajes de progreso
            **kwargs: Parámetros específicos del algoritmo
        """
        super().__init__(problema, random_seed)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.verbose = verbose
        # Agregar parámetros específicos del algoritmo
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """
        Ejecuta el algoritmo de optimización.
        
        Returns:
            Tuple[mejor_solucion, mejor_fitness]
        """
        # Inicializar población
        population = self._rng.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([self._evaluate(ind) for ind in population])
        
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        # Ciclo principal
        for t in range(self.max_iter):
            # Implementar lógica del algoritmo aquí
            
            # Actualizar mejor solución
            if fitness[best_idx] < best_fitness:
                best_solution = population[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            # Registrar historial
            self.history.append(best_fitness)
            
            # Mostrar progreso
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
        
        return best_solution, best_fitness
```

### Requisitos para Implementaciones

#### 1. **Herencia de BaseOptimizer**

Tu algoritmo DEBE heredar de `BaseOptimizer`, que proporciona:
- `self.problema`: Instancia del problema
- `self.dim`: Dimensión del problema (3 para levitador)
- `self.bounds`: Límites del espacio de búsqueda
- `self.lb`, `self.ub`: Límites inferior y superior como arrays
- `self._rng`: Generador de números aleatorios (para reproducibilidad)
- `self._evaluate()`: Método para evaluar soluciones
- `self.evaluations`: Contador de evaluaciones
- `self.history`: Lista para registrar evolución del fitness

#### 2. **Documentación Completa**

Cada algoritmo DEBE incluir:

- **Docstring de clase** con:
  - Nombre completo y acrónimo del algoritmo
  - Descripción de la inspiración (biológica, física, etc.)
  - Referencia bibliográfica completa
  - Pseudocódigo simplificado del algoritmo
  
- **Docstring de `__init__`** con:
  - Descripción de cada parámetro
  - Valores por defecto recomendados
  
- **Docstring de `optimize`** con:
  - Descripción del proceso
  - Tipo de retorno

#### 3. **Reproducibilidad**

- Usar `self._rng` en lugar de `np.random` para todas las operaciones aleatorias
- Permitir `random_seed` como parámetro en `__init__`
- Ejemplo:
  ```python
  # ✅ CORRECTO
  value = self._rng.random()
  indices = self._rng.choice(n, size=k, replace=False)
  
  # ❌ INCORRECTO
  value = np.random.random()
  indices = np.random.choice(n, size=k, replace=False)
  ```

#### 4. **Límites del Espacio de Búsqueda**

- Respetar `self.lb` y `self.ub` en todo momento
- Usar `np.clip()` para mantener soluciones dentro de límites
- Ejemplo:
  ```python
  new_solution = current_solution + perturbation
  new_solution = np.clip(new_solution, self.lb, self.ub)
  ```

#### 5. **Evaluación de Fitness**

- Usar SIEMPRE `self._evaluate(solution)` en lugar de `self.problema.fitness_function()`
- Esto permite el conteo automático de evaluaciones
- Ejemplo:
  ```python
  # ✅ CORRECTO
  fitness = self._evaluate(individual)
  
  # ❌ INCORRECTO
  fitness = self.problema.fitness_function(individual)
  ```

#### 6. **Registro de Historial**

- Agregar el mejor fitness de cada iteración a `self.history`
- Esto permite análisis de convergencia
- Ejemplo:
  ```python
  for t in range(self.max_iter):
      # ... lógica del algoritmo ...
      self.history.append(best_fitness)
  ```

#### 7. **Mensajes de Progreso**

- Proporcionar parámetro `verbose` (default: `True`)
- Mostrar progreso cada 10 iteraciones
- Formato consistente:
  ```python
  if self.verbose and t % 10 == 0:
      print(f"  Iter {t:3d}: Mejor = {best_fitness:.6e}")
  ```

### Ejemplo Completo: Particle Swarm Optimization

```python
class ParticleSwarmOptimization(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) - Optimización por Enjambre de Partículas.
    
    Inspirado en el comportamiento social de bandadas de aves y cardúmenes de peces.
    Cada partícula ajusta su velocidad basándose en su mejor posición personal
    (pbest) y la mejor posición global del enjambre (gbest).
    
    Referencia: Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization."
    Proceedings of ICNN'95 - International Conference on Neural Networks, 4, 1942-1948.
    
    Pseudocódigo:
    1. Inicializar posiciones y velocidades de partículas
    2. Para cada iteración:
       a. Evaluar fitness de cada partícula
       b. Actualizar pbest de cada partícula
       c. Actualizar gbest del enjambre
       d. Actualizar velocidades: v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
       e. Actualizar posiciones: x = x + v
    3. Retornar gbest
    """
    
    def __init__(self, problema: LevitadorBenchmark,
                 n_particles: int = 30,
                 max_iter: int = 100,
                 w: float = 0.7,
                 c1: float = 1.5,
                 c2: float = 1.5,
                 random_seed: Optional[int] = None,
                 verbose: bool = True):
        """
        Inicializa PSO.
        
        Args:
            problema: Instancia de LevitadorBenchmark
            n_particles: Número de partículas en el enjambre
            max_iter: Número máximo de iteraciones
            w: Inercia (peso de la velocidad anterior)
            c1: Coeficiente cognitivo (atracción a pbest)
            c2: Coeficiente social (atracción a gbest)
            random_seed: Semilla para reproducibilidad
            verbose: Si True, muestra progreso
        """
        super().__init__(problema, random_seed)
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """Ejecuta PSO."""
        # Inicializar posiciones y velocidades
        positions = self._rng.uniform(self.lb, self.ub, (self.n_particles, self.dim))
        velocities = self._rng.uniform(-1, 1, (self.n_particles, self.dim))
        
        # Evaluar fitness inicial
        fitness = np.array([self._evaluate(p) for p in positions])
        
        # Inicializar pbest y gbest
        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        
        gbest_idx = np.argmin(fitness)
        gbest_position = positions[gbest_idx].copy()
        gbest_fitness = fitness[gbest_idx]
        
        # Ciclo principal
        for t in range(self.max_iter):
            for i in range(self.n_particles):
                # Actualizar velocidad
                r1 = self._rng.random(self.dim)
                r2 = self._rng.random(self.dim)
                
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Actualizar posición
                positions[i] = positions[i] + velocities[i]
                positions[i] = np.clip(positions[i], self.lb, self.ub)
                
                # Evaluar nueva posición
                fitness[i] = self._evaluate(positions[i])
                
                # Actualizar pbest
                if fitness[i] < pbest_fitness[i]:
                    pbest_positions[i] = positions[i].copy()
                    pbest_fitness[i] = fitness[i]
                    
                    # Actualizar gbest
                    if fitness[i] < gbest_fitness:
                        gbest_position = positions[i].copy()
                        gbest_fitness = fitness[i]
            
            self.history.append(gbest_fitness)
            
            if self.verbose and t % 10 == 0:
                print(f"  Iter {t:3d}: Mejor = {gbest_fitness:.6e}")
        
        return gbest_position, gbest_fitness
```

### Dónde Agregar tu Implementación

Agrega tu algoritmo en el archivo `example_optimization.py` siguiendo estos pasos:

1. **Ubicación:** Agregar después de las implementaciones existentes
2. **Orden:** Mantener orden alfabético por nombre del algoritmo
3. **Separación:** Usar el separador estándar:
   ```python
   # =============================================================================
   # NOMBRE DEL ALGORITMO
   # =============================================================================
   ```

### Ejemplo de Uso

Después de implementar tu algoritmo, agrega un ejemplo de uso al final de `example_optimization.py`:

```python
def ejemplo_mi_algoritmo():
    """Ejemplo de uso de Mi Algoritmo."""
    print("\n" + "="*60)
    print("EJEMPLO: Mi Algoritmo")
    print("="*60)
    
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    
    algo = MiAlgoritmo(
        problema, 
        pop_size=30, 
        max_iter=100,
        random_seed=42
    )
    
    print("\nEjecutando optimización...")
    mejor_sol, mejor_error = algo.optimize()
    
    print("\n🏆 Resultado:")
    print(f"  k0 = {mejor_sol[0]:.6f} H")
    print(f"  k  = {mejor_sol[1]:.6f} H")
    print(f"  a  = {mejor_sol[2]:.6f} m")
    print(f"  Error: {mejor_error:.6e}")
    print(f"  Evaluaciones: {algo.evaluations}")
    
    return mejor_sol
```

### Testing

Aunque no es obligatorio para la contribución inicial, es recomendable probar tu algoritmo:

```python
# Test básico
def test_mi_algoritmo():
    problema = LevitadorBenchmark(random_seed=42, verbose=False)
    algo = MiAlgoritmo(problema, pop_size=10, max_iter=5, random_seed=42)
    
    mejor_sol, mejor_error = algo.optimize()
    
    # Verificaciones básicas
    assert len(mejor_sol) == 3
    assert mejor_error > 0
    assert all(lb <= x <= ub for x, (lb, ub) in zip(mejor_sol, problema.bounds))
    assert len(algo.history) == 5  # max_iter
    
    print("✅ Test pasado")
```

### Checklist de Contribución

Antes de enviar tu Pull Request, verifica que tu implementación cumple con:

- [ ] Hereda de `BaseOptimizer`
- [ ] Incluye docstring completo con referencia bibliográfica
- [ ] Usa `self._rng` para todas las operaciones aleatorias
- [ ] Respeta `self.lb` y `self.ub` con `np.clip()`
- [ ] Usa `self._evaluate()` para evaluar fitness
- [ ] Registra `self.history` en cada iteración
- [ ] Incluye parámetro `verbose` con mensajes de progreso
- [ ] Proporciona ejemplo de uso
- [ ] Los nombres de variables y comentarios son claros
- [ ] El código sigue el estilo PEP 8
- [ ] Has probado el algoritmo con el benchmark

---

## Estándares de Código

### Estilo de Código

- Seguir [PEP 8](https://www.python.org/dev/peps/pep-0008/) para Python
- Usar nombres descriptivos para variables y funciones
- Límite de 100 caracteres por línea (preferible 80)
- Usar comillas simples `'` para strings (excepto docstrings)

### Comentarios y Documentación

```python
# ✅ BUEN ESTILO
def update_velocity(self, particle_idx: int) -> np.ndarray:
    """
    Actualiza la velocidad de una partícula según PSO.
    
    Args:
        particle_idx: Índice de la partícula
        
    Returns:
        Nueva velocidad como array numpy
    """
    # Componente cognitiva (atracción a pbest)
    r1 = self._rng.random(self.dim)
    cognitive = self.c1 * r1 * (self.pbest[particle_idx] - self.positions[particle_idx])
    
    # Componente social (atracción a gbest)
    r2 = self._rng.random(self.dim)
    social = self.c2 * r2 * (self.gbest - self.positions[particle_idx])
    
    return self.w * self.velocities[particle_idx] + cognitive + social
```

### Imports

Orden de imports:
1. Biblioteca estándar de Python
2. Bibliotecas de terceros
3. Imports locales del proyecto

```python
# Biblioteca estándar
from typing import Tuple, Optional
import logging

# Terceros
import numpy as np
from scipy.optimize import minimize

# Locales
from levitador_benchmark import LevitadorBenchmark
from example_optimization import BaseOptimizer
```

### Type Hints

Usar type hints para claridad:

```python
def optimize(self) -> Tuple[np.ndarray, float]:
    """..."""
    pass

def __init__(self, problema: LevitadorBenchmark, 
             pop_size: int = 30,
             max_iter: int = 100,
             random_seed: Optional[int] = None) -> None:
    """..."""
    pass
```

---

## Proceso de Pull Request

### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub
# Luego clona tu fork
git clone https://github.com/TU_USUARIO/levitador-benchmark.git
cd levitador-benchmark
```

### 2. Crear Branch

```bash
# Crear branch con nombre descriptivo
git checkout -b add-particle-swarm-algorithm

# O para correcciones
git checkout -b fix-differential-evolution-bounds
```

### 3. Hacer Cambios

- Implementa tu algoritmo siguiendo las guías anteriores
- Prueba tu implementación localmente
- Asegúrate de que todo funciona correctamente

### 4. Commit

Usa mensajes de commit claros y descriptivos:

```bash
# ✅ BUENOS mensajes
git commit -m "Add Particle Swarm Optimization implementation"
git commit -m "Fix bounds checking in Differential Evolution"
git commit -m "Add documentation for Grey Wolf Optimizer"

# ❌ MALOS mensajes
git commit -m "Update code"
git commit -m "Fix bug"
git commit -m "Changes"
```

### 5. Push y Pull Request

```bash
# Push tu branch
git push origin add-particle-swarm-algorithm
```

Luego en GitHub:
1. Navega a tu fork
2. Click en "New Pull Request"
3. Selecciona tu branch
4. Completa la descripción del PR

### Plantilla de Pull Request

```markdown
## Descripción

Implementación de [Nombre del Algoritmo] basado en [breve descripción de la inspiración].

## Tipo de cambio

- [ ] Nueva metaheurística
- [ ] Corrección de bug
- [ ] Mejora de documentación
- [ ] Otra (especificar): _____

## Algoritmo

**Nombre:** Particle Swarm Optimization (PSO)
**Referencia:** Kennedy, J., & Eberhart, R. (1995)
**Características:**
- Tamaño de población configurable
- Parámetros: w, c1, c2
- Soporta reproducibilidad con random_seed

## Testing

- [x] Probado con datos sintéticos
- [ ] Probado con datos experimentales
- [x] Ejemplo de uso incluido
- [x] Documentación completa

## Resultados

Resultados en 100 iteraciones con pop_size=30:
- Mejor MSE: 1.23e-08
- Evaluaciones: 3000
- Tiempo: ~45 segundos

## Checklist

- [x] El código sigue las guías de estilo del proyecto
- [x] He revisado mi propio código
- [x] He comentado áreas complejas
- [x] Incluye documentación completa
- [x] No introduce warnings
- [x] He probado que funciona correctamente
```

---

## Reportar Bugs

### Antes de Reportar

1. Revisa los [issues existentes](https://github.com/JRavenelco/levitador-benchmark/issues)
2. Verifica que usas la última versión del código
3. Prueba con un ambiente limpio (virtualenv nuevo)

### Información a Incluir

Usa esta plantilla:

```markdown
## Descripción del Bug

Descripción clara y concisa del problema.

## Pasos para Reproducir

1. Importar módulo X
2. Ejecutar función Y con parámetros Z
3. Observar error

## Comportamiento Esperado

Descripción de lo que debería suceder.

## Comportamiento Actual

Descripción de lo que realmente sucede.

## Código Mínimo Reproducible

```python
from levitador_benchmark import LevitadorBenchmark
problema = LevitadorBenchmark()
# ... código que causa el error
```

## Error/Traceback

```
Traceback completo aquí
```

## Entorno

- OS: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- Python: [e.g., 3.9.7]
- NumPy: [e.g., 1.21.0]
- SciPy: [e.g., 1.7.0]

## Información Adicional

Cualquier otra información relevante.
```

---

## Sugerencias de Mejoras

### Proponer Nuevas Características

Para proponer mejoras o nuevas características:

1. Abre un Issue con label "enhancement"
2. Describe claramente la mejora propuesta
3. Explica el caso de uso
4. Sugiere una posible implementación (opcional)

### Ejemplo

```markdown
## Título: Agregar soporte para optimización multiobjetivo

### Motivación

Muchos problemas reales tienen múltiples objetivos a optimizar simultáneamente.

### Propuesta

Extender `LevitadorBenchmark` para soportar:
1. Múltiples funciones objetivo
2. Frente de Pareto
3. Métricas de evaluación (IGD, hypervolume, etc.)

### Casos de Uso

- Minimizar MSE y tiempo de convergencia simultáneamente
- Balance entre precisión y robustez

### Implementación Sugerida

```python
class MultiObjectiveBenchmark(LevitadorBenchmark):
    def fitness_function(self, individuo):
        mse = super().fitness_function(individuo)
        # Implementar evaluación de robustez aquí
        robustness = self._evaluate_robustness(individuo)
        return [mse, robustness]
    
    def _evaluate_robustness(self, individuo):
        """
        Evalúa la robustez de la solución ante perturbaciones.
        
        Returns:
            float: Métrica de robustez (placeholder)
        """
        # Ejemplo: evaluar con ruido en los parámetros
        perturbations = []
        for _ in range(5):
            perturbed = individuo + np.random.normal(0, 0.01, len(individuo))
            perturbations.append(super().fitness_function(perturbed))
        return np.std(perturbations)  # Variabilidad como medida de robustez
```

### Alternativas Consideradas

- Usar biblioteca existente (pymoo, deap)
- Implementación desde cero
```

---

## Recursos Adicionales

### Algoritmos Metaheurísticos Populares

Si buscas ideas de algoritmos para implementar:

**Basados en Evolución:**
- Genetic Algorithm (GA) ✅ *Ya implementado*
- Differential Evolution (DE) ✅ *Ya implementado*
- Evolution Strategies (ES)
- Covariance Matrix Adaptation (CMA-ES)

**Basados en Enjambres:**
- Particle Swarm Optimization (PSO)
- Ant Colony Optimization (ACO)
- Artificial Bee Colony (ABC) ✅ *Ya implementado*
- Firefly Algorithm (FA)

**Basados en Física:**
- Simulated Annealing (SA)
- Gravitational Search Algorithm (GSA)
- Black Hole Algorithm (BHA)

**Inspirados en Animales:**
- Grey Wolf Optimizer (GWO) ✅ *Ya implementado*
- Whale Optimization Algorithm (WOA)
- Bat Algorithm (BA)
- Cuckoo Search (CS)

**Algoritmos Recientes (2020+):**
- Honey Badger Algorithm (HBA) ✅ *Ya implementado*
- Arithmetic Optimization Algorithm (AOA)
- Aquila Optimizer (AO)
- Reptile Search Algorithm (RSA)

### Referencias

**Libros:**
- Yang, X. S. (2014). *Nature-Inspired Optimization Algorithms*. Elsevier.
- Talbi, E. G. (2009). *Metaheuristics: From Design to Implementation*. Wiley.

**Reviews:**
- Slowik, A., & Kwasnicka, H. (2020). "Evolutionary algorithms and their applications to engineering problems." *Neural Computing and Applications*, 32, 12363-12379.

**Benchmarks Relacionados:**
- CEC Competitions: https://www3.ntu.edu.sg/home/epnsugan/
- BBOB: https://numbbo.github.io/coco/

---

## Preguntas Frecuentes

### ¿Puedo usar bibliotecas externas en mi implementación?

**Preferiblemente no.** Las implementaciones deben usar solo NumPy y SciPy para mantener el proyecto ligero. Si tu algoritmo requiere una biblioteca específica, discútelo primero en un Issue.

### ¿Qué tan optimizado debe estar mi código?

No necesita ser extremadamente optimizado, pero debe ser **razonablemente eficiente**. Evita operaciones O(n³) innecesarias o bucles que puedan vectorizarse.

### ¿Puedo implementar variantes de algoritmos existentes?

¡Sí! Puedes agregar variantes (e.g., "PSO con coeficientes adaptativos") como clases separadas. Asegúrate de documentar claramente las diferencias con la versión original.

### ¿Necesito tests unitarios?

No son obligatorios para la contribución inicial, pero son bienvenidos. El proyecto eventualmente añadirá tests para todos los algoritmos.

### ¿En qué idioma debo documentar?

El proyecto usa **español** para comentarios y documentación. Los nombres de variables/funciones pueden estar en inglés si es convención en el campo (e.g., `fitness`, `crossover`).

---

## Contacto

- **Issues:** https://github.com/JRavenelco/levitador-benchmark/issues
- **Email:** jesus.santana@uaq.mx
- **ORCID:** [0000-0002-6183-7379](https://orcid.org/0000-0002-6183-7379)

---

## Reconocimientos

Gracias a todos los contribuyentes que ayudan a mejorar este benchmark:

<!-- Se actualizará automáticamente -->

---

**¡Gracias por contribuir al Levitador Magnético Benchmark!** 🧲

Tu aporte ayuda a la comunidad de investigación en optimización y metaheurísticas.
