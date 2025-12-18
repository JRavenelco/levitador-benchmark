# Guía de Evaluación en Lote (Batch Evaluation)

## 📋 Resumen

Esta guía documenta la nueva funcionalidad de evaluación en lote implementada en el Levitador Benchmark, que permite optimizar significativamente el rendimiento de algoritmos de optimización poblacionales.

## 🚀 Características

### Métodos Disponibles

#### 1. `fitness_function(solucion)` - Evaluación Individual
```python
solucion = [0.036, 0.0035, 0.005]
error = problema.fitness_function(solucion)
```
- **Uso**: Evaluación de una solución individual
- **Speedup**: Baseline (1.0x)
- **Recomendado para**: Evaluaciones puntuales, debugging

#### 2. `evaluate_batch_vectorized(poblacion)` - Evaluación Vectorizada
```python
import numpy as np
poblacion = np.random.uniform(lb, ub, (50, 3))
fitness = problema.evaluate_batch_vectorized(poblacion)
```
- **Uso**: Poblaciones pequeñas y medianas (<100 individuos)
- **Speedup**: 1.03-1.04x (3-4% más rápido)
- **Ventajas**:
  - Validación vectorizada de restricciones
  - Menor overhead que parallelización
  - Más eficiente para poblaciones pequeñas

#### 3. `evaluate_batch(poblacion, n_jobs=-1)` - Evaluación Paralela
```python
# Usar todos los CPUs disponibles
fitness = problema.evaluate_batch(poblacion, n_jobs=-1)

# Usar número específico de CPUs
fitness = problema.evaluate_batch(poblacion, n_jobs=4)

# Evaluación secuencial (equivalente a evaluate_batch_vectorized)
fitness = problema.evaluate_batch(poblacion, n_jobs=1)
```
- **Uso**: Poblaciones grandes (>100 individuos)
- **Speedup**: 1.5-2.1x (hasta 2x más rápido)
- **Ventajas**:
  - Procesamiento paralelo real con multiprocessing
  - Escala con el número de CPUs
  - Ideal para poblaciones grandes

## 📊 Resultados de Rendimiento

### Benchmark Completo
```
Población | Individual | Vectorized | Parallel | Speedup Parallel
----------|-----------|------------|----------|------------------
10        | 0.0837s   | 0.0816s    | 0.0558s  | 1.50x
30        | 0.2469s   | 0.2400s    | 0.1323s  | 1.87x
50        | 0.4191s   | 0.4022s    | 0.2038s  | 2.06x
100       | 0.8121s   | 0.7872s    | 0.3829s  | 2.12x
```

### Recomendaciones

| Tamaño de Población | Método Recomendado | Razón |
|---------------------|-------------------|-------|
| < 30 | `evaluate_batch_vectorized()` | Menor overhead |
| 30-100 | `evaluate_batch_vectorized()` | Balance óptimo |
| > 100 | `evaluate_batch(n_jobs=-1)` | Máximo speedup |

## 💡 Ejemplos de Uso

### Ejemplo 1: Algoritmo Personalizado
```python
from levitador_benchmark import LevitadorBenchmark
import numpy as np

problema = LevitadorBenchmark(random_seed=42)
lb, ub = problema.get_bounds_array()

# Algoritmo simple con batch evaluation
pop_size = 50
mejor_error = float('inf')

for iteracion in range(100):
    poblacion = np.random.uniform(lb, ub, (pop_size, 3))
    
    # Evaluar en lote (mucho más rápido)
    fitness = problema.evaluate_batch_vectorized(poblacion)
    
    idx_mejor = np.argmin(fitness)
    if fitness[idx_mejor] < mejor_error:
        mejor_error = fitness[idx_mejor]
        mejor_solucion = poblacion[idx_mejor]
```

### Ejemplo 2: Integración con Algoritmos Existentes
```python
from example_optimization import DifferentialEvolution

# Los algoritmos ya están optimizados para usar batch evaluation
de = DifferentialEvolution(
    problema, 
    pop_size=50, 
    max_iter=100,
    random_seed=42
)

mejor_sol, mejor_error = de.optimize()
```

### Ejemplo 3: PySwarms con Batch Evaluation
```python
import pyswarms as ps

def fitness_swarm(particles):
    # Usar batch evaluation en lugar de loop
    return problema.evaluate_batch_vectorized(particles)

optimizer = ps.single.GlobalBestPSO(
    n_particles=30,
    dimensions=3,
    options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
    bounds=(lb, ub)
)

best_cost, best_pos = optimizer.optimize(fitness_swarm, iters=100)
```

## 🔧 Implementación Técnica

### Optimizaciones Implementadas

1. **Validación Vectorizada**
   - Pre-validación de restricciones usando operaciones NumPy
   - Evita evaluar individuos inválidos
   - Reduce tiempo de ejecución en ~3-4%

2. **Procesamiento Paralelo**
   - Usa `multiprocessing.Pool` para evaluación paralela
   - Distribución automática de trabajo entre CPUs
   - Speedup casi lineal con número de CPUs

3. **Pre-asignación de Memoria**
   - Arrays de resultados pre-asignados
   - Evita realocaciones dinámicas
   - Mejora localidad de caché

### Constantes
```python
from levitador_benchmark import PENALTY_VALUE

# PENALTY_VALUE = 1e9
# Valor de penalización para soluciones inválidas
```

## 🧪 Testing

Todos los métodos han sido exhaustivamente probados:

```bash
# Ejecutar tests de batch evaluation
pytest tests/test_batch_evaluation.py -v

# Ejecutar benchmark de rendimiento
python benchmark_batch_performance.py

# Ejecutar ejemplos
python example_batch_usage.py
```

### Cobertura de Tests
- ✅ Evaluación individual vs batch (consistencia)
- ✅ Evaluación secuencial vs paralela (consistencia)
- ✅ Manejo de soluciones inválidas
- ✅ Poblaciones vacías
- ✅ Poblaciones grandes
- ✅ Preservación de orden

## ⚠️ Consideraciones

### Cuando usar Evaluación Paralela
- ✅ Poblaciones grandes (>100 individuos)
- ✅ Sistema con múltiples CPUs
- ✅ Evaluaciones costosas (muchos puntos temporales)

### Cuando NO usar Evaluación Paralela
- ❌ Poblaciones pequeñas (<30 individuos)
- ❌ Sistema con CPU único
- ❌ Overhead de multiprocessing > beneficio

### Reproducibilidad
Los tres métodos garantizan resultados idénticos:
```python
# Todos estos dan el mismo resultado
r1 = [problema.fitness_function(ind.tolist()) for ind in pop]
r2 = problema.evaluate_batch_vectorized(pop)
r3 = problema.evaluate_batch(pop, n_jobs=-1)

assert np.allclose(r1, r2)
assert np.allclose(r2, r3)
```

## 📚 Referencias

- Código fuente: `levitador_benchmark.py`
- Tests: `tests/test_batch_evaluation.py`
- Ejemplos: `example_batch_usage.py`
- Benchmarks: `benchmark_batch_performance.py`

## 🤝 Contribuir

Si encuentras formas de optimizar aún más la evaluación en lote, ¡las contribuciones son bienvenidas!

Áreas de mejora potencial:
- GPU acceleration con CuPy/JAX
- Batch ODE solving para mayor eficiencia
- Caché inteligente de simulaciones similares
- Evaluación distribuida para clusters

---

*Última actualización: 2024*
