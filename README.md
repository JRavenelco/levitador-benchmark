# 🧲 Levitador Magnético Benchmark

**Problema de optimización real para algoritmos bio-inspirados y metaheurísticas.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

---

## 📋 Descripción

Este benchmark proporciona un **problema de optimización del mundo real** basado en un sistema de levitación magnética. El objetivo es identificar los parámetros físicos de un electroimán que minimizan el error entre un modelo dinámico (gemelo digital) y datos experimentales reales.

A diferencia de funciones de prueba sintéticas (Rosenbrock, Rastrigin, etc.), este problema:

- ✅ Proviene de un **sistema físico real**
- ✅ Tiene **restricciones físicas naturales**
- ✅ Incluye **datos experimentales** para validación
- ✅ Es **multimodal** y presenta retos de convergencia

### 🆕 Nuevo: Framework Modular de Optimización

El repositorio ahora incluye un **framework modular** con 8 algoritmos bio-inspirados implementados:
- Random Search (baseline)
- Differential Evolution
- Genetic Algorithm
- Grey Wolf Optimizer
- Artificial Bee Colony
- Honey Badger Algorithm
- Shrimp Optimization Algorithm
- Tianji Horse Racing Strategy

---

## 🎯 El Problema de Optimización

### Modelo Físico

El sistema consiste en una esfera de acero suspendida por un electroimán. La inductancia del electroimán varía con la distancia según:

$$L(y) = k_0 + \frac{k}{1 + y/a}$$

Donde:
| Parámetro | Descripción | Unidad |
|-----------|-------------|--------|
| $k_0$ | Inductancia base | H |
| $k$ | Coeficiente de inductancia | H |
| $a$ | Parámetro geométrico | m |
| $y$ | Posición de la esfera | m |

### Objetivo

Encontrar $[k_0, k, a]$ que minimicen el **Error Cuadrático Medio (MSE)** entre:
- La trayectoria simulada por el modelo
- Los datos experimentales reales

### Espacio de Búsqueda

| Variable | Límite Inferior | Límite Superior |
|----------|-----------------|-----------------|
| $k_0$ | 0.0001 | 0.1 |
| $k$ | 0.0001 | 0.1 |
| $a$ | 0.0001 | 0.05 |

---

## 🚀 Instalación

### Requisitos
- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib (para visualización)
- PyYAML (para configuraciones)

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/JRavenelco/levitador-benchmark.git
cd levitador-benchmark

# Instalar dependencias
pip install -r requirements.txt
```

---

## 💻 Uso

### Opción 1: Script de Benchmark (Recomendado)

El script de benchmark permite comparar múltiples algoritmos fácilmente:

```bash
# Ejecutar benchmark completo con configuración por defecto
python scripts/run_benchmark.py --config config/default.yaml

# Test rápido con pocos trials
python scripts/run_benchmark.py --config config/quick_test.yaml

# Ejecutar solo un algoritmo específico
python scripts/run_benchmark.py --config config/default.yaml --optimizer GreyWolfOptimizer

# Comparación completa (30 trials por algoritmo)
python scripts/run_benchmark.py --config config/full_comparison.yaml
```

**Salida del benchmark:**
- Resultados en `results/` (JSON con métricas)
- Gráficas de convergencia
- Box plots comparativos
- Métricas de rendimiento
- Comparación de tiempos de ejecución

### Opción 2: Uso Programático (Python)

#### Ejemplo Básico

```python
from levitador_benchmark import LevitadorBenchmark
from src.optimization import GreyWolfOptimizer

# Crear instancia del problema
problema = LevitadorBenchmark()

# Crear y ejecutar optimizador
optimizer = GreyWolfOptimizer(problema, pop_size=30, max_iter=100, random_seed=42)
best_solution, best_fitness = optimizer.optimize()

print(f"Mejor solución: k0={best_solution[0]:.6f}, k={best_solution[1]:.6f}, a={best_solution[2]:.6f}")
print(f"Error MSE: {best_fitness:.6e}")
```

#### Comparar Múltiples Algoritmos

```python
from levitador_benchmark import LevitadorBenchmark
from src.optimization import (
    DifferentialEvolution, GreyWolfOptimizer, 
    ArtificialBeeColony, HoneyBadgerAlgorithm
)

problema = LevitadorBenchmark(random_seed=42)

algorithms = {
    'DE': DifferentialEvolution(problema, pop_size=30, max_iter=50, random_seed=42),
    'GWO': GreyWolfOptimizer(problema, pop_size=30, max_iter=50, random_seed=42),
    'ABC': ArtificialBeeColony(problema, pop_size=30, max_iter=50, random_seed=42),
    'HBA': HoneyBadgerAlgorithm(problema, pop_size=30, max_iter=50, random_seed=42),
}

results = {}
for name, algo in algorithms.items():
    print(f"\nRunning {name}...")
    best_sol, best_fit = algo.optimize()
    results[name] = best_fit
    print(f"  Fitness: {best_fit:.6e}")

# Mostrar ranking
for name in sorted(results, key=results.get):
    print(f"{name}: {results[name]:.6e}")
```

### Opción 3: Jupyter Notebook (Interactivo)

Abre el notebook de demostración:

```bash
jupyter notebook notebooks/parameter_identification_demo.ipynb
```

El notebook incluye:
- Ejemplos de uso de cada algoritmo
- Visualización de convergencia
- Comparación estadística
- Análisis de resultados

---

## 📁 Estructura del Repositorio

```
levitador-benchmark/
├── README.md                           # Este archivo
├── LICENSE                             # Licencia MIT
├── requirements.txt                    # Dependencias
├── levitador_benchmark.py              # Clase principal del benchmark
├── example_optimization.py             # Ejemplos legacy (compatibilidad)
│
├── src/                                # Código fuente modular
│   ├── optimization/                   # Algoritmos de optimización
│   │   ├── base_optimizer.py          # Clase base abstracta
│   │   ├── random_search.py           # Random Search
│   │   ├── differential_evolution.py  # Differential Evolution
│   │   ├── genetic_algorithm.py       # Genetic Algorithm
│   │   ├── grey_wolf.py               # Grey Wolf Optimizer
│   │   ├── artificial_bee_colony.py   # Artificial Bee Colony
│   │   ├── honey_badger.py            # Honey Badger Algorithm
│   │   ├── shrimp.py                  # Shrimp Optimizer
│   │   └── tianji.py                  # Tianji Horse Racing
│   │
│   ├── visualization/                  # Utilidades de visualización
│   │   └── plots.py                   # Funciones de gráficas
│   │
│   ├── utils/                         # Utilidades generales
│   │   └── config_loader.py           # Cargador de configuraciones YAML
│   │
│   ├── data/                          # Módulo de datos
│   └── models/                        # Módulo de modelos
│
├── config/                            # Configuraciones YAML
│   ├── default.yaml                   # Configuración por defecto
│   ├── quick_test.yaml               # Test rápido
│   └── full_comparison.yaml          # Comparación completa
│
├── scripts/                           # Scripts ejecutables
│   └── run_benchmark.py              # Script principal de benchmark
│
├── notebooks/                         # Jupyter notebooks
│   └── parameter_identification_demo.ipynb
│
├── data/                              # Datos experimentales
│   └── datos_levitador.txt           # Datos del levitador real
│
├── tests/                             # Tests unitarios
│   └── test_benchmark.py             # Tests del benchmark
│
└── docs/                              # Documentación adicional
    ├── DOE_experimentos.md
    └── formato_datos.md
```

---

## 🔧 Configuración (YAML)

Los algoritmos se configuran mediante archivos YAML. Ejemplo:

```yaml
# config/default.yaml
benchmark:
  data_path: "data/datos_levitador.txt"
  random_seed: 42
  verbose: true

optimizers:
  GreyWolfOptimizer:
    pop_size: 30
    max_iter: 100
    random_seed: 42
    verbose: true
  
  DifferentialEvolution:
    pop_size: 30
    max_iter: 100
    F: 0.8
    CR: 0.9
    random_seed: 42
    verbose: true

benchmark_settings:
  n_trials: 10
  save_history: true
  output_dir: "results"
```

---

## 📊 Algoritmos Implementados

### 1. Random Search
Búsqueda aleatoria (baseline).
- **Clase:** `RandomSearch`
- **Parámetros:** `n_iterations`

### 2. Differential Evolution (DE)
Evolución Diferencial clásica (DE/rand/1/bin).
- **Clase:** `DifferentialEvolution`
- **Parámetros:** `pop_size`, `max_iter`, `F`, `CR`
- **Referencia:** Storn & Price (1997)

### 3. Genetic Algorithm (GA)
Algoritmo genético con selección por torneo y BLX-alpha.
- **Clase:** `GeneticAlgorithm`
- **Parámetros:** `pop_size`, `generations`, `crossover_prob`, `mutation_prob`

### 4. Grey Wolf Optimizer (GWO)
Inspirado en la jerarquía y caza de lobos grises.
- **Clase:** `GreyWolfOptimizer`
- **Parámetros:** `pop_size`, `max_iter`
- **Referencia:** Mirjalili et al. (2014)

### 5. Artificial Bee Colony (ABC)
Basado en el comportamiento de abejas melíferas.
- **Clase:** `ArtificialBeeColony`
- **Parámetros:** `pop_size`, `max_iter`, `limit`
- **Referencia:** Karaboga (2005)

### 6. Honey Badger Algorithm (HBA)
Inspirado en el comportamiento del tejón de miel.
- **Clase:** `HoneyBadgerAlgorithm`
- **Parámetros:** `pop_size`, `max_iter`, `beta`
- **Referencia:** Hashim et al. (2022)

### 7. Shrimp Optimization Algorithm (SOA)
Basado en el comportamiento del camarón mantis.
- **Clase:** `ShrimpOptimizer`
- **Parámetros:** `pop_size`, `max_iter`

### 8. Tianji Horse Racing Strategy
Estrategia china antigua aplicada a optimización.
- **Clase:** `TianjiOptimizer`
- **Parámetros:** `pop_size`, `max_iter`

---

## 📈 Resultados de Referencia

Valores de referencia obtenidos experimentalmente:

| Parámetro | Valor Estimado |
|-----------|----------------|
| $k_0$ | 0.0363 H |
| $k$ | 0.0035 H |
| $a$ | 0.0052 m |

Los algoritmos bien sintonizados deberían converger a soluciones cercanas con MSE < 1e-8.

---

## 🧪 Tests

Ejecutar tests unitarios:

```bash
# Instalar pytest si no está instalado
pip install pytest

# Ejecutar tests
pytest tests/test_benchmark.py -v
```

---

## 📚 Citar este Trabajo

```bibtex
@software{levitador_benchmark,
  author = {Santana-Ramírez, José de Jesús},
  title = {Levitador Magnético Benchmark: Problema de Optimización Real para Metaheurísticas},
  year = {2024},
  url = {https://github.com/JRavenelco/levitador-benchmark},
  note = {Universidad Autónoma de Querétaro},
  orcid = {0000-0002-6183-7379}
}
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -m 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

### Agregar un Nuevo Algoritmo

Para agregar un nuevo optimizador:

1. Crea un archivo en `src/optimization/mi_algoritmo.py`
2. Hereda de `BaseOptimizer`
3. Implementa el método `optimize()`
4. Agrega el algoritmo a `src/optimization/__init__.py`
5. Agrega configuración en `config/default.yaml`
6. Actualiza la documentación

Ejemplo:

```python
from .base_optimizer import BaseOptimizer
import numpy as np

class MiAlgoritmo(BaseOptimizer):
    def __init__(self, problema, param1=10, **kwargs):
        super().__init__(problema, **kwargs)
        self.param1 = param1
    
    def optimize(self):
        # Tu implementación aquí
        best_solution = ...
        best_fitness = ...
        return best_solution, best_fitness
```

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

- **Autor:** José de Jesús Santana Ramírez
- **ORCID:** [0000-0002-6183-7379](https://orcid.org/0000-0002-6183-7379)
- **Institución:** Doctorado en Ingeniería, Universidad Autónoma de Querétaro
- **Email:** jesus.santana@uaq.mx

---

## 🎓 Reconocimientos

Este trabajo es parte de la investigación doctoral en la Universidad Autónoma de Querétaro sobre control y optimización de sistemas no lineales.
