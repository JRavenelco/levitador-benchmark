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
- NumPy
- SciPy
- Pandas (para cargar datos)
- Matplotlib (opcional, para visualización)

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/JRavenelco/levitador-benchmark.git
cd levitador-benchmark

# Instalar dependencias
pip install numpy scipy pandas matplotlib
```

---

## 🏗️ Arquitectura Modular

El repositorio incluye un framework modular completo para ejecutar y comparar múltiples algoritmos de optimización:

```
levitador-benchmark/
├── src/
│   ├── optimization/          # Algoritmos de optimización
│   │   ├── base_optimizer.py  # Clase base abstracta
│   │   ├── random_search.py
│   │   ├── differential_evolution.py
│   │   ├── genetic_algorithm.py
│   │   ├── grey_wolf_optimizer.py
│   │   ├── artificial_bee_colony.py
│   │   ├── honey_badger.py
│   │   ├── shrimp_optimizer.py
│   │   └── tianji_optimizer.py
│   ├── visualization/         # Utilidades de visualización
│   │   ├── convergence_plot.py
│   │   └── comparison_plots.py
│   └── utils/                 # Utilidades generales
│       └── config_loader.py
├── config/                    # Configuraciones YAML
│   ├── default.yaml
│   ├── quick_test.yaml
│   └── full_comparison.yaml
├── scripts/
│   └── run_benchmark.py       # Script principal de benchmark
└── notebooks/
    └── parameter_identification_demo.ipynb
```

### Algoritmos Disponibles

| Algoritmo | Clase | Referencia |
|-----------|-------|------------|
| **Random Search** | `RandomSearch` | Baseline algorithm |
| **Differential Evolution** | `DifferentialEvolution` | Storn & Price (1997) |
| **Genetic Algorithm** | `GeneticAlgorithm` | Holland (1975) |
| **Grey Wolf Optimizer** | `GreyWolfOptimizer` | Mirjalili et al. (2014) |
| **Artificial Bee Colony** | `ArtificialBeeColony` | Karaboga (2005) |
| **Honey Badger Algorithm** | `HoneyBadgerAlgorithm` | Hashim et al. (2022) |
| **Shrimp Optimizer** | `ShrimpOptimizer` | Novel algorithm |
| **Tianji Horse Racing** | `TianjiOptimizer` | Ancient Chinese strategy |

---

## 💻 Uso

### Opción 1: CLI - Script de Benchmark

La forma más rápida de comparar algoritmos es usar el script CLI:

```bash
# Ejecución rápida (pocos algoritmos, pocas iteraciones)
python scripts/run_benchmark.py --config config/quick_test.yaml

# Comparación completa (todos los algoritmos, muchas iteraciones)
python scripts/run_benchmark.py --config config/full_comparison.yaml

# Ejecución personalizada
python scripts/run_benchmark.py --algorithms DE GA GWO --trials 10
```

**Salidas generadas:**
- 📊 Gráficas de convergencia
- 📦 Boxplots de comparación
- ⏱️ Comparación de tiempos de ejecución
- 📄 Estadísticas en JSON
- 💾 Resultados crudos en NPZ

### Opción 2: Python API - Uso Programático

#### Ejemplo Básico

```python
from levitador_benchmark import LevitadorBenchmark

# 1. Crear instancia del problema
problema = LevitadorBenchmark()

# 2. Evaluar una solución candidata
solucion = [0.036, 0.0035, 0.005]  # [k0, k, a]
error = problema.fitness_function(solucion)

print(f"Error MSE: {error:.6e}")
```

#### Usando Algoritmos del Framework

```python
from levitador_benchmark import LevitadorBenchmark
from src.optimization import DifferentialEvolution, GreyWolfOptimizer

# Crear problema
problema = LevitadorBenchmark(random_seed=42)

# Opción 1: Differential Evolution
de = DifferentialEvolution(
    problema, 
    pop_size=30, 
    max_iter=100, 
    F=0.8, 
    CR=0.9,
    random_seed=42
)
best_sol, best_fitness = de.optimize()
print(f"DE - Best fitness: {best_fitness:.6e}")

# Opción 2: Grey Wolf Optimizer
gwo = GreyWolfOptimizer(
    problema,
    pop_size=30,
    max_iter=100,
    random_seed=42
)
best_sol, best_fitness = gwo.optimize()
print(f"GWO - Best fitness: {best_fitness:.6e}")

# Acceder al historial de convergencia
convergence = gwo.get_convergence_curve()
```

#### Comparando Múltiples Algoritmos

```python
from src.optimization import (
    DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony
)
from src.visualization import plot_convergence

# Configurar problema
problema = LevitadorBenchmark(random_seed=42)

# Ejecutar algoritmos
algorithms = {
    'DE': DifferentialEvolution(problema, pop_size=30, max_iter=50, random_seed=42),
    'GA': GeneticAlgorithm(problema, pop_size=30, generations=50, random_seed=42),
    'GWO': GreyWolfOptimizer(problema, pop_size=30, max_iter=50, random_seed=42),
    'ABC': ArtificialBeeColony(problema, pop_size=30, max_iter=50, random_seed=42)
}

results = {}
histories = {}

for name, algo in algorithms.items():
    print(f"Running {name}...")
    best_sol, best_fit = algo.optimize()
    results[name] = best_fit
    histories[name] = algo.get_convergence_curve()
    print(f"  {name}: {best_fit:.6e}")

# Visualizar convergencia
plot_convergence(histories, save_path='comparison.png')
```

#### Usando Configuraciones YAML

```python
from src.utils import load_config
from src.optimization import ALGORITHM_REGISTRY

# Cargar configuración
config = load_config('config/default.yaml')

# Obtener configuración de un algoritmo
de_config = config['algorithms']['DifferentialEvolution']

# Crear optimizador desde configuración
problema = LevitadorBenchmark()
optimizer = DifferentialEvolution(problema, **de_config)
best_sol, best_fit = optimizer.optimize()
```

#### Opción 3: Demo Interactivo - Jupyter Notebook

Para una experiencia interactiva con explicaciones paso a paso:

```bash
jupyter notebook notebooks/parameter_identification_demo.ipynb
```

---

## ⚙️ Configuración

El framework usa archivos YAML para configurar experimentos. Tres configuraciones predefinidas están disponibles:

### `config/quick_test.yaml`
Configuración rápida para pruebas y depuración:
- 2 ensayos por algoritmo
- Poblaciones pequeñas (15 individuos)
- Pocas iteraciones (20)
- Solo algoritmos principales (DE, GA, RandomSearch)

### `config/default.yaml`
Configuración balanceada para uso general:
- 5 ensayos por algoritmo
- Poblaciones medianas (30 individuos)
- Iteraciones moderadas (100)
- Todos los algoritmos habilitados

### `config/full_comparison.yaml`
Configuración completa para investigación:
- 10 ensayos por algoritmo
- Poblaciones grandes (50 individuos)
- Muchas iteraciones (200)
- Todos los algoritmos habilitados

### Estructura de Configuración

```yaml
benchmark:
  data_path: "data/datos_levitador.txt"
  random_seed: 42
  noise_level: 1e-5

optimization:
  n_trials: 5
  save_results: true
  output_dir: "results"

algorithms:
  DifferentialEvolution:
    enabled: true
    pop_size: 30
    max_iter: 100
    F: 0.8
    CR: 0.9
    random_seed: 42
    verbose: false

visualization:
  plot_convergence: true
  plot_boxplot: true
  save_plots: true
  plot_dir: "plots"
  dpi: 300
```

### Crear Configuración Personalizada

```yaml
# my_config.yaml
benchmark:
  data_path: "data/datos_levitador.txt"
  random_seed: 123

algorithms:
  DifferentialEvolution:
    enabled: true
    pop_size: 50
    max_iter: 150
  
  GreyWolfOptimizer:
    enabled: true
    pop_size: 40
    max_iter: 120
```

Ejecutar con configuración personalizada:

```bash
python scripts/run_benchmark.py --config my_config.yaml
```

---

## 🔄 Compatibilidad Hacia Atrás

El archivo original `example_optimization.py` se mantiene funcional para compatibilidad:

```python
from example_optimization import (
    RandomSearch, DifferentialEvolution, GeneticAlgorithm,
    GreyWolfOptimizer, ArtificialBeeColony, HoneyBadgerAlgorithm
)

# Uso idéntico al original
problema = LevitadorBenchmark()
algo = DifferentialEvolution(problema, pop_size=30, max_iter=100)
best_sol, best_fit = algo.optimize()
```

### Con Datos Experimentales Reales

```python
problema = LevitadorBenchmark("data/datos_levitador.txt")
```

### Control de Reproducibilidad

```python
# Usar semilla para resultados reproducibles
problema = LevitadorBenchmark(random_seed=42)

# Configurar nivel de ruido para datos sintéticos
problema = LevitadorBenchmark(noise_level=1e-4, random_seed=42)
```

### Integración con Algoritmos de Optimización

#### Evolución Diferencial (SciPy)

```python
from scipy.optimize import differential_evolution
from levitador_benchmark import LevitadorBenchmark

problema = LevitadorBenchmark()

resultado = differential_evolution(
    problema.fitness_function,
    problema.bounds,
    strategy='best1bin',
    maxiter=100,
    popsize=20,
    disp=True
)

print(f"Mejor solución: {resultado.x}")
print(f"Error final: {resultado.fun:.6e}")
```

#### Algoritmo Genético (DEAP)

```python
from deap import base, creator, tools, algorithms
from levitador_benchmark import LevitadorBenchmark
import numpy as np

problema = LevitadorBenchmark()

# Configuración DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Generador de individuos
def create_individual():
    return [np.random.uniform(lb, ub) for lb, ub in problema.bounds]

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (problema.fitness_function(ind),))
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Ejecutar
pop = toolbox.population(n=50)
result = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=50, verbose=True)
```

#### Enjambre de Partículas (PySwarms)

```python
import pyswarms as ps
from levitador_benchmark import LevitadorBenchmark
import numpy as np

problema = LevitadorBenchmark()
lb, ub = problema.get_bounds_array()

# Función wrapper para PySwarms (espera matriz de partículas)
def fitness_swarm(particles):
    return np.array([problema.fitness_function(p) for p in particles])

# Configurar PSO
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = ps.single.GlobalBestPSO(
    n_particles=30,
    dimensions=3,
    options=options,
    bounds=(lb, ub)
)

# Ejecutar
best_cost, best_pos = optimizer.optimize(fitness_swarm, iters=100)
print(f"Mejor posición: {best_pos}")
print(f"Mejor costo: {best_cost:.6e}")
```

---

## 📊 Visualización de Resultados

```python
from levitador_benchmark import LevitadorBenchmark

problema = LevitadorBenchmark()
mejor_solucion = [0.0363, 0.0035, 0.0052]

# Generar gráfica comparativa
problema.visualize_solution(mejor_solucion, save_path="resultado.png")
```


---

## 📁 Estructura del Repositorio

```
levitador-benchmark/
├── README.md                    # Este archivo
├── LICENSE                      # Licencia MIT
├── requirements.txt             # Dependencias del proyecto
├── levitador_benchmark.py       # Clase principal del benchmark
├── example_optimization.py      # Ejemplos de algoritmos
├── tutorial_metaheuristicas.ipynb  # Notebook tutorial interactivo
├── data/
│   └── datos_levitador.txt      # Datos experimentales reales
├── docs/
│   └── formato_datos.md         # Descripción del formato de datos
├── tests/
│   └── test_benchmark.py        # Tests unitarios (pytest)
└── videos/                      # Videos explicativos
    ├── 01_problema_fisico.mp4
    ├── 02_funcion_fitness.mp4
    └── 03_como_optimizar.mp4
```

---

## 🔬 Detalles Físicos

### Ecuaciones del Sistema

El modelo dinámico se basa en las ecuaciones de **Euler-Lagrange**:

**Ecuación Mecánica (Newton):**
$$m\ddot{y} = \frac{1}{2}\frac{\partial L}{\partial y}i^2 + mg$$

**Ecuación Eléctrica (Kirchhoff):**
$$L(y)\frac{di}{dt} + \frac{\partial L}{\partial y}\dot{y}i + Ri = u$$

### Constantes del Sistema

| Constante | Valor | Descripción |
|-----------|-------|-------------|
| $m$ | 0.018 kg | Masa de la esfera |
| $g$ | 9.81 m/s² | Aceleración gravitacional |
| $R$ | 2.72 Ω | Resistencia de la bobina |

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

## 🔬 Diseño de Experimentos (DOE)

El repositorio incluye un DOE estructurado para generar datos experimentales diversos.

### Experimentos Disponibles

| Fase | Experimentos | Descripción |
|------|--------------|-------------|
| **1** | E01, E02, E07, E08, E11 | Caracterización básica (escalones, senoidales) |
| **2** | E03-E06, E09-E10 | Caracterización extendida (rampas, pulsos) |
| **3** | V01-V06 | Validación (repeticiones) |
| **4** | E12 | Robustez (PRBS) |

### Ejecutar Experimentos

```bash
# Listar experimentos disponibles
python experimentos_doe.py --listar

# Ejecutar en modo simulación (sin hardware)
python experimentos_doe.py --fase 1 --simular

# Ejecutar experimento específico
python experimentos_doe.py --experimento E01

# Ejecutar todos los experimentos
python experimentos_doe.py --todos
```

### Documentación Completa

Ver [docs/DOE_experimentos.md](docs/DOE_experimentos.md) para:
- Definición de factores y niveles
- Protocolo experimental
- Métricas a calcular
- Análisis posterior

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si usas este benchmark en tu investigación:

1. Reporta tus resultados abriendo un Issue
2. Comparte mejoras al código via Pull Request
3. Cita este trabajo en tus publicaciones

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

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

---

## 📧 Contacto

- **Autor:** José de Jesús Santana Ramírez
- **ORCID:** [0000-0002-6183-7379](https://orcid.org/0000-0002-6183-7379)
- **Institución:** Doctorado en Ingeniería, Universidad Autónoma de Querétaro
- **Email:** jesus.santana@uaq.mx

---

## 🧠 KAN-PINN: Observador Neuronal con Física

### Descripción

Además de la optimización de parámetros, este proyecto incluye un **observador de estado basado en KAN-PINN** (Kolmogorov-Arnold Networks + Physics-Informed Neural Networks) para estimar la posición de la esfera sin sensor directo.

### Arquitectura

```
Entradas: [i, L_est, u]
    │
    ▼
┌─────────────────────────────┐
│  KAN Layer 1: 3 → 32        │  B-splines + Residual
├─────────────────────────────┤
│  KAN Layer 2: 32 → 32       │  B-splines + Residual
├─────────────────────────────┤
│  KAN Layer 3: 32 → 1        │  B-splines + Residual
└─────────────────────────────┘
    │
    ▼
Salida: y (posición estimada)
```

### Pérdida Física (PINN)

La red se entrena minimizando:

$$\mathcal{L} = \mathcal{L}_{datos} + \lambda \mathcal{L}_{física}$$

Donde la pérdida física impone la consistencia con el modelo de inductancia:

$$L(y) = k_0 + \frac{k}{1 + y/a}$$

### Resultados del Entrenamiento

| Métrica | Valor |
|---------|-------|
| Correlación | 0.589 |
| MAE | 2.88 mm |
| Datasets | 5 (~13k muestras) |

### Uso del Observador

```python
from pinn.kan_observador import KANObservador
import torch

# Cargar modelo entrenado
model = KANObservador(hidden=32, depth=2, num_knots=8)
checkpoint = torch.load('pinn/kan_observador_*.pt')
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Inferencia
X = torch.tensor([[i, L_est, u]])  # [corriente, inductancia, voltaje]
y_estimado = model(X)
```

### Validación con Metaheurísticos

Los parámetros $[k_0, k, a]$ identificados por metaheurísticos pueden usarse para:
1. **Validar** el modelo físico del KAN-PINN
2. **Comparar** estimación KAN vs fórmula analítica
3. **Mejorar** la pérdida física con parámetros más precisos

---

## 🎬 Videos Explicativos

### 1. El Problema Físico
![Problema Físico](videos/01_problema_fisico.gif)

### 2. Función de Fitness (MSE)
![Función Fitness](videos/02_funcion_fitness.gif)

### 3. Arquitectura KAN-PINN
![Arquitectura KAN](videos/03_arquitectura_kan.gif)

### 4. Algoritmos Metaheurísticos
![Metaheurísticos](videos/04_metaheuristicos.gif)

*Animaciones generadas con Manim*

---

