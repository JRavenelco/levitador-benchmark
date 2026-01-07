# 🧲 Levitador Magnético Benchmark

**Problema de optimización real para algoritmos bio-inspirados y metaheurísticas con pipeline de dos fases para identificación de parámetros y observación KAN-PINN.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)]()

---

## 📋 Descripción

Este benchmark proporciona un **problema de optimización del mundo real** basado en un sistema de levitación magnética. El repositorio incluye un pipeline completo de dos fases:

1. **Fase 1: Identificación de Parámetros Físicos** - Optimización con metaheurísticas para identificar parámetros del sistema (inductancia y resistencia)
2. **Fase 2: Entrenamiento KAN-PINN** - Red neuronal informada por física (Physics-Informed) para observación sensorless de posición

A diferencia de funciones de prueba sintéticas (Rosenbrock, Rastrigin, etc.), este problema:

- ✅ Proviene de un **sistema físico real**
- ✅ Tiene **restricciones físicas naturales**
- ✅ Incluye **datos experimentales** para validación
- ✅ Es **multimodal** y presenta retos de convergencia
- ✅ Integra **estimación de resistencia sin sensor de temperatura**
- ✅ Permite **entrenamiento de observadores neuronales**

---

## 🎯 Pipeline de Dos Fases

### Fase 1: Identificación de Parámetros Físicos

**Objetivo:** Identificar los parámetros físicos del sistema usando metaheurísticas.

**Parámetros a optimizar:**
- `K0`: Numerador de inductancia [H]
- `A`: Parámetro geométrico [m]  
- `R0`: Resistencia base [Ω]
- `α`: Coeficiente de temperatura [1/°C]

**Modelo Físico:**

Inductancia (función no lineal de la posición):
```
L(y) = K0 / (1 + y/A)
```

Resistencia (estimada sin sensor de temperatura):
```
R(t) ≈ R0 * (1 + α*ΔT(t))
```

Donde ΔT(t) se aproxima mediante calentamiento Joule: ΔT ∝ ∫ i²(t) dt

**Ecuaciones del sistema:**
- Mecánica: `m·ÿ = (1/2)·(∂L/∂y)·i² + m·g`
- Eléctrica: `L(y)·(di/dt) + (∂L/∂y)·ẏ·i + R(t)·i = u`

**Estimación de R(t) vía Ley de Kirchhoff:**

Sin sensor de temperatura, la resistencia se estima usando:
```
R_est(t) = (u(t) - dφ̂(t)/dt) / i(t)
```

donde `φ̂(t) = L(y(t)) · i(t)` es el flujo magnético estimado.

**Función de Fitness:**

La función objetivo minimiza el error cuadrático medio (MSE) entre las trayectorias simuladas y reales:

```python
MSE_total = 0.8 * MSE_posición + 0.2 * MSE_corriente
```

Características del fitness:
- **Simulación dinámica**: Integra las ecuaciones diferenciales del sistema con los parámetros candidatos
- **Ponderación balanceada**: Prioriza el ajuste de posición (80%) sobre corriente (20%)
- **Suavizado adaptativo**: Aplica filtros Savitzky-Golay para reducir ruido experimental
- **Submuestreo configurable**: Acelera la optimización 10-50x sin pérdida significativa de precisión
- **Detección de fallos**: Retorna penalización alta (1e10) si la simulación diverge o viola restricciones físicas

**Métodos de Optimización (Metaheurísticas):**

Los algoritmos metaheurísticos exploran el espacio de parámetros de forma inteligente:

1. **Evolución Diferencial (DE)**: Usa vectores diferencia entre miembros de la población para generar nuevos candidatos. Excelente balance exploración-explotación.

2. **Grey Wolf Optimizer (GWO)**: Simula la jerarquía de caza de lobos grises con líderes alfa, beta y delta guiando la búsqueda.

3. **Artificial Bee Colony (ABC)**: Inspirado en el comportamiento de abejas melíferas. Divide la población en exploradoras, trabajadoras y observadoras.

4. **Algoritmo Honey Badger (HBA)**: Modela el comportamiento de búsqueda del tejón de miel, alternando entre excavación intensa y exploración.

5. **Shrimp Optimizer (SOA)**: Basado en el comportamiento adaptativo de camarones en diferentes condiciones ambientales.

6. **Tianji Optimizer**: Inspirado en la estrategia china antigua de carreras de caballos, enfocado en aprovechar ventajas locales.

7. **Algoritmo Genético (GA)**: Evolución artificial con selección por torneo, cruce BLX-α y mutación gaussiana.

8. **Random Search**: Búsqueda aleatoria como baseline de comparación.

**Características Avanzadas:**

- **Evaluación en paralelo**: Usa múltiples núcleos CPU para evaluar poblaciones
- **Múltiples trials**: Ejecuta cada algoritmo varias veces para análisis estadístico robusto
- **Diagnóstico de residuales**: Analiza la calidad del ajuste mediante residuales de posición y corriente
- **Comparación automática**: Genera reportes comparativos con valores teóricos de referencia

### Fase 2: Entrenamiento KAN-PINN (Observador Sensorless)

**Objetivo:** Entrenar una red neuronal KAN (Kolmogorov-Arnold Network) informada por física para estimar la posición sin sensor directo.

**Arquitectura de dos etapas:**

1. **Etapa 1 - Observador de Flujo:**
   - Entrada: (u, i)
   - Salida: φ̂ (flujo estimado)
   - Pérdida: MSE + Kirchhoff (u = R·i + dφ/dt)

2. **Etapa 2 - Predictor de Posición:**
   - Entrada: (u, i, φ̂)
   - Salida: ŷ (posición estimada)
   - Pérdida: MSE + PINN (φ̂ = L*(ŷ)·i) usando K0*, A* de Fase 1

**Características clave:**
- Usa capa HiPPO-LegS para captura temporal
- KAN con B-splines y conexiones residuales
- Curriculum learning para peso PINN
- Sin data leakage entre etapas

---

## 🚀 Instalación

### Requisitos
- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib (visualización)
- PyYAML (configuración)
- PyTorch >= 1.12 (opcional, solo para KAN-PINN)

### Instalación rápida

```bash
# Clonar el repositorio
git clone https://github.com/JRavenelco/levitador-benchmark.git
cd levitador-benchmark

# Instalar dependencias básicas
pip install numpy scipy pandas matplotlib pyyaml

# Para KAN-PINN (Fase 2), instalar PyTorch:
pip install torch
```

### ⚡ Quick Start - Benchmark Completo

Para ejecutar rápidamente un benchmark completo de todos los algoritmos:

```bash
# Ejecutar benchmark completo (todos los algoritmos, 5 trials)
python scripts/run_full_optimization.py

# Ver resultados generados
ls -l results/optimization_comparison/
cat results/optimization_comparison/BENCHMARK_REPORT.md
```

Este script compara automáticamente 8 algoritmos metaheurísticos y genera reportes detallados con visualizaciones. Ver [sección completa](#-benchmark-completo-de-optimización) para más detalles.

---

## 🏗️ Arquitectura Modular

El repositorio incluye un framework modular completo para el pipeline de dos fases:

```
levitador-benchmark/
├── src/
│   ├── benchmarks/             # Benchmarks de optimización
│   │   ├── parameter_benchmark.py   # Fase 1: Identificación de parámetros
│   │   └── kanpinn_benchmark.py     # Fase 2: Hyperparams KAN-PINN
│   ├── kan_pinn/               # Módulo KAN-PINN (requiere PyTorch)
│   │   ├── hippo_layer.py      # Capa HiPPO-LegS
│   │   ├── kan_layer.py        # Capa KAN con B-splines
│   │   ├── flux_observer.py    # Etapa 1: Observador de flujo
│   │   ├── position_predictor.py  # Etapa 2: Predictor de posición
│   │   ├── physics_loss.py     # Pérdidas físicas
│   │   └── trainer.py          # Entrenador con curriculum learning
│   ├── optimization/           # Algoritmos de optimización
│   │   ├── base_optimizer.py   # Clase base abstracta
│   │   ├── random_search.py
│   │   ├── differential_evolution.py
│   │   ├── genetic_algorithm.py
│   │   ├── grey_wolf_optimizer.py
│   │   ├── artificial_bee_colony.py
│   │   ├── honey_badger.py
│   │   ├── shrimp_optimizer.py
│   │   └── tianji_optimizer.py
│   ├── visualization/          # Utilidades de visualización
│   │   ├── convergence_plot.py
│   │   └── comparison_plots.py
│   └── utils/                  # Utilidades generales
│       └── config_loader.py
├── config/                     # Configuraciones YAML
│   ├── pipeline_config.yaml    # Pipeline completo (Fase 1 + 2)
│   ├── kanpinn_default.yaml    # Config KAN-PINN
│   ├── default.yaml            # Config optimización estándar
│   ├── quick_test.yaml
│   └── full_comparison.yaml
├── scripts/
│   ├── optimize_parameters.py  # Script Fase 1
│   ├── train_kanpinn.py        # Script Fase 2
│   ├── pipeline_identificacion_kanpinn.py  # Orquestador completo
│   └── run_benchmark.py        # Benchmark original
├── data/
│   ├── datos_levitador.txt     # Datos experimentales
│   └── sesiones_kan_pinn/      # Datasets para KAN-PINN
└── notebooks/
    └── KAN_SENSORLESS_REAL.ipynb  # Demo KAN-PINN
```

### Algoritmos Disponibles

Los siguientes algoritmos metaheurísticos están implementados y optimizados para la identificación de parámetros:

| Algoritmo | Clase | Referencia | Características Clave |
|-----------|-------|------------|----------------------|
| **Differential Evolution** | `DifferentialEvolution` | Storn & Price (1997) | Mutación basada en diferencias vectoriales, excelente para espacios continuos |
| **Grey Wolf Optimizer** | `GreyWolfOptimizer` | Mirjalili et al. (2014) | Jerarquía de liderazgo, balance exploración-explotación |
| **Artificial Bee Colony** | `ArtificialBeeColony` | Karaboga (2005) | Múltiples roles (exploradoras, trabajadoras, observadoras) |
| **Honey Badger Algorithm** | `HoneyBadgerAlgorithm` | Hashim et al. (2022) | Búsqueda adaptativa con alternancia excavación/exploración |
| **Shrimp Optimizer** | `ShrimpOptimizer` | Novel algorithm | Comportamiento adaptativo multi-fase |
| **Tianji Horse Racing** | `TianjiOptimizer` | Ancient Chinese strategy | Estrategia de aprovechamiento de ventajas locales |
| **Genetic Algorithm** | `GeneticAlgorithm` | Holland (1975) | Selección por torneo, cruce BLX-α, mutación gaussiana |
| **Random Search** | `RandomSearch` | Baseline algorithm | Búsqueda aleatoria uniforme (referencia de comparación) |

**Notas de Implementación:**

Todos los algoritmos comparten la interfaz común `BaseOptimizer` que proporciona:
- Gestión automática de límites (bounds enforcement)
- Registro de historial de convergencia
- Contador de evaluaciones de fitness
- Soporte para semillas aleatorias (reproducibilidad)
- Modo verbose para depuración

**Recomendaciones de Uso:**

- **Para convergencia rápida**: Differential Evolution (DE) o Grey Wolf Optimizer (GWO)
- **Para robustez**: Artificial Bee Colony (ABC) o Honey Badger (HBA)  
- **Para exploración exhaustiva**: Usar múltiples algoritmos y comparar resultados
- **Para problemas de alta dimensionalidad**: DE o SOA
- **Para baseline/comparación**: Random Search


---

## 💻 Uso del Pipeline

### Pipeline Completo: Fase 1 + Fase 2

```bash
# Ejecutar pipeline completo (identificación + entrenamiento)
python scripts/pipeline_identificacion_kanpinn.py --config config/pipeline_config.yaml

# Solo Fase 1 (identificación de parámetros)
python scripts/pipeline_identificacion_kanpinn.py --phase1-only

# Solo Fase 2 (entrenamiento KAN-PINN con parámetros existentes)
python scripts/pipeline_identificacion_kanpinn.py --phase2-only \
    --use-params results/parameter_identification/parametros_optimos.json
```

### Fase 1: Identificación de Parámetros

```bash
# Ejecución con configuración completa
python scripts/optimize_parameters.py --config config/pipeline_config.yaml

# Ejecución rápida con algoritmos específicos
python scripts/optimize_parameters.py --algorithms DE GWO ABC --trials 10

# Ejecución personalizada
python scripts/optimize_parameters.py \
    --data data/datos_levitador.txt \
    --algorithms DE GWO HBA SOA Tianji GA RandomSearch \
    --trials 5 \
    --output results/my_optimization
```

**Salidas generadas:**
- 📄 `parametros_optimos.json` - Parámetros óptimos [K0, A, R0, α]
- 📄 `optimization_results.json` - Estadísticas de todos los algoritmos
- 📊 `convergence_*.png` - Curvas de convergencia por algoritmo
- 📊 `comparison_boxplot.png` - Comparación de rendimiento
- 📊 `best_solution.png` - Visualización de la mejor solución

### Fase 2: Entrenamiento KAN-PINN

```bash
# Entrenar con configuración por defecto
python scripts/train_kanpinn.py --config config/kanpinn_default.yaml

# Usar parámetros de Fase 1
python scripts/train_kanpinn.py \
    --config config/kanpinn_default.yaml \
    --use-params results/parameter_identification/parametros_optimos.json

# Entrenar solo una etapa
python scripts/train_kanpinn.py --stage 1  # Solo observador de flujo
python scripts/train_kanpinn.py --stage 2  # Solo predictor de posición
```

**Nota:** Fase 2 requiere PyTorch. La implementación completa está basada en el notebook `KAN_SENSORLESS_REAL.ipynb`.

### Python API - Fase 1

```python
from src.benchmarks import ParameterBenchmark
from src.optimization import DifferentialEvolution, GreyWolfOptimizer

# Crear problema de identificación de parámetros
problema = ParameterBenchmark(
    data_path='data/datos_levitador.txt',
    subsample_factor=20,  # Submuestreo para velocidad
    verbose=True
)

print(f"Optimizing {problema.dim} parameters: {problema.variable_names}")
print(f"Bounds: {problema.bounds}")

# Usar Differential Evolution
de = DifferentialEvolution(
    problema,
    pop_size=30,
    max_iter=100,
    F=0.8,
    CR=0.9,
    random_seed=42
)

best_sol, best_fitness = de.optimize()
print(f"Best parameters: K0={best_sol[0]:.6f}, A={best_sol[1]:.6f}, "
      f"R0={best_sol[2]:.4f}, α={best_sol[3]:.6f}")
print(f"Best fitness: {best_fitness:.6e}")

# Visualizar solución
problema.visualize_solution(best_sol, save_path='results/solution.png')

# Estimar curva de resistencia
R_curve = problema.estimate_resistance_curve(best_sol[0], best_sol[1])
print(f"R(t) range: [{R_curve.min():.3f}, {R_curve.max():.3f}] Ω")
```

### Entendiendo el Proceso de Optimización Metaheurística

**¿Cómo funcionan los metaheurísticos en este problema?**

1. **Inicialización**: Cada algoritmo crea una población de soluciones candidatas `θ = [K0, A, R0, α]` dentro de los límites físicos definidos.

2. **Evaluación**: Para cada candidato:
   - Se simula el sistema dinámico completo usando las EDOs
   - Se compara la trayectoria simulada con los datos experimentales
   - Se calcula el MSE como medida de calidad (fitness)

3. **Evolución/Búsqueda**: Los algoritmos usan diferentes estrategias bio-inspiradas:
   - **DE**: Combina vectores diferencia para generar mutantes
   - **GWO**: Sigue a los mejores "lobos" (soluciones) del grupo
   - **ABC**: Abejas exploradoras buscan nuevas fuentes de alimento (soluciones)
   - **Otros**: Cada algoritmo implementa su propia metáfora de búsqueda

4. **Convergencia**: El proceso se repite hasta:
   - Alcanzar un MSE suficientemente bajo (< 1e-7)
   - Completar el número máximo de iteraciones
   - Detectar estancamiento (sin mejora significativa)

**Ejemplo de Salida de Convergencia:**

```
Iteration 10/100: Best fitness = 3.45e-06
Iteration 20/100: Best fitness = 1.23e-06  
Iteration 30/100: Best fitness = 4.56e-07
Iteration 40/100: Best fitness = 2.31e-07
Iteration 50/100: Best fitness = 8.92e-08  ✓ Target reached!

Final parameters:
  K0 = 0.036234 H   (theoretical: 0.0363)
  A  = 0.005123 m   (theoretical: 0.0052)
  R0 = 2.718 Ω      (estimated)
  α  = 0.00387 /°C  (estimated)
```

**Ventajas de Usar Múltiples Algoritmos:**

- Diferentes algoritmos tienen fortalezas en diferentes regiones del espacio de búsqueda
- La comparación permite identificar el método más robusto para este problema específico
- Los resultados estadísticos (media, desviación) revelan la estabilidad del algoritmo
- El análisis de convergencia muestra qué algoritmos requieren más evaluaciones

### Python API - Compatibilidad con Benchmark Original

El benchmark original (`LevitadorBenchmark`) sigue funcionando para problemas simples:

```python
from levitador_benchmark import LevitadorBenchmark

# Problema original (3 parámetros: k0, k, a)
problema = LevitadorBenchmark()

# Evaluar una solución candidata
solucion = [0.036, 0.0035, 0.005]  # [k0, k, a]
error = problema.fitness_function(solucion)

print(f"Error MSE: {error:.6e}")
```

---

## 🚀 Benchmark Completo de Optimización

### Ejecución Rápida

Para ejecutar un benchmark completo comparando todos los algoritmos metaheurísticos disponibles:

```bash
# Ejecutar benchmark completo con configuración por defecto
python scripts/run_full_optimization.py

# Ejecutar con configuración personalizada
python scripts/run_full_optimization.py --config config/full_optimization.yaml

# Ejecutar con más trials para mejor análisis estadístico
python scripts/run_full_optimization.py --trials 10

# Ejecutar con semilla diferente para reproducibilidad
python scripts/run_full_optimization.py --seed 123
```

### ¿Qué hace este script?

El script `run_full_optimization.py` ejecuta un benchmark exhaustivo que:

1. **Carga datos experimentales** de `data/datos_levitador.txt`
2. **Ejecuta todos los algoritmos disponibles**:
   - Differential Evolution (DE)
   - Grey Wolf Optimizer (GWO)
   - Artificial Bee Colony (ABC)
   - Honey Badger Algorithm (HBA)
   - Shrimp Optimizer (SOA)
   - Tianji Optimizer (Tianji)
   - Genetic Algorithm (GA)
   - Random Search (Random)

3. **Configura cada algoritmo con parámetros optimizados**:
   - `pop_size`: 50 individuos
   - `max_iter`: 200 iteraciones
   - Parámetros específicos bien ajustados

4. **Ejecuta múltiples trials** (default: 5) por algoritmo para estadísticas robustas

5. **Compara con valores teóricos de referencia**:
   - k₀ = 0.0363 H
   - k = 0.0035 H
   - a = 0.0052 m

6. **Genera reportes y visualizaciones**:
   - 📊 Curvas de convergencia comparativas
   - 📦 Boxplot de rendimiento
   - 📋 Tabla de comparación con valores teóricos
   - 📄 Reporte detallado en markdown (`BENCHMARK_REPORT.md`)

### Resultados Generados

Todos los resultados se guardan en `results/optimization_comparison/`:

```
results/optimization_comparison/
├── BENCHMARK_REPORT.md          # Reporte completo en markdown
├── optimization_results.json     # Resultados en formato JSON
├── convergence_curves.png        # Curvas de convergencia
├── performance_boxplot.png       # Boxplot comparativo
└── comparison_table.png          # Tabla con valores teóricos
```

### Interpretación de Resultados

El reporte incluye:

1. **Ranking de algoritmos** - Ordenados por mejor MSE obtenido
2. **Estadísticas detalladas** - Media, desviación estándar, mejor, peor
3. **Comparación con teóricos** - Errores porcentuales para cada parámetro
4. **Criterios de éxito**:
   - ✅ MSE < 1e-7
   - ✅ Parámetros dentro del 10% de valores teóricos

### Ejemplo de Salida

```
🏆 BEST ALGORITHM:
   DE (DifferentialEvolution)
   MSE: 2.345678e-08
   k₀ = 0.036234 H  (theoretical: 0.0363)
   k  = 0.003487 H  (theoretical: 0.0035)
   a  = 0.005123 m  (theoretical: 0.0052)

✓ SUCCESS CRITERIA:
   MSE < 1e-07: ✅ PASS
   Parameters within 10%: ✅ PASS
```

### Interpretación de Resultados de Metaheurísticos

**Métricas Clave:**

1. **MSE (Mean Squared Error)**: Medida principal de calidad del ajuste
   - Excelente: MSE < 1e-7
   - Bueno: 1e-7 < MSE < 1e-6
   - Aceptable: 1e-6 < MSE < 1e-5
   - Requiere ajuste: MSE > 1e-5

2. **Convergencia**: Iteraciones necesarias para alcanzar el óptimo
   - Convergencia rápida: < 30 iteraciones
   - Convergencia normal: 30-80 iteraciones
   - Convergencia lenta: > 80 iteraciones

3. **Robustez**: Consistencia entre múltiples trials
   - Desviación estándar baja (< 10% de la media) indica alta robustez
   - Desviación estándar alta sugiere sensibilidad a inicialización

4. **Comparación con Valores Teóricos**:
   - Los parámetros K0 y A pueden validarse con valores de referencia
   - R0 y α son estimados (no hay medición directa de temperatura)
   - Error porcentual < 10% indica identificación exitosa

**Diagnóstico de Problemas Comunes:**

| Síntoma | Posible Causa | Solución |
|---------|---------------|----------|
| MSE estancado en valor alto | Mínimo local, población pequeña | Aumentar `pop_size`, cambiar algoritmo |
| Convergencia muy lenta | Parámetros conservadores | Ajustar F, CR (DE) o tasas de mutación |
| Resultados inconsistentes | Sensibilidad a ruido en datos | Aumentar `smoothing_window`, validar datos |
| Parámetros fuera de rango físico | Bounds incorrectos | Revisar límites en configuración |
| Simulación falla (fitness = 1e10) | Parámetros causan inestabilidad numérica | Ajustar tolerancias ODE, revidar bounds |

**Visualizaciones Generadas:**

1. **Curvas de Convergencia** (`convergence_*.png`): Muestra la evolución del mejor fitness vs. iteraciones
   - Línea descendente suave indica búsqueda eficiente
   - Línea con muchas mesetas sugiere dificultad en escapar mínimos locales

2. **Boxplot de Comparación** (`comparison_boxplot.png`): Compara distribución de fitness entre algoritmos
   - Caja más baja = mejor desempeño promedio
   - Caja más pequeña = mayor robustez

3. **Visualización de Solución** (`best_solution.png`): Compara trayectorias simuladas vs. reales
   - Superposición cercana indica buen ajuste
   - Divergencias revelan limitaciones del modelo o ruido en datos

### Configuración Personalizada

Puedes crear tu propia configuración editando `config/full_optimization.yaml`:

```yaml
# Ajustar número de trials
benchmark:
  n_trials: 10  # Más trials = mejor estadística

# Ajustar parámetros de optimización
optimization:
  pop_size: 100    # Más individuos = mejor exploración
  max_iter: 300    # Más iteraciones = mejor convergencia

# Habilitar/deshabilitar algoritmos
algorithms:
  DifferentialEvolution:
    enabled: true
    pop_size: 50
    max_iter: 200
  
  RandomSearch:
    enabled: false  # Deshabilitar si no es necesario
```

---

## 💻 Uso del Benchmark Original (3 parámetros)

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

### Mejores Prácticas para Optimización Metaheurística

**1. Configuración de Población e Iteraciones:**

La relación entre población y número de iteraciones afecta el presupuesto total de evaluaciones:

```
Evaluaciones Totales ≈ pop_size × max_iter
```

Recomendaciones por complejidad del problema:
- **Problema simple (3 parámetros)**: `pop_size=30`, `max_iter=50-100`
- **Problema moderado (4 parámetros + R(t))**: `pop_size=50`, `max_iter=100-200`
- **Problema complejo (muchos parámetros)**: `pop_size=100`, `max_iter=200-500`

**2. Ajuste de Hiperparámetros por Algoritmo:**

| Algoritmo | Parámetro Crítico | Valor Recomendado | Efecto |
|-----------|-------------------|-------------------|---------|
| DE | F (mutation factor) | 0.5-0.9 | Mayor F → más exploración |
| DE | CR (crossover rate) | 0.7-0.95 | Mayor CR → más diversidad |
| GWO | a (linearly decreased) | 2→0 | Controla exploración vs explotación |
| ABC | limit (abandonment) | pop_size × dim | Mayor limit → más persistencia |
| GA | crossover_prob | 0.7-0.9 | Mayor prob → más recombinación |
| GA | mutation_prob | 0.1-0.3 | Mayor prob → más diversidad |

**3. Estrategias de Aceleración:**

Para problemas con datos experimentales largos (>10,000 muestras):

- **Submuestreo**: `subsample_factor=10-50` reduce tiempo ~10-50x
- **Evaluación paralela**: Automática si múltiples núcleos disponibles
- **Early stopping**: Configurar tolerancia de convergencia

```python
problema = ParameterBenchmark(
    data_path='data/datos_levitador.txt',
    subsample_factor=20,  # Usa solo 1 de cada 20 muestras
    verbose=True
)
```

**4. Reproducibilidad:**

Siempre usar semillas aleatorias para experimentos reproducibles:

```python
optimizer = DifferentialEvolution(
    problema,
    pop_size=30,
    max_iter=100,
    random_seed=42  # ← Garantiza resultados reproducibles
)
```

Para múltiples trials con diferentes semillas:
```python
for trial in range(5):
    seed = base_seed + trial
    optimizer = DifferentialEvolution(problema, random_seed=seed)
    best_sol, best_fit = optimizer.optimize()
```

**5. Validación de Resultados:**

Después de la optimización, siempre:

1. **Verificar convergencia**: Revisar curva de convergencia para detectar estancamiento
2. **Validar físicamente**: Los parámetros deben estar en rangos razonables
3. **Comparar múltiples runs**: Ejecutar 5-10 trials para evaluar robustez
4. **Visualizar ajuste**: Comparar trayectorias simuladas vs. reales
5. **Analizar residuales**: Verificar que los errores sean aleatorios, no sistemáticos

**6. Debugging de Optimización:**

Si el algoritmo no converge:

```python
# Activar modo verbose para ver progreso detallado
optimizer = DifferentialEvolution(problema, verbose=True)

# Revisar historial de convergencia
history = optimizer.get_convergence_curve()
print(f"Mejora final: {history[0]} → {history[-1]}")

# Verificar que fitness se evalúa correctamente
test_solution = [0.036, 0.005, 2.5, 0.004]  # Valores razonables
fitness = problema.fitness_function(test_solution)
print(f"Test fitness: {fitness}")  # Debe ser finito, no 1e10
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


---

## 🗺️ Mapa Mental: Arquitectura del Pipeline

```
╔════════════════════════════════════════════════════════════════════════════╗
║                  LEVITADOR MAGNÉTICO - PIPELINE DE DOS FASES               ║
╚════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│  ENTRADA: Datos Experimentales (t, y, i, u)                                │
│  ▪ datos_levitador.txt (identificación parámetros)                         │
│  ▪ sesiones_kan_pinn/*.txt (entrenamiento KAN-PINN)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                       FASE 1: IDENTIFICACIÓN DE PARÁMETROS                ┃
┃                                                                            ┃
┃  Objetivo: Identificar θ = [K0, A, R0, α]                                 ┃
┃                                                                            ┃
┃  ┌──────────────────────────────────────────────────────────────────────┐ ┃
┃  │  MODELOS FÍSICOS                                                     │ ┃
┃  │                                                                      │ ┃
┃  │  ▪ Inductancia:  L(y) = K0 / (1 + y/A)                              │ ┃
┃  │                  ∂L/∂y = -K0 / (A·(1 + y/A)²)                       │ ┃
┃  │                                                                      │ ┃
┃  │  ▪ Resistencia (sin sensor de temperatura):                         │ ┃
┃  │                  R(t) ≈ R0·(1 + α·ΔT(t))                            │ ┃
┃  │                  ΔT(t) ∝ ∫ i²(t) dt (Joule heating)                 │ ┃
┃  │                                                                      │ ┃
┃  │  ▪ Estimación vía Kirchhoff:                                        │ ┃
┃  │                  R_est(t) = (u(t) - dφ̂/dt) / i(t)                   │ ┃
┃  │                  donde φ̂ = L(y)·i                                   │ ┃
┃  └──────────────────────────────────────────────────────────────────────┘ ┃
┃                                                                            ┃
┃  ┌──────────────────────────────────────────────────────────────────────┐ ┃
┃  │  ECUACIONES DINÁMICAS                                                │ ┃
┃  │                                                                      │ ┃
┃  │  ▪ Mecánica:    m·ÿ = (1/2)·(∂L/∂y)·i² + m·g                        │ ┃
┃  │  ▪ Eléctrica:   L(y)·(di/dt) + (∂L/∂y)·ẏ·i + R(t)·i = u            │ ┃
┃  └──────────────────────────────────────────────────────────────────────┘ ┃
┃                                                                            ┃
┃  ┌──────────────────────────────────────────────────────────────────────┐ ┃
┃  │  METAHEURÍSTICOS (ParameterBenchmark)                               │ ┃
┃  │                                                                      │ ┃
┃  │  ▪ Differential Evolution (DE)     ▪ Honey Badger (HBA)            │ ┃
┃  │  ▪ Grey Wolf Optimizer (GWO)       ▪ Shrimp Optimizer (SOA)        │ ┃
┃  │  ▪ Artificial Bee Colony (ABC)     ▪ Tianji Optimizer              │ ┃
┃  │  ▪ Genetic Algorithm (GA)          ▪ Random Search                 │ ┃
┃  │                                                                      │ ┃
┃  │  Fitness: MSE(y_simulada(θ), y_real)                                │ ┃
┃  └──────────────────────────────────────────────────────────────────────┘ ┃
┃                                                                            ┃
┃  SALIDA: parametros_optimos.json → [K0*, A*, R0*, α*]                     ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                     │
                                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                   FASE 2: ENTRENAMIENTO KAN-PINN (Sensorless)             ┃
┃                                                                            ┃
┃  Objetivo: Entrenar observador neuronal para estimar posición sin sensor  ┃
┃                                                                            ┃
┃  ┌──────────────────────────────────────────────────────────────────────┐ ┃
┃  │  ETAPA 1: OBSERVADOR DE FLUJO (FluxObserver)                        │ ┃
┃  │                                                                      │ ┃
┃  │    Entrada: (u, i)                                                  │ ┃
┃  │              │                                                       │ ┃
┃  │              ▼                                                       │ ┃
┃  │         ┌─────────┐                                                 │ ┃
┃  │         │ HiPPO-8 │  (captura temporal online)                      │ ┃
┃  │         └────┬────┘                                                 │ ┃
┃  │              │                                                       │ ┃
┃  │              ▼                                                       │ ┃
┃  │         ┌─────────┐                                                 │ ┃
┃  │         │  KAN    │  (B-splines + residual)                        │ ┃
┃  │         │ 3 → 32  │                                                 │ ┃
┃  │         └────┬────┘                                                 │ ┃
┃  │              │                                                       │ ┃
┃  │              ▼                                                       │ ┃
┃  │    Salida: φ̂ (flujo estimado)                                      │ ┃
┃  │                                                                      │ ┃
┃  │    Pérdida: L = w_data·MSE(φ̂, φ) + w_kirch·|u - R·i - dφ̂/dt|²     │ ┃
┃  └──────────────────────────────────────────────────────────────────────┘ ┃
┃                                                                            ┃
┃  ┌──────────────────────────────────────────────────────────────────────┐ ┃
┃  │  ETAPA 2: PREDICTOR DE POSICIÓN (PositionPredictor)                │ ┃
┃  │                                                                      │ ┃
┃  │    Entrada: (u, i, φ̂)  ← flujo de Etapa 1                          │ ┃
┃  │              │                                                       │ ┃
┃  │              ▼                                                       │ ┃
┃  │         ┌─────────┐                                                 │ ┃
┃  │         │  KAN    │  (sin HiPPO, usa φ̂ directamente)               │ ┃
┃  │         │ 3 → 32  │                                                 │ ┃
┃  │         │  → 32   │                                                 │ ┃
┃  │         │  → 1    │                                                 │ ┃
┃  │         └────┬────┘                                                 │ ┃
┃  │              │                                                       │ ┃
┃  │              ▼                                                       │ ┃
┃  │    Salida: ŷ (posición estimada)                                   │ ┃
┃  │                                                                      │ ┃
┃  │    Pérdida PINN (usando K0*, A* de Fase 1):                        │ ┃
┃  │         L = w_data·MSE(ŷ, y) + w_pinn·|φ̂ - L*(ŷ)·i|²               │ ┃
┃  │                                                                      │ ┃
┃  │    Curriculum Learning: w_pinn va de 0.1 → 5.0                     │ ┃
┃  └──────────────────────────────────────────────────────────────────────┘ ┃
┃                                                                            ┃
┃  SALIDA: Modelos entrenados (.pt) + predicciones + métricas               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RESULTADO FINAL:                                                           │
│  ▪ Parámetros físicos identificados: [K0*, A*, R0*, α*]                    │
│  ▪ Observador de posición sensorless entrenado                             │
│  ▪ Estimación de R(t) sin sensor de temperatura                            │
│  ▪ Visualizaciones y métricas de convergencia                              │
└─────────────────────────────────────────────────────────────────────────────┘

CARACTERÍSTICAS CLAVE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▪ NO hay sensor de temperatura → R(t) se estima vía Kirchhoff
▪ NO hay data leakage → Etapa 2 usa φ̂ de Etapa 1 (no y_sensor)
▪ Restricciones físicas garantizadas: K0 > 0, A > 0, R0 > 0
▪ Submuestreo configurable para optimización rápida
▪ Pérdidas físicas: Kirchhoff (Etapa 1) + PINN Euler-Lagrange (Etapa 2)
▪ 8 algoritmos metaheurísticos disponibles
▪ Framework modular y extensible
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---
