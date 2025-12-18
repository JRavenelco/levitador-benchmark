# Datasets para KAN-PINN y Metaheurísticos

**Autor:** José de Jesús Santana Ramírez  
**ORCID:** [0000-0002-6183-7379](https://orcid.org/0000-0002-6183-7379)  
**Fecha:** 17-Dic-2024

## Descripción

Estos datasets contienen datos experimentales del levitador magnético capturados en tiempo real durante experimentos de control. Son útiles para:

1. **Metaheurísticos**: Identificar parámetros $[k_0, k, a]$ del modelo de inductancia
2. **KAN-PINN**: Entrenar observadores de posición basados en redes neuronales con física

## Formato de Datos

Archivos TSV (tab-separated) con las siguientes columnas:

| Columna | Variable | Unidad | Descripción |
|---------|----------|--------|-------------|
| 1 | `t` | s | Tiempo |
| 2 | `y` | m | Posición medida (sensor) |
| 3 | `y_obs` | m | Posición estimada (observador) |
| 4 | `dy_obs` | m/s | Velocidad estimada |
| 5 | `i` | A | Corriente de la bobina |
| 6 | `u` | V | Voltaje de control |
| 7 | `yd` | m | Referencia de posición |

## Archivos Disponibles

| Archivo | Tipo de Experimento | Muestras | Duración |
|---------|---------------------|----------|----------|
| `dataset_constante_*.txt` | Referencia constante 5mm | ~3000 | 30s |
| `dataset_escalon_*.txt` | Escalones 4mm → 6mm → 5mm | ~3000 | 30s |
| `dataset_senoidal_*.txt` | Senoidal 5mm ± 1mm | ~3000 | 30s |
| `dataset_chirp_*.txt` | Chirp 0.1-2 Hz | ~3500 | 35s |
| `dataset_multiescalon_*.txt` | Múltiples escalones aleatorios | ~3500 | 35s |

## Uso para Metaheurísticos

```python
import numpy as np
from levitador_benchmark import LevitadorBenchmark

# Cargar dataset real
data = np.loadtxt('data/sesiones_kan_pinn/dataset_escalon_20251217_205858.txt', skiprows=1)
t, y, y_obs, dy_obs, i, u, yd = data.T

# Usar con benchmark
benchmark = LevitadorBenchmark()
benchmark.set_experimental_data(t, y, i, u)

# Optimizar parámetros
from scipy.optimize import differential_evolution
result = differential_evolution(benchmark.fitness_function, benchmark.bounds)
print(f"k0={result.x[0]:.4f}, k={result.x[1]:.4f}, a={result.x[2]:.4f}")
```

## Uso para KAN-PINN

```python
import torch
import numpy as np

# Cargar múltiples datasets
datasets = []
for file in glob('data/sesiones_kan_pinn/*.txt'):
    data = np.loadtxt(file, skiprows=1)
    datasets.append(data)

# Combinar y preparar para entrenamiento
all_data = np.vstack(datasets)
i, u, y = all_data[:, 4], all_data[:, 5], all_data[:, 1]

# Calcular inductancia estimada L = φ/i
# (ver pinn/kan_observador.py para detalles)
```

## Parámetros del Sistema

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| $R$ | 2.72 Ω | Resistencia de la bobina |
| $k_0$ | 0.0704 H | Inductancia base (identificado) |
| $k$ | 0.0327 H | Coeficiente inductancia |
| $a$ | 0.0052 m | Parámetro geométrico |
| $T_s$ | 0.01 s | Período de muestreo (100 Hz) |

## Licencia

MIT - Ver LICENSE en el repositorio principal.
