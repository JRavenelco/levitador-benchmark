# KAN-PINN JAX GPU Optimization Notebook

## 🚀 Google Colab Notebook para Optimización en GPU

Este notebook (`KAN_PINN_JAX_GPU_Optimization.ipynb`) implementa un sistema completo de identificación de parámetros físicos del levitador magnético usando JAX y Differential Evolution en GPU.

### 📋 Características Principales

- ✅ **Auto-configuración completa**: Instala dependencias automáticamente
- ✅ **Carga flexible de datos**: Soporta formato estándar y KAN-PINN
- ✅ **GPU Acceleration**: Differential Evolution completamente vectorizado en JAX
- ✅ **Modelo físico completo**: Ecuaciones del levitador magnético
- ✅ **Visualizaciones ricas**: Convergencia, comparación modelo vs datos, inductancia
- ✅ **Guardado automático**: JSON + gráficas con descarga automática
- ✅ **Comparativa GPU vs CPU**: Demuestra el speedup obtenido

### 🎯 Cómo Usar

#### Opción 1: Abrir Directamente en Colab

1. Ve a [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook
3. Sube `KAN_PINN_JAX_GPU_Optimization.ipynb`
4. **Activar GPU**: Runtime → Change runtime type → GPU (T4)
5. **Ejecutar todo**: Runtime → Run all

#### Opción 2: Desde GitHub

1. Abre el notebook directamente desde GitHub:
   ```
   https://colab.research.google.com/github/JRavenelco/levitador-benchmark/blob/main/KAN_PINN_JAX_GPU_Optimization.ipynb
   ```
2. **Activar GPU**: Runtime → Change runtime type → GPU (T4)
3. **Ejecutar todo**: Runtime → Run all

### ⚙️ Estructura del Notebook

1. **Configuración del Entorno** - Instalación de JAX y dependencias
2. **Clonar Repositorio** - Obtiene datos del repo
3. **Carga de Datos** - Funciones para formato estándar y KAN-PINN
4. **Transferencia a GPU** - Convierte datos a JAX arrays
5. **Modelo Físico** - Implementación vectorizada del levitador
6. **Función de Fitness** - MSE vectorizado
7. **Differential Evolution** - Optimización GPU
8. **Ejecutar Optimización** - Con configuración ajustable
9. **Simular Resultados** - Con parámetros identificados
10. **Visualización** - Gráficas de convergencia y comparación
11. **Guardar Resultados** - JSON + PNG con descarga automática
12. **GPU vs CPU** - Comparativa de velocidad (opcional)
13. **Resumen** - Próximos pasos y referencias

### 📊 Datasets Disponibles

El notebook puede usar diferentes datasets:

- `data/datos_levitador.txt` - **Datos estándar** (por defecto)
- `data/sesiones_kan_pinn/dataset_escalon_*.txt` - Respuesta a escalón
- `data/sesiones_kan_pinn/dataset_senoidal_*.txt` - Señal senoidal
- `data/sesiones_kan_pinn/dataset_chirp_*.txt` - Chirp
- `data/sesiones_kan_pinn/dataset_multiescalon_*.txt` - Múltiples escalones
- `data/sesiones_kan_pinn/dataset_constante_*.txt` - Entrada constante

Para cambiar el dataset, modifica la variable `DATA_FILE` en la celda correspondiente.

### 🎛️ Parámetros Configurables

En la celda de "Ejecutar Optimización" puedes ajustar:

```python
POP_SIZE = 100        # Tamaño de población (50-200)
MAX_ITER = 200        # Generaciones (100-500)
F_MUTATION = 0.8      # Factor de mutación (0.5-0.9)
CR_CROSSOVER = 0.9    # Probabilidad de cruce (0.7-0.95)
SUBSAMPLE = 10        # Submuestreo de datos (1-20)
```

### ⏱️ Tiempos Esperados

- **GPU T4 (Colab)**: ~2-5 minutos (100 ind, 200 gen)
- **GPU A100**: ~1-2 minutos
- **CPU**: ~10-30 minutos

### 📦 Resultados Generados

Al finalizar, se generan:

1. **`optimization_results.json`** - Parámetros y métricas
2. **`convergencia.png`** - Gráfica de convergencia
3. **`comparacion_modelo_datos.png`** - Comparación completa

Los archivos se descargan automáticamente en Colab.

### 🐛 Troubleshooting

**❌ "No GPU detected"**
- Solución: Runtime → Change runtime type → GPU (T4)

**❌ "git clone failed"**
- El notebook descargará los datos automáticamente
- También puedes subir tus propios archivos de datos

**❌ "JAX import error"**
- Re-ejecuta la celda de instalación
- Reinicia el runtime si es necesario

**❌ Optimización muy lenta**
- Aumenta `SUBSAMPLE` (ej: 20)
- Reduce `POP_SIZE` o `MAX_ITER`

### 📚 Referencias

- **Repositorio**: [levitador-benchmark](https://github.com/JRavenelco/levitador-benchmark)
- **JAX**: https://jax.readthedocs.io
- **Differential Evolution**: Storn & Price (1997)

### 🤝 Contribuir

¿Encontraste un bug o tienes una mejora? Abre un issue o PR en el repositorio.

### 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) en el repositorio.

---

**Creado para el proyecto de Levitador Magnético** 🧲
