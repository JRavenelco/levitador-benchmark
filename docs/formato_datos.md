# 📊 Formato de Datos Experimentales

## Descripción General

Los datos experimentales del levitador magnético se capturan durante la operación en lazo cerrado del sistema. El controlador ya está implementado en el sistema real, y el objetivo del benchmark es identificar los parámetros físicos del modelo.

## Archivo de Datos

**Ubicación:** `data/datos_levitador.txt`

### Formato

El archivo es un CSV con separadores de espacio/tabulador. Contiene las siguientes columnas:

| Columna | Nombre | Unidad | Descripción |
|---------|--------|--------|-------------|
| 1 | `t` | s | Tiempo (segundos) |
| 2 | `y` | m | Posición de la esfera (metros) |
| 3 | `v` | m/s | Velocidad de la esfera |
| 4 | `i` | A | Corriente en la bobina |
| 5 | `u` | V | Voltaje aplicado (entrada de control) |
| 6 | `ref` | m | Referencia de posición |

### Ejemplo de Lectura

```python
import pandas as pd

# Leer datos
data = pd.read_csv('data/datos_levitador.txt', sep=r'\s+', header=None)
data.columns = ['t', 'y', 'v', 'i', 'u', 'ref']

print(f"Duración: {data['t'].iloc[-1]:.2f} s")
print(f"Muestras: {len(data)}")
print(f"Frecuencia de muestreo: {len(data)/data['t'].iloc[-1]:.0f} Hz")
```

## Condiciones Experimentales

### Sistema Físico
- **Esfera:** Acero, masa m = 0.018 kg
- **Electroimán:** Bobina con R = 2.72 Ω
- **Sensor:** Posición óptica (rango 0-25 mm)

### Operación
- **Modo:** Lazo cerrado (controlador PID activo)
- **Referencia:** Variable (escalones, rampa, senoidal)
- **Frecuencia de muestreo:** ~1000 Hz

## Uso en el Benchmark

El benchmark utiliza las columnas `t`, `y`, y `u`:

- `t`: Vector de tiempo para la simulación
- `y`: Posición real (target a igualar)
- `u`: Voltaje de entrada (señal de control conocida)

La simulación resuelve las ecuaciones diferenciales del modelo con los parámetros candidatos `[k0, k, a]` y compara la posición simulada con la posición real `y`.

## Generación de Nuevos Datos

Si deseas capturar nuevos datos experimentales:

1. Conectar el sistema de adquisición de datos
2. Ejecutar el controlador en lazo cerrado
3. Guardar los datos en formato CSV con las 6 columnas
4. Nombrar el archivo con fecha: `datos_levitador_YYYYMMDD_HHMMSS.txt`

---

*Documentación del proyecto Levitador Benchmark*
