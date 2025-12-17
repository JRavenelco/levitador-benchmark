# 🔬 Diseño de Experimentos (DOE) - Levitador Magnético

## Objetivo

Generar un conjunto de datos experimentales diverso y robusto para:
1. **Validar** el modelo matemático del levitador
2. **Caracterizar** la respuesta dinámica del sistema
3. **Proveer** datos variados para el benchmark de optimización

---

## Factores Experimentales

### Factor A: Tipo de Referencia
| Nivel | Descripción | Código |
|-------|-------------|--------|
| A1 | Escalón (step) | `escalon` |
| A2 | Rampa (ramp) | `rampa` |
| A3 | Senoidal (sine) | `senoidal` |
| A4 | Pulso (pulse) | `pulso` |
| A5 | Escalera (stair) | `escalera` |

### Factor B: Posición de Equilibrio
| Nivel | Valor (mm) | Código |
|-------|-----------|--------|
| B1 | 3.0 | `y0_3mm` |
| B2 | 4.0 | `y0_4mm` |
| B3 | 5.0 | `y0_5mm` |
| B4 | 6.0 | `y0_6mm` |

### Factor C: Amplitud de Perturbación
| Nivel | Valor (mm) | Código |
|-------|-----------|--------|
| C1 | 0.5 | `amp_05mm` |
| C2 | 1.0 | `amp_1mm` |
| C3 | 1.5 | `amp_15mm` |

### Factor D: Frecuencia (solo para senoidal)
| Nivel | Valor (Hz) | Código |
|-------|-----------|--------|
| D1 | 0.1 | `f_01hz` |
| D2 | 0.5 | `f_05hz` |
| D3 | 1.0 | `f_1hz` |

---

## Diseño Factorial Fraccional

### Experimentos Básicos (Caracterización)

| # | Tipo | y₀ (mm) | Amplitud | Duración | Descripción |
|---|------|---------|----------|----------|-------------|
| E01 | Escalón ↓ | 5.0 → 4.0 | 1.0 mm | 30s | Respuesta a escalón descendente |
| E02 | Escalón ↑ | 4.0 → 5.0 | 1.0 mm | 30s | Respuesta a escalón ascendente |
| E03 | Escalón ↓ | 5.0 → 3.5 | 1.5 mm | 30s | Escalón grande descendente |
| E04 | Escalón ↑ | 3.5 → 5.0 | 1.5 mm | 30s | Escalón grande ascendente |
| E05 | Rampa ↓ | 5.0 → 4.0 | 1.0 mm | 30s | Rampa lenta (10s) |
| E06 | Rampa ↑ | 4.0 → 5.0 | 1.0 mm | 30s | Rampa lenta ascendente |
| E07 | Senoidal | 5.0 | 0.5 mm | 40s | Baja frecuencia (0.1 Hz) |
| E08 | Senoidal | 5.0 | 0.5 mm | 30s | Media frecuencia (0.5 Hz) |
| E09 | Senoidal | 5.0 | 0.5 mm | 20s | Alta frecuencia (1.0 Hz) |
| E10 | Pulso | 5.0 | 1.0 mm | 30s | Pulsos cada 5s |
| E11 | Escalera | 5.0→3.0 | steps 0.5 mm | 40s | 4 escalones |
| E12 | PRBS | 5.0 | ±0.5 mm | 60s | Señal pseudo-aleatoria |

### Experimentos de Validación (Repeticiones)

| # | Basado en | Repeticiones | Propósito |
|---|-----------|--------------|-----------|
| V01-V03 | E01 | 3 | Validar reproducibilidad escalón |
| V04-V06 | E08 | 3 | Validar reproducibilidad senoidal |

### Experimentos de Robustez

| # | Condición | Descripción |
|---|-----------|-------------|
| R01 | Perturbación externa | Golpe suave durante operación |
| R02 | Arranque frío | Sistema encendido desde reposo |
| R03 | Operación prolongada | 5 minutos continuo |

---

## Matriz de Experimentos Recomendada

### Fase 1: Caracterización Básica (Prioridad Alta)
```
E01, E02, E07, E08, E11
```

### Fase 2: Caracterización Extendida (Prioridad Media)
```
E03, E04, E05, E06, E09, E10
```

### Fase 3: Validación (Prioridad Alta)
```
V01-V06 (repeticiones de E01 y E08)
```

### Fase 4: Robustez (Prioridad Baja)
```
R01, R02, R03
```

---

## Protocolo Experimental

### Preparación
1. Verificar conexión del levitador (COM port)
2. Verificar que la esfera esté limpia y centrada
3. Esperar 2 minutos de calentamiento del electroimán
4. Verificar funcionamiento del sensor de posición

### Ejecución
1. Ejecutar script de adquisición con parámetros del experimento
2. Esperar estabilización (5s) antes del cambio de referencia
3. Registrar condiciones ambientales (temperatura, hora)
4. Guardar archivo con nomenclatura estándar

### Nomenclatura de Archivos
```
exp_{ID}_{tipo}_{y0}mm_{amp}mm_{fecha}_{hora}.txt

Ejemplos:
exp_E01_escalon_5mm_1mm_20251217_180000.txt
exp_E08_senoidal_5mm_05mm_05hz_20251217_181500.txt
```

---

## Métricas a Calcular

Para cada experimento, calcular:

| Métrica | Símbolo | Descripción |
|---------|---------|-------------|
| Tiempo de subida | tᵣ | 10% → 90% del valor final |
| Tiempo de asentamiento | tₛ | Error < 2% del valor final |
| Sobreimpulso | Mp | Máximo % sobre valor final |
| Error en estado estable | eₛₛ | Error promedio después de tₛ |
| ISE | ∫e²dt | Integral del error cuadrático |
| IAE | ∫\|e\|dt | Integral del error absoluto |

---

## Análisis Posterior

### 1. Identificación del Sistema
- Ajuste de modelo de primer/segundo orden
- Estimación de parámetros (k₀, k, a)
- Validación cruzada con datos diferentes

### 2. Análisis de Variabilidad
- ANOVA para repeticiones
- Intervalos de confianza de parámetros
- Detección de outliers

### 3. Caracterización Frecuencial
- Diagrama de Bode experimental
- Estimación de ancho de banda
- Función de transferencia

---

## Tiempo Estimado

| Fase | Experimentos | Tiempo |
|------|--------------|--------|
| Preparación | - | 10 min |
| Fase 1 | 5 exp × 30s | 5 min |
| Fase 2 | 6 exp × 30s | 5 min |
| Fase 3 | 6 exp × 30s | 5 min |
| Fase 4 | 3 exp × 60s | 5 min |
| **Total** | **20 exp** | **~30 min** |

---

## Referencias

- Montgomery, D. C. (2017). *Design and Analysis of Experiments*
- Box, G. E., Hunter, J. S., & Hunter, W. G. (2005). *Statistics for Experimenters*
