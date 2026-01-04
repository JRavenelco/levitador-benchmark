#!/usr/bin/env python3
"""
Preparador de Datos de Calibración para Identificación de Parámetros
=====================================================================

Este script procesa los datos crudos del levitador (formato levitador_sensorless_kan.cpp)
y los prepara para el benchmark JAX.

Formato de entrada (del C++):
    t, yd, y, y_est_final, ie, u

Formato de salida (para benchmark):
    t, y, i, u

Uso:
    python scripts/prepare_calibration_data.py --input datos_raw.txt --output data/calibration_data.txt
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_raw_data(filepath: str) -> dict:
    """
    Carga datos crudos del levitador.
    Formato: t, yd, y, y_est_final, ie, u
    """
    print(f"Cargando datos de {filepath}...")
    
    # Intentar cargar con diferentes delimitadores
    try:
        data = np.loadtxt(filepath, delimiter='\t')
    except:
        try:
            data = np.loadtxt(filepath, delimiter=',')
        except:
            data = np.loadtxt(filepath)
    
    print(f"Shape: {data.shape}")
    
    if data.shape[1] >= 6:
        # Formato completo: t, yd, y, y_est, ie, u
        return {
            't': data[:, 0],
            'yd': data[:, 1],      # Setpoint
            'y': data[:, 2],       # Posición real (sensor)
            'y_est': data[:, 3],   # Posición estimada
            'i': data[:, 4],       # Corriente
            'u': data[:, 5],       # Voltaje de control
        }
    elif data.shape[1] >= 4:
        # Formato mínimo: t, y, i, u
        return {
            't': data[:, 0],
            'y': data[:, 1],
            'i': data[:, 2],
            'u': data[:, 3],
        }
    else:
        raise ValueError(f"Formato de datos no reconocido. Columnas: {data.shape[1]}")


def analyze_data_quality(data: dict) -> dict:
    """
    Analiza la calidad de los datos para identificación de parámetros.
    """
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE CALIDAD DE DATOS")
    print("="*60)
    
    metrics = {}
    
    # Duración
    t = data['t']
    duration = t[-1] - t[0]
    dt = np.median(np.diff(t))
    metrics['duration_s'] = duration
    metrics['dt_s'] = dt
    metrics['n_samples'] = len(t)
    print(f"\n⏱️  Duración: {duration:.2f}s ({len(t)} muestras, dt={dt*1000:.1f}ms)")
    
    # Posición
    y = data['y']
    y_mm = y * 1000
    metrics['y_min_mm'] = np.min(y_mm)
    metrics['y_max_mm'] = np.max(y_mm)
    metrics['y_range_mm'] = np.max(y_mm) - np.min(y_mm)
    metrics['y_std_mm'] = np.std(y_mm)
    
    print(f"\n📏 Posición (y):")
    print(f"   Rango: {np.min(y_mm):.2f} - {np.max(y_mm):.2f} mm")
    print(f"   Variación: {metrics['y_range_mm']:.2f} mm")
    print(f"   Desv. estándar: {metrics['y_std_mm']:.3f} mm")
    
    # Corriente
    i = data['i']
    metrics['i_min_A'] = np.min(i)
    metrics['i_max_A'] = np.max(i)
    metrics['i_mean_A'] = np.mean(i)
    
    print(f"\n⚡ Corriente (i):")
    print(f"   Rango: {np.min(i):.4f} - {np.max(i):.4f} A")
    print(f"   Promedio: {np.mean(i):.4f} A")
    
    # Voltaje
    u = data['u']
    metrics['u_min_V'] = np.min(u)
    metrics['u_max_V'] = np.max(u)
    
    print(f"\n🔋 Voltaje (u):")
    print(f"   Rango: {np.min(u):.2f} - {np.max(u):.2f} V")
    
    # Diagnóstico de calidad
    print("\n" + "-"*60)
    print("🔍 DIAGNÓSTICO:")
    
    issues = []
    
    # Check 1: ¿Hay variación en posición?
    if metrics['y_range_mm'] < 0.5:
        issues.append("⚠️  CRÍTICO: Posición casi constante. Se necesitan cambios de setpoint.")
        metrics['quality'] = 'BAD'
    elif metrics['y_range_mm'] < 1.5:
        issues.append("⚠️  ADVERTENCIA: Poca variación en posición. Ideal > 2mm de rango.")
        
    # Check 2: ¿Corriente razonable?
    if metrics['i_mean_A'] < 0.1:
        issues.append("⚠️  CRÍTICO: Corriente muy baja. Verificar Rs o escala.")
    elif metrics['i_mean_A'] < 0.3:
        issues.append("⚠️  ADVERTENCIA: Corriente baja para levitación típica.")
        
    # Check 3: ¿Duración suficiente?
    if duration < 5:
        issues.append("⚠️  ADVERTENCIA: Datos muy cortos. Ideal > 30s con múltiples setpoints.")
        
    # Check 4: ¿Hay dinámica visible?
    dy_dt = np.gradient(y, t)
    if np.max(np.abs(dy_dt)) < 0.001:
        issues.append("⚠️  CRÍTICO: Sin dinámica observable. El sistema parece estático.")
        metrics['quality'] = 'BAD'
    
    if not issues:
        print("✅ Datos de buena calidad para identificación.")
        metrics['quality'] = 'GOOD'
    else:
        for issue in issues:
            print(f"   {issue}")
        if 'quality' not in metrics:
            metrics['quality'] = 'FAIR'
    
    return metrics


def extract_dynamic_segments(data: dict, min_velocity_mm_s: float = 0.5) -> list:
    """
    Extrae segmentos con dinámica activa (cambios de setpoint).
    """
    t = data['t']
    y = data['y'] * 1000  # mm
    
    dy_dt = np.abs(np.gradient(y, t))
    
    # Encontrar índices donde hay movimiento
    moving = dy_dt > min_velocity_mm_s
    
    # Expandir regiones para incluir contexto
    segments = []
    in_segment = False
    start_idx = 0
    
    for i in range(len(moving)):
        if moving[i] and not in_segment:
            start_idx = max(0, i - 50)  # 50 muestras de contexto
            in_segment = True
        elif not moving[i] and in_segment:
            end_idx = min(len(moving) - 1, i + 100)  # 100 muestras post-transición
            if end_idx - start_idx > 100:  # Segmento mínimo
                segments.append((start_idx, end_idx))
            in_segment = False
    
    return segments


def plot_data(data: dict, output_path: str = None):
    """
    Visualiza los datos para inspección.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    t = data['t']
    
    # Posición
    ax = axes[0]
    ax.plot(t, data['y'] * 1000, 'b-', label='y (sensor)', linewidth=0.8)
    if 'yd' in data:
        ax.plot(t, data['yd'] * 1000, 'r--', label='yd (setpoint)', linewidth=1)
    if 'y_est' in data:
        ax.plot(t, data['y_est'] * 1000, 'g-', label='y_est', alpha=0.7, linewidth=0.8)
    ax.set_ylabel('Posición [mm]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title('Datos del Levitador Magnético')
    
    # Corriente
    ax = axes[1]
    ax.plot(t, data['i'], 'orange', linewidth=0.8)
    ax.set_ylabel('Corriente [A]')
    ax.grid(True, alpha=0.3)
    
    # Voltaje
    ax = axes[2]
    ax.plot(t, data['u'], 'purple', linewidth=0.8)
    ax.set_ylabel('Voltaje [V]')
    ax.set_xlabel('Tiempo [s]')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"\n📊 Gráfica guardada en: {output_path}")
    
    plt.show()


def convert_to_benchmark_format(data: dict, output_path: str):
    """
    Convierte datos al formato esperado por el benchmark JAX.
    Formato: t, y, i, u
    """
    t = data['t']
    y = data['y']
    i = data['i']
    u = data['u']
    
    # Stack en matriz
    output = np.column_stack([t, y, i, u])
    
    # Guardar
    np.savetxt(output_path, output, fmt='%.6f', delimiter='\t',
               header='t[s]\ty[m]\ti[A]\tu[V]', comments='')
    
    print(f"\n💾 Datos convertidos guardados en: {output_path}")
    print(f"   Formato: t, y, i, u")
    print(f"   Muestras: {len(t)}")


def print_acquisition_protocol():
    """
    Imprime el protocolo de adquisición de datos de calibración.
    """
    print("""
╔══════════════════════════════════════════════════════════════════╗
║     PROTOCOLO DE ADQUISICIÓN DE DATOS DE CALIBRACIÓN            ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  OBJETIVO: Capturar datos con DINÁMICA para identificación       ║
║                                                                   ║
║  PASOS:                                                           ║
║                                                                   ║
║  1. Iniciar levitador_sensorless_kan.exe                         ║
║     - Modo recomendado: MAESTRO (M) para datos con sensor real   ║
║                                                                   ║
║  2. Esperar estabilización inicial (~5s en setpoint 5mm)         ║
║                                                                   ║
║  3. SECUENCIA DE SETPOINTS (presionar teclas):                   ║
║     ┌─────────┬───────────┬────────────┐                         ║
║     │ Tecla   │ Setpoint  │ Esperar    │                         ║
║     ├─────────┼───────────┼────────────┤                         ║
║     │   2     │   5mm     │   10s      │                         ║
║     │   1     │   4mm     │   10s      │                         ║
║     │   3     │   6mm     │   10s      │                         ║
║     │   2     │   5mm     │   10s      │                         ║
║     │   4     │   7mm     │   10s      │                         ║
║     │   2     │   5mm     │   10s      │                         ║
║     │   5     │   8mm     │   10s      │                         ║
║     │   2     │   5mm     │   10s      │                         ║
║     └─────────┴───────────┴────────────┘                         ║
║                                                                   ║
║  4. Presionar 'Q' para terminar y guardar                        ║
║                                                                   ║
║  DURACIÓN TOTAL: ~80 segundos                                    ║
║  ARCHIVO DE SALIDA: datos_levitador_[timestamp].txt              ║
║                                                                   ║
║  IMPORTANTE:                                                      ║
║  - Los cambios de setpoint generan la DINÁMICA necesaria         ║
║  - Cada transición revela información sobre K0, A, R0            ║
║  - Más transiciones = mejor identificación                       ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(
        description='Preparador de datos de calibración para benchmark JAX'
    )
    parser.add_argument('--input', '-i', type=str, 
                        help='Archivo de datos crudos del levitador')
    parser.add_argument('--output', '-o', type=str,
                        default='data/calibration_data.txt',
                        help='Archivo de salida para benchmark')
    parser.add_argument('--plot', action='store_true',
                        help='Mostrar gráfica de datos')
    parser.add_argument('--protocol', action='store_true',
                        help='Mostrar protocolo de adquisición')
    
    args = parser.parse_args()
    
    if args.protocol or not args.input:
        print_acquisition_protocol()
        if not args.input:
            return
    
    # Cargar y analizar
    data = load_raw_data(args.input)
    metrics = analyze_data_quality(data)
    
    # Graficar
    if args.plot:
        plot_path = args.output.replace('.txt', '_preview.png')
        plot_data(data, plot_path)
    
    # Convertir formato
    if metrics.get('quality') != 'BAD':
        convert_to_benchmark_format(data, args.output)
        print("\n✅ Datos listos para benchmark JAX:")
        print(f"   python scripts/benchmark_jax.py --data {args.output}")
    else:
        print("\n❌ DATOS NO APTOS para identificación.")
        print("   Por favor, adquiera nuevos datos siguiendo el protocolo.")
        print("   Ejecute: python scripts/prepare_calibration_data.py --protocol")


if __name__ == '__main__':
    main()
