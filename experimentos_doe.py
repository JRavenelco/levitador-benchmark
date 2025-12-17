"""
Automatización de Experimentos DOE - Levitador Magnético
=========================================================

Este script ejecuta automáticamente la batería de experimentos
definida en el DOE para caracterizar el sistema.

Uso:
    python experimentos_doe.py --fase 1
    python experimentos_doe.py --experimento E01
    python experimentos_doe.py --todos

Autor: Jesús (Doctorado UAQ)
"""

import asyncio
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Optional, List, Dict
import matplotlib.pyplot as plt

# Importar el sistema de adquisición
try:
    from adquisicion_datos import LevitadorDataAcquisition
except ImportError:
    print("⚠️  Ejecutar desde el directorio del benchmark")


@dataclass
class Experimento:
    """Define un experimento del DOE."""
    id: str
    nombre: str
    tipo: str
    y0: float  # Posición inicial [m]
    y1: float  # Posición final [m] (o amplitud para senoidal)
    duracion: float  # Segundos
    frecuencia: float = 0.0  # Hz (solo para senoidal)
    t_transicion: float = 5.0  # Tiempo antes del cambio [s]
    descripcion: str = ""


# =============================================================================
# PERFILES DE REFERENCIA PARA CADA TIPO DE EXPERIMENTO
# =============================================================================

def perfil_escalon(t: float, exp: Experimento) -> float:
    """Escalón: y0 → y1 en t_transicion."""
    if t < exp.t_transicion:
        return exp.y0
    else:
        return exp.y1


def perfil_rampa(t: float, exp: Experimento) -> float:
    """Rampa: transición lineal de y0 a y1."""
    t_inicio = exp.t_transicion
    t_fin = exp.duracion - 5  # Dejar 5s al final
    
    if t < t_inicio:
        return exp.y0
    elif t < t_fin:
        progreso = (t - t_inicio) / (t_fin - t_inicio)
        return exp.y0 + progreso * (exp.y1 - exp.y0)
    else:
        return exp.y1


def perfil_senoidal(t: float, exp: Experimento) -> float:
    """Senoidal: oscila alrededor de y0 con amplitud (y1-y0)."""
    if t < exp.t_transicion:
        return exp.y0
    else:
        amplitud = abs(exp.y1 - exp.y0)
        return exp.y0 + amplitud * np.sin(2 * np.pi * exp.frecuencia * (t - exp.t_transicion))


def perfil_pulso(t: float, exp: Experimento) -> float:
    """Pulso: alterna entre y0 y y1 cada 5 segundos."""
    if t < exp.t_transicion:
        return exp.y0
    else:
        t_rel = t - exp.t_transicion
        periodo = 5.0  # 5 segundos por pulso
        if (t_rel // periodo) % 2 == 0:
            return exp.y0
        else:
            return exp.y1


def perfil_escalera(t: float, exp: Experimento) -> float:
    """Escalera: pasos discretos de y0 a y1."""
    if t < exp.t_transicion:
        return exp.y0
    
    n_pasos = 4
    t_total = exp.duracion - exp.t_transicion - 5
    t_paso = t_total / n_pasos
    t_rel = t - exp.t_transicion
    
    paso_actual = min(int(t_rel / t_paso), n_pasos - 1)
    delta = (exp.y1 - exp.y0) / (n_pasos - 1)
    
    return exp.y0 + paso_actual * delta


def perfil_prbs(t: float, exp: Experimento, seed: int = 42) -> float:
    """PRBS: señal pseudo-aleatoria binaria."""
    if t < exp.t_transicion:
        return exp.y0
    
    np.random.seed(seed + int(t * 10))  # Reproducible pero variante
    periodo = 0.5  # Cambio cada 0.5s
    t_rel = t - exp.t_transicion
    idx = int(t_rel / periodo)
    np.random.seed(seed + idx)
    
    amplitud = abs(exp.y1 - exp.y0)
    if np.random.random() > 0.5:
        return exp.y0 + amplitud
    else:
        return exp.y0 - amplitud


# =============================================================================
# DEFINICIÓN DE EXPERIMENTOS DOE
# =============================================================================

EXPERIMENTOS_DOE = {
    # Fase 1: Caracterización Básica
    "E01": Experimento("E01", "Escalon Descendente", "escalon", 
                       0.005, 0.004, 30, descripcion="Escalón 5→4mm"),
    "E02": Experimento("E02", "Escalon Ascendente", "escalon",
                       0.004, 0.005, 30, descripcion="Escalón 4→5mm"),
    "E03": Experimento("E03", "Escalon Grande Desc", "escalon",
                       0.005, 0.0035, 30, descripcion="Escalón 5→3.5mm"),
    "E04": Experimento("E04", "Escalon Grande Asc", "escalon",
                       0.0035, 0.005, 30, descripcion="Escalón 3.5→5mm"),
    "E05": Experimento("E05", "Rampa Descendente", "rampa",
                       0.005, 0.004, 30, descripcion="Rampa 5→4mm"),
    "E06": Experimento("E06", "Rampa Ascendente", "rampa",
                       0.004, 0.005, 30, descripcion="Rampa 4→5mm"),
    "E07": Experimento("E07", "Senoidal 0.1Hz", "senoidal",
                       0.005, 0.0055, 40, frecuencia=0.1, descripcion="Seno lento"),
    "E08": Experimento("E08", "Senoidal 0.5Hz", "senoidal",
                       0.005, 0.0055, 30, frecuencia=0.5, descripcion="Seno medio"),
    "E09": Experimento("E09", "Senoidal 1.0Hz", "senoidal",
                       0.005, 0.0055, 20, frecuencia=1.0, descripcion="Seno rápido"),
    "E10": Experimento("E10", "Pulsos", "pulso",
                       0.005, 0.004, 30, descripcion="Pulsos cada 5s"),
    "E11": Experimento("E11", "Escalera", "escalera",
                       0.005, 0.003, 40, descripcion="4 escalones"),
    "E12": Experimento("E12", "PRBS", "prbs",
                       0.005, 0.0055, 60, descripcion="Pseudo-aleatorio"),
}

# Validaciones (repeticiones)
for i in range(1, 4):
    EXPERIMENTOS_DOE[f"V0{i}"] = Experimento(
        f"V0{i}", f"Validacion Escalon {i}", "escalon",
        0.005, 0.004, 30, descripcion=f"Repetición {i} de E01"
    )
    EXPERIMENTOS_DOE[f"V0{i+3}"] = Experimento(
        f"V0{i+3}", f"Validacion Senoidal {i}", "senoidal",
        0.005, 0.0055, 30, frecuencia=0.5, descripcion=f"Repetición {i} de E08"
    )

PERFILES = {
    "escalon": perfil_escalon,
    "rampa": perfil_rampa,
    "senoidal": perfil_senoidal,
    "pulso": perfil_pulso,
    "escalera": perfil_escalera,
    "prbs": perfil_prbs,
}


# =============================================================================
# EJECUCIÓN DE EXPERIMENTOS
# =============================================================================

class DOERunner:
    """Ejecutor de experimentos DOE."""
    
    def __init__(self, puerto: str = 'COM1', output_dir: str = 'data/doe'):
        self.puerto = puerto
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resultados = []
    
    async def ejecutar_experimento(self, exp: Experimento, simular: bool = False) -> Optional[pd.DataFrame]:
        """Ejecuta un experimento individual."""
        print(f"\n{'='*60}")
        print(f"🔬 Experimento: {exp.id} - {exp.nombre}")
        print(f"   Tipo: {exp.tipo}")
        print(f"   y₀: {exp.y0*1000:.1f}mm → y₁: {exp.y1*1000:.1f}mm")
        print(f"   Duración: {exp.duracion}s")
        if exp.frecuencia > 0:
            print(f"   Frecuencia: {exp.frecuencia}Hz")
        print(f"{'='*60}")
        
        # Crear función de perfil
        perfil_fn = PERFILES.get(exp.tipo, perfil_escalon)
        perfil = lambda t: perfil_fn(t, exp)
        
        if simular:
            # Modo simulación (sin hardware)
            df = self._simular_experimento(exp, perfil)
        else:
            # Modo real
            daq = LevitadorDataAcquisition(port=self.puerto)
            daq.set_reference(exp.y0)
            df = await daq.acquire(exp.duracion, reference_profile=perfil)
        
        if df is not None and len(df) > 0:
            # Guardar datos
            filepath = self._guardar_datos(exp, df)
            
            # Calcular métricas
            metricas = self._calcular_metricas(exp, df)
            self.resultados.append({'experimento': exp.id, **metricas})
            
            print(f"\n✅ Completado: {len(df)} muestras guardadas")
            print(f"   Archivo: {filepath}")
            
            return df
        else:
            print(f"❌ Error en experimento {exp.id}")
            return None
    
    def _simular_experimento(self, exp: Experimento, perfil: Callable) -> pd.DataFrame:
        """Simula un experimento sin hardware."""
        print("   ⚠️  MODO SIMULACIÓN (sin hardware)")
        
        Ts = 0.01
        t = np.arange(0, exp.duracion, Ts)
        yd = np.array([perfil(ti) for ti in t])
        
        # Simular respuesta con algo de ruido y dinámica
        y = np.zeros_like(yd)
        tau = 0.05  # Constante de tiempo
        for i in range(1, len(t)):
            y[i] = y[i-1] + (yd[i-1] - y[i-1]) * Ts / tau
        y += np.random.normal(0, 1e-5, len(t))
        
        # Corriente aproximada
        i_sim = 0.5 + 0.3 * (0.005 - y) / 0.005 + np.random.normal(0, 0.01, len(t))
        
        return pd.DataFrame({
            't': t,
            'y': y,
            'yd': yd,
            'i': i_sim,
            'id': i_sim * 0.95,
            'u': 5 + 3 * (0.005 - y) / 0.005
        })
    
    def _guardar_datos(self, exp: Experimento, df: pd.DataFrame) -> str:
        """Guarda datos con nomenclatura estándar DOE."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Crear nombre descriptivo
        y0_str = f"{exp.y0*1000:.0f}mm"
        y1_str = f"{exp.y1*1000:.0f}mm"
        
        if exp.frecuencia > 0:
            filename = f"exp_{exp.id}_{exp.tipo}_{y0_str}_{exp.frecuencia}hz_{timestamp}.txt"
        else:
            filename = f"exp_{exp.id}_{exp.tipo}_{y0_str}_{y1_str}_{timestamp}.txt"
        
        filepath = self.output_dir / filename
        
        # Guardar en formato compatible
        df.to_csv(filepath, sep='\t', index=False, header=False,
                  columns=['t', 'yd', 'y', 'id', 'i', 'u'])
        
        return str(filepath)
    
    def _calcular_metricas(self, exp: Experimento, df: pd.DataFrame) -> Dict:
        """Calcula métricas de respuesta del sistema."""
        t = df['t'].values
        y = df['y'].values
        yd = df['yd'].values
        error = yd - y
        
        # ISE e IAE
        Ts = t[1] - t[0] if len(t) > 1 else 0.01
        ise = np.sum(error**2) * Ts
        iae = np.sum(np.abs(error)) * Ts
        
        # Error en estado estable (últimos 20% de datos)
        n_ss = int(0.2 * len(error))
        ess = np.mean(np.abs(error[-n_ss:])) if n_ss > 0 else np.nan
        
        # Para escalones, calcular tiempo de subida y asentamiento
        tr, ts, mp = np.nan, np.nan, np.nan
        
        if exp.tipo == "escalon":
            # Buscar cambio de referencia
            dyd = np.diff(yd)
            idx_cambio = np.where(np.abs(dyd) > 1e-6)[0]
            
            if len(idx_cambio) > 0:
                idx_start = idx_cambio[0]
                y_inicial = y[idx_start]
                y_final = yd[-1]
                delta_y = y_final - y_inicial
                
                if abs(delta_y) > 1e-6:
                    # Tiempo de subida (10% a 90%)
                    y_10 = y_inicial + 0.1 * delta_y
                    y_90 = y_inicial + 0.9 * delta_y
                    
                    try:
                        if delta_y > 0:
                            idx_10 = np.where(y[idx_start:] >= y_10)[0][0] + idx_start
                            idx_90 = np.where(y[idx_start:] >= y_90)[0][0] + idx_start
                        else:
                            idx_10 = np.where(y[idx_start:] <= y_10)[0][0] + idx_start
                            idx_90 = np.where(y[idx_start:] <= y_90)[0][0] + idx_start
                        tr = t[idx_90] - t[idx_10]
                    except IndexError:
                        pass
                    
                    # Sobreimpulso
                    y_post = y[idx_start:]
                    if delta_y > 0:
                        mp = (np.max(y_post) - y_final) / abs(delta_y) * 100
                    else:
                        mp = (y_final - np.min(y_post)) / abs(delta_y) * 100
                    mp = max(0, mp)
                    
                    # Tiempo de asentamiento (±2%)
                    banda = 0.02 * abs(delta_y)
                    dentro_banda = np.abs(y[idx_start:] - y_final) < banda
                    if np.any(dentro_banda):
                        idx_ts = np.where(dentro_banda)[0]
                        # Encontrar cuando se queda dentro permanentemente
                        for i in range(len(idx_ts) - 1, -1, -1):
                            if np.all(dentro_banda[idx_ts[i]:]):
                                ts = t[idx_start + idx_ts[i]] - t[idx_start]
                                break
        
        return {
            'ISE': ise,
            'IAE': iae,
            'ess': ess,
            'tr': tr,
            'ts': ts,
            'Mp': mp,
        }
    
    async def ejecutar_fase(self, fase: int, simular: bool = False):
        """Ejecuta todos los experimentos de una fase."""
        if fase == 1:
            ids = ["E01", "E02", "E07", "E08", "E11"]
        elif fase == 2:
            ids = ["E03", "E04", "E05", "E06", "E09", "E10"]
        elif fase == 3:
            ids = [f"V0{i}" for i in range(1, 7)]
        elif fase == 4:
            ids = ["E12"]  # Solo PRBS como "robustez"
        else:
            print(f"❌ Fase {fase} no válida (usar 1-4)")
            return
        
        print(f"\n{'#'*60}")
        print(f"# FASE {fase}: {len(ids)} experimentos")
        print(f"{'#'*60}")
        
        for exp_id in ids:
            if exp_id in EXPERIMENTOS_DOE:
                await self.ejecutar_experimento(EXPERIMENTOS_DOE[exp_id], simular)
                await asyncio.sleep(2)  # Pausa entre experimentos
        
        self._mostrar_resumen()
    
    async def ejecutar_todos(self, simular: bool = False):
        """Ejecuta todos los experimentos del DOE."""
        for fase in [1, 2, 3, 4]:
            await self.ejecutar_fase(fase, simular)
    
    def _mostrar_resumen(self):
        """Muestra resumen de resultados."""
        if not self.resultados:
            return
        
        print(f"\n{'='*60}")
        print("📊 RESUMEN DE RESULTADOS")
        print(f"{'='*60}")
        
        df = pd.DataFrame(self.resultados)
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4e}" if pd.notna(x) else "N/A"))
        
        # Guardar resumen
        resumen_path = self.output_dir / "resumen_doe.csv"
        df.to_csv(resumen_path, index=False)
        print(f"\n💾 Resumen guardado: {resumen_path}")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

def visualizar_experimento(filepath: str):
    """Visualiza los datos de un experimento."""
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['t', 'yd', 'y', 'id', 'i', 'u'])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(df['t'], df['y']*1000, 'b-', label='y medida')
    axes[0].plot(df['t'], df['yd']*1000, 'r--', label='yd referencia')
    axes[0].set_ylabel('Posición [mm]')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Experimento: {Path(filepath).stem}')
    
    axes[1].plot(df['t'], df['i'], 'g-', label='i medida')
    axes[1].plot(df['t'], df['id'], 'r--', label='id referencia')
    axes[1].set_ylabel('Corriente [A]')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df['t'], df['u'], 'orange')
    axes[2].set_xlabel('Tiempo [s]')
    axes[2].set_ylabel('Voltaje [V]')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath.replace('.txt', '.png'), dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='DOE del Levitador Magnético')
    parser.add_argument('--puerto', default='COM1', help='Puerto serie')
    parser.add_argument('--fase', type=int, help='Ejecutar fase específica (1-4)')
    parser.add_argument('--experimento', help='Ejecutar experimento específico (ej: E01)')
    parser.add_argument('--todos', action='store_true', help='Ejecutar todos los experimentos')
    parser.add_argument('--simular', action='store_true', help='Modo simulación (sin hardware)')
    parser.add_argument('--listar', action='store_true', help='Listar experimentos disponibles')
    args = parser.parse_args()
    
    if args.listar:
        print("\n📋 EXPERIMENTOS DISPONIBLES")
        print("="*60)
        for exp_id, exp in EXPERIMENTOS_DOE.items():
            print(f"  {exp_id}: {exp.nombre} ({exp.tipo}) - {exp.descripcion}")
        return
    
    runner = DOERunner(puerto=args.puerto)
    
    if args.experimento:
        if args.experimento in EXPERIMENTOS_DOE:
            await runner.ejecutar_experimento(
                EXPERIMENTOS_DOE[args.experimento], 
                simular=args.simular
            )
        else:
            print(f"❌ Experimento '{args.experimento}' no encontrado")
            print("   Usa --listar para ver experimentos disponibles")
    elif args.fase:
        await runner.ejecutar_fase(args.fase, simular=args.simular)
    elif args.todos:
        await runner.ejecutar_todos(simular=args.simular)
    else:
        parser.print_help()
        print("\n💡 Ejemplos:")
        print("   python experimentos_doe.py --listar")
        print("   python experimentos_doe.py --fase 1 --simular")
        print("   python experimentos_doe.py --experimento E01")
        print("   python experimentos_doe.py --todos --simular")


if __name__ == '__main__':
    asyncio.run(main())
