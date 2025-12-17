"""
Levitador Magnético Benchmark
==============================
Problema de optimización para algoritmos bio-inspirados y metaheurísticas.

Objetivo: Identificar los parámetros físicos (k0, k, a) de la función de 
inductancia no lineal L(y) = k0 + k/(1 + y/a) que minimizan el error entre 
la simulación del gemelo digital y los datos experimentales reales.

Autor: Jesús (Doctorado)
Licencia: MIT
"""

import numpy as np
from scipy.integrate import odeint
from pathlib import Path
from typing import Tuple, List, Optional


class LevitadorBenchmark:
    """
    Problema de Optimización: Identificación de Parámetros del Levitador Magnético.
    
    El objetivo es encontrar los parámetros físicos (k0, k, a) de la función 
    de inductancia no lineal L(y) que minimicen el error entre la simulación
    y los datos reales.
    
    Modelo de Inductancia:
        L(y) = k0 + k / (1 + y/a)
    
    Donde:
        - k0: Inductancia base [H]
        - k:  Coeficiente de inductancia [H]
        - a:  Parámetro geométrico [m]
        - y:  Posición de la esfera [m]
    
    Uso típico:
        >>> problema = LevitadorBenchmark("datos_levitador.txt")
        >>> error = problema.fitness_function([0.036, 0.0035, 0.005])
    """
    
    def __init__(self, datos_reales_path: Optional[str] = None):
        """
        Inicializa el benchmark.
        
        Args:
            datos_reales_path: Ruta al archivo de datos experimentales.
                               Si es None, usa datos sintéticos de ejemplo.
        """
        # Constantes Físicas Fijas (no se optimizan) - DEFINIR PRIMERO
        self.m = 0.018  # Masa de la esfera [kg]
        self.g = 9.81   # Gravedad [m/s²]
        self.R = 2.72   # Resistencia de la bobina [Ω]
        
        # 1. Cargar datos reales (Ground Truth)
        if datos_reales_path and Path(datos_reales_path).exists():
            self._load_real_data(datos_reales_path)
        else:
            self._generate_synthetic_data()

        # 2. Definir el Espacio de Búsqueda (Límites)
        # Basados en rangos físicamente razonables
        self.bounds = [
            (1e-4, 0.1),   # k0: Inductancia base [H]
            (1e-4, 0.1),   # k:  Coeficiente [H]
            (1e-4, 0.05)   # a:  Parámetro geométrico [m]
        ]
        self.dim = 3  # Dimensión del problema
        
        # Nombres de las variables (para visualización)
        self.variable_names = ['k0', 'k', 'a']
        
        # Valores de referencia (si se conocen)
        self.reference_solution = [0.0363, 0.0035, 0.0052]
    
    def _load_real_data(self, path: str):
        """Carga datos experimentales desde archivo."""
        import pandas as pd
        
        # Formato típico: t, yd, y, ied, ie, u (separado por tabs)
        try:
            df = pd.read_csv(path, sep=r"\t|→", engine="python", header=None, dtype=float)
            if df.shape[1] >= 6:
                df = df.iloc[:, -6:]
            df.columns = ["t", "yd", "y", "ied", "ie", "u"]
            
            self.t_real = df["t"].to_numpy(dtype=np.float64)
            self.y_real = df["y"].to_numpy(dtype=np.float64)
            self.i_real = df["ie"].to_numpy(dtype=np.float64)
            self.u_real = df["u"].to_numpy(dtype=np.float64)
            
            print(f"✓ Datos cargados: {len(self.t_real)} puntos")
            print(f"  Rango temporal: [{self.t_real[0]:.3f}, {self.t_real[-1]:.3f}] s")
            print(f"  Rango posición: [{self.y_real.min()*1000:.2f}, {self.y_real.max()*1000:.2f}] mm")
            
        except Exception as e:
            print(f"⚠ Error cargando datos: {e}")
            print("  Usando datos sintéticos...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Genera datos sintéticos para pruebas."""
        print("ℹ Usando datos sintéticos de ejemplo")
        
        # Parámetros "objetivo" ocultos
        k0_true, k_true, a_true = 0.0363, 0.0035, 0.0052
        
        # Tiempo de simulación
        self.t_real = np.linspace(0, 0.5, 500)
        
        # Voltaje: escalón de 12V
        self.u_real = 12.0 * np.ones_like(self.t_real)
        
        # Simular respuesta "verdadera"
        estado0 = [0.015, 0.0, 0.0]  # y0=15mm, v0=0, i0=0
        
        def modelo_true(estado, t):
            y, v, i = estado
            u = 12.0
            denom = 1 + (y / a_true)
            L = k0_true + (k_true / denom)
            dL_dy = -(k_true / (a_true * denom**2))
            L = max(L, 1e-6)
            
            F_mag = 0.5 * dL_dy * (i**2)
            dy_dt = v
            dv_dt = (F_mag + self.m * self.g) / self.m
            di_dt = (u - self.R*i - i*dL_dy*v) / L
            
            return [dy_dt, dv_dt, di_dt]
        
        sol = odeint(modelo_true, estado0, self.t_real)
        
        # Añadir ruido realista
        self.y_real = sol[:, 0] + np.random.normal(0, 1e-5, len(self.t_real))
        self.i_real = sol[:, 2] + np.random.normal(0, 0.01, len(self.t_real))

    def _modelo_dinamico(self, estado, t, k0, k, a):
        """
        Modelo dinámico del levitador (Ecuaciones de Euler-Lagrange).
        
        Args:
            estado: [y, v, i] - posición, velocidad, corriente
            t: tiempo actual
            k0, k, a: parámetros de inductancia a evaluar
        
        Returns:
            [dy/dt, dv/dt, di/dt]
        """
        y, v, i = estado
        
        # Limitar posición para evitar singularidades
        y = max(y, 1e-6)
        a = max(a, 1e-6)
        
        # Interpolamos el voltaje real al tiempo t actual
        u = np.interp(t, self.t_real, self.u_real)
        
        # Modelo de Inductancia (lo que queremos optimizar)
        denom = 1 + (y / a)
        denom = max(denom, 1e-6)  # Evitar división por cero
        L = k0 + (k / denom)
        dL_dy = -(k / (a * denom**2))
        
        # Evitar valores extremos
        L = max(L, 1e-6)
        dL_dy = np.clip(dL_dy, -1e6, 1e6)
        
        # Ecuación Mecánica (Newton)
        F_mag = 0.5 * dL_dy * (i**2)  # Fuerza magnética
        dy_dt = v
        dv_dt = (F_mag + self.m * self.g) / self.m
        
        # Ecuación Eléctrica (Kirchhoff)
        di_dt = (u - self.R*i - i*dL_dy*v) / L
        
        # Limitar derivadas para estabilidad numérica
        dv_dt = np.clip(dv_dt, -1e4, 1e4)
        di_dt = np.clip(di_dt, -1e6, 1e6)
        
        return [dy_dt, dv_dt, di_dt]

    def fitness_function(self, individuo: List[float]) -> float:
        """
        FUNCIÓN OBJETIVO PARA ALGORITMOS BIO-INSPIRADOS.
        
        Esta es la función que los algoritmos genéticos, PSO, DE, etc.
        deben minimizar.
        
        Args:
            individuo: Vector [k0, k, a] con los parámetros a evaluar
        
        Returns:
            Error (MSE) entre simulación y datos reales.
            Valores más bajos = mejor ajuste.
            Retorna 1e9 si hay error o violación de restricciones.
        
        Ejemplo:
            >>> error = problema.fitness_function([0.036, 0.0035, 0.005])
            >>> print(f"MSE: {error:.6e}")
        """
        k0, k, a = individuo
        
        # Validar restricciones físicas (valores positivos)
        if any(x <= 0 for x in individuo):
            return 1e9  # Penalización: individuo inválido
        
        # Verificar que está dentro de los límites
        for val, (lb, ub) in zip(individuo, self.bounds):
            if val < lb or val > ub:
                return 1e9  # Fuera de límites

        # Condición inicial (tomada de los datos reales)
        y0 = self.y_real[0] if len(self.y_real) > 0 else 0.01
        estado_inicial = [y0, 0.0, 0.0]  # [y0, v0, i0]
        
        try:
            # Simular dinámica con los parámetros propuestos
            solucion = odeint(
                self._modelo_dinamico, 
                estado_inicial, 
                self.t_real, 
                args=(k0, k, a),
                full_output=False
            )
            
            y_sim = solucion[:, 0]
            
            # Calcular Error Cuadrático Medio (MSE)
            error = np.mean((self.y_real - y_sim)**2)
            
            # Si la simulación explota (NaNs o infinitos), penalizar
            if np.isnan(error) or np.isinf(error):
                return 1e9
            
            return float(error)
            
        except Exception:
            return 1e9  # Si falla el integrador, penalizar

    def evaluate_batch(self, population: np.ndarray) -> np.ndarray:
        """
        Evalúa una población completa (útil para algoritmos paralelos).
        
        Args:
            population: Matriz (n_individuos, 3) con parámetros
        
        Returns:
            Vector de fitness para cada individuo
        """
        return np.array([self.fitness_function(ind) for ind in population])

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna límites en formato array (útil para scipy.optimize).
        
        Returns:
            (lower_bounds, upper_bounds)
        """
        bounds_arr = np.array(self.bounds)
        return bounds_arr[:, 0], bounds_arr[:, 1]

    def visualize_solution(self, individuo: List[float], save_path: str = None):
        """
        Visualiza la solución comparando simulación vs datos reales.
        
        Args:
            individuo: [k0, k, a] solución a visualizar
            save_path: Ruta para guardar la figura (opcional)
        """
        import matplotlib.pyplot as plt
        
        k0, k, a = individuo
        
        # Simular
        y0 = self.y_real[0]
        sol = odeint(self._modelo_dinamico, [y0, 0, 0], self.t_real, args=(k0, k, a))
        y_sim = sol[:, 0]
        
        # Calcular error
        mse = np.mean((self.y_real - y_sim)**2)
        
        # Graficar
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.t_real * 1000, self.y_real * 1000, 'b-', 
                label='Datos Reales', linewidth=2)
        ax.plot(self.t_real * 1000, y_sim * 1000, 'r--', 
                label='Simulación', linewidth=2)
        
        ax.set_xlabel('Tiempo [ms]', fontsize=12)
        ax.set_ylabel('Posición [mm]', fontsize=12)
        ax.set_title(f'Levitador Magnético - MSE: {mse:.2e}\n'
                     f'k0={k0:.4f}, k={k:.4f}, a={a:.4f}', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Figura guardada en: {save_path}")
        
        plt.show()

    def __repr__(self):
        return (f"LevitadorBenchmark(dim={self.dim}, "
                f"n_samples={len(self.t_real)}, "
                f"bounds={self.bounds})")


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def create_benchmark(data_path: str = None) -> LevitadorBenchmark:
    """
    Función de fábrica para crear el benchmark.
    
    Args:
        data_path: Ruta a datos experimentales (opcional)
    
    Returns:
        Instancia de LevitadorBenchmark lista para usar
    """
    return LevitadorBenchmark(data_path)


def run_quick_test():
    """Ejecuta una prueba rápida del benchmark."""
    print("=" * 60)
    print("LEVITADOR MAGNÉTICO - TEST RÁPIDO")
    print("=" * 60)
    
    # Crear benchmark con datos sintéticos
    benchmark = LevitadorBenchmark()
    
    print(f"\n{benchmark}")
    print(f"\nEspacio de búsqueda:")
    for name, (lb, ub) in zip(benchmark.variable_names, benchmark.bounds):
        print(f"  {name}: [{lb}, {ub}]")
    
    # Evaluar solución de referencia
    ref = benchmark.reference_solution
    error_ref = benchmark.fitness_function(ref)
    print(f"\nSolución de referencia: {ref}")
    print(f"Error (MSE): {error_ref:.6e}")
    
    # Evaluar una solución aleatoria
    np.random.seed(42)
    random_sol = [np.random.uniform(lb, ub) for lb, ub in benchmark.bounds]
    error_rand = benchmark.fitness_function(random_sol)
    print(f"\nSolución aleatoria: {[f'{x:.4f}' for x in random_sol]}")
    print(f"Error (MSE): {error_rand:.6e}")
    
    print("\n" + "=" * 60)
    print("✓ Test completado")
    

if __name__ == "__main__":
    run_quick_test()
