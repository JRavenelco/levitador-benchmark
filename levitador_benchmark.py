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
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Configurar logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)


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
    
    def __init__(self, datos_reales_path: Optional[str] = None, random_seed: Optional[int] = None,
                 noise_level: float = 1e-5, verbose: bool = True):
        """
        Inicializa el benchmark.
        
        Configura el problema de optimización cargando datos experimentales o 
        generando datos sintéticos, y define el espacio de búsqueda para los 
        parámetros de inductancia.

        Parameters
        ----------
        datos_reales_path : str, optional
            Ruta al archivo de datos experimentales. Si es None, se generan 
            datos sintéticos de ejemplo (default: None).
        random_seed : int, optional
            Semilla para el generador de números aleatorios para garantizar 
            reproducibilidad. Si es None, se usa un estado aleatorio 
            (default: None).
        noise_level : float, optional
            Nivel de ruido (desviación estándar) para datos sintéticos 
            (default: 1e-5).
        verbose : bool, optional
            Si True, muestra mensajes informativos durante la inicialización 
            (default: True).

        Notes
        -----
        El benchmark define automáticamente:
        - Espacio de búsqueda (bounds) para los parámetros [k0, k, a]
        - Datos de referencia (t_real, y_real, i_real, u_real)
        - Constantes físicas fijas (m, g, R)
        - Solución de referencia conocida
        """
        self._verbose = verbose
        self._noise_level = noise_level
        
        # Configurar generador de números aleatorios para reproducibilidad
        if random_seed is not None:
            self._rng = np.random.default_rng(random_seed)
        else:
            self._rng = np.random.default_rng()
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
        """
        Carga datos experimentales desde archivo.
        
        Lee un archivo de datos experimentales en formato tabular y extrae las 
        señales temporales necesarias para el benchmark: tiempo, posición, 
        corriente y voltaje.

        Parameters
        ----------
        path : str
            Ruta al archivo de datos experimentales. El archivo debe contener 
            al menos 6 columnas separadas por tabuladores o flechas: 
            [t, yd, y, ied, ie, u].

        Notes
        -----
        Si ocurre un error al cargar los datos, se generan automáticamente 
        datos sintéticos como respaldo.
        
        El formato esperado del archivo es:
        - Columnas: t, yd, y, ied, ie, u
        - Separadores: tabuladores o símbolo →
        - Sin encabezados
        """
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
            
            if self._verbose:
                logger.info(f"Datos cargados: {len(self.t_real)} puntos")
                logger.info(f"Rango temporal: [{self.t_real[0]:.3f}, {self.t_real[-1]:.3f}] s")
                logger.info(f"Rango posición: [{self.y_real.min()*1000:.2f}, {self.y_real.max()*1000:.2f}] mm")
            
        except Exception as e:
            logger.warning(f"Error cargando datos: {e}")
            logger.warning("Usando datos sintéticos...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """
        Genera datos sintéticos para pruebas.
        
        Crea datos sintéticos de respuesta del sistema usando parámetros 
        conocidos (k0_true, k_true, a_true) y añade ruido gaussiano para 
        simular mediciones reales.

        Notes
        -----
        Los parámetros "verdaderos" ocultos usados son:
        - k0 = 0.0363 H (inductancia base)
        - k = 0.0035 H (coeficiente de inductancia)
        - a = 0.0052 m (parámetro geométrico)
        
        El nivel de ruido se controla mediante el atributo `_noise_level` 
        definido en la inicialización.
        
        Los datos generados incluyen:
        - t_real : array de tiempos
        - y_real : posición de la esfera (con ruido)
        - i_real : corriente del electroimán (con ruido)
        - u_real : voltaje de entrada (escalón de 12V)
        """
        if self._verbose:
            logger.info("Usando datos sintéticos de ejemplo")
        
        # Parámetros "objetivo" ocultos
        k0_true, k_true, a_true = 0.0363, 0.0035, 0.0052
        
        # Tiempo de simulación
        self.t_real = np.linspace(0, 0.5, 500)
        
        # Voltaje: escalón de 12V
        self.u_real = 12.0 * np.ones_like(self.t_real)
        
        # Simular respuesta "verdadera"
        estado0 = [0.015, 0.0, 0.0]  # y0=15mm, v0=0, i0=0
        
        # Usamos el mismo modelo dinámico que en fitness_function para consistencia perfecta
        sol = odeint(
            self._modelo_dinamico, 
            estado0, 
            self.t_real, 
            args=(k0_true, k_true, a_true)
        )
        
        # Añadir ruido realista (usando el generador con semilla)
        self.y_real = sol[:, 0] + self._rng.normal(0, self._noise_level, len(self.t_real))
        self.i_real = sol[:, 2] + self._rng.normal(0, self._noise_level * 1000, len(self.t_real))

    def _modelo_dinamico(self, estado, t, k0, k, a):
        """
        Modelo dinámico del levitador (Ecuaciones de Euler-Lagrange).
        
        Implementa las ecuaciones diferenciales que gobiernan la dinámica del 
        sistema de levitación magnética, incluyendo la ecuación mecánica 
        (Newton) y la ecuación eléctrica (Kirchhoff).

        Parameters
        ----------
        estado : array_like
            Vector de estado [y, v, i] donde:
            - y : posición de la esfera [m]
            - v : velocidad de la esfera [m/s]
            - i : corriente en la bobina [A]
        t : float
            Tiempo actual [s]
        k0 : float
            Inductancia base [H]
        k : float
            Coeficiente de inductancia [H]
        a : float
            Parámetro geométrico [m]

        Returns
        -------
        list of float
            Vector de derivadas [dy/dt, dv/dt, di/dt] donde:
            - dy/dt : velocidad [m/s]
            - dv/dt : aceleración [m/s²]
            - di/dt : derivada de la corriente [A/s]

        Notes
        -----
        El modelo usa la función de inductancia no lineal:
        L(y) = k0 + k / (1 + y/a)
        
        Las ecuaciones son:
        - Mecánica: m*dv/dt = 0.5*dL/dy*i² + m*g
        - Eléctrica: L(y)*di/dt + dL/dy*v*i + R*i = u(t)
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
        Función objetivo para algoritmos bio-inspirados.
        
        Esta es la función que los algoritmos genéticos, PSO, DE, etc. deben 
        minimizar. Calcula el Error Cuadrático Medio (MSE) entre la trayectoria 
        simulada y los datos experimentales reales.

        Parameters
        ----------
        individuo : list of float
            Vector [k0, k, a] con los parámetros de inductancia a evaluar:
            - k0 : inductancia base [H]
            - k : coeficiente de inductancia [H]
            - a : parámetro geométrico [m]

        Returns
        -------
        float
            Error cuadrático medio (MSE) entre simulación y datos reales. 
            Valores más bajos indican mejor ajuste. Retorna 1e9 si hay error 
            numérico o violación de restricciones.

        Notes
        -----
        La función penaliza automáticamente:
        - Valores negativos o cero en cualquier parámetro
        - Parámetros fuera de los límites definidos en `self.bounds`
        - Errores numéricos (NaN, Inf) durante la simulación
        - Fallas en el integrador ODE

        Examples
        --------
        >>> problema = LevitadorBenchmark()
        >>> error = problema.fitness_function([0.036, 0.0035, 0.005])
        >>> print(f"MSE: {error:.6e}")
        MSE: 1.234567e-06
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
        
        Evalúa múltiples soluciones candidatas en un solo llamado, útil para 
        algoritmos de optimización basados en población como algoritmos 
        genéticos, PSO, o evolución diferencial.

        Parameters
        ----------
        population : np.ndarray
            Matriz de dimensiones (n_individuos, 3) donde cada fila contiene 
            los parámetros [k0, k, a] de un individuo.

        Returns
        -------
        np.ndarray
            Vector de dimensión (n_individuos,) con el valor de fitness (MSE) 
            para cada individuo. Valores más bajos indican mejor ajuste.

        Examples
        --------
        >>> problema = LevitadorBenchmark()
        >>> poblacion = np.array([[0.036, 0.0035, 0.005],
        ...                        [0.040, 0.0030, 0.004]])
        >>> fitness = problema.evaluate_batch(poblacion)
        >>> print(fitness)
        [1.23e-06 5.67e-05]
        """
        # For small populations, sequential is faster due to process overhead
        # Threshold determined empirically for typical fitness evaluation times
        if len(population) < 100:
            return np.array([self.fitness_function(ind) for ind in population])
        
        # Use n-1 cores to leave one free for the system
        n_cores = max(1, multiprocessing.cpu_count() - 1)
        
        # Convert to list for ProcessPoolExecutor.map
        pop_list = [ind for ind in population]
        
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            results = list(executor.map(self.fitness_function, pop_list))
            
        return np.array(results)

    def get_bounds_array(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retorna límites en formato array (útil para scipy.optimize).
        
        Convierte los límites del espacio de búsqueda al formato requerido por 
        optimizadores como scipy.optimize.differential_evolution.

        Returns
        -------
        lower_bounds : np.ndarray
            Array de dimensión (3,) con los límites inferiores [k0_min, k_min, a_min].
        upper_bounds : np.ndarray
            Array de dimensión (3,) con los límites superiores [k0_max, k_max, a_max].

        Examples
        --------
        >>> problema = LevitadorBenchmark()
        >>> lb, ub = problema.get_bounds_array()
        >>> print(f"k0: [{lb[0]}, {ub[0]}]")
        k0: [0.0001, 0.1]
        """
        bounds_arr = np.array(self.bounds)
        return bounds_arr[:, 0], bounds_arr[:, 1]

    def visualize_solution(self, individuo: List[float], save_path: str = None):
        """
        Visualiza la solución comparando simulación vs datos reales.
        
        Genera una gráfica que compara la trayectoria simulada usando los 
        parámetros propuestos contra los datos experimentales reales.

        Parameters
        ----------
        individuo : list of float
            Vector [k0, k, a] con los parámetros de inductancia a visualizar.
        save_path : str, optional
            Ruta donde guardar la figura. Si es None, solo se muestra sin 
            guardar (default: None).

        Notes
        -----
        La gráfica incluye:
        - Datos reales (línea azul continua)
        - Simulación con parámetros propuestos (línea roja discontinua)
        - Error MSE en el título
        - Valores de los parámetros en el título

        Examples
        --------
        >>> problema = LevitadorBenchmark()
        >>> solucion = [0.0363, 0.0035, 0.0052]
        >>> problema.visualize_solution(solucion, save_path="resultado.png")
        ✓ Figura guardada en: resultado.png
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
        """
        Representación en cadena del objeto LevitadorBenchmark.

        Returns
        -------
        str
            Cadena descriptiva con la dimensión del problema, número de 
            muestras y límites del espacio de búsqueda.
        """
        return (f"LevitadorBenchmark(dim={self.dim}, "
                f"n_samples={len(self.t_real)}, "
                f"bounds={self.bounds})")


# =============================================================================
# FUNCIONES DE CONVENIENCIA
# =============================================================================

def create_benchmark(data_path: str = None) -> LevitadorBenchmark:
    """
    Función de fábrica para crear el benchmark.
    
    Crea y retorna una instancia configurada de LevitadorBenchmark lista para 
    ser usada en algoritmos de optimización.

    Parameters
    ----------
    data_path : str, optional
        Ruta al archivo de datos experimentales. Si es None, se usan datos 
        sintéticos (default: None).

    Returns
    -------
    LevitadorBenchmark
        Instancia del benchmark lista para evaluar soluciones.

    Examples
    --------
    >>> # Con datos sintéticos
    >>> problema = create_benchmark()
    >>> 
    >>> # Con datos experimentales
    >>> problema = create_benchmark("data/datos_levitador.txt")
    """
    return LevitadorBenchmark(data_path)


def run_quick_test():
    """
    Ejecuta una prueba rápida del benchmark.
    
    Función de demostración que crea una instancia del benchmark con datos 
    sintéticos, muestra información del problema y evalúa tanto la solución 
    de referencia como una solución aleatoria.

    Notes
    -----
    Esta función es útil para:
    - Verificar que el benchmark funciona correctamente
    - Entender la estructura del problema
    - Ver ejemplos de evaluación de soluciones
    
    La función imprime:
    - Información del problema (dimensión, límites)
    - Solución de referencia y su error MSE
    - Solución aleatoria y su error MSE

    Examples
    --------
    >>> run_quick_test()
    ============================================================
    LEVITADOR MAGNÉTICO - TEST RÁPIDO
    ============================================================
    ...
    ✓ Test completado
    """
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
