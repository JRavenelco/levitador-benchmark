"""
Adquisición de Datos del Levitador Magnético Real
==================================================

Este script permite capturar datos experimentales del prototipo físico
del levitador magnético para usarlos en el benchmark.

Requisitos:
- Prototipo del levitador conectado vía puerto serie (COM1)
- Controlador PID ejecutándose en el microcontrolador

Uso:
    python adquisicion_datos.py --duracion 30 --referencia 0.005

Autor: Jesús (Doctorado UAQ)
"""

import asyncio
import serial
import time
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt


class LevitadorDataAcquisition:
    """
    Sistema de adquisición de datos para el levitador magnético.
    
    Captura datos en tiempo real del sistema físico:
    - t: tiempo [s]
    - y: posición de la esfera [m]
    - i: corriente en la bobina [A]
    - u: voltaje de control [V]
    - yd: referencia de posición [m]
    """
    
    def __init__(self, port: str = 'COM1', baudrate: int = 115200):
        """
        Args:
            port: Puerto serie donde está conectado el levitador
            baudrate: Velocidad de comunicación
        """
        self.port = port
        self.baudrate = baudrate
        self.running = False
        
        # Parámetros de control (iguales que en el microcontrolador)
        self.Ts = 0.01  # Periodo de muestreo [s]
        self.kp = 100.0
        self.ki = 50.0
        self.kd = 1.5
        self.kpi = 12.0
        self.kii = 3000.0
        self.Vref = 9.86
        self.Iref = 0.827
        self.Rs = 2.2
        
        # Escalas de conversión
        self.esc_pos = 0.05 / 1023.0      # ADC a metros
        self.esc_corriente = 5.0 / (self.Rs * 1023.0)  # ADC a amperios
        self.esc_pwm = 254.0 / self.Vref  # Voltaje a PWM
        
        # Buffers de datos
        self.data = {
            't': [], 'y': [], 'yd': [], 'i': [], 'id': [], 'u': []
        }
        
        # Variables de estado
        self._reset_state()
    
    def _reset_state(self):
        """Reinicia las variables de estado del controlador."""
        self.t = 0
        self.y = 0
        self.y_1 = 0
        self.ef = 0
        self.ef_1 = 0
        self.integral = 0
        self.intei = 0
        self.yd = 0.005  # Referencia inicial [m]
        self.flagcom = 0
        self.pv = 0
        self.i_raw = 0
    
    def set_reference(self, yd: float):
        """Establece la referencia de posición."""
        self.yd = yd
        print(f"📍 Referencia: {yd*1000:.2f} mm")
    
    async def acquire(self, duration: float, reference_profile: callable = None):
        """
        Adquiere datos durante un tiempo especificado.
        
        Args:
            duration: Duración de la adquisición [s]
            reference_profile: Función opcional que define yd(t)
        
        Returns:
            DataFrame con los datos capturados
        """
        self._reset_state()
        self.data = {k: [] for k in self.data}
        
        print(f"🔌 Conectando a {self.port}...")
        
        try:
            ser = serial.Serial(self.port, self.baudrate, timeout=0.1)
            print(f"✅ Conectado. Adquiriendo por {duration}s...")
            self.running = True
            
            start_time = time.time()
            
            while self.running and (time.time() - start_time) < duration:
                # Leer byte
                if ser.in_waiting:
                    received = ser.read(1)
                    await self._process_byte(received, reference_profile)
                
                await asyncio.sleep(0.001)
            
            ser.close()
            print(f"✅ Adquisición completada: {len(self.data['t'])} muestras")
            
        except serial.SerialException as e:
            print(f"❌ Error de conexión: {e}")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(self.data)
        return df
    
    async def _process_byte(self, received: bytes, reference_profile: callable):
        """Procesa un byte recibido del puerto serie."""
        if self.flagcom != 0:
            self.flagcom += 1
        
        if received == b'\xAA' and self.flagcom == 0:
            self.pv = 0
            self.i_raw = 0
            self.flagcom = 1
        
        if self.flagcom == 2:
            self.pv = (self.pv << 8) + ord(received)
        
        if self.flagcom == 3:
            self.pv = (self.pv << 8) + ord(received)
        
        if self.flagcom == 4:
            self.i_raw = (self.i_raw << 8) + ord(received)
        
        if self.flagcom == 5:
            self.i_raw = (self.i_raw << 8) + ord(received)
            
            # Actualizar referencia si hay perfil
            if reference_profile:
                self.yd = reference_profile(self.t)
            
            # Conversiones
            self.ef_1 = self.ef
            self.y_1 = self.y
            self.y = self.esc_pos * self.pv
            ie = self.esc_corriente * self.i_raw
            self.ef = self.yd - self.y
            
            # Controlador PID posición
            proporcional = self.kp * self.ef
            derivativa = self.kd * (self.ef - self.ef_1) / self.Ts
            
            if -self.Iref < self.integral < self.Iref:
                self.integral += self.ki * self.Ts * self.ef
            else:
                self.integral = np.clip(self.integral, -0.95*self.Iref, 0.95*self.Iref)
            
            id_ref = proporcional + self.integral + derivativa
            id_ref = np.clip(id_ref, -self.Iref, 0)
            ied = -id_ref
            
            # Controlador PI corriente
            ei = ied - ie
            propi = self.kpi * ei
            
            if -self.Vref < self.intei < self.Vref:
                self.intei += self.kii * self.Ts * ei
            else:
                self.intei = np.clip(self.intei, -0.95*self.Vref, 0.95*self.Vref)
            
            u = np.clip(propi + self.intei, 0, self.Vref)
            
            # Guardar datos
            self.data['t'].append(self.t)
            self.data['y'].append(self.y)
            self.data['yd'].append(self.yd)
            self.data['i'].append(ie)
            self.data['id'].append(ied)
            self.data['u'].append(u)
            
            self.t += self.Ts
            self.flagcom = 0
    
    def save_data(self, df: pd.DataFrame, output_dir: str = 'data'):
        """
        Guarda los datos en archivo con timestamp.
        
        Args:
            df: DataFrame con los datos
            output_dir: Directorio de salida
        
        Returns:
            Ruta del archivo guardado
        """
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"datos_levitador_{timestamp}.txt"
        filepath = Path(output_dir) / filename
        
        # Guardar en formato compatible con el benchmark
        df.to_csv(filepath, sep='\t', index=False, header=False,
                  columns=['t', 'yd', 'y', 'id', 'i', 'u'])
        
        print(f"💾 Datos guardados: {filepath}")
        return str(filepath)
    
    def plot_data(self, df: pd.DataFrame, save_path: str = None):
        """Visualiza los datos capturados."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Posición
        axes[0].plot(df['t'], df['y']*1000, 'b-', label='y (medida)')
        axes[0].plot(df['t'], df['yd']*1000, 'r--', label='yd (referencia)')
        axes[0].set_ylabel('Posición [mm]')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('Datos Experimentales del Levitador')
        
        # Corriente
        axes[1].plot(df['t'], df['i'], 'g-', label='i (medida)')
        axes[1].plot(df['t'], df['id'], 'r--', label='id (referencia)')
        axes[1].set_ylabel('Corriente [A]')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Voltaje
        axes[2].plot(df['t'], df['u'], 'orange', label='u (control)')
        axes[2].set_xlabel('Tiempo [s]')
        axes[2].set_ylabel('Voltaje [V]')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"📊 Gráfica guardada: {save_path}")
        
        plt.show()


# =============================================================================
# Perfiles de referencia predefinidos
# =============================================================================

def perfil_escalon(t, y0=0.005, y1=0.0045, t_cambio=10):
    """Escalón de posición."""
    return y1 if t > t_cambio else y0

def perfil_senoidal(t, y0=0.005, amplitud=0.001, frecuencia=0.5):
    """Referencia senoidal."""
    return y0 + amplitud * np.sin(2 * np.pi * frecuencia * t)

def perfil_rampa(t, y0=0.005, y1=0.003, t_inicio=5, t_fin=15):
    """Rampa lineal."""
    if t < t_inicio:
        return y0
    elif t < t_fin:
        return y0 + (y1 - y0) * (t - t_inicio) / (t_fin - t_inicio)
    else:
        return y1


# =============================================================================
# Main
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='Adquisición de datos del levitador')
    parser.add_argument('--puerto', default='COM1', help='Puerto serie (default: COM1)')
    parser.add_argument('--duracion', type=float, default=20, help='Duración en segundos')
    parser.add_argument('--referencia', type=float, default=0.005, help='Referencia de posición [m]')
    parser.add_argument('--perfil', choices=['constante', 'escalon', 'senoidal', 'rampa'],
                        default='constante', help='Tipo de perfil de referencia')
    parser.add_argument('--graficar', action='store_true', help='Mostrar gráfica al terminar')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧲 ADQUISICIÓN DE DATOS - LEVITADOR MAGNÉTICO")
    print("=" * 60)
    
    # Crear adquisidor
    daq = LevitadorDataAcquisition(port=args.puerto)
    daq.set_reference(args.referencia)
    
    # Seleccionar perfil
    if args.perfil == 'constante':
        perfil = None
    elif args.perfil == 'escalon':
        perfil = lambda t: perfil_escalon(t, y0=args.referencia, y1=args.referencia*0.9)
    elif args.perfil == 'senoidal':
        perfil = lambda t: perfil_senoidal(t, y0=args.referencia)
    elif args.perfil == 'rampa':
        perfil = lambda t: perfil_rampa(t, y0=args.referencia)
    
    # Adquirir
    df = await daq.acquire(args.duracion, reference_profile=perfil)
    
    if df is not None and len(df) > 0:
        # Guardar
        filepath = daq.save_data(df)
        
        # Estadísticas
        print(f"\n📈 Estadísticas:")
        print(f"   Muestras: {len(df)}")
        print(f"   Duración: {df['t'].iloc[-1]:.2f} s")
        print(f"   Posición: [{df['y'].min()*1000:.2f}, {df['y'].max()*1000:.2f}] mm")
        print(f"   Corriente: [{df['i'].min():.3f}, {df['i'].max():.3f}] A")
        
        # Graficar
        if args.graficar:
            daq.plot_data(df)
        
        print(f"\n✅ Archivo listo para usar con el benchmark:")
        print(f"   problema = LevitadorBenchmark('{filepath}')")
    
    return df


if __name__ == '__main__':
    asyncio.run(main())
