import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = r'c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\levitador valentin\KAN_VALIDATION.csv'

try:
    df = pd.read_csv(file_path)
    
    print("="*60)
    print("📊 ANÁLISIS DE KAN_VALIDATION.csv")
    print("="*60)
    
    t = df['t'].values
    y = df['y_sensor'].values  # Position in meters
    i = df['i'].values         # Current in Amperes
    u = df['u'].values         # Voltage
    
    duration = t[-1] - t[0]
    dt = np.mean(np.diff(t))
    
    print(f"⏱️  Duración: {duration:.2f}s ({len(df)} muestras)")
    print(f"    dt promedio: {dt*1000:.2f}ms")
    
    # Position Analysis
    y_mm = y * 1000
    y_range = np.max(y_mm) - np.min(y_mm)
    y_std = np.std(y_mm)
    
    print(f"\n📏 Posición (y_sensor):")
    print(f"   Rango: {np.min(y_mm):.2f} - {np.max(y_mm):.2f} mm")
    print(f"   Variación: {y_range:.2f} mm")
    print(f"   Desv. Std: {y_std:.3f} mm")
    
    # Current Analysis
    print(f"\n⚡ Corriente (i):")
    print(f"   Rango: {np.min(i):.4f} - {np.max(i):.4f} A")
    print(f"   Promedio: {np.mean(i):.4f} A")
    
    # Voltage Analysis
    print(f"\n🔋 Voltaje (u):")
    print(f"   Rango: {np.min(u):.2f} - {np.max(u):.2f} V")
    
    # Verdict
    print("\n" + "-"*60)
    print("🔍 VEREDICTO:")
    
    reasons = []
    is_good = True
    
    if y_range < 2.0:
        reasons.append(f"❌ Poca variación en posición ({y_range:.2f}mm < 2.0mm). El sistema parece estático.")
        is_good = False
    else:
        reasons.append(f"✅ Variación de posición suficiente ({y_range:.2f}mm).")
        
    if np.mean(i) < 0.1:
        reasons.append(f"⚠️ Corriente muy baja ({np.mean(i):.3f}A). ¿Está levitando o en reposo?")
    
    if len(t) < 1000:
        reasons.append("⚠️ Pocos datos (<10s).")
    
    if is_good:
        print("✅ LOS DATOS PARECEN ÚTILES para identificación (contienen dinámica).")
    else:
        print("❌ LOS DATOS TIENEN LIMITACIONES severas.")
    
    for r in reasons:
        print("   " + r)
        
except Exception as e:
    print(f"Error al leer el archivo: {e}")
