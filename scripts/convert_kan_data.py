import pandas as pd
import numpy as np
import sys

input_path = r'c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\levitador valentin\KAN_VALIDATION.csv'
output_path = r'c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\levitador-benchmark\data\kan_validation_data.txt'

try:
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Extract required columns for benchmark: t, y, i, u
    # CSV columns: t,y_sensor,y_kan,phi_est,i,u,error_mm
    
    benchmark_data = df[['t', 'y_sensor', 'i', 'u']].copy()
    
    # Check for NaNs
    if benchmark_data.isnull().values.any():
        print("Warning: NaNs found, dropping...")
        benchmark_data = benchmark_data.dropna()
        
    # Save as space/tab separated txt
    print(f"Saving to {output_path}...")
    np.savetxt(output_path, benchmark_data.values, fmt='%.6f', delimiter='\t', 
               header='t\ty\ti\tu', comments='')
    
    print("Conversion successful.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
