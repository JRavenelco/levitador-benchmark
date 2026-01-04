#!/usr/bin/env python3
"""
KAN-PINN Training Script
=========================

Phase 2: Train KAN-PINN (sensorless position observer) using identified parameters.

This script trains a two-stage KAN-PINN model:
- Stage 1: Flux Observer (u, i) -> phi_hat
- Stage 2: Position Predictor (u, i, phi_hat) -> y_hat

Usage:
    python scripts/train_kanpinn.py --config config/kanpinn_default.yaml
"""

import sys
import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.flux_observer import FluxObserverHiPPO
from src.models.position_predictor import PositionPredictor
from src.data.sequence_dataset import SeqDataset
from src.utils.config_loader import load_config
from src.utils.physics import LevitationPhysics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path, config, physics):
    """
    Load data (TXT or CSV) and preprocess for training.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}...")
    
    # Load data based on extension
    if data_path.suffix == '.txt':
        # Assuming format: t, y, i, u (columns 0, 1, 2, 3)
        try:
            data = np.loadtxt(data_path)
            t = data[:, 0]
            y_raw = data[:, 1]
            i_raw = data[:, 2]
            u_raw = data[:, 3]
            df = pd.DataFrame({'t': t, 'y_sensor': y_raw, 'i': i_raw, 'u': u_raw})
        except Exception as e:
            raise ValueError(f"Error loading TXT data: {e}")
    else:
        # Assume CSV with columns 'u', 'i', 'y_sensor'
        df = pd.read_csv(data_path)
        if 'y_sensor' not in df.columns and 'y' in df.columns:
            df['y_sensor'] = df['y']
    
    # Filter valid range (as in notebook)
    # mask = (df['i'] > 0.1) & (df['y_sensor'] > 0.002) & (df['y_sensor'] < 0.015)
    # Using config or defaults if not present
    # For robustness, we'll use a broad filter or all data if specified
    if 'data' in config and 'filter' in config['data'] and config['data']['filter']:
        mask = (df['i'] > 0.1) & (df['y_sensor'] > 0.002) & (df['y_sensor'] < 0.015)
        df = df[mask].reset_index(drop=True)
        logger.info(f"Filtered data to {len(df)} samples")

    u_raw = df['u'].values.astype(np.float32)
    i_raw = df['i'].values.astype(np.float32)
    y_raw = df['y_sensor'].values.astype(np.float32)

    # Calculate Target Flux (phi = L(y) * i) for Stage 1 training
    L_vals = physics.inductance_L(y_raw)
    phi_target = (L_vals * i_raw).astype(np.float32)

    # Normalization statistics
    stats = {
        'u': (float(u_raw.mean()), float(u_raw.std())),
        'i': (float(i_raw.mean()), float(i_raw.std())),
        'phi': (float(phi_target.mean()), float(phi_target.std())),
        'y': (float(y_raw.mean()), float(y_raw.std()))
    }

    # Normalize
    u_n = (u_raw - stats['u'][0]) / stats['u'][1]
    i_n = (i_raw - stats['i'][0]) / stats['i'][1]
    phi_n = (phi_target - stats['phi'][0]) / stats['phi'][1]
    y_n = (y_raw - stats['y'][0]) / stats['y'][1]

    # Create dataset dictionary
    # We keep raw values for physics loss calculations
    processed_arrays = {
        'X': np.stack([u_n, i_n], axis=1), # [N, 2]
        'phi_target': phi_n[:, None],      # [N, 1]
        'u_raw': u_raw[:, None],           # [N, 1]
        'i_raw': i_raw[:, None],           # [N, 1]
        'y_raw': y_raw[:, None],           # [N, 1]
        'y_target': y_n[:, None]           # [N, 1] for Stage 2
    }
    
    return processed_arrays, stats

def train_flux_observer(model, dataset, config, physics, device, stats):
    """
    Stage 1: Train Flux Observer using Kirchhoff loss.
    """
    logger.info("Starting Stage 1: Flux Observer Training")
    
    cfg = config['stage1']
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=False, drop_last=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_loss = float('inf')
    history = {'loss': [], 'data': [], 'kirch': []}
    
    # physics object has R0 if loaded from params, but usually fixed R=2.2 or optimized R0
    # Notebook used R = 2.2 constant.
    R_val = getattr(physics, 'R0', 2.2) 
    dt = 0.01 # Fixed dt
    
    phi_m, phi_s = stats['phi']
    
    model.train()
    model.to(device)
    
    for epoch in range(cfg['epochs']):
        ep_loss, ep_data, ep_kirch = 0, 0, 0
        n_batch = 0
        
        # Curriculum: increase Kirchhoff weight
        # e.g. ramp up over first 25% of epochs
        w_kirch = min(cfg['w_kirch'], epoch / (cfg['epochs'] * 0.25 + 1) * cfg['w_kirch'])
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False):
            X_b = batch['X'].to(device)
            phi_b = batch['phi_target'].to(device)
            u_b = batch['u_raw'].to(device)
            i_b = batch['i_raw'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            phi_pred_n = model(X_b)
            
            # Loss 1: Data (MSE)
            loss_data = nn.functional.mse_loss(phi_pred_n, phi_b)
            
            # Loss 2: Kirchhoff
            # Desnormalize phi
            phi_pred = phi_pred_n * phi_s + phi_m
            
            # Finite difference dphi/dt (assuming sequence order is preserved)
            # [B, S, 1]
            dphi_dt = (phi_pred[:, 1:] - phi_pred[:, :-1]) / dt
            
            # Residual: u - (R*i + dphi/dt)
            # Align time steps: u and i at t+1 match dphi/dt between t and t+1 ?
            # Usually dphi/dt at t approx (phi[t+1]-phi[t])/dt. 
            # Kirchhoff: u[t] = R*i[t] + dphi/dt[t]
            # Notebook: residual = u_b[:, 1:] - (R * i_b[:, 1:] + dphi_dt)
            # This aligns u[t+1] with dphi/dt[t->t+1]. Let's stick to notebook logic.
            
            residual = u_b[:, 1:] - (R_val * i_b[:, 1:] + dphi_dt)
            loss_kirch = torch.mean(residual ** 2)
            
            loss = loss_data + w_kirch * loss_kirch
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
            
            ep_loss += loss.item()
            ep_data += loss_data.item()
            ep_kirch += loss_kirch.item()
            n_batch += 1
            
        scheduler.step()
        
        ep_loss /= n_batch
        history['loss'].append(ep_loss)
        history['data'].append(ep_data/n_batch)
        history['kirch'].append(ep_kirch/n_batch)
        
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(model.state_dict(), Path(config['output_dir']) / 'flux_observer.pt')
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Ep {epoch+1} | Loss: {ep_loss:.6f} | Data: {ep_data/n_batch:.6f} | Kirch: {ep_kirch/n_batch:.6f}")
            
    return history

def train_position_predictor(flux_model, pos_model, dataset_arrays, config, physics, device, stats):
    """
    Stage 2: Train Position Predictor using PINN (Euler-Lagrange).
    Uses FIXED Flux Model to generate inputs.
    """
    logger.info("Starting Stage 2: Position Predictor Training")
    
    # Generate Estimated Flux for the whole dataset
    # We need to process sequentially
    flux_model.eval()
    dataset_full = SeqDataset(dataset_arrays, config['model']['seq_len'])
    loader_eval = DataLoader(dataset_full, batch_size=config['stage1']['batch_size'], shuffle=False, drop_last=False)
    
    phi_est_list = []
    
    logger.info("Generating estimated flux with trained observer...")
    
    # Actually, simpler approach: Train on batches, computing phi_est on the fly.
    # This acts as data augmentation/noise injection if flux model has dropout (it doesn't).
    
    cfg = config['stage2']
    # Use same dataset as stage 1, but we'll modify inputs in the loop
    loader = DataLoader(dataset_full, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)
    
    optimizer = optim.AdamW(pos_model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_loss = float('inf')
    history = {'loss': [], 'data': [], 'pinn': []}
    
    dt = 0.01
    y_m, y_s = stats['y']
    
    pos_model.train()
    pos_model.to(device)
    
    for epoch in range(cfg['epochs']):
        ep_loss, ep_data, ep_pinn = 0, 0, 0
        n_batch = 0
        
        lambda_pinn = min(cfg['w_pinn'], epoch / (cfg['epochs'] * 0.5 + 1) * cfg['w_pinn'])
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{cfg['epochs']}", leave=False):
            # Inputs
            X_b = batch['X'].to(device) # [B, S, 2] (u, i)
            y_b = batch['y_target'].to(device) # [B, S, 1] normalized
            i_phys = batch['i_raw'].to(device) # [B, S, 1]
            
            # 1. Get Flux Estimate (Frozen)
            with torch.no_grad():
                phi_est_n = flux_model(X_b) # [B, S, 1]
            
            # 2. Prepare Input for Position Predictor: (u, i, phi_est)
            # X_b is (u, i), we append phi_est
            X_pos = torch.cat([X_b, phi_est_n], dim=2) # [B, S, 3]
            
            optimizer.zero_grad()
            
            # Forward
            y_pred_n = pos_model(X_pos)
            
            # Loss 1: Data (MSE)
            loss_data = nn.functional.mse_loss(y_pred_n, y_b)
            
            # Loss 2: PINN (Euler-Lagrange)
            # Desnormalize y
            y_phys = y_pred_n * y_s + y_m
            
            # Derivatives (Central difference)
            y_curr = y_phys[:, 1:-1]
            y_next = y_phys[:, 2:]
            y_prev = y_phys[:, :-2]
            
            # Acceleration
            acc = (y_next - 2*y_curr + y_prev) / (dt**2)
            
            # Align current (center)
            i_center = i_phys[:, 1:-1]
            
            # Force Magnetic
            # F = 0.5 * i^2 * dL/dy
            # dL/dy = -K0 / (A * (1 + y/A)^2)
            
            # dL_dy calculation
            K0, A = physics.K0, physics.A
            denom = 1 + y_curr / A
            dL_dy = -K0 / (A * (denom ** 2))
            F_mag = 0.5 * (i_center ** 2) * dL_dy
            
            # Residual: m*a + m*g - F_mag = 0
            residual = physics.m * acc - physics.m * physics.g - F_mag
            
            loss_pinn = torch.mean(residual ** 2)
            
            loss = loss_data + lambda_pinn * loss_pinn
            
            loss.backward()
            nn.utils.clip_grad_norm_(pos_model.parameters(), cfg['clip_grad_norm'])
            optimizer.step()
            
            ep_loss += loss.item()
            ep_data += loss_data.item()
            ep_pinn += loss_pinn.item()
            n_batch += 1
            
        scheduler.step()
        
        ep_loss /= n_batch
        history['loss'].append(ep_loss)
        history['data'].append(ep_data/n_batch)
        history['pinn'].append(ep_pinn/n_batch)
        
        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save(pos_model.state_dict(), Path(config['output_dir']) / 'position_predictor.pt')
            
        if (epoch + 1) % 10 == 0:
            logger.info(f"Ep {epoch+1} | Loss: {ep_loss:.6f} | Data: {ep_data/n_batch:.6f} | PINN: {ep_pinn/n_batch:.6f}")

    return history

def plot_history(history, title, save_path):
    plt.figure(figsize=(10, 5))
    for k, v in history.items():
        plt.semilogy(v, label=k)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Phase 2: KAN-PINN Training')
    parser.add_argument('--config', type=str, default='config/kanpinn_default.yaml', help='Config file')
    parser.add_argument('--data', type=str, default='data/datos_levitador.txt', help='Data file')
    parser.add_argument('--params', type=str, default='results/parametros_optimos.json', help='Optimized parameters')
    parser.add_argument('--output', type=str, default='results/kanpinn_training', help='Output directory')
    args = parser.parse_args()
    
    # Setup
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = load_config(args.config)
    # Inject output dir into config for easier access
    config['output_dir'] = args.output
    
    # Physics
    if Path(args.params).exists():
        with open(args.params, 'r') as f:
            params_dict = json.load(f)
        logger.info(f"Loaded physics parameters from {args.params}")
    else:
        logger.warning(f"Params file {args.params} not found. Using defaults.")
        params_dict = {}
    physics = LevitationPhysics(params_dict)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Data
    processed_arrays, stats = load_and_preprocess_data(args.data, config, physics)
    
    # Save stats
    torch.save(stats, output_path / 'normalization_stats.pt')
    
    # Dataset
    seq_len = config['model']['seq_len']
    dataset = SeqDataset(processed_arrays, seq_len)
    logger.info(f"Dataset created: {len(dataset)} sequences")
    
    # Models
    hippo_n = config['model']['hippo_n']
    grid_size = config['model']['grid_size']
    
    flux_model = FluxObserverHiPPO(hippo_n=hippo_n, grid_size=grid_size)
    pos_model = PositionPredictor(grid_size=grid_size)
    
    # Train Stage 1
    hist1 = train_flux_observer(flux_model, dataset, config, physics, device, stats)
    plot_history(hist1, 'Flux Observer Training', output_path / 'loss_flux_observer.png')
    
    # Train Stage 2
    # Load best flux model for stage 2
    flux_model.load_state_dict(torch.load(output_path / 'flux_observer.pt'))
    hist2 = train_position_predictor(flux_model, pos_model, processed_arrays, config, physics, device, stats)
    plot_history(hist2, 'Position Predictor Training', output_path / 'loss_position_predictor.png')
    
    logger.info("Training Complete!")
    logger.info(f"Models saved to {output_path}")

if __name__ == '__main__':
    main()
