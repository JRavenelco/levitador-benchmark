"""
ENTRENAMIENTO UNIFICADO: HiPPO-KAN + PINN (2 ETAPAS)
====================================================
Combina lo mejor de:
1. Arquitectura 2 Etapas (Colab): FluxObserver -> PositionPredictor
2. Física Avanzada (Local): HiPPO Vectorizado + Loss Lagrangiano/Acción
3. Parámetros Actualizados: Sincronizados con levitador_sensorless_kan.cpp

Autor: Cascade
Fecha: Enero 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# =============================================================================
# 1. CONFIGURACIÓN FÍSICA (Sincronizada con C++ y v3_hippo)
# =============================================================================
# Valores tomados de entrenar_kan_pinn_v3_hippo.py / CBR_InitPosition.h
R_NOMINAL = 2.20      # Resistencia Real [Ohm] (era 16.0 en versión anterior incorrecta)
K0 = 0.0363           # Inductancia base [H]
K_COEFF = 0.0035      # Coeficiente K [H*m]
A_COEFF = 0.0052      # Parámetro geométrico [m]
MASS = 0.009          # Masa [kg] (9g)
G = 9.81              # Gravedad [m/s^2]
VREF = 9.86
DT = 0.01             # Tiempo de muestreo [s]

# Hiperparámetros
HIPPO_N = 8
GRID_SIZE = 10
SEQ_LEN = 32
BATCH_SIZE = 256  # Increased batch size for larger dataset
EPOCHS_STAGE1 = 200  # Adjusted for larger dataset
EPOCHS_STAGE2 = 400  # Adjusted for larger dataset

def _env_int(name, default):
    v = os.getenv(name, "")
    return default if v == "" else int(v)

def _env_float(name, default):
    v = os.getenv(name, "")
    return default if v == "" else float(v)

def _env_bool(name, default):
    v = os.getenv(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")

BATCH_SIZE = _env_int("BATCH_SIZE", BATCH_SIZE)
EPOCHS_STAGE1 = _env_int("EPOCHS_STAGE1", EPOCHS_STAGE1)
EPOCHS_STAGE2 = _env_int("EPOCHS_STAGE2", EPOCHS_STAGE2)
NUM_WORKERS = _env_int("NUM_WORKERS", 0)
USE_AMP = _env_bool("USE_AMP", False)

# =============================================================================
# 2.1. RUTA 2 (Baldwin-lite): domain randomization / multi-sesion
# =============================================================================
AUGMENT = True
R_PERT = 0.30          # +/- 30% (deriva térmica y variación entre sesiones)
U_NOISE_STD = 0.10     # [V] ruido en voltaje para robustez
I_NOISE_STD = 0.01     # [A] ruido en corriente para robustez
P_SAT_HI = 0.05        # probabilidad de clipear u a Vref (simular saturación)
P_SAT_LO = 0.05        # probabilidad de clipear u a 0 (simular recorte)
YREF_STEP_THRESH = 2e-4  # 0.2mm en metros

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# =============================================================================
# 2. MODELOS (HiPPO + KAN)
# =============================================================================

class KANLayer(nn.Module):
    def __init__(self, in_feat, out_feat, grid_size=10, k=3):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.k = k
        self.grid_size = grid_size
        
        # Grid
        grid = torch.linspace(-2.0, 2.0, grid_size + 1)
        h = 4.0 / grid_size
        pad = torch.linspace(-2.0 - k*h, -2.0 - h, k)
        pad_end = torch.linspace(2.0 + h, 2.0 + k*h, k)
        self.register_buffer("grid", torch.cat([pad, grid, pad_end]))
        
        self.base = nn.Linear(in_feat, out_feat)
        self.spline_w = nn.Parameter(torch.randn(out_feat, in_feat, grid_size + k) * 0.1)

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        for p in range(1, self.k + 1):
            denom1 = grid[p:-1] - grid[:-p-1] + 1e-8
            term1 = (x - grid[:-p-1]) / denom1 * bases[..., :-1]
            denom2 = grid[p+1:] - grid[1:-p] + 1e-8
            term2 = (grid[p+1:] - x) / denom2 * bases[..., 1:]
            bases = term1 + term2
        return bases

    def forward(self, x):
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_feat)
        base = self.base(nn.functional.silu(x_flat))
        spline = torch.einsum("bi...,oi...->bo", self.b_splines(x_flat), self.spline_w)
        return (base + spline).reshape(*original_shape, self.out_feat)

class HiPPOLayer(nn.Module):
    def __init__(self, N=8, dt=DT):
        super().__init__()
        self.N = N
        A = np.zeros((N, N))
        for n in range(N):
            for m in range(N):
                if n > m: A[n, m] = np.sqrt(2*n+1)*np.sqrt(2*m+1)
                elif n == m: A[n, m] = n+1
        
        I = np.eye(N)
        # Discretización Bilineal (Tustin)
        A_d = np.linalg.inv(I - (dt/2)*A) @ (I + (dt/2)*A)
        B_d = np.linalg.inv(I - (dt/2)*A) @ (dt * np.sqrt(2*np.arange(N)+1))
        
        self.register_buffer("A_d", torch.tensor(A_d, dtype=torch.float32))
        self.register_buffer("B_d", torch.tensor(B_d, dtype=torch.float32))

    def forward(self, x_seq):
        # x_seq: [B, S, F]
        B_size, S_size, F_size = x_seq.shape
        c = torch.zeros(B_size, F_size, self.N, device=x_seq.device)
        history = []
        
        # Vectorizado sobre Batch y Features
        for t in range(S_size):
            x_t = x_seq[:, t, :] # [B, F]
            # c: [B, F, N]
            c = torch.einsum('bfn,nm->bfm', c, self.A_d) + self.B_d.unsqueeze(0).unsqueeze(0) * x_t.unsqueeze(-1)
            history.append(c.reshape(B_size, -1)) # [B, F*N]
            
        return torch.stack(history, dim=1)

class FluxObserver(nn.Module):
    def __init__(self, hippo_n=HIPPO_N):
        super().__init__()
        self.hippo = HiPPOLayer(N=hippo_n)
        # Input: u, i -> 2 features
        self.kan1 = KANLayer(2 * hippo_n, 32, grid_size=GRID_SIZE)
        self.kan2 = KANLayer(32, 16, grid_size=GRID_SIZE)
        self.kan3 = KANLayer(16, 1, grid_size=GRID_SIZE)

    def forward(self, x): # x: [B, S, 2] (u, i)
        h = self.hippo(x)
        h = self.kan1(h)
        h = self.kan2(h)
        return self.kan3(h) # [B, S, 1] phi_hat

class PositionPredictor(nn.Module):
    def __init__(self, hippo_n=HIPPO_N):
        super().__init__()
        # Input: u, i, phi -> 3 features
        # Usa HiPPO para capturar dinámica de orden superior (velocidad, aceleración implícita)
        self.hippo = HiPPOLayer(N=hippo_n)
        
        # KAN procesa la memoria HiPPO (3 features * N coeficientes)
        self.kan1 = KANLayer(3 * hippo_n, 32, grid_size=GRID_SIZE)
        self.kan2 = KANLayer(32, 16, grid_size=GRID_SIZE)
        self.kan3 = KANLayer(16, 1, grid_size=GRID_SIZE)

    def forward(self, x): # x: [B, S, 3] (u, i, phi)
        h = self.hippo(x) # [B, S, 3*N]
        h = self.kan1(h)
        h = self.kan2(h)
        return self.kan3(h) # [B, S, 1] y_hat

# =============================================================================
# 3. FUNCIONES FÍSICAS Y PÉRDIDAS
# =============================================================================

def inductance_model(y):
    return K0 + (K_COEFF / (1.0 + y / A_COEFF))

def position_from_inductance(L):
    denom = (L - K0)
    denom = torch.clamp(denom, min=1e-6)
    y = A_COEFF * (K_COEFF / denom - 1.0)
    return torch.clamp(y, 0.001, 0.025)

def dL_dy_model(y):
    return -(K_COEFF / (A_COEFF * (1.0 + y / A_COEFF)**2))

def physics_losses(y_pred, i_seq, u_seq, phi_est):
    """
    Calcula pérdidas físicas vectorizadas
    """
    # 1. Derivadas
    y_dot = (y_pred[:, 1:] - y_pred[:, :-1]) / DT
    y_ddot = (y_dot[:, 1:] - y_dot[:, :-1]) / DT
    
    # Alinear
    y = y_pred[:, 1:-1]
    i = i_seq[:, 1:-1]
    u = u_seq[:, 1:-1]
    phi = phi_est[:, 1:-1]
    
    # --- PÉRDIDA DE ACCIÓN (LAGRANGIANO) ---
    L_y = inductance_model(y)
    T_mec = 0.5 * MASS * y_dot[:, 1:]**2
    T_mag = 0.5 * L_y * i**2
    V = MASS * G * y
    Lagrangian = (T_mec + T_mag) - V
    loss_action = torch.mean(torch.abs(Lagrangian))
    
    # --- PÉRDIDA SANTANA (CONSISTENCIA DE FLUJO) ---
    phi_phys = L_y * i
    loss_santana = torch.mean((phi - phi_phys)**2)
    
    # --- PÉRDIDA EULER-LAGRANGE (DINÁMICA) ---
    dL_dy = dL_dy_model(y)
    F_mag = 0.5 * i**2 * dL_dy # Fuerza magnética (negativa/atractiva implícita en dL/dy)
    # Ecuación: m*y_ddot = F_mag + m*g (si y apunta abajo)
    residual_dyn = MASS * y_ddot - (F_mag + MASS * G) 
    loss_dynamics = torch.mean(residual_dyn**2)
    
    return loss_action, loss_santana, loss_dynamics

# =============================================================================
# 4. PREPARACIÓN DE DATOS
# =============================================================================

def load_data():
    print("Cargando datos...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "data", "mega_controlled_dataset.txt"),
        os.path.join(script_dir, "..", "levitador-benchmark", "data", "mega_controlled_dataset.txt"),
        os.path.join(script_dir, "..", "..", "levitador-benchmark", "data", "mega_controlled_dataset.txt"),
        r"c:\Users\jesus\Documents\Doctorado\Experimentos\CRio DAQ\cDAQ_9174\levitador-benchmark\data\mega_controlled_dataset.txt",
    ]

    mega_path = None
    for p in candidates:
        p2 = os.path.normpath(p)
        if os.path.exists(p2):
            mega_path = p2
            break

    if mega_path:
        print(f"Usando archivo: {mega_path}")
        df = pd.read_csv(mega_path, sep=r'\s+', header=None, names=['t', 'y_ref', 'y_sensor', 'i', 'u'])
    else:
        print("Dataset Mega no encontrado, buscando locales...")
        if os.path.exists(os.path.join(script_dir, "KAN_VALIDATION.csv")):
            file_path = os.path.join(script_dir, "KAN_VALIDATION.csv")
            df = pd.read_csv(file_path)
        else:
            file_path = os.path.join(script_dir, "CBR_KAN_TRAIN.csv")
            df = pd.read_csv(file_path)
        
    print(f"Datos cargados: {len(df)}")
    
    # Recalcular phi_target con los parámetros NUEVOS
    y_s = df['y_sensor'].values
    i_s = df['i'].values
    L_vals = inductance_model(y_s)
    phi_new = L_vals * i_s
    df['phi_target'] = phi_new

    # Derivar task_id (pseudo-sesiones) a partir de cambios en y_ref
    # Esto permite muestrear secuencias consistentes dentro del mismo setpoint/escenario.
    y_ref = df['y_ref'].values
    yref_jump = np.abs(np.diff(y_ref)) > YREF_STEP_THRESH
    task_id = np.concatenate([[0], np.cumsum(yref_jump).astype(np.int64)])
    df['task_id'] = task_id
    
    # Filtro de estabilidad (ajustado para no ser tan agresivo si el dataset es bueno)
    # Filtrar solo valores físicos imposibles o ruido extremo
    mask = (df['i'] > 0.01) & (df['y_sensor'] > 0.001) & (df['y_sensor'] < 0.025)
    df = df[mask].reset_index(drop=True)
    print(f"Datos filtrados: {len(df)}")

    if 'task_id' in df.columns:
        print(f"Tareas (task_id) detectadas: {df['task_id'].nunique()}")
    
    return df

class SequenceDataset(Dataset):
    def __init__(self, u, i, phi, y, task_id=None, seq_len=SEQ_LEN):
        self.u = torch.FloatTensor(u)
        self.i = torch.FloatTensor(i)
        self.phi = torch.FloatTensor(phi)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

        if task_id is None:
            self.task_id = torch.zeros(len(self.u), dtype=torch.int64)
        else:
            self.task_id = torch.as_tensor(task_id, dtype=torch.int64)

        # Precomputar starts válidos: secuencia completa dentro del mismo task_id
        valid = []
        for start in range(0, len(self.u) - self.seq_len):
            tid0 = self.task_id[start].item()
            if int(self.task_id[start + self.seq_len - 1].item()) == int(tid0):
                valid.append(start)
        self.valid_starts = valid
        
    def __len__(self):
        return len(self.valid_starts)
        
    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.seq_len
        return (
            self.u[start:end],
            self.i[start:end],
            self.phi[start:end],
            self.y[start:end],
            self.task_id[start]  # task_id escalar
        )


def augment_phys(u_phys, i_phys):
    if not AUGMENT:
        return u_phys, i_phys

    u_phys = u_phys + torch.randn_like(u_phys) * U_NOISE_STD
    i_phys = i_phys + torch.randn_like(i_phys) * I_NOISE_STD

    # Simular saturación de forma aleatoria
    m_hi = (torch.rand_like(u_phys) < P_SAT_HI)
    m_lo = (torch.rand_like(u_phys) < P_SAT_LO)
    u_phys = torch.where(m_hi, torch.full_like(u_phys, VREF), u_phys)
    u_phys = torch.where(m_lo, torch.zeros_like(u_phys), u_phys)
    u_phys = torch.clamp(u_phys, 0.0, VREF)

    i_phys = torch.clamp(i_phys, min=0.0)
    return u_phys, i_phys

# =============================================================================
# 5. ENTRENAMIENTO PRINCIPAL
# =============================================================================

def main():
    df = load_data()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Normalización (Estadísticas Globales)
    stats = {
        'u_mean': df['u'].mean(), 'u_std': df['u'].std(),
        'i_mean': df['i'].mean(), 'i_std': df['i'].std(),
        'phi_mean': df['phi_target'].mean(), 'phi_std': df['phi_target'].std(),
        'y_mean': df['y_sensor'].mean(), 'y_std': df['y_sensor'].std(),
    }
    
    print("\nEstadísticas de Normalización:")
    for k, v in stats.items(): print(f"  {k}: {v:.6f}")
    
    # Datos Normalizados
    u_n = (df['u'].values - stats['u_mean']) / stats['u_std']
    i_n = (df['i'].values - stats['i_mean']) / stats['i_std']
    phi_n = (df['phi_target'].values - stats['phi_mean']) / stats['phi_std']
    y_n = (df['y_sensor'].values - stats['y_mean']) / stats['y_std']
    
    task_id = df['task_id'].values if 'task_id' in df.columns else None
    dataset = SequenceDataset(u_n, i_n, phi_n, y_n, task_id=task_id)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    
    # -------------------------------------------------------------------------
    # ETAPA 1: FLUX OBSERVER (u, i) -> phi
    # -------------------------------------------------------------------------
    print("\n>>> INICIANDO ETAPA 1: FLUX OBSERVER")
    flux_model = FluxObserver().to(DEVICE)
    opt_flux = optim.AdamW(flux_model.parameters(), lr=1e-3)
    sched_flux = optim.lr_scheduler.CosineAnnealingLR(opt_flux, T_max=EPOCHS_STAGE1)

    scaler_flux = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))
    
    for epoch in range(EPOCHS_STAGE1):
        flux_model.train()
        total_loss = 0
        
        for u_b, i_b, _phi_b, y_b, _task in loader:
            u_b = u_b.unsqueeze(-1).to(DEVICE)
            i_b = i_b.unsqueeze(-1).to(DEVICE)
            y_b = y_b.unsqueeze(-1).to(DEVICE)

            # Desnormalizar -> augmentar en físico -> renormalizar para la red
            u_phys = u_b * stats['u_std'] + stats['u_mean']
            i_phys = i_b * stats['i_std'] + stats['i_mean']
            u_phys, i_phys = augment_phys(u_phys, i_phys)
            u_in = (u_phys - stats['u_mean']) / stats['u_std']
            i_in = (i_phys - stats['i_mean']) / stats['i_std']

            with torch.cuda.amp.autocast(enabled=(USE_AMP and torch.cuda.is_available())):
                # Input: [u, i]
                x_in = torch.cat([u_in, i_in], dim=-1)
                phi_pred = flux_model(x_in)

                # Loss 1: Data (MSE vs Target calculado, coherente con i_phys augmentada)
                y_true_phys = y_b * stats['y_std'] + stats['y_mean']
                phi_target_phys = inductance_model(y_true_phys) * i_phys
                phi_target_norm = (phi_target_phys - stats['phi_mean']) / stats['phi_std']
                loss_mse = nn.functional.mse_loss(phi_pred, phi_target_norm)

                # Loss 2: Kirchhoff (Física)
                phi_phys = phi_pred * stats['phi_std'] + stats['phi_mean']

                dphi_dt = (phi_phys[:, 1:] - phi_phys[:, :-1]) / DT
                # Variación de resistencia (deriva térmica / sesion)
                R_batch = R_NOMINAL * (1.0 + (2.0 * torch.rand_like(u_phys[:, :1]) - 1.0) * R_PERT)
                res_kirch = u_phys[:, :-1] - (R_batch * i_phys[:, :-1] + dphi_dt)
                loss_kirch = torch.mean(res_kirch**2)

                loss = loss_mse + 0.1 * loss_kirch

            opt_flux.zero_grad(set_to_none=True)
            scaler_flux.scale(loss).backward()
            scaler_flux.step(opt_flux)
            scaler_flux.update()
            total_loss += loss.item()
            
        sched_flux.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} | Flux Loss: {total_loss/len(loader):.6f}")
            
    # Guardar Flux Observer
    torch.save(flux_model.state_dict(), "flux_observer_final.pt")
    
    # -------------------------------------------------------------------------
    # ETAPA 2: POSITION PREDICTOR (u, i, phi_est) -> delta-y
    # -------------------------------------------------------------------------
    # Para soportar augmentación coherente (u/i saturado + ruido), calculamos phi_est
    # on-the-fly con el FluxObserver congelado.
    flux_model.eval()
    stage2_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"Dataset Etapa 2 preparado: {len(dataset)} secuencias")

    print("\n>>> INICIANDO ETAPA 2: POSITION PREDICTOR (HiPPO-KAN)")
    pos_model = PositionPredictor().to(DEVICE)
    opt_pos = optim.AdamW(pos_model.parameters(), lr=1e-3)
    sched_pos = optim.lr_scheduler.CosineAnnealingLR(opt_pos, T_max=EPOCHS_STAGE2)

    scaler_pos = torch.cuda.amp.GradScaler(enabled=(USE_AMP and torch.cuda.is_available()))
    
    for epoch in range(EPOCHS_STAGE2):
        pos_model.train()
        total_loss = 0
        l_mse_accum = 0
        l_phys_accum = 0
        
        for u_b, i_b, _phi_b, y_b, _task in stage2_loader:
            u_b = u_b.unsqueeze(-1).to(DEVICE)
            i_b = i_b.unsqueeze(-1).to(DEVICE)
            y_b = y_b.unsqueeze(-1).to(DEVICE)

            # Desnormalizar -> augmentar -> renormalizar
            u_phys = u_b * stats['u_std'] + stats['u_mean']
            i_phys = i_b * stats['i_std'] + stats['i_mean']
            u_phys, i_phys = augment_phys(u_phys, i_phys)
            u_in = (u_phys - stats['u_mean']) / stats['u_std']
            i_in = (i_phys - stats['i_mean']) / stats['i_std']

            with torch.no_grad():
                x_flux = torch.cat([u_in, i_in], dim=-1)
                phi_est_pred = flux_model(x_flux)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and torch.cuda.is_available())):
                # Input Stage 2: [u, i, phi_est]
                x_pos = torch.cat([u_in, i_in, phi_est_pred], dim=-1)
                y_pred = pos_model(x_pos)

                # Física: y_true/phi_est desnormalizados
                y_true_phys = y_b * stats['y_std'] + stats['y_mean']
                phi_p_phys = phi_est_pred * stats['phi_std'] + stats['phi_mean']

                L_inst = phi_p_phys / torch.clamp(i_phys, min=0.01)
                y_base_phys = position_from_inductance(torch.clamp(L_inst, 0.01, 0.20))

                delta_true_phys = y_true_phys - y_base_phys
                delta_true_norm = delta_true_phys / stats['y_std']

                loss_mse = nn.functional.mse_loss(y_pred, delta_true_norm)

                y_p_phys = y_base_phys + (y_pred * stats['y_std'])

                l_act, l_sant, l_dyn = physics_losses(y_p_phys, i_phys, u_phys, phi_p_phys)

                w_phys = min(0.1, epoch / 500.0)
                loss = loss_mse + w_phys * (l_act + l_sant + 0.01 * l_dyn)

            opt_pos.zero_grad(set_to_none=True)
            scaler_pos.scale(loss).backward()
            scaler_pos.step(opt_pos)
            scaler_pos.update()
            
            total_loss += loss.item()
            l_mse_accum += loss_mse.item()
            l_phys_accum += (l_act + l_sant).item()
            
        sched_pos.step()
        if (epoch+1) % 100 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} | Pos Loss: {total_loss/len(stage2_loader):.6f} (MSE: {l_mse_accum/len(stage2_loader):.6f}, Phys: {l_phys_accum/len(stage2_loader):.6f})")

    # Guardar Position Predictor
    torch.save(pos_model.state_dict(), "position_predictor_final.pt")
    
    # Guardar Pipeline Completo para Export
    torch.save({
        'flux_model': flux_model.state_dict(),
        'pos_model': pos_model.state_dict(),
        'norm': {
            'u': (stats['u_mean'], stats['u_std']),
            'i': (stats['i_mean'], stats['i_std']),
            'phi': (stats['phi_mean'], stats['phi_std']),
            'y': (stats['y_mean'], stats['y_std'])
        },
        'pos_output_is_delta': True,
        'ruta2_domain_randomization': True
    }, "sensorless_pipeline_final.pt")
    
    print("\n✅ Entrenamiento Completo. Modelos guardados.")
    print("   - flux_observer_final.pt")
    print("   - position_predictor_final.pt")
    print("   - sensorless_pipeline_final.pt")

if __name__ == "__main__":
    main()
