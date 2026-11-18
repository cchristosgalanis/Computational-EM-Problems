import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

import Config as cfg
import pinn_function as pfn


NUM_CORES = 8
os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
torch.set_num_threads(NUM_CORES)

# class to create activation function Sine
class Sine(nn.Module):
    def __init__(self, omega_0=30.0):
        super(Sine, self).__init__()
        self.omega_0 = omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


# class for PINN 
class TEz_solver(nn.Module):
    def __init__(self, layers, first_omega0=30.0, hidden_omega0=5.0):
        super(TEz_solver, self).__init__()

        self.first_omega0 = first_omega0
        self.hidden_omega0 = hidden_omega0

        self.x_min, self.x_max = cfg.x_start, cfg.x_end
        self.y_min, self.y_max = cfg.y_start, cfg.y_end
        self.t_min, self.t_max = cfg.t_start, cfg.t_end

        layer_sizes = [3] + layers + [3]
        self.network = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            is_first_layer = (i == 0)
            is_last_layer = (i == len(layer_sizes) - 2)

            linear_layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            omega = self.first_omega0 if is_first_layer else self.hidden_omega0
            
            # linear state
            self.init_weights(linear_layer, is_first_layer, omega)
            
            # linear state push to network
            self.network.append(linear_layer)

            # activation function
            if not is_last_layer:
                self.network.append(Sine(omega_0=omega))

    # initialize weights
    def init_weights(self, linear: nn.Linear, is_first: bool, omega_0: float):
        with torch.no_grad():
            num_input = linear.weight.size(1)

            if is_first:
                # first layer
                bound = 1.0 / num_input
            else:
                # hidden layers l > 1
                bound = np.sqrt(6.0 / num_input) / omega_0

            linear.weight.uniform_(-bound, bound)
            linear.bias.zero_()

    # normalize function
    def normalize_inputs(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_norm = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_norm, y_norm, t_norm], dim=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # normalize
        h = self.normalize_inputs(x, y, t)

        # pass through layers
        for layer in self.network:
            h = layer(h)

        # get outputs 
        Ex = h[:, 0:1]
        Ey = h[:, 1:2]
        Hz = h[:, 2:3]

        return Ex, Ey, Hz


"""
    
    Time-Merching approach


"""



def predict_tez_at_time(trained_models, x_grid, y_grid, t_value, device):
    num_windows = len(trained_models)
    dt_window = (cfg.t_end - cfg.t_start) / num_windows

    window_idx = int((t_value - cfg.t_start) / dt_window)
    window_idx = min(max(window_idx, 0), num_windows - 1) 

    active_model = trained_models[window_idx]
    active_model.eval()

    nx, ny = x_grid.shape
    x_flat = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_flat = torch.tensor(y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.full_like(x_flat, fill_value=t_value)

    with torch.no_grad():
        Ex_pred, Ey_pred, Hz_pred = active_model(x_flat, y_flat, t_flat)

    Ex_2d = Ex_pred.cpu().numpy().reshape(nx, ny)
    Ey_2d = Ey_pred.cpu().numpy().reshape(nx, ny)
    Hz_2d = Hz_pred.cpu().numpy().reshape(nx, ny)

    return Ex_2d, Ey_2d, Hz_2d


if __name__ == "__main__":
    device = torch.device("cpu")
    print(f"Using device: {device}")

    hidden_layers = [64, 64, 64, 64]
    model = TEz_solver(hidden_layers, first_omega0=30.0, hidden_omega0=5.0)

    # model training
    num_windows = 15 
    trained_models = pfn.train_time_marching(
        model=model,
        num_windows=num_windows,
        adam_epochs=2000,
        lbfgs_max_iter=1000,
        lr_adam=1e-4,
        lr_lbfgs=0.05,
        device=str(device)
    )

    x_vals = np.linspace(cfg.x_start, cfg.x_end, 200)
    y_vals = np.linspace(cfg.y_start, cfg.y_end, 200)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')

    target_times = np.linspace(cfg.t_start, cfg.t_end, 4)

    # csv files
    all_snapshots_data = []
    
    for snap_id, t_val in enumerate(target_times):
        print(f"Exporting field data for Snapshot {snap_id} (t = {t_val:.3e} s)...")
        Ex_2d, Ey_2d, Hz_2d = predict_tez_at_time(trained_models, X_grid, Y_grid, t_val, device)
        
        df_snap = pd.DataFrame({
            'snapshot_id': snap_id,
            'time_s': t_val,
            'x': X_grid.flatten(),
            'y': Y_grid.flatten(),
            'Hz': Hz_2d.flatten(),
            'Ex': Ex_2d.flatten(),
            'Ey': Ey_2d.flatten()
        })
        all_snapshots_data.append(df_snap)

    final_df = pd.concat(all_snapshots_data, ignore_index=True)
    csv_filename = "PINN_fields_snapshots.csv"
    final_df.to_csv(csv_filename, index=False)
    print(f" Field data successfully exported to {csv_filename}")


    #plots
    fig, axes = plt.subplots(3, 4, figsize=(22, 12), dpi=150)
    field_names = ['Ex', 'Ey', 'Hz']

    for col_idx, t_val in enumerate(target_times):
        print(f"Plotting fields at t = {t_val:.3e} s...")
        Ex_2d, Ey_2d, Hz_2d = predict_tez_at_time(trained_models, X_grid, Y_grid, t_val, device)
        fields_data = [Ex_2d, Ey_2d, Hz_2d]
        
        for row_idx, field_data in enumerate(fields_data):
            ax = axes[row_idx, col_idx]
            
            vmax = np.max(np.abs(field_data)) + 1e-12
            im = ax.pcolormesh(X_grid, Y_grid, field_data, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

            ax.axvline(x=cfg.x_interface, color='black', linestyle='--', linewidth=1.2)

            pml_rect = plt.Rectangle(
                (cfg.x_start + cfg.d_pml_x, cfg.y_start + cfg.d_pml_y),
                (cfg.x_end - cfg.x_start) - 2 * cfg.d_pml_x,
                (cfg.y_end - cfg.y_start) - 2 * cfg.d_pml_y,
                linewidth=1.2, edgecolor='gray', facecolor='none', linestyle=':', alpha=0.7
            )
            ax.add_patch(pml_rect)

            ax.set_title(f"{field_names[row_idx]} Field (Snapshot {col_idx})", fontsize=11)
            
            if row_idx == 2:
                ax.set_xlabel("x (m)")
            else:
                ax.set_xticklabels([])
                
            if col_idx == 0:
                ax.set_ylabel("y (m)")
            else:
                ax.set_yticklabels([])
                
            ax.set_aspect('equal')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            if row_idx == 0 and col_idx == 0:
                ax.axvline(x=cfg.x_interface, color='black', linestyle='--', linewidth=1.2, label='Interface (x=1.0)')
                dummy_rect = plt.Rectangle((0,0), 0, 0, linewidth=1.2, edgecolor='gray', facecolor='none', linestyle=':', label='PML Boundary')
                ax.add_patch(dummy_rect)
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle("TEz Mode Wave Propagation - All Fields (SIREN PINN)", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig("TEz_SIREN_PML_All_Fields.png", dpi=300, bbox_inches='tight')
    plt.show()