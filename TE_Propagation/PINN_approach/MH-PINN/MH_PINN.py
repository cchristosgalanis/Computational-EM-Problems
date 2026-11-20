import torch 
import numpy as np
import torch.nn as nn
import Config as cfg
import pandas as pd 
import os
import functions as fn 
import matplotlib.pylab as plt 

# SIREN activation function
class Sine(nn.Module):
    def __init__(self, omega0 = 30.0):
        super(Sine, self).__init__()
        self.omega0 = omega0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x);


# MH-PINN solver class
class MH_PINN(nn.Module):
    def __init__(self, hidden_layers=[9], first_omega0=30.0, hidden_omega0=5.0, R_in=1.2, L=1.5, center=(1.0,0.5),
                 gauss_pulse=(2.0, 0.5, 0.3, 0.1), t_start = 0.0): # gauss_pulse = (H-, mean_x, mean_y, sigma_g)
        
        super(MH_PINN, self).__init__()

        # Geometry variables & Coordinate Mapping
        self.R_in = R_in
        self.L = L
        self.xc = float(center[0]) # get x
        self.yc = float(center[1]) # get y
        self.t_start = t_start
        self.prev_model = None


        self.H0, self.mean_x, self.mean_y, self.sigma_g = gauss_pulse

        # Stream1 : FCNN Stream ((xi, theta, t) -> inputs | 1 output N (Neural envelope))

        layer_sizes = [3] + hidden_layers + [1]
        self.network = nn.ModuleList()

        for i in range(len(layer_sizes)-1):
            is_first = (i==0)
            is_last = (i == len(layer_sizes)-2)
            linear = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            omega = first_omega0 if (is_first) else hidden_omega0

            # initialize SIREN weights
            self._init_siren_weights(linear, is_first, omega)
            self.network.append(linear)

            if not is_last:
                self.network.append(Sine(omega0=omega))


    def _init_siren_weights(self, linear: nn.Linear, is_first: bool, omega0: float):
        with torch.no_grad():
            num_in = linear.weight.size(1)
            if is_first:
                bound = 1.0 / num_in
            else:
                bound = np.sqrt(6.0 / num_in) / omega0

            linear.weight.uniform_(-bound, bound)
            linear.bias.zero_()


    # Stream2 : Deterministic Coordinate Mapping 

    # compute r 
    def map_r(self, xi: torch.Tensor) -> torch.Tensor:
        return self.R_in + self.L * (1.0+xi) / (1.0-xi + 1e-8) # r= Rin + L * (1+ξ)/(1-ξ)

    # compute jacobian and jacobian's derivative
    def map_jacobian(self, xi: torch.Tensor) -> torch.Tensor:
        J = (2*self.L) / ((1-xi + 1e-8)**2)
        J_der = (4*self.L) / ((1-xi + 1e-8)**3)
        return J, J_der

    # compute cartesian coordinates to polar coordinates
    def cart_to_xi_theta(self, x:torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor, torch.Tensor]:
        dx = x - self.xc
        dy = y - self.yc

        r = torch.sqrt(dx**2 + dy**2 + 1e-12)
        theta = torch.remainder(torch.atan2(dy,dx), 2.0 * np.pi)

        # reverse | xi = (r-Rin-L) / (r-Rin+L)
        xi = (r - self.R_in - self.L) / (r - self.R_in + self.L + 1e-8)
        xi = torch.clamp(xi, min=-1.0, max=0.999)
        return r, xi, theta


    # Hard constraints | Deterministc Funtioncs Φ(r) & H0(r, theta)
    # Φ(r) factor
    def radiation_factor(self, r: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(r + 1e-8)

    # initial pulse factor
    def gaussian_pulse(self, r: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        x = self.xc + r * torch.cos(theta)
        y = self.yc + r * torch.sin(theta)

        return self.H0 * torch.exp(-((x - self.mean_x)**2 + (y - self.mean_y)**2) / (2.0 * self.sigma_g**2))


    # Forward pass
    def forward(self, xi: torch.Tensor, theta: torch.Tensor, t: torch.Tensor, prev_model=None) -> torch.Tensor:
        # normalize inputs for FCNN stream
        theta_norm = (theta / np.pi) - 1.0
        t_norm = t - self.t_start # norm

        inputs = torch.cat([xi, theta_norm, t_norm], dim=1)

        # predict Neural envelope | stream1
        h = inputs
        for layer in self.network:
            h = layer(h)
        N_envelope = h # [N,1]

        # mapping | stream2
        r = self.map_r(xi)
        Phi = self.radiation_factor(r)

        # hard constraints module
        active_prev = prev_model if prev_model is not None else self.prev_model

        if active_prev is None:
            H0 = self.gaussian_pulse(r, theta)
        else:
            with torch.no_grad():
                H0 = active_prev(xi, theta, torch.full_like(t, self.t_start))

        # E constuct
        E_envelope = (H0 / Phi) + (t**2) * N_envelope

        Hz_pred = Phi * E_envelope


        return Hz_pred

    

def predict_hz_at_time(trained_models, x_grid, y_grid, t_value, device):
    num_windows = len(trained_models)
    dt_window = (cfg.t_end - cfg.t_start) / num_windows

    window_idx = int((t_value - cfg.t_start) / dt_window)
    window_idx = min(max(window_idx, 0), num_windows - 1)

    active_model = trained_models[window_idx]
    prev_model = trained_models[window_idx - 1] if window_idx > 0 else None
    active_model.eval()

    nx, ny = x_grid.shape
    x_flat = torch.tensor(x_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    y_flat = torch.tensor(y_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.full_like(x_flat, fill_value=t_value)

    with torch.no_grad():
        _, xi_flat, theta_flat = active_model.cart_to_xi_theta(x_flat, y_flat)
        Hz_pred = active_model(xi_flat, theta_flat, t_flat, prev_model=prev_model)

    return Hz_pred.cpu().numpy().reshape(nx, ny)


if __name__ == "__main__":
    NUM_CORES = 8
    os.environ["OMP_NUM_THREADS"] = str(NUM_CORES)
    os.environ["MKL_NUM_THREADS"] = str(NUM_CORES)
    torch.set_num_threads(NUM_CORES)

    device = torch.device("cpu")
    print(f"Using device: {device}")

    hidden_layers = [128, 128, 128, 128]
    model = MH_PINN(
        hidden_layers=hidden_layers,
        first_omega0=30.0,
        hidden_omega0=5.0,
        R_in=1.2,
        L=1.5,
        center=(1.0, 0.5),
        gauss_pulse=(2.0, 0.5, 0.3, 0.1),
        t_start=cfg.t_start
    )

    # training
    num_windows = 15
    trained_models = fn.train_time_marching(
        model=model,
        num_windows=num_windows,
        adam_epochs=2000,
        lbfgs_max_iter=1000,
        lr_adam=1e-4,
        lr_lbfgs=0.05,
        device=str(device)
    )

    # plot and csv
    x_vals = np.linspace(cfg.x_start, cfg.x_end, 200)
    y_vals = np.linspace(cfg.y_start, cfg.y_end, 200)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals, indexing='ij')

    target_times = np.linspace(cfg.t_start, cfg.t_end, 4)

    # csv
    all_snapshots_data = []
    for snap_id, t_val in enumerate(target_times):
        print(f"Exporting Hz data for Snapshot {snap_id} (t = {t_val:.3e} s)...")
        Hz_2d = predict_hz_at_time(trained_models, X_grid, Y_grid, t_val, device)

        df_snap = pd.DataFrame({
            'snapshot_id': snap_id,
            'time_s': t_val,
            'x': X_grid.flatten(),
            'y': Y_grid.flatten(),
            'Hz': Hz_2d.flatten()
        })
        all_snapshots_data.append(df_snap)

    final_df = pd.concat(all_snapshots_data, ignore_index=True)
    csv_filename = "MH_PINN_Hz_snapshots.csv"
    final_df.to_csv(csv_filename, index=False)
    print(f"Field data successfully exported to {csv_filename}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5), dpi=150)

    for col_idx, t_val in enumerate(target_times):
        print(f"Plotting Hz at t = {t_val:.3e} s...")
        Hz_2d = predict_hz_at_time(trained_models, X_grid, Y_grid, t_val, device)

        ax = axes[col_idx]
        vmax = np.max(np.abs(Hz_2d)) + 1e-12
        im = ax.pcolormesh(X_grid, Y_grid, Hz_2d, shading='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)

        ax.axvline(x=cfg.x_interface, color='black', linestyle='--', linewidth=1.2, label='Interface (x=1.0)')
        ax.set_title(f"Hz Field (Snapshot {col_idx})\nt = {t_val:.2e} s", fontsize=11)
        ax.set_xlabel("x (m)")
        if col_idx == 0:
            ax.set_ylabel("y (m)")
            ax.legend(loc='upper right', fontsize=8)
        else:
            ax.set_yticklabels([])

        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("TEz Mode Wave Propagation - Hz Field (MH-PINN)", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("MH_PINN_Hz_Snapshots.png", dpi=300, bbox_inches='tight')
    plt.show()
    

            




        
