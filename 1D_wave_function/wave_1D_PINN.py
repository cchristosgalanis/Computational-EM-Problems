import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import PINN_functions as pfn

# class to create activation function Sine
class Sine(nn.Module):
    def __init__(self, omega_0 = 30.0):
        super(Sine, self).__init__()

        self.omega_0 = omega_0

    def forward(self,x):
        return torch.sin(self.omega_0 * x)

# class for PINN 
class wave_1D_PINN(nn.Module):
    def __init__(self, layers, first_omega0 = 30.0, hidden_omega0 = 5.0):
        super(wave_1D_PINN, self).__init__()

        self.first_omega0 = first_omega0
        self.hiddenen_omega0 = hidden_omega0

        self.x_min, self.x_max = 0.0, 5.0
        self.t_min, self.t_max = 0.0, 2.0

        self.linears = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i],layers[i+1]))

        # mathematical initialization
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for i, linear in enumerate(self.linears):
                num_input = linear.weight.size(1)

                if i == 0:
                    # first layer
                    bound = 1.0 / num_input
                    linear.weight.uniform_(-bound,bound)
                else:
                    # hidden layers l > 1
                    bound = np.sqrt(6.0 / num_input) / self.hiddenen_omega0
                    linear.weight.uniform_(-bound, bound)

                # put biases to 0
                linear.bias.zero_()

    def forward(self, X):
        x = X[:, 0:1]
        t = X[:, 1:2]

        # normalize to [-1,1]
        x_norm = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        t_norm = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0

        H = torch.cat([x_norm, t_norm], dim=1)

        for i in range(len(self.linears) - 1):
            H = self.linears[i](H)
            if i == 0:
                H = torch.sin(self.first_omega0 * H)
            else:
                H = torch.sin(self.hiddenen_omega0 * H)

        E = self.linears[-1](H)
        return E


# ------------------------------------------------------------------------------------------------------------

"""
    block of code with vanilla approach

"""

# if __name__ == "__main__":
#     # mps for apple silicon else CPU
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # model initialization
#     layers = [2, 100, 100, 100, 100, 1] 
#     model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

#     X_pde, X_ic, X_bc = pfn.create_points(pfn.N_pde, pfn.N_ic, pfn.N_bc, pfn.x_start, pfn.x_end, pfn.total_time)

#     X_pde = X_pde.to(device)
#     X_ic = X_ic.to(device)
#     X_bc = X_bc.to(device)

#     trained_model, loss_history = pfn.wave_solver_PINN(model, X_pde, X_ic, X_bc, pfn.epochs)

#     # test PINN with c and t of numerical approach
#     df_fdtd = pd.read_csv("FDTD_wave_equation_results_100FourierTerms.csv")
#     x_fdtd = df_fdtd["x"].values
#     E_fdtd = df_fdtd["E"].values

#     # create collocation points for PINN
#     x_tensor = torch.tensor(x_fdtd, dtype=torch.float32).view(-1,1).to(device)
#     t_tensor = torch.full_like(x_tensor,1.6).to(device) # t tensor used for plot L2_norm

#     model.eval()
#     with torch.no_grad():
#         X_test = torch.cat([x_tensor, t_tensor], dim=1)
#         E_pinn_tensor = model(X_test)
#         E_pinn = E_pinn_tensor.cpu().numpy().flatten()

#     # compute relative L2 norm
#     l2_error = np.linalg.norm(E_pinn - E_fdtd) / np.linalg.norm(E_fdtd)

#     print(f"Relative L2 error between PINN and FDTD: {l2_error * 100:.3f}%")

#     # create plot with FDTD and PINNs approach
#     plt.figure(figsize=(10, 6), dpi=150)
#     plt.plot(x_fdtd, E_fdtd, label='FDTD (Ground Truth)', color='black', linestyle='--', linewidth=2)
#     plt.plot(x_fdtd, E_pinn, label='PINN (Prediction)', color='mediumblue', linewidth=2, alpha=0.8)

#     plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
#     plt.title(f'Comparison: PINN (100 neurons) vs FDTD (100 terms) at t=1.6\nRelative L2 Error: {l2_error * 100:.2f}%')
#     plt.xlabel('Position (x)')
#     plt.ylabel('Electric Field Amplitude E(x)')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.fill_between(x_fdtd, E_fdtd, E_pinn, color='red', alpha=0.2, label='Error Margin') 
#     plt.tight_layout()
#     plt.savefig("PINN(100 neurons)_FDTD(100 terms).png", dpi=300,bbox_inches='tight')
#     plt.show()      

#     pfn.plot_pinn_results(trained_model, loss_history, pfn.x_start, pfn.x_end, pfn.total_time, device, grid_res=1000) 

# --------------------------------------------------------------------------------------------------------------------------------------------

"""
    code for time-merching approach for solution

"""

# if __name__ == "__main__":
#     # mps for apple silicon else CPU
#     device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # model initialization
#     layers = [2, 100, 100, 100, 100, 1] 
#     model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

#     dt = 0.5
#     models_list = pfn.time_merching_solution(model, pfn.x_start, pfn.x_end, pfn.total_time, dt, pfn.epochs)

#     target_times = [0.4, 0.6, 1.0, 1.6, 2.0]

#     df_fdtd = pd.read_csv("FDTD_wave_equation_results_100FourierTerms.csv")
    
#     for t_val in target_times:
#         print(f"\n--- Evaluating and Plotting for t = {t_val} ---")
#         df_current = df_fdtd[np.isclose(df_fdtd['t'], t_val, atol=1e-4)] 
        
#         x_fdtd = df_current["x"].values
#         E_fdtd = df_current["E"].values

#         x_tensor = torch.tensor(x_fdtd, dtype=torch.float32).view(-1, 1).to(device)

#         E_pinn = pfn.predict_wave(models_list, x_tensor, t_value=t_val, dt=dt)

#         l2_error = np.linalg.norm(E_pinn - E_fdtd) / np.linalg.norm(E_fdtd)
#         print(f"Relative L2 error at t={t_val}: {l2_error * 100:.3f}%")

#         plt.figure(figsize=(10, 6), dpi=150)
#         plt.plot(x_fdtd, E_fdtd, label='FDTD (Ground Truth)', color='black', linestyle='--', linewidth=2)
#         plt.plot(x_fdtd, E_pinn, label='PINN (Prediction)', color='mediumblue', linewidth=2, alpha=0.8)

#         plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
#         plt.title(f'Comparison: PINN Time-Marching vs FDTD at t={t_val}\nRelative L2 Error: {l2_error * 100:.2f}%')
#         plt.xlabel('Position (x)')
#         plt.ylabel('Electric Field Amplitude E(x)')
#         plt.legend()
#         plt.grid(True, linestyle='--', alpha=0.5)
#         plt.fill_between(x_fdtd, E_fdtd, E_pinn, color='red', alpha=0.2, label='Error Margin') 
#         plt.tight_layout()
#         plt.savefig(f"PINN_TimeMarching_t{t_val}.png", dpi=300, bbox_inches='tight')
#         plt.show()


# ------------------------------------------------------------------------------------------------------------------------

"""
    block of code for NTK trace values approach for spectral bias problem
"""

# if __name__ == "__main__":
#     # mps for apple silicon else CPU
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # model initialization
#     layers = [2, 100, 100, 100, 100, 1] 
#     model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

#     X_pde, X_ic, X_bc = pfn.create_points(pfn.N_pde, pfn.N_ic, pfn.N_bc, pfn.x_start, pfn.x_end, pfn.total_time)

#     X_pde = X_pde.to(device)
#     X_ic = X_ic.to(device)
#     X_bc = X_bc.to(device)

#     # training to whole time domain
#     trained_model, loss_history = pfn.wave_solver_PINN_NTK(model, X_pde, X_ic, X_bc, pfn.epochs)

#     df_fdtd = pd.read_csv("FDTD_wave_equation_results_100FourierTerms.csv")
    
#     target_times = [0.4, 0.6, 1.0, 1.6, 2.0]
    
#     model.eval()
#     with torch.no_grad():
#         for t_val in target_times:
#             print(f"\n--- Evaluating and Plotting for t = {t_val} ---")
            
#             df_current = df_fdtd[np.isclose(df_fdtd['t'], t_val, atol=1e-4)] 
            
#             if df_current.empty:
#                 break
                
#             x_fdtd = df_current["x"].values
#             E_fdtd = df_current["E"].values

#             # create collocation points for PINN
#             x_tensor = torch.tensor(x_fdtd, dtype=torch.float32).view(-1,1).to(device)
#             t_tensor = torch.full_like(x_tensor, t_val).to(device) 

#             X_test = torch.cat([x_tensor, t_tensor], dim=1)
#             E_pinn_tensor = model(X_test)
#             E_pinn = E_pinn_tensor.cpu().numpy().flatten()

#             # compute relative L2 norm
#             l2_error = np.linalg.norm(E_pinn - E_fdtd) / np.linalg.norm(E_fdtd)

#             print(f"Relative L2 error between NTK-PINN and FDTD at t={t_val}: {l2_error * 100:.3f}%")

#             # create plot with FDTD and PINNs approach
#             plt.figure(figsize=(10, 6), dpi=150)
#             plt.plot(x_fdtd, E_fdtd, label='FDTD (Ground Truth)', color='black', linestyle='--', linewidth=2)
#             plt.plot(x_fdtd, E_pinn, label='NTK-PINN (Prediction)', color='mediumblue', linewidth=2, alpha=0.8)

#             plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
#             plt.title(f'Comparison: NTK-PINN vs FDTD at t={t_val}\nRelative L2 Error: {l2_error * 100:.2f}%')
#             plt.xlabel('Position (x)')
#             plt.ylabel('Electric Field Amplitude E(x)')
#             plt.legend()
#             plt.grid(True, linestyle='--', alpha=0.5)
#             plt.fill_between(x_fdtd, E_fdtd, E_pinn, color='red', alpha=0.2, label='Error Margin') 
#             plt.tight_layout()
            
#             plt.savefig(f"PINN_NTK_FDTD_t{t_val}.png", dpi=300, bbox_inches='tight')
#             plt.show()      

#     pfn.plot_pinn_results(trained_model, loss_history, pfn.x_start, pfn.x_end, pfn.total_time, device, grid_res=1000)

# ---------------------------------------------------------------------------------------------------------------------

""" 
    block of code to create NTK approach to heatmap

"""

# if __name__ == "__main__":
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     layers = [2, 100, 100, 100, 100, 1] 
#     model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

#     X_pde, X_ic, X_bc = pfn.create_points(pfn.N_pde, pfn.N_ic, pfn.N_bc, pfn.x_start, pfn.x_end, pfn.total_time)
#     X_pde, X_ic, X_bc = X_pde.to(device), X_ic.to(device), X_bc.to(device)

#     trained_model, loss_history = pfn.wave_solver_PINN_NTK(model, X_pde, X_ic, X_bc, pfn.epochs)

#     print("\n--- Generating Heatmaps ---")
#     df_fdtd = pd.read_csv("FDTD_heatmap.csv")
    
#     x_test = torch.tensor(df_fdtd['x'].values, dtype=torch.float32).view(-1, 1).to(device)
#     t_test = torch.tensor(df_fdtd['t'].values, dtype=torch.float32).view(-1, 1).to(device)
#     X_test_full = torch.cat([x_test, t_test], dim=1)

#     model.eval()
#     with torch.no_grad():
#         E_pinn_full = model(X_test_full).cpu().numpy().flatten()

#     df_fdtd['E_pinn'] = E_pinn_full
#     df_fdtd['Abs_Error'] = np.abs(df_fdtd['E'] - df_fdtd['E_pinn'])

#     heatmap_pred = df_fdtd.pivot(index='t', columns='x', values='E_pinn')
#     heatmap_err = df_fdtd.pivot(index='t', columns='x', values='Abs_Error')

#     num_x_ticks, num_y_ticks = 6, 5
#     x_tick_positions = np.linspace(0, len(heatmap_pred.columns) - 1, num_x_ticks)
#     x_tick_labels = np.round(np.linspace(0, 5.0, num_x_ticks), 1)
#     y_tick_positions = np.linspace(0, len(heatmap_pred.index) - 1, num_y_ticks)
#     y_tick_labels = np.round(np.linspace(0, 2.0, num_y_ticks), 1)

#     plt.figure(figsize=(10, 6))
#     ax1 = sns.heatmap(heatmap_pred, cmap='jet', cbar_kws={'label': 'Electric Field Amplitude E(x,t)'})
#     ax1.invert_yaxis()
#     ax1.set_xticks(x_tick_positions)
#     ax1.set_xticklabels(x_tick_labels, rotation=0)
#     ax1.set_yticks(y_tick_positions)
#     ax1.set_yticklabels(y_tick_labels, rotation=0)
#     plt.title('NTK-PINN Prediction', fontsize=14, pad=15)
#     plt.xlabel('Space (x)', fontsize=12)
#     plt.ylabel('Time (t)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('NTK_PINN_Prediction_Heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     ax2 = sns.heatmap(heatmap_err, cmap='inferno', cbar_kws={'label': 'Absolute Error |E_real - E_pinn|'})
#     ax2.invert_yaxis()
#     ax2.set_xticks(x_tick_positions)
#     ax2.set_xticklabels(x_tick_labels, rotation=0)
#     ax2.set_yticks(y_tick_positions)
#     ax2.set_yticklabels(y_tick_labels, rotation=0)
#     plt.title('Absolute Error (NTK-PINN vs FDTD)', fontsize=14, pad=15)
#     plt.xlabel('Space (x)', fontsize=12)
#     plt.ylabel('Time (t)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('NTK_PINN_AbsoluteError_Heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     print("[Success] Heatmaps generated and saved.")


# -----------------------------------------------------------------------------------------

"""
    block of code. for vanilla approach and get heatmaps

"""


# if __name__ == "__main__":
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     layers = [2, 100, 100, 100, 100, 1] 
#     model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

#     X_pde, X_ic, X_bc = pfn.create_points(pfn.N_pde, pfn.N_ic, pfn.N_bc, pfn.x_start, pfn.x_end, pfn.total_time)
#     X_pde, X_ic, X_bc = X_pde.to(device), X_ic.to(device), X_bc.to(device)

#     trained_model, loss_history = pfn.wave_solver_PINN(model, X_pde, X_ic, X_bc, pfn.epochs)

#     print("\n--- Generating Vanilla PINN Heatmaps ---")
#     df_fdtd = pd.read_csv("FDTD_heatmap.csv")
    
#     x_test = torch.tensor(df_fdtd['x'].values, dtype=torch.float32).view(-1, 1).to(device)
#     t_test = torch.tensor(df_fdtd['t'].values, dtype=torch.float32).view(-1, 1).to(device)
#     X_test_full = torch.cat([x_test, t_test], dim=1)

#     model.eval()
#     with torch.no_grad():
#         E_pinn_full = model(X_test_full).cpu().numpy().flatten()

#     df_fdtd['E_pinn'] = E_pinn_full
#     df_fdtd['Abs_Error'] = np.abs(df_fdtd['E'] - df_fdtd['E_pinn'])

#     heatmap_pred = df_fdtd.pivot(index='t', columns='x', values='E_pinn')
#     heatmap_err = df_fdtd.pivot(index='t', columns='x', values='Abs_Error')

#     num_x_ticks, num_y_ticks = 6, 5
#     x_tick_positions = np.linspace(0, len(heatmap_pred.columns) - 1, num_x_ticks)
#     x_tick_labels = np.round(np.linspace(0, 5.0, num_x_ticks), 1)
#     y_tick_positions = np.linspace(0, len(heatmap_pred.index) - 1, num_y_ticks)
#     y_tick_labels = np.round(np.linspace(0, 2.0, num_y_ticks), 1)

#     plt.figure(figsize=(10, 6))
#     ax1 = sns.heatmap(heatmap_pred, cmap='jet', cbar_kws={'label': 'Electric Field Amplitude E(x,t)'})
#     ax1.invert_yaxis()
#     ax1.set_xticks(x_tick_positions)
#     ax1.set_xticklabels(x_tick_labels, rotation=0)
#     ax1.set_yticks(y_tick_positions)
#     ax1.set_yticklabels(y_tick_labels, rotation=0)
#     plt.title('Vanilla PINN Prediction', fontsize=14, pad=15)
#     plt.xlabel('Space (x)', fontsize=12)
#     plt.ylabel('Time (t)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('Vanilla_PINN_Prediction_Heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     plt.figure(figsize=(10, 6))
#     ax2 = sns.heatmap(heatmap_err, cmap='inferno', cbar_kws={'label': 'Absolute Error |E_real - E_pinn|'})
#     ax2.invert_yaxis()
#     ax2.set_xticks(x_tick_positions)
#     ax2.set_xticklabels(x_tick_labels, rotation=0)
#     ax2.set_yticks(y_tick_positions)
#     ax2.set_yticklabels(y_tick_labels, rotation=0)
#     plt.title('Absolute Error (Vanilla PINN vs FDTD)', fontsize=14, pad=15)
#     plt.xlabel('Space (x)', fontsize=12)
#     plt.ylabel('Time (t)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig('Vanilla_PINN_AbsoluteError_Heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     print("[Success] Vanilla PINN heatmaps generated and saved.")


# ----------------------------------------------------------------------------------------

"""
    block of code for time merching approach and get heatmas

"""


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    layers = [2, 100, 100, 100, 100, 1] 
    model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

    dt_marching = 0.5  
    print(f"\n--- Training PINN with Time Marching (dt={dt_marching}) ---")
    models_list = pfn.time_merching_solution(
        model, pfn.x_start, pfn.x_end, pfn.total_time, dt_marching, pfn.epochs
    )

    print("\n--- Generating Time Marching PINN Heatmaps ---")
    df_fdtd = pd.read_csv("FDTD_heatmap.csv")
    
    unique_times = df_fdtd['t'].unique()
    
    E_pinn_full = np.zeros(len(df_fdtd))
    
    for t_val in unique_times:
        t_indices = df_fdtd.index[df_fdtd['t'] == t_val].tolist()
        
        x_vals = df_fdtd.loc[t_indices, 'x'].values
        x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1).to(device)
        
        E_pred_t = pfn.predict_wave(models_list, x_tensor, t_value=t_val, dt=dt_marching)
        
        E_pinn_full[t_indices] = E_pred_t

    df_fdtd['E_pinn'] = E_pinn_full
    df_fdtd['Abs_Error'] = np.abs(df_fdtd['E'] - df_fdtd['E_pinn'])

    heatmap_pred = df_fdtd.pivot(index='t', columns='x', values='E_pinn')
    heatmap_err = df_fdtd.pivot(index='t', columns='x', values='Abs_Error')

    num_x_ticks, num_y_ticks = 6, 5
    x_tick_positions = np.linspace(0, len(heatmap_pred.columns) - 1, num_x_ticks)
    x_tick_labels = np.round(np.linspace(0, 5.0, num_x_ticks), 1)
    y_tick_positions = np.linspace(0, len(heatmap_pred.index) - 1, num_y_ticks)
    y_tick_labels = np.round(np.linspace(0, 2.0, num_y_ticks), 1)

    plt.figure(figsize=(10, 6))
    ax1 = sns.heatmap(heatmap_pred, cmap='jet', cbar_kws={'label': 'Electric Field Amplitude E(x,t)'})
    ax1.invert_yaxis()
    ax1.set_xticks(x_tick_positions)
    ax1.set_xticklabels(x_tick_labels, rotation=0)
    ax1.set_yticks(y_tick_positions)
    ax1.set_yticklabels(y_tick_labels, rotation=0)
    plt.title('Time Marching PINN Prediction', fontsize=14, pad=15)
    plt.xlabel('Space (x)', fontsize=12)
    plt.ylabel('Time (t)', fontsize=12)
    plt.tight_layout()
    plt.savefig('TimeMarching_PINN_Prediction_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    ax2 = sns.heatmap(heatmap_err, cmap='inferno', cbar_kws={'label': 'Absolute Error |E_real - E_pinn|'})
    ax2.invert_yaxis()
    ax2.set_xticks(x_tick_positions)
    ax2.set_xticklabels(x_tick_labels, rotation=0)
    ax2.set_yticks(y_tick_positions)
    ax2.set_yticklabels(y_tick_labels, rotation=0)
    plt.title('Absolute Error (Time Marching PINN vs FDTD)', fontsize=14, pad=15)
    plt.xlabel('Space (x)', fontsize=12)
    plt.ylabel('Time (t)', fontsize=12)
    plt.tight_layout()
    plt.savefig('TimeMarching_PINN_AbsoluteError_Heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("[Success] Time Marching PINN heatmaps generated and saved.")