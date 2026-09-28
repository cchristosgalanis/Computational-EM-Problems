import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

            
if __name__ == "__main__":
    # mps for apple silicon else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # model initialization
    layers = [2, 100, 100, 100, 100, 1] 
    model = wave_1D_PINN(layers, first_omega0=30.0, hidden_omega0=5.0).to(device)

    X_pde, X_ic, X_bc = pfn.create_points(pfn.N_pde, pfn.N_ic, pfn.N_bc, pfn.x_start, pfn.x_end, pfn.total_time)

    X_pde = X_pde.to(device)
    X_ic = X_ic.to(device)
    X_bc = X_bc.to(device)

    trained_model, loss_history = pfn.wave_solver_PINN(model, X_pde, X_ic, X_bc, pfn.epochs)

    # test PINN with c and t of numerical approach
    df_fdtd = pd.read_csv("FDTD_wave_equation_results_100FourierTerms.csv")
    x_fdtd = df_fdtd["x"].values
    E_fdtd = df_fdtd["E"].values

    # create collocation points for PINN
    x_tensor = torch.tensor(x_fdtd, dtype=torch.float32).view(-1,1).to(device)
    t_tensor = torch.full_like(x_tensor,1.5).to(device) # t tensor used for plot L2_norm

    model.eval()
    with torch.no_grad():
        X_test = torch.cat([x_tensor, t_tensor], dim=1)
        E_pinn_tensor = model(X_test)
        E_pinn = E_pinn_tensor.cpu().numpy().flatten()

    # compute relative L2 norm
    l2_error = np.linalg.norm(E_pinn - E_fdtd) / np.linalg.norm(E_fdtd)

    print(f"Relative L2 error between PINN and FDTD: {l2_error * 100:.3f}%")

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(x_fdtd, E_fdtd, label='FDTD (Ground Truth)', color='black', linestyle='--', linewidth=2)
    plt.plot(x_fdtd, E_pinn, label='PINN (Prediction)', color='mediumblue', linewidth=2, alpha=0.8)

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f'Comparison: PINN (100 neurons) vs FDTD (100 terms) at t=1.5\nRelative L2 Error: {l2_error * 100:.2f}%')
    plt.xlabel('Position (x)')
    plt.ylabel('Electric Field Amplitude E(x)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.fill_between(x_fdtd, E_fdtd, E_pinn, color='red', alpha=0.2, label='Error Margin') 
    plt.tight_layout()
    plt.savefig("PINN(100 neurons)_FDTD(100 terms).png", dpi=300,bbox_inches='tight')
    plt.show()      

    pfn.plot_pinn_results(trained_model, loss_history, pfn.x_start, pfn.x_end, pfn.total_time, device, grid_res=1000)



