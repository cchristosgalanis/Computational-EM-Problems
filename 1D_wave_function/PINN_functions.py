import torch 
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

"""
    this file contains functions that are used for PINN approach for 1D wave function. 
    Also contains parameters for each function.

"""

# parameters for functions
C = 5.0 # velocity for wave function
mi  = 2.5 # variables for Gaussian pulse
sigma = 0.45 # variables for Gaussing pulse
N_pde = 10000
N_ic = 1000
N_bc = 1000
total_time = 1.0 # total time of simulation 
x_start = 0.0 
x_end = 5.0
epochs = 5000 # adam and L-BFGS iterations


def create_points(N_pde, N_ic, N_bc, x_start, x_end, total_time):
    # create collocation points | (x,t) E (0,1)
    X_pde = torch.rand(N_pde,2)

    # get this values to real problem
    X_pde[:,0] = x_start + X_pde[:,0] * (x_end - x_start) # x parameter
    X_pde[:,1] = X_pde[:,1] * total_time # t parameter

    X_pde.requires_grad = True # for being able to use AD

    # points for initial condition (x,t) E [0,L] x [0]
    X_ic = torch.zeros(N_ic,2)
    X_ic[:,0] = x_start + torch.rand(N_ic) * (x_end - x_start)
    X_ic[:,1] = 0.0 
    X_ic.requires_grad = True 

    # points for Dirichlet Boundary Conditions
    points_per_bound = N_bc // 2 # seperate space to 2 (left and right)

    X_left = torch.zeros(points_per_bound, 2)
    X_left[:,0] = x_start
    X_left[:,1] = torch.rand(points_per_bound) * total_time

    X_right = torch.zeros(points_per_bound,2)
    X_right[:,0] = x_end
    X_right[:,1] = torch.rand(points_per_bound) * total_time

    X_bc = torch.cat([X_left,X_right], dim=0)

    return X_pde, X_ic, X_bc


# create function to compute total loss
def compute_loss(model, X_pde, X_ic, X_bc):
    """
        function to compute total loss
    """

# ----------------------------------------------------------------------------------------------------
    # PDE | physics loss
# ----------------------------------------------------------------------------------------------------

    E = model(X_pde)
    # compute 1st order of E with respect to X_pde
    dE_DXpde = torch.autograd.grad(
        outputs=E,
        inputs=X_pde,
        grad_outputs=torch.ones_like(E),
        retain_graph=True,
        create_graph=True
    )[0]

    # seperate derivatives
    E_x = dE_DXpde[:, 0:1] # θΕ / θx
    E_t = dE_DXpde[:, 1:2] # θΕ / θt

    # compute 2nd order derivatives
    E_xx = torch.autograd.grad(
        outputs=E_x,
        inputs=X_pde,
        grad_outputs=torch.ones_like(E_x),
        retain_graph=True,
        create_graph=True
    )[0][:, 0:1]


    E_tt = torch.autograd.grad(
        outputs=E_t,
        inputs=X_pde,
        grad_outputs=torch.ones_like(E_t),
        retain_graph=True,
        create_graph=True
    )[0][:, 1:2]

    # PDE function 
    f_pde = (1.0 / C**2) * E_tt -  E_xx

    # Physics loo
    loss_pde = torch.mean(f_pde**2)

# ----------------------------------------------------------------------------------------------------
    # Initial State (t=0) Loss where Gaussian pulse
# ----------------------------------------------------------------------------------------------------

    # Gaussian pulse 
    G = torch.exp(-((X_ic[:,0] - mi)**2) / (2*(sigma**2)))
    G = G.reshape(-1,1)

    # E_initial state (t=0)
    E_ic = model(X_ic)

    # initial state loss
    loss_ic = torch.mean((E_ic - G)**2)

# ----------------------------------------------------------------------------------------------------
    # Initial State (t=0) Loss where velocity is 0
# ----------------------------------------------------------------------------------------------------

    dE_dt = torch.autograd.grad(
        outputs=E_ic,
        inputs=X_ic,
        grad_outputs=torch.ones_like(E_ic),
        retain_graph=True,
        create_graph=True
    )[0][:, 1:2]

    # initial velocity loss
    loss_icvel = torch.mean(dE_dt**2)

# ----------------------------------------------------------------------------------------------------
    # Dirichlet Boundary Conditions (0,t) = 0 and (L,t) = 0
# ----------------------------------------------------------------------------------------------------

    loss_bc = torch.mean((model(X_bc)**2))

    return  loss_pde + 1e3 * loss_ic + loss_icvel + 1e3*loss_bc


# create training function 
def wave_solver_PINN(model, X_pde, X_ic, X_bc, epochs):
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(),lr = 1e-4)

    print("\n Neural network training ... \n")

    loss_history = []
    for epoch in range(epochs):
        # set grad to zero
        optimizer.zero_grad()

        # compute loss
        loss = compute_loss(model, X_pde, X_ic, X_bc)
        loss.backward()

        # opitmizer
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 500 == 0:
            curr_loss = loss.item()
            print(f"Epoch: {epoch} | Loss: {curr_loss:.6e} \n")

    print("Training finished\n")

    # using LBFGS optimizer of now on
    optim_lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter = 5000, # max step number
        tolerance_grad = 1e-7, # gradient -> 0
        tolerance_change = 1e-9, # no change
        history_size = 50, # previous steps in buffer
        lr = 0.05
    )  

    lbfgs_iter = 0
    def closure():
        nonlocal lbfgs_iter
        optim_lbfgs.zero_grad()

        # compute loss
        loss_lbfgs = compute_loss(model,X_pde, X_ic, X_bc)
        loss_lbfgs.backward()

        loss_history.append(loss_lbfgs.item())

        if lbfgs_iter % 100 == 0:
            print(f"L-BFGS Step: {lbfgs_iter:04d} | Loss: {loss_lbfgs.item():.6e}")

        lbfgs_iter += 1

        return loss_lbfgs

    optim_lbfgs.step(closure)
    print("\n Train finished\n")

    return model, loss_history


# function to plot total loss and PINNs solution approach
def plot_pinn_results(model, loss_history, x_start, x_end, T_total, device, grid_res=1000):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color='blue', label='Total Loss')
    plt.yscale('log')
    plt.xlabel('Epochs (Adam + L-BFGS)')
    plt.ylabel('Loss (Log Scale)')
    plt.title('PINN Training Convergence (t=1.5)')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pinn_loss_history_(100 neurons)_1.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create a dense 1D spatial grid
    x_axis = torch.linspace(x_start, x_end, grid_res).reshape(-1, 1)
    
    t_axis = torch.full_like(x_axis, T_total)
    
    X_test = torch.cat([x_axis, t_axis], dim=1)
    
    with torch.no_grad():
        E_pred = model(X_test.to(device)).cpu().detach().numpy().flatten()        
    x_np = x_axis.numpy().flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(x_np, E_pred, color='darkblue', linewidth=2, label=f'E(x, t={T_total})')
    plt.fill_between(x_np, E_pred, 0, color='blue', alpha=0.1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    plt.xlabel('Position (x)', fontsize=12)
    plt.ylabel('Electric Field Amplitude E(x)', fontsize=12)
    plt.title('PINN 1D Wave Equation - Final Snapshot (t=1.5)', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('pinn_1d_snapshot_(100 neurons)_1.png', dpi=300, bbox_inches='tight')
    plt.show()
