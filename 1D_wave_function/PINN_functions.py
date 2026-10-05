import torch 
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

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
total_time = 2.0 # total time of simulation 
x_start = 0.0 
x_end = 5.0
dt = 0.5 # # parameter for time merching
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


# create points for time merching
def create_points_merch(N_pde, N_ic, N_bc, x_start, x_end, t_start, t_end):
    # create collocation points | (x,t) E (0,1)
    X_pde = torch.rand(N_pde,2)

    # get this values to real problem
    X_pde[:,0] = x_start + X_pde[:,0] * (x_end - x_start) # x parameter
    X_pde[:,1] = t_start + X_pde[:,1] * (t_end - t_start) # t parameter

    X_pde.requires_grad = True # for being able to use AD

    # points for initial condition (x,t) E [0,L] x [0]
    X_ic = torch.zeros(N_ic,2)
    X_ic[:,0] = x_start + torch.rand(N_ic) * (x_end - x_start)
    X_ic[:,1] = t_start
    X_ic.requires_grad = True 

    # points for Dirichlet Boundary Conditions
    points_per_bound = N_bc // 2 # seperate space to 2 (left and right)

    X_left = torch.zeros(points_per_bound, 2)
    X_left[:,0] = x_start
    X_left[:,1] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

    X_right = torch.zeros(points_per_bound,2)
    X_right[:,0] = x_end
    X_right[:,1] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

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

    # definition of lbfgs algorithm 
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


# function to compute loss for time merching
def compute_loss_marching(model, target_pos, target_vel, X_pde, X_ic, X_bc):
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
    # Dirichlet Boundary Conditions (0,t) = 0 and (L,t) = 0
# ----------------------------------------------------------------------------------------------------

    loss_bc = torch.mean((model(X_bc)**2))

# ----------------------------------------------------------------------------------------------------

    # using curr and pre model to compute loss for new initial conditions
    E_curr = model(X_ic)
    dE_curr_dt = torch.autograd.grad(
        E_curr, X_ic,
        grad_outputs = torch.ones_like(E_curr),
        create_graph=True
    )[0][:, 1:2]

    # compute loss for initial conditions
    loss_ic_pos = torch.mean((E_curr - target_pos)**2)
    loss_ic_vel = torch.mean((dE_curr_dt - target_vel)**2)

    return loss_pde + 1e3 * loss_bc + 1e3 * loss_ic_pos + loss_ic_vel

# create function for time-merching solution (Casuality Problem)
def time_merching_solution(model,x_start,x_end,total_time,dt,epochs):
    device = next(model.parameters()).device

    # set time for each model
    K = int(total_time / dt) # outter loop for each model
    print("\n Starting model training with Time-Merching approach...\n")

    models = []
    prev_model = None
    k = 0

    for k in range(K):
        t_start = k * dt
        t_end = (k+1) * dt
        X_pde, X_ic, X_bc = create_points_merch(N_pde, N_ic, N_bc, x_start, x_end, t_start,t_end)

        X_pde = X_pde.to(device)
        X_ic = X_ic.to(device)
        X_bc = X_bc.to(device)

        target_pos = None
        target_vel = None
        if k > 0:
            E_prev = prev_model(X_ic)
            dE_prev_dt = torch.autograd.grad(
                E_prev, X_ic, 
                grad_outputs=torch.ones_like(E_prev), 
                create_graph=False,
                retain_graph=True
            )[0][:, 1:2]
            
            target_pos = E_prev.detach()
            target_vel = dE_prev_dt.detach()

        optimizer = optim.Adam(model.parameters(), lr = 1e-4)

        for epoch in range(epochs):
            optimizer.zero_grad()

            if k == 0:
                loss = compute_loss(model, X_pde, X_ic, X_bc)
            else:
                loss = compute_loss_marching(model, target_pos, target_vel, X_pde, X_ic, X_bc)

            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                print(f"Epoch: {epoch} | Loss: {loss.item():.6e}")

        print(f"Model window {k} training finished with Adam | Continue with L-BFGS \n")

            # train with L-BFGS
        optim_lbfgs = optim.LBFGS(
                model.parameters(),
                max_iter=5000, 
                tolerance_grad=1e-7, 
                tolerance_change=1e-9, 
                history_size=50, 
                lr=0.05
        )

        lbfgs_iter = 0
        def closure():
            nonlocal lbfgs_iter
            optim_lbfgs.zero_grad()

            if k == 0:
                loss_lbfgs = compute_loss(model, X_pde, X_ic, X_bc)
            else:
                loss_lbfgs = compute_loss_marching(model, target_pos, target_vel, X_pde, X_ic, X_bc)
                
            loss_lbfgs.backward()

            if lbfgs_iter % 100 == 0:
                print(f"L-BFGS Step: {lbfgs_iter:04d} | Loss: {loss_lbfgs.item():.6e}")
            lbfgs_iter += 1
            return loss_lbfgs

        optim_lbfgs.step(closure)

            # keep previous model | freeze weights
        prev_model = copy.deepcopy(model)
        models.append(prev_model)
        prev_model.eval()
        for param in prev_model.parameters():
            param.requires_grad = False

    return models 

# function used to get models prediction 
def predict_wave(models_list, x_tensor, t_value, dt):
    k = int(t_value / dt)
    k = min(k, len(models_list) - 1) 
    
    target_model = models_list[k]
    target_model.eval()

    device = next(target_model.parameters()).device
    
    x_tensor = x_tensor.to(device)
    
    if x_tensor.dim() == 1:
        x_tensor = x_tensor.unsqueeze(1)
        
    N = x_tensor.shape[0]
  
    t_tensor = torch.full((N, 1), t_value, dtype=torch.float32, device=device)
    
    X_test = torch.cat([x_tensor, t_tensor], dim=1)
    
    with torch.no_grad():
        E_pred = target_model(X_test)
        
    return E_pred.cpu().numpy().flatten()

# ---- operators ----

def pde_operator(model, x):
    E = model(x)

    dE = torch.autograd.grad(E,
                             x,
                             torch.ones_like(E),
                            create_graph=True)[0]

    E_x = dE[:, 0:1]
    E_t = dE[:, 1:2]

    E_xx = torch.autograd.grad(dE[:, 0:1],
                                x,
                                torch.ones_like(E_x),
                                create_graph=True)[0][:, 0:1]

    E_tt = torch.autograd.grad(dE[:, 1:2],
                               x,
                               torch.ones_like(E_t),
                               create_graph=True)[0][:, 1:2]

    return (1.0 / C**2) * E_tt - E_xx


def ic_pos_operator(model, x):
    G = torch.exp(-((x[:, 0:1] - mi)**2) / (2 * (sigma**2)))
    return model(x) - G


def ic_vel_operator(model, x):
    E = model(x)
    dE = torch.autograd.grad(E,
                             x,
                             torch.ones_like(E),
                             create_graph=True)[0]
    return dE[:, 1:2]    


def bc_operator(model, x):
    return model(x)


# function to compute ntk traces values
def compute_ntk_traces(model, X_points, operator_fn, num_samples):
    """
        compute average trace of NTK matrix for given operator function
    """

    sample_size = min(X_points.shape[0], num_samples)
    idx = torch.randperm(X_points.shape[0])[:sample_size]
    X_sample = X_points[idx].clone().requires_grad_(True)

    trace = 0.0
    for i in range(sample_size):
        model.zero_grad()
        target = operator_fn(model, X_sample[i:i+1])

        # compute gradients of target with respect to all model's weights
        gradients = torch.autograd.grad(target, model.parameters(),allow_unused=True)

        trace += sum(torch.sum(g**2) for g in gradients if g is not None)

    return trace / sample_size


# function to compute adaptive loss weights based on NTK traces
def compute_adaptive_loss(model, X_pde, X_ic, X_bc, lam_pde, lam_ic_pos, lam_ic_vel, lam_bc):
    # pde loss
    f_pde = pde_operator(model, X_pde)
    loss_pde = torch.mean(f_pde**2)

    # ic position loss
    f_ic_pos = ic_pos_operator(model, X_ic)
    loss_ic_pos = torch.mean(f_ic_pos**2)

    # ic velocity loss
    f_ic_vel = ic_vel_operator(model, X_ic)
    loss_ic_vel = torch.mean(f_ic_vel**2)

    # bc loss
    f_bc = bc_operator(model, X_bc)
    loss_bc = torch.mean(f_bc**2)

    # total loss
    total_loss = (lam_pde*loss_pde + 
                  lam_ic_pos*loss_ic_pos + 
                  lam_ic_vel*loss_ic_vel +
                  lam_bc*loss_bc)

    return total_loss, loss_pde, loss_ic_pos, loss_ic_vel, loss_bc


#   NTK PINNs approach to train neural network
def wave_solver_PINN_NTK(model, X_pde, X_ic, X_bc, epochs, update_freq=100, alpha=0.9):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    loss_history = []

    # initialize dynamic weights
    lam_pde, lam_ic_pos, lam_ic_vel, lam_bc = 1.0, 1.0, 1.0, 1.0

    for epoch in range(epochs):
        if epoch % update_freq == 0:
            tr_pde = compute_ntk_traces(model, X_pde, pde_operator,100)
            tr_ic_pos = compute_ntk_traces(model, X_ic, ic_pos_operator,100)
            tr_ic_vel = compute_ntk_traces(model, X_ic, ic_vel_operator,100)
            tr_bc = compute_ntk_traces(model, X_bc, bc_operator,100)

            tr_total = tr_pde + tr_ic_pos + tr_ic_vel + tr_bc

            # using equations of paper
            lam_pde_1 = tr_total / (tr_pde + 1e-8)
            lam_ic_pos_1 = tr_total / (tr_ic_pos + 1e-8)
            lam_ic_vel_1 = tr_total / (tr_ic_vel + 1e-8)
            lam_bc_1 = tr_total / (tr_bc + 1e-8)

            # using moving averate
            lam_pde = alpha * lam_pde + (1 - alpha) * lam_pde_1
            lam_ic_pos = alpha * lam_ic_pos + (1 - alpha) * lam_ic_pos_1
            lam_ic_vel = alpha * lam_ic_vel + (1 - alpha) * lam_ic_vel_1
            lam_bc = alpha * lam_bc + (1 - alpha) * lam_bc_1

        optimizer.zero_grad()
        total_loss, loss_pde, loss_ic_pos, loss_ic_vel, loss_bc = compute_adaptive_loss(model, X_pde, X_ic, X_bc, lam_pde, lam_ic_pos, lam_ic_vel, lam_bc)

        total_loss.backward()
        optimizer.step()    
        loss_history.append(total_loss.item())

        if epoch % 500 == 0:
            print(f"Epoch: {epoch} | Total Loss: {total_loss.item():.6e} | PDE Loss: {loss_pde.item():.6e} | IC Pos Loss: {loss_ic_pos.item():.6e} | IC Vel Loss: {loss_ic_vel.item():.6e} | BC Loss: {loss_bc.item():.6e}")  
            print(f"   [NTK \u03bb weights] PDE: {lam_pde:.1f} | IC_pos: {lam_ic_pos:.1f} | IC_vel: {lam_ic_vel:.1f} | BC: {lam_bc:.1f}")

        print("\n--- Adam Training Finished. Starting L-BFGS with Frozen NTK Weights ---\n")

        # L-BFGS
        optim_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=5000, 
            tolerance_grad=1e-7, 
            tolerance_change=1e-9, 
            history_size=50, 
            lr=0.05
        )  

        lbfgs_iter = 0
        def closure():
            nonlocal lbfgs_iter
            optim_lbfgs.zero_grad()
            total_loss, _, _, _, _ = compute_adaptive_loss(
                model, X_pde, X_ic, X_bc, lam_pde, lam_ic_pos, lam_ic_vel, lam_bc
            )
            total_loss.backward()
            loss_history.append(total_loss.item())

            if lbfgs_iter % 100 == 0:
                print(f"L-BFGS Step: {lbfgs_iter:04d} | Loss: {total_loss.item():.6e}")
            lbfgs_iter += 1
            return total_loss

        optim_lbfgs.step(closure)
        print("\n--- Training Finished Completely ---\n")

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

    # create a dense 1D spatial grid
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
