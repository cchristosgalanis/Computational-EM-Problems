import torch
import numpy as np
import Config as cfg
import copy

# function to create collocation points for PDE
def collocation_points(N_pde: int, t_start: float, t_end: float, device: str="cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # get config values
    xi_min = cfg.xi_min
    xi_max = cfg.xi_max

    pde_points = torch.rand(N_pde, 3, dtype=torch.float32, device=device)

    # xi sampling in [-1, xi_max]
    pde_points[:,0] = xi_min + pde_points[:,0] * (xi_max - xi_min)

    # theta sampling in [0,2π]
    pde_points[:,1] = pde_points[:,1] * (2.0*np.pi)

    # time sampling
    t = t_start + torch.rand(cfg.N_pde, dtype=torch.float32, device=device) * (t_end - t_start)
    pde_points[:,2] = t_start + pde_points[:,2] * (t_end - t_start)

    pde_points.requires_grad_(True)

    return pde_points


# function to compute residual function
def residual_function(model, pde_points, prev_model=None) -> torch.Tensor:
    """
        By using specific approach, there is need only for PDE residual function
    """

    xi = pde_points[:, 0:1]
    theta = pde_points[:, 1:2]
    t = pde_points[:, 2:3]

    # get mode's predictions
    Hz = model(xi, theta, t, prev_model=prev_model)
    ones = torch.ones_like(Hz)

    # First Derivatives
    dHz_dt = torch.autograd.grad(
        Hz, t, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    dHz_dxi = torch.autograd.grad(
        Hz, xi, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    dHz_dtheta = torch.autograd.grad(
        Hz, theta, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    # Second order Derivatives
    d2Hz_dt2 = torch.autograd.grad(
        dHz_dt, t, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    d2Hz_dxi2 = torch.autograd.grad(
        dHz_dxi, xi, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    d2Hz_dtheta2 = torch.autograd.grad(
        dHz_dtheta, theta, grad_outputs=ones, create_graph=True, retain_graph=True
    )[0]

    # compute geometry mapping parameters
    r = model.map_r(xi)
    J, J_der = model.map_jacobian(xi)

    # compute Laplacian in (ξ,θ)
    laplacian_Hz = ((1.0/J**2) * d2Hz_dxi2) + ((1.0/(r * J)) - (J_der / J**3)) * (dHz_dxi) + ((1.0/r**2) * d2Hz_dtheta2)

    # create space for dielectric value
    x_pos = model.xc + r * torch.cos(theta)
    eps_r = torch.where(x_pos >= cfg.x_interface, cfg.eps_r, 1.0)
    c_2 = (cfg.c0**2) / eps_r

    # PDE residual function
    R = d2Hz_dt2 - c_2 * laplacian_Hz

    return torch.mean(R**2)


# training function
def train_time_marching(
    model,
    num_windows: int = 15,
    adam_epochs: int = 2000,
    lbfgs_max_iter: int = 1000,
    lr_adam: float = 1e-4,
    lr_lbfgs: float = 0.05,
    device: str = "cpu"
):
    trained_models = []
    dt_window = (cfg.t_end - cfg.t_start) / num_windows
    prev_model = None

    current_model = model.to(device)

    for w in range(num_windows):
        t_w_start = cfg.t_start + w * dt_window
        t_w_end = t_w_start + dt_window

        print(f"Time Window {w+1}/{num_windows}: t in [{t_w_start:.3e}, {t_w_end:.3e}] s")

        current_model.t_start = t_w_start
        current_model.prev_model = prev_model

        # create collocation points
        pde_points = collocation_points(cfg.N_pde, t_w_start, t_w_end, device=device)

        # Adam optimizer
        print("\n ===== Adam Training ===")
        optimizer_adam = torch.optim.Adam(current_model.parameters(), lr=lr_adam)
        current_model.train()

        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()
            loss = residual_function(current_model, pde_points, prev_model=prev_model)
            loss.backward()
            optimizer_adam.step()

            if epoch == 1 or epoch % 500 == 0 or epoch == adam_epochs:
                print(f"Window [{w+1}/{num_windows}] | Adam Epoch: {epoch:04d}/{adam_epochs} | Loss: {loss.item():.6e}")

        # L-BFGS
        print("\n === L-BFGS Optimizer ===")
        optimizer_lbfgs = torch.optim.LBFGS(
            current_model.parameters(),
            lr=lr_lbfgs,
            max_iter=lbfgs_max_iter,
            history_size=50,
            line_search_fn="strong_wolfe"
        )

        lbfgs_step = [0]
        def closure():
            optimizer_lbfgs.zero_grad()
            loss = residual_function(current_model, pde_points, prev_model=prev_model)
            loss.backward()
            if lbfgs_step[0] % 100 == 0:
                print(f"Window [{w+1}/{num_windows}] | L-BFGS Step: {lbfgs_step[0]:04d} | Loss: {loss.item():.6e}")
            lbfgs_step[0] += 1
            return loss

        optimizer_lbfgs.step(closure)

        finished_model = copy.deepcopy(current_model)
        finished_model.eval()
        for param in finished_model.parameters():
            param.requires_grad = False

        trained_models.append(finished_model)
        prev_model = finished_model

        current_model = copy.deepcopy(current_model).to(device)
        for param in current_model.parameters():
            param.requires_grad = True

    return trained_models

