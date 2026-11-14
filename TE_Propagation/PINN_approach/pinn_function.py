import numpy as np
import torch
import torch.optim as optim
import copy
import Config as cfg

"""  
    =====================================================================================================================

                    this file contains functions to implement PINN solver for TEz mode 
    
    =====================================================================================================================

"""

"""
    # function to create points for simulation | approach problem through Time-Merching perception

"""
def create_points(N_pde, N_ic, N_bc, x_start, x_end, y_start, y_end,t_start, t_end):
    """
        creating collocation points for PDE 
    """
    # assume that points (x,y) -> XYT_pde
    XYT_pde = torch.rand(N_pde, 3)

    XYT_pde[:,0] = x_start + XYT_pde[:,0] * (x_end - x_start) # x-axe 
    XYT_pde[:,1] = y_start + XYT_pde[:,1] * (y_end - y_start) # y-axe
    XYT_pde[:,2] = t_start + XYT_pde[:,2] * (t_end - t_start) # time

    XYT_pde.requires_grad = True # to use AD

    """
        creating initial condition points
    """

    XYT_initial = torch.rand(N_ic, 3)
    XYT_initial[:,0] = x_start + XYT_initial[:,0] * (x_end - x_start)
    XYT_initial[:,1] = y_start + XYT_initial[:,1] * (y_end - y_start)
    XYT_initial[:,2] = t_start

    XYT_initial.requires_grad = True # for AD

    """
        creating points for boundary condtions
    """

    points_per_bound = N_bc // 4

    # X-axe: left position x=0
    XYT_left = torch.zeros(points_per_bound, 3)
    XYT_left[:,0] = x_start
    XYT_left[:,1] = y_start + torch.rand(points_per_bound) * (y_end - y_start)
    XYT_left[:,2] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

    # X-axe: right position x=Lx
    XYT_right = torch.zeros(points_per_bound, 3)
    XYT_right[:,0] = x_end
    XYT_right[:,1] = y_start + torch.rand(points_per_bound) * (y_end - y_start)
    XYT_right[:,2] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

    # Y-axe: bottom position y=0
    XYT_bottom = torch.zeros(points_per_bound,3)
    XYT_bottom[:,0] = x_start + torch.rand(points_per_bound) * (x_end - x_start)
    XYT_bottom[:,1] = y_start
    XYT_bottom[:,2] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

    # Y-axe: upper position y=Ly
    XYT_top = torch.zeros(points_per_bound,3)
    XYT_top[:,0] = x_start + torch.rand(points_per_bound) * (x_end - x_start)
    XYT_top[:,1] = y_end
    XYT_top[:,2] = t_start + torch.rand(points_per_bound) * (t_end - t_start)

    XYT_bc = torch.cat([XYT_left, XYT_right, XYT_bottom, XYT_top], dim=0)
    XYT_bc.requires_grad = True

    return XYT_pde, XYT_initial, XYT_bc


"""
    ------------------------------------------------------------------------------------------------------------------------
                    function to implement residual function for NN
    ------------------------------------------------------------------------------------------------------------------------

"""

def residual_function(model, XYT_pde, XYT_initial, XYT_bc, prev_model=None):
    
    """
        ------------------------------------------------------------------------------------------------------------------------
                    residual function for PDE term
        ------------------------------------------------------------------------------------------------------------------------
    """

    def compute_sigma(x, y, x_start, x_end, y_start, y_end, d_pml, Z_0, m=3):
        """
            function to create PML conductance
            sigma_e = 0 -> in collocation points | sigma_e != 0 -> in PML
            sigma_m = 0 -> in collocation points | sigma_m != 0 -> in PML
        """
        dx_left = torch.clamp((x_start + d_pml) - x, min=0.0)
        dx_right = torch.clamp(x - (x_end - d_pml), min=0.0)
        sigma_x = cfg.sigma_max * ((dx_left + dx_right)/d_pml) ** m

        dy_bottom = torch.clamp((y_start + d_pml) - y, min=0.0)
        dy_top = torch.clamp(y - (y_end - d_pml), min=0.0)
        sigma_y = cfg.sigma_max * ((dy_bottom + dy_top) / d_pml) ** m

        # sigma* = (Z0^2/ε_r) * sigma
        impendace = (Z_0**2) / eps_r_spatial
        sigma_mx = impendace * sigma_x
        sigma_my = impendace * sigma_y

        return sigma_x, sigma_y, sigma_mx, sigma_my

    # get points for x, y and t
    x_pde = XYT_pde[:, 0:1]
    y_pde = XYT_pde[:, 1:2]
    t_pde = XYT_pde[:, 2:3]

    # normalize parameters to O(1)
    Z_0 = np.sqrt(cfg.mu_0 / cfg.eps_0) 

    # get model's prediction for PDE 
    Ex, Ey, Hz = model(x_pde, y_pde, t_pde)
    ones = torch.ones_like(Ex)

    # compute first order detivatives
    # ---- Ex component ----
    dEx_dt = torch.autograd.grad(
            Ex,
            t_pde,
            grad_outputs=ones,
            create_graph=True
    )[0]

    dEx_dy = torch.autograd.grad(
            Ex, 
            y_pde,
            grad_outputs=ones,
            create_graph=True
    )[0]

    # ---- Ey componte ----
    dEy_dt = torch.autograd.grad(
        Ey,
        t_pde,
        grad_outputs=ones,
        create_graph=True,
    )[0]

    dEy_dx = torch.autograd.grad(
        Ey,
        x_pde,
        grad_outputs=ones,
        create_graph=True
    )[0]

    # ---- Hz component ----

    dHz_dt = torch.autograd.grad(
        Hz,
        t_pde,
        grad_outputs=ones,
        create_graph=True
    )[0]

    dHz_dx = torch.autograd.grad(
        Hz,
        x_pde,
        grad_outputs=ones,
        create_graph=True
    )[0]

    dHz_dy = torch.autograd.grad(
        Hz,
        y_pde,
        grad_outputs=ones,
        create_graph=True
    )[0]

    eps_r_spatial = torch.where(x_pde >= cfg.x_interface, cfg.eps_r, 1.0)
    # function to compute conductances for PML
    sigma_x, sigma_y, sigma_mx, sigma_my = compute_sigma(
        x_pde, y_pde,
        cfg.x_start, cfg.x_end, cfg.y_start, cfg.y_end,
        cfg.d_pml,Z_0, m=3
    )

    # residual function's PDE terms |  normalized approach -> div with c_0 to get tau = c_0 * t
    f_1 = eps_r_spatial * (dEx_dt / cfg.c0) + (Z_0 * sigma_y) * Ex - (Z_0) * dHz_dy
    f_2 = eps_r_spatial * (dEy_dt / cfg.c0) + (Z_0 * sigma_x) * Ey + (Z_0) * dHz_dx
    f_3 = (Z_0) * (dHz_dt / cfg.c0) + ((sigma_mx + sigma_my) / Z_0) * Hz - dEx_dy + dEy_dx

    # PDE loss function term
    loss_pde = torch.mean(f_1**2) + torch.mean(f_2**2) + torch.mean(f_3**2)

    """
        initial condition loss function
    """

    x_initial = XYT_initial[:, 0:1]
    y_initial = XYT_initial[:, 1:2]
    t_initial = XYT_initial[:, 2:3]

    if prev_model is None:
        Hz_0 = cfg.H0 * torch.exp(-(((x_initial - cfg.mean_x)**2) + ((y_initial - cfg.mean_y)**2)) / (2.0 * (cfg.sigma_g**2)))
        Ex_0 = torch.zeros_like(Hz_0)
        Ey_0 = torch.zeros_like(Hz_0)
    else:
        with torch.no_grad():
            Ex_0, Ey_0, Hz_0 = prev_model(x_initial, y_initial, t_initial)

    Ex_init, Ey_init, Hz_init = model(x_initial, y_initial, t_initial)
    loss_ic = torch.mean((Ex_init - Ex_0)**2) + torch.mean((Ey_init - Ey_0)**2) + torch.mean((Hz_init - Hz_0)**2)


    """
        boundary conditios loss function
    """

    x_boundaries = XYT_bc[:, 0:1]
    y_boundaries = XYT_bc[:, 1:2]
    t_boundaries = XYT_bc[:, 2:3]

    Ex_bc, Ey_bc, Hz_bc = model(x_boundaries, y_boundaries, t_boundaries)

    # boundaries loss term
    loss_bc = torch.mean(Ex_bc**2) + torch.mean(Ey_bc**2) + torch.mean(Hz_bc**2)

    return loss_pde + 1e3*loss_ic + 1e3*loss_bc


"""
        ------------------------------------------------------------------------------------------------------------------------
                        Model solver | Time merching approach
         ------------------------------------------------------------------------------------------------------------------------

"""

import copy
import torch
import torch.optim as optim
import Config as cfg


def train_time_marching(model,num_windows: int,adam_epochs: int,lbfgs_max_iter: int = 5000,lr_adam: float = 1e-4,lr_lbfgs: float = 0.05,
                        device: str = "mps" if torch.backends.mps.is_available() else "cpu"):
    """
            training solver with time-merching approach
    """
    model.to(device)
    dt_window = (cfg.t_end - cfg.t_start) / num_windows

    trained_models = []
    prev_model = None

    for w in range(num_windows):
        t_w_start = cfg.t_start + w * dt_window
        t_w_end = t_w_start + dt_window

        print(f"\n{'='*65}")
        print(f"Time Windoow {w + 1}/{num_windows}: t in [{t_w_start:.3e}, {t_w_end:.3e}] s")
        print(f"{'='*65}")

        # update boundaries for curr timw windoe
        model.t_min = t_w_start
        model.t_max = t_w_end

        # create points for curr window
        XYT_pde, XYT_initial, XYT_bc = create_points(
            cfg.N_pde, cfg.N_ic, cfg.N_bc,
            cfg.x_start, cfg.x_end,
            cfg.y_start, cfg.y_end,
            t_w_start, t_w_end
        )

        XYT_pde = XYT_pde.detach().to(device).requires_grad_(True)
        XYT_initial = XYT_initial.detach().to(device).requires_grad_(True)
        XYT_bc = XYT_bc.detach().to(device).requires_grad_(True)

        # Adam optimizer training
        print("\n ===== Adam Training === \n")
        optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)
        scheduler = optim.lr_scheduler.StepLR(optimizer_adam, step_size=max(1, adam_epochs // 3), gamma=0.5)

        model.train()
        for epoch in range(1, adam_epochs + 1):
            optimizer_adam.zero_grad()

            loss_adam = residual_function(model, XYT_pde, XYT_initial, XYT_bc, prev_model=prev_model)
            loss_adam.backward()

            optimizer_adam.step()
            scheduler.step()

            if epoch % 500 == 0 or epoch == 1:
                print(f"Window [{w+1}/{num_windows}] | Adam Epoch: {epoch:04d}/{adam_epochs} | Loss: {loss_adam.item():.6e}")

        # L-BFGS optimizer 
        print("\n === L-BFGS Optimizer ====")
        optim_lbfgs = optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            lr=lr_lbfgs,
            line_search_fn="strong_wolfe"
        )

        lbfgs_iter = 0

        def closure():
            nonlocal lbfgs_iter
            optim_lbfgs.zero_grad()

            loss_lbfgs = residual_function(model, XYT_pde, XYT_initial, XYT_bc, prev_model=prev_model)
            loss_lbfgs.backward()

            if lbfgs_iter % 100 == 0:
                print(f"Window [{w+1}/{num_windows}] | L-BFGS Step: {lbfgs_iter:04d} | Loss: {loss_lbfgs.item():.6e}")
            lbfgs_iter += 1
            return loss_lbfgs

        optim_lbfgs.step(closure)

        # freeze model's weights
        prev_model = copy.deepcopy(model)
        prev_model.eval()
        for param in prev_model.parameters():
            param.requires_grad = False

        trained_models.append(prev_model)

    return trained_models





    













