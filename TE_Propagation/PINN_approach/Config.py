import numpy as np

"""

    this file contains constant values for TEz mode problem 

"""

pi = np.pi

# geometry parameters
x_start = 0.0
x_end = 2.0
x_interface = 1.0 # dielectric interface (e_r=12)
y_start = 0.0
y_end = 1.0
t_start = 0.0
t_end = 15e-9 # total time of simulation
d_pml = 0.2

# problem variables
c0 = 2.998e8 # speed of light
eps_0 = 8.854e-12 # dielectric value of empty space
mu_0 = 4.0*pi*1.0e-7 # magnetic value of empty space
eps_r = 12.0 # dielectric value interface
H0 = 2.0 # Ho for initial magnetic gaussing pulse
mean_x = 0.5 
mean_y = 0.5
sigma_g = 0.1 # sigma_x = sigma_y -> circular polarization
sigma_max = 0.1

# number of points for every condition
N_pde = 25000 # points for PDE
N_ic = 2000 # points for initial condition
N_bc = 2000 # points for boundary conditions | PML approach