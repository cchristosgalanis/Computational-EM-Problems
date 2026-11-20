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
t_end = 5e-9 # total time of simulation

# problem variables
c0 = 2.998e8 # speed of light
eps_0 = 8.854e-12 # dielectric value of empty space
mu_0 = 4.0*pi*1.0e-7 # magnetic value of empty space
eps_r = 12.0 # dielectric value interface
sigma_max = 0.1

Nx_points = 1500
Ny_points = 1000
dx = (x_end - x_start) / (Nx_points - 1)
dy = (y_end - y_start) / (Ny_points - 1)

# number of points for every condition
N_pde = 5000 # points for PDE

# Mapping Coordinates parameters
xi_min = -1.0
xi_max = 0.99 # not 1.0 to avoid inf values