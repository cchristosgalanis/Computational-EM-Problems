#ifndef CONFIG_HPP
#define CONFIG_HPP

#include<cmath>

namespace Config {
    // pi computation
    const double pi = std::acos(-1.0);
    // geometry parameters
    const double x_start = 0.0;
    const double x_end = 2.0;
    const double y_start = 0.0;
    const double y_end = 1.0;
    const int Nx_points = 1500;
    const int Ny_points = 100;
    const double x0 = 1.0;
    // PML boundaries
    const int N_pml_x = 10;
    const int N_pml_y = 10;
    //PML parameters
    const double sigma_max = 0.05;
    const int pml_order = 3;
    // problem variables
    const double T_simulation = 12e-9; // time of simulation
    const double c0 = 2.998e8; // speed of light
    const double eps_0 = 8.854e-12; 
    const double mu_0 = (4.0*pi)*10e-7;
    const double eps_r = 12.0;
    const double Ho = 2.0; // Ho for magnetic Gaussian pulse as initial state
    const double mean_x = 0.5; // Average value for x axe for Gaussian pulse
    const double mean_y = 0.3; // average value for y axe for Gaussian pulse
    const double sigma_g = 0.1; // sigma_x = sigma_y -> circular polarization

    // compute steps for each axe
    const double dx = (x_end - x_start) / (Nx_points - 1);
    const double dy = (y_end - y_start) / (Ny_points - 1);
    const double dt = 0.99 / (c0 * std::sqrt(std::pow(1.0/dx,2) + std::pow(1.0/dy,2)));
}

#endif