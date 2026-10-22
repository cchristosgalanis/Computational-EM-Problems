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
    const int Nx_points = 10;
    const int Ny_points = 10;
    // problem variables
    const double c0 = 2.998e8; // speed of light
    const double eps_0 = 8.854e-12; 
    const double mu_0 = (4.0*pi)*10e-7;
    const double eps_r = 2.0;

    // compute steps for each axe
    const double dx = (x_end - x_start) / (Nx_points - 1);
    const double dy = (y_end - y_start) / (Ny_points - 1);
}

#endif