#ifndef CONFIG_HPP
#define CONFIG_HPP

#include<cmath>

namespace Config {
    int N_points = 300; // parameter for creating grid | rectangle
    int N_terms = 50; // parameter for Fourier 
    int N_sim = 50; // parameter for Simpon's rule 
    double x_start = 0.0;
    double x_end = 5.0;
    double c = 5.0; // velocity of wave function parameter
    const double mi  = 0.5;
    const double sigma = 0.45;
    const double pi = std::acos(-1.0);
    double T_total = 50.0; // simulation time for FDTD

    double func(double x, int n){
        return std::exp(-std::pow(x - mi,2) / (2 * std::pow(sigma,2))) * std::sin(n * pi * x / x_end);
    }
}

#endif