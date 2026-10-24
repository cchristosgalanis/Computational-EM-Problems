#include"FDTD_solver.hpp"
#include"Config.hpp"
#include<cmath>
#include<iostream>
#include<vector>

int main(){
    // getting parameters of problem
    int N_points = Config::Nx_points * Config::Ny_points;

    // vectors 
    std::vector<double> eps(N_points, 0.0); // dielectric values
    std::vector<double> mu(N_points, 0.0); // μ0 values
    std::vector<double> sigma_ex(N_points, 0.0); // cond value for electric field to x axe
    std::vector<double> sigma_ey(N_points, 0.0); // cond value for electric field to y axe
    std::vector<double> sigma_mx(N_points, 0.0); // cond value for magnetic field to x axe
    std::vector<double> sigma_my(N_points, 0.0); // cond value for magnetic field to y axe
    
    // vectors for coef variables for update equations
    std::vector<double> coef_ex1(N_points, 0.0);
    std::vector<double> coef_ex2(N_points, 0.0);
    std::vector<double> coef_ey1(N_points, 0.0);
    std::vector<double> coef_ey2(N_points, 0.0);


    // creating object for solver
    FDTD_solver solver;

    // creating grid
    solver.create_grid(Config::Nx_points, Config::Ny_points, Config::eps_0, Config::eps_r, eps.data(),mu.data(), sigma_ex.data(), sigma_ey.data(), sigma_mx.data(), sigma_my.data());

    for (int i=0; i<Config::Nx_points; i++){
        for (int j=0; j<Config::Ny_points; j++){
            double x = i * Config::dx;
            double y = j * Config::dy;

            int idx = i * Config::Ny_points + j;

            std::cout << "(" << x << "," << y << "): " << " ε: " << eps[idx] << " μ: " << mu[idx] << " σ_x: " << sigma_ex[idx] << " σ_ y: " << sigma_ey[idx] << std::endl;
        }
    }
}