#include"FDTD_solver.hpp"
#include"Config.hpp"
#include<cmath>
#include<iostream>
#include<vector>

int main(){
    // getting parameters of problem
    int N_points = Config::Nx_points * Config::Ny_points;

    // vectors 
    std::vector<double> eps(N_points, 0.0);

    // creating object for solver
    FDTD_solver solver;

    // creating grid
    solver.create_grid(Config::Nx_points, Config::Ny_points, Config::eps_0, Config::eps_r, eps.data());

    for (int i=0; i<Config::Nx_points; i++){
        for (int j=0; j<Config::Ny_points; j++){
            double x = i * Config::dx;
            double y = j * Config::dy;

            int idx = i * Config::Ny_points + j;

            std::cout << "(" << x << "," << y << "): " << eps[idx] << std::endl;
        }
    }
}