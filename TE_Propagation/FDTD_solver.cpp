#include"FDTD_solver.hpp"
#include<cmath>
#include<vector>
/*
    This file contains functions to compute FDTD approach to solution
*/

// function to create 2D grid
void FDTD_solver::create_grid(int Nx_points, int Ny_points, double eps_0, double eps_r, double *eps){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j; // flatten memory
            
            // first half is air
            if (i < Nx_points/2){
                eps[idx] = eps_0;
            }
            // material with dielectric value
            else {
                eps[idx] = eps_0 * eps_r;
            }
        }
    }
}