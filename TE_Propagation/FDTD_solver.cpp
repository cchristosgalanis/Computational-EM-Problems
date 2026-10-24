#include"FDTD_solver.hpp"
#include<cmath>
#include<vector>
/*
    This file contains functions to compute FDTD approach to solution
*/

// function to create 2D grid
void FDTD_solver::create_grid(int Nx_points, int Ny_points, double eps_0, double eps_r, double *eps, double *mu, double *sigma_ex, double *sigma_ey, double *sigma_mx, double *sigma_my){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j; // flatten memory
            
            // real value of x
            double x = Config::x_start + (i * Config::dx);

            // all space has μ0
            mu[idx] = Config::mu_0;

            if (x < Config::x0){
                eps[idx] = eps_0;
                sigma_ex[idx] = 0.0;
                sigma_ey[idx] = 0.0;
                sigma_mx[idx] = 0.0;
                sigma_my[idx] = 0.0;
            }
            else {
                eps[idx] = eps_0 * eps_r;
                sigma_ex[idx] = 0.0;
                sigma_ey[idx] = 0.0;
                sigma_mx[idx] = 0.0;
                sigma_my[idx] = 0.0;
            }

            // PML boundaries creation
            if (i < Config::N_pml_x){
                sigma_ex[idx] = Config::sigma_max * std::pow((double)(Config::N_pml_x - i)/Config::N_pml_x,Config::pml_order);
                sigma_mx[idx] = sigma_ex[idx] * (Config::mu_0/eps_0);
            }
            else if (i >= Nx_points - Config::N_pml_x){
                sigma_ex[idx] = Config::sigma_max * std::pow((double)(i - (Nx_points-Config::N_pml_x)+1)/Config::N_pml_x,Config::pml_order);
                sigma_mx[idx] = sigma_ex[idx] * (Config::mu_0/eps_0);
            }

            if (j < Config::N_pml_y){
                sigma_ey[idx] = Config::sigma_max * std::pow((double)(Config::N_pml_y-j)/Config::N_pml_y, Config::pml_order);
                sigma_my[idx] = sigma_ey[idx] * (Config::mu_0/eps_0);
            }
            else if (j >= Ny_points - Config::N_pml_y){
                sigma_ey[idx] = Config::sigma_max * std::pow((double)(j-(Ny_points - Config::N_pml_y)+1)/Config::N_pml_y, Config::pml_order);
                sigma_my[idx] = sigma_ey[idx] * (Config::mu_0/eps_0);
            }
        }
    }
}

// function to compute coef values for update equations
void FDTD_solver::coef_var_electric(int Nx_points, int Ny_points, const double *sigma_ex, const double *sigma_ey, const double *eps, double *coef_ex1, double *coef_ex2, double *coef_ey1, double *coef_ey2){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j;
            
            // C_ex1
            coef_ex1[idx] = (2*eps[idx] - sigma_ey[idx]*Config::dt) / (2*eps[idx] + sigma_ey[idx]*Config::dt);

            // C_ex2
            coef_ex2[idx] = (2 * Config::dt) / ((2*eps[idx] + sigma_ey[idx]*Config::dt)*Config::dy);

            // C_ey1
            coef_ey1[idx] = (2*eps[idx] - sigma_ex[idx]*Config::dt) / (2*eps[idx] + sigma_ex[idx]*Config::dt);

            // C_ey2
            coef_ey2[idx] = (2*Config::dt) / ((2*eps[idx] + sigma_ex[idx]*Config::dt)*Config::dx);
        }
    }
}

void coef_var_magnetic(int Nx_points, int Ny_points, const double *sigma_mx, const double *sigma_my, const double *mu, double *coef_hzx1, double *coef_hzx2, double *coef_hzy1, double *coef_hzy2){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j;

            // C_hzx1
            coef_hzx1[idx] = (2*mu[idx] - sigma_mx[idx]*Config::dt) / (2*mu[idx] + sigma_mx[idx]*Config::dt);

            // C_hzx2
            coef_hzx2[idx] = (2*Config::dt) / ((2*mu[idx] + sigma_mx[idx]*Config::dt)*Config::dx);

            // C_hzy1
            coef_hzy1[idx] = (2*mu[idx] - sigma_my[idx]*Config::dt) / (2*mu[idx] + sigma_my[idx]*Config::dt);

            // C_hzy2
            coef_hzy2[idx] = (2*Config::dt) / ((2*mu[idx] + sigma_my[idx]*Config::dt)*Config::dy);
        }
    }
}
