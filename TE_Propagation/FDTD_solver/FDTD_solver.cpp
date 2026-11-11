#include"FDTD_solver.hpp"
#include<cmath>
#include<vector>
#include <fstream>
#include <iostream>

/*
    This file contains functions to compute FDTD approach to solution
*/

// function to create 2D grid
void FDTD_solver::create_grid(int Nx_points, int Ny_points, double eps_0, double *eps_x, double *eps_y, double *mu, double *sigma_ex, double *sigma_ey, double *sigma_mx, double *sigma_my){
    int x_interface = std::round((Config::x0 - Config::x_start) / Config::dx);

    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j; // flatten memory
            double x = Config::x_start + (i*Config::dx);

            // all space has μ0
            mu[idx] = Config::mu_0;

            // dielectric space and dielectric interface
            if (i == x_interface){
                eps_x[idx] = eps_0 * ((1.0 + Config::eps_r)/2.0);
                sigma_ex[idx] = 0.0;
                sigma_ey[idx] = 0.0;
                sigma_mx[idx] = 0.0;
                sigma_my[idx] = 0.0;
            }
            else if (i < x_interface){
                eps_x[idx] = eps_0;
                sigma_ex[idx] = 0.0;
                sigma_ey[idx] = 0.0;
                sigma_mx[idx] = 0.0;
                sigma_my[idx] = 0.0;
            }
            else{
                eps_x[idx] = eps_0 * Config::eps_r;
                sigma_ex[idx] = 0.0;
                sigma_ey[idx] = 0.0;
                sigma_mx[idx] = 0.0;
                sigma_my[idx] = 0.0;
            }

            // Ey
            if (x + Config::dx/2.0 < Config::x0) {
                eps_y[idx] = eps_0; // empty space
            } 
            else if (std::abs(x + Config::dx/2.0 - Config::x0) <= Config::dx/2.0) {
                eps_y[idx] = eps_0 * (1.0 + Config::eps_r) / 2.0; //interface
            } 
            else {
                eps_y[idx] = eps_0 * Config::eps_r; // dielectric
            }

            // PML boundaries creation
            if (i < Config::N_pml_x){
                sigma_ex[idx] = Config::sigma_max * std::pow((double)(Config::N_pml_x - i)/Config::N_pml_x,Config::pml_order);
                sigma_mx[idx] = sigma_ex[idx] * (Config::mu_0/eps_x[idx]);
            }
            else if (i >= Nx_points - Config::N_pml_x){
                sigma_ex[idx] = Config::sigma_max * std::pow((double)(i - (Nx_points-Config::N_pml_x)+1)/Config::N_pml_x,Config::pml_order);
                sigma_mx[idx] = sigma_ex[idx] * (Config::mu_0/eps_x[idx]);
            }

            if (j < Config::N_pml_y){
                sigma_ey[idx] = Config::sigma_max * std::pow((double)(Config::N_pml_y-j)/Config::N_pml_y, Config::pml_order);
                sigma_my[idx] = sigma_ey[idx] * (Config::mu_0/eps_y[idx]);
            }
            else if (j >= Ny_points - Config::N_pml_y){
                sigma_ey[idx] = Config::sigma_max * std::pow((double)(j-(Ny_points - Config::N_pml_y)+1)/Config::N_pml_y, Config::pml_order);
                sigma_my[idx] = sigma_ey[idx] * (Config::mu_0/eps_y[idx]);
            }
        }
    }
}

// function to compute coef values for update equations
void FDTD_solver::coef_var_electric(int Nx_points, int Ny_points, const double *sigma_ex, const double *sigma_ey, const double *eps_x, double *eps_y, double *coef_ex1, double *coef_ex2, double *coef_ey1, double *coef_ey2){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j;
            
            // C_ex1
            coef_ex1[idx] = (2.0 * eps_y[idx] - sigma_ey[idx]*Config::dt) / (2.0 * eps_y[idx] + sigma_ey[idx]*Config::dt);

            // C_ex2
            coef_ex2[idx] = (2.0 * Config::dt) / ((2.0 * eps_y[idx] + sigma_ey[idx]*Config::dt)*Config::dy);

            // C_ey1
            coef_ey1[idx] = (2.0 * eps_x[idx] - sigma_ex[idx]*Config::dt) / (2.0 * eps_x[idx] + sigma_ex[idx]*Config::dt);

            // C_ey2
            coef_ey2[idx] = (2.0 * Config::dt) / ((2.0 * eps_x[idx] + sigma_ex[idx]*Config::dt)*Config::dx);
        }
    }
}

// functon to compute coef values for magnetic field, Berenger approach to get Hxz and Hzy for magnetic field
void FDTD_solver::coef_var_magnetic(int Nx_points, int Ny_points, const double *sigma_mx, const double *sigma_my, const double *mu, double *coef_hzx1, double *coef_hzx2, double *coef_hzy1, double *coef_hzy2){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            int idx = i * Ny_points + j;

            // C_hzx1
            coef_hzx1[idx] = (2.0*mu[idx] - sigma_mx[idx]*Config::dt) / (2.0*mu[idx] + sigma_mx[idx]*Config::dt);

            // C_hzx2
            coef_hzx2[idx] = (2.0*Config::dt) / ((2.0*mu[idx] + sigma_mx[idx]*Config::dt)*Config::dx);

            // C_hzy1
            coef_hzy1[idx] = (2.0*mu[idx] - sigma_my[idx]*Config::dt) / (2.0*mu[idx] + sigma_my[idx]*Config::dt);

            // C_hzy2
            coef_hzy2[idx] = (2.0*Config::dt) / ((2.0*mu[idx] + sigma_my[idx]*Config::dt)*Config::dy);
        }
    }
}


// // main function for FDTD approach 
//  // function of FDTD approach 
// void FDTD_solver::FDTD_solver(int Nx_points, int Ny_points, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
//     const double *coef_hzx1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double* __restrict__ Ex_future, double* __restrict__ Ex_present, double* __restrict__ Ey_future, double* __restrict__ Ey_present,
//     double* __restrict__ Hzx_future, double* __restrict__ Hzy_future, double* __restrict__ Hzx_present, double* __restrict__ Hzy_present, double* __restrict__ Hz_future, double* __restrict__ Hz_present){
//         /*
//             all of thea arguments are vectors alreary computed in the main pipeline. We have a TE mode, and using PML as absorption method, thus 
//             we use Berenger approach, so we can absorp waves with θi angle, thats why we have Hzx, and Hzy. 
        
//         */

//         // compute N time steps
//         int N_steps = std::ceil(Config::T_simulation / Config::dt);

//         // initialize Ex, Ey and Hz (Hzx + Hzy)
//         for (int i=0; i<Nx_points; i++){
//             for (int j=0; j<Ny_points; j++){
//                 double x = Config::x_start + i * Config::dx; // real value of x axe
//                 double y = Config::y_start + j * Config::dy; // real value of y axe

//                 int idx = i * Ny_points + j;

//                 // initilaize Hz 
//                 /* 
//                     using Berenger method so we have to compute Hozx and Hozy and add them together to get 
//                     Hoz value for initial condition
//                 */
//                Hzx_present[idx] = (Config::Ho / 2) * std::exp(-(std::pow((x - Config::mean_x),2))/(2*std::pow(Config::sigma_g,2)));
//                Hzy_present[idx] = (Config::Ho / 2) * std::exp(-(std::pow((y - Config::mean_y),2))/(2*std::pow(Config::sigma_g,2)));
//                Hz_present[idx] = Hzx_present[idx] + Hzy_present[idx];

//             }
//         }

//         for (int n=0; n<N_steps; n++){
//             // magnetic field update equatiosn
//             for (int i=0; i<Nx_points-1; i++){
//                 for (int j=0; j<Ny_points-1; j++){
//                     int idx = i * Ny_points + j;

//                     Hzx_future[idx] = coef_hzx1[idx] * Hzx_present[idx] - coef_hzx2[idx]*(Ey_present[idx+Ny_points] - Ey_present[idx]);
//                     Hzy_future[idx] = coef_hzy1[idx] * Hzy_present[idx] + coef_hzy2[idx]*(Ex_present[idx+1] - Ex_present[idx]);
//                     Hz_future[idx] = Hzx_future[idx] + Hzy_future[idx];
//                 }
//             }

//             // electric field update equations
//             for (int i=1; i<Nx_points; i++){
//                 for (int j=1; j<Ny_points; j++){
//                     int idx = i * Ny_points + j;

//                     Ex_future[idx] = coef_ex1[idx] * Ex_present[idx] + coef_ex2[idx]*(Hz_future[idx] - Hz_future[idx-1]);
//                     Ey_future[idx] = coef_ey1[idx] * Ey_present[idx] - coef_ey2[idx] * (Hz_future[idx] - Hz_future[idx-Ny_points]);
//                 }
//             }
            
//             for (int i=0; i<Nx_points; i++){
//                 for (int j=0; j<Ny_points; j++){
//                     int idx = i * Ny_points + j;
//                     Ex_present[idx] = Ex_future[idx];
//                     Ey_present[idx] = Ey_future[idx];
//                     Hzx_present[idx] = Hzx_future[idx]; 
//                     Hzy_present[idx] = Hzy_future[idx];
//                     Hz_present[idx] = Hz_future[idx];
//                 }
//             }
//         }
// }

/* 
    implementation of functions that create snapshot datas for solver's stability

*/

// function to create initial Gaussian pulse
void FDTD_solver::initial_pulse(int Nx_points, int Ny_points, double *Ex_present, double *Ey_present, double *Hzx_present, double *Hzy_present, double *Hz_present){
    for (int i=0; i<Nx_points; i++){
        for (int j=0; j<Ny_points; j++){
            double x = Config::x_start + (i*Config::dx);
            double y = Config::y_start + (j*Config::dy);

            int idx = i * Ny_points + j;

            // initilaize Hz 
            /* 
                using Berenger method so we have to compute Hozx and Hozy and add them together to get 
                Hoz value for initial condition
            */
            double x_pulse = std::pow(x-Config::mean_x,2);
            double y_pulse = std::pow(y-Config::mean_y,2);

            double Gaussian_pulse = Config::Ho * std::exp(-(x_pulse + y_pulse) / (2.0 * std::pow(Config::sigma_g, 2)));

            // split energy
            Hzx_present[idx] = Gaussian_pulse / 2.0;
            Hzy_present[idx] = Gaussian_pulse / 2.0;

            Hz_present[idx] = Hzx_present[idx] + Hzy_present[idx];
        }
    }
}

void FDTD_solver::snapshots_FDTD(int Nx_points, int Ny_points, int steps_to_run, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
            const double *coef_hzx1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double *Ex_future, double *Ex_present, double *Ey_future, double *Ey_present,
            double *hzx_future, double *hzy_future, double *hzx_present, double *hzy_present, double *Hz_future, double *Hz_present){

    for (int n=0; n<steps_to_run; n++){
        // Hz (Hzx + Hzy) update equations
        for (int i=0; i<Nx_points-1; i++){
            for (int j=0; j<Ny_points-1; j++){
                int idx = i * Ny_points + j;
                hzx_future[idx] = coef_hzx1[idx] * hzx_present[idx] - coef_hzx2[idx] * (Ey_present[idx + Ny_points] - Ey_present[idx]);
                hzy_future[idx] = coef_hzy1[idx] * hzy_present[idx] + coef_hzy2[idx] * (Ex_present[idx + 1] - Ex_present[idx]);
                Hz_future[idx] = hzx_future[idx] + hzy_future[idx];
            }
        }

        // Ex update equations
        for (int i=0; i<Nx_points-1; i++){
            for (int j=1; j<Ny_points-1; j++){
                int idx = i * Ny_points + j;
                Ex_future[idx] = coef_ex1[idx] * Ex_present[idx] + coef_ex2[idx] * (Hz_future[idx] - Hz_future[idx - 1]);
            }
        }

        // Ey update equations
        for (int i = 1; i < Nx_points - 1; i++) {
            for (int j = 0; j < Ny_points - 1; j++) { 
                int idx = i * Ny_points + j;
                Ey_future[idx] = coef_ey1[idx] * Ey_present[idx] - coef_ey2[idx] * (Hz_future[idx] - Hz_future[idx - Ny_points]);
            }
        }
        
        // buffer swap
        for (int i=0; i<Nx_points; i++){
            for (int j=0; j<Ny_points; j++){
                int idx = i * Ny_points + j;
                Ex_present[idx] = Ex_future[idx];
                Ey_present[idx] = Ey_future[idx];
                hzx_present[idx] = hzx_future[idx]; 
                hzy_present[idx] = hzy_future[idx];
                Hz_present[idx] = Hz_future[idx];
            }
        }
    }
}


// function to call solver and get snapshots
void FDTD_solver::FDTD(int Nx_points, int Ny_points, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
    const double *coef_hzx1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double* __restrict__ Ex_future, double* __restrict__ Ex_present, double* __restrict__ Ey_future, double* __restrict__ Ey_present,
    double* __restrict__ hzx_future, double* __restrict__ hzy_future, double* __restrict__ hzx_present, double* __restrict__ hzy_present, double* __restrict__ Hz_future, double* __restrict__ Hz_present){
        
        int N_steps = std::ceil(Config::T_simulation / Config::dt);
        initial_pulse(Nx_points, Ny_points, Ex_present, Ey_present, hzx_present, hzy_present, Hz_present);

        snapshots_FDTD(Nx_points, Ny_points, N_steps, coef_ex1, coef_ex2, coef_ey1, coef_ey2, coef_hzx1, coef_hzx2, coef_hzy1, coef_hzy2,
                       Ex_future, Ex_present, Ey_future, Ey_present, hzx_future, hzy_future, hzx_present, hzy_present, Hz_future, Hz_present);
}


