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

// functon to compute coef values for magnetic field, Berenger approach to get Hxz and Hzy for magnetic field
void FDTD_solver::coef_var_magnetic(int Nx_points, int Ny_points, const double *sigma_mx, const double *sigma_my, const double *mu, double *coef_hzx1, double *coef_hzx2, double *coef_hzy1, double *coef_hzy2){
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


// main function for FDTD approach 
 // function of FDTD approach 
void FDTD_solver::FDTD(int Nx_points, int Ny_points, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
    const double *coef_hxz1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double *Ex_future, double *Ex_present, double *Ey_future, double *Ey_present,
    double *Hzx_future, double *Hzy_future, double *Hzx_present, double *Hzy_present, double *Hz_future, double *Hz_present){
        /*
            all of thea arguments are vectors alreary computed in the main pipeline. We have a TE mode, and using PML as absorption method, thus 
            we use Berenger approach, so we can absorp waves with θi angle, thats why we have Hzx, and Hzy. 
        
        */

        // compute N time steps
        int N_steps = std::ceil(Config::T_simulation / Config::dt);

        // initialize Ex, Ey and Hz (Hzx + Hzy)
        for (int i=0; i<Nx_points; i++){
            for (int j=0; j<Ny_points; j++){
                double x = Config::x_start + i * Config::dx; // real value of x axe
                double y = Config::y_start + j * Config::dy; // real value of y axe

                int idx = i * Ny_points + j;

                // initilaize Hz 
                /* 
                    using Berenger method so we have to compute Hozx and Hozy and add them together to get 
                    Hoz value for initial condition
                */
               Hzx_present[idx] = (Config::Ho / 2) * std::exp(-(std::pow((x - Config::mean_x),2))/(2*std::pow(Config::sigma_g,2)));
               Hzy_present[idx] = (Config::Ho / 2) * std::exp(-(std::pow((y - Config::mean_y),2))/(2*std::pow(Config::sigma_g,2)));
               Hz_present[idx] = Hzx_present[idx] + Hzy_present[idx];

               // initialize electric field
               Ex_present[idx] = 0.0;
               Ey_present[idx] = 0.0;
            }
        }

        for (int n=0; n<N_steps; n++){
            // magnetic field update equatiosn
            for (int i=0; i<Nx_points-1; i++){
                for (int j=0; j<Ny_points-1; j++){
                    int idx = i * Ny_points + j;

                    Hzx_future[idx] = coef_hxz1[idx] * Hzx_present[idx] - coef_hzx2[idx]*(Ey_present[idx+Ny_points] - Ey_present[idx]);
                    Hzy_future[idx] = coef_hzy1[idx] * Hzy_present[idx] + coef_hzy2[idx]*(Ex_present[idx+1] - Ex_present[idx]);
                    Hz_future[idx] = Hzx_future[idx] + Hzy_future[idx];
                }
            }

            // electric field update equations
            for (int i=1; i<Nx_points; i++){
                for (int j=1; j<Ny_points; j++){
                    int idx = i * Ny_points + j;

                    Ex_future[idx] = coef_ex1[idx] * Ex_present[idx] + coef_ex2[idx]*(Hz_future[idx] - Hz_future[idx-1]);
                    Ey_future[idx] = coef_ey1[idx] * Ey_present[idx] - coef_ey2[idx] * (Hz_future[idx] - Hz_future[idx-Ny_points]);
                }
            }
            
            for (int i=0; i<Nx_points; i++){
                for (int j=0; j<Ny_points; j++){
                    int idx = i * Ny_points + j;
                    Ex_present[idx] = Ex_future[idx];
                    Ey_present[idx] = Ey_future[idx];
                    Hzx_present[idx] = Hzx_future[idx]; 
                    Hzy_present[idx] = Hzy_future[idx];
                    Hz_present[idx] = Hz_future[idx];
                }
            }
        }
}