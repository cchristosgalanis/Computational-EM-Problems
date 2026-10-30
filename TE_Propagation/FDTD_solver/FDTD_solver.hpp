#ifndef FDTD_SOLVER_HPP
#define FDTD_SOLVER_HPP

#include"Config.hpp"

class FDTD_solver{
    public:
        // function to create space of simulation
        void create_grid(int Nx_points, int Ny_points, double eps_0, double eps_r, double *eps, double *mu, double *sigma_ex, double *sigma_ey, double *sigma_mx, double *sigma_my);

        // funtion to compute coef values
        // coefficients for electric field
        void coef_var_electric(int Nx_points, int Ny_points,const double *sigma_ex, const double *sigma_ey, const double *eps, double *coef_ex1, double *coef_ex2, double *coef_ey1, double *coef_ey2);
        // coefficients for magnetic field
        void coef_var_magnetic(int Nx_points, int Ny_points, const double *sigma_mx, const double *sigma_my, const double *mu, double *coef_hzx1, double *coef_hzx2, double *coef_hzy1, double *coef_hzy2);

        // function of FDTD approach | main function for solver
        void FDTD(int Nx_points, int Ny_points, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
        const double *coef_hxz1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double *Ex_future, double *Ex_present, double *Ey_future, double *Ey_present,
        double *Hzx_future, double *Hzy_future, double *Hzx_present, double *Hzy_present, double *Hz_future, double *Hz_present); 

        /* 
            functions to get snapshos during simulation in order to check solver's stability
        */

        // function to create initial Gaussian pulse
        void initial_pulse(int Nx_points, int Ny_points, double *Ex_present, double *Ey_present, double *Hzx_present, double *Hzy_present, double *Hz_present);

        // function to create datas during simulation
        void snapshots_FDTD(int Nx_points, int Ny_points, int steps_to_run, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
            const double *coef_hxz1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, double *Ex_future, double *Ex_present, double *Ey_future, double *Ey_present,
            double *Hzx_future, double *Hzy_future, double *Hzx_present, double *Hzy_present, double *Hz_future, double *Hz_present);
    
};

#endif