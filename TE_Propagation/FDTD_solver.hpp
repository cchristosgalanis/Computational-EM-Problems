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

        // function of FDTD approach 
        void FDTD(int Nx_points, int Ny_points, double *H_present, double *H_future, double *E_present, double *E_future); 
};

#endif