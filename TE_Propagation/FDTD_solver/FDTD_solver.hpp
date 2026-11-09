#ifndef FDTD_SOLVER_HPP
#define FDTD_SOLVER_HPP

#include "Config.hpp"

class FDTD_solver {
    public:
        // function to create space of simulation
        void create_grid(int Nx_points, int Ny_points, double eps_0, double eps_r, double *eps_x, double *eps_y, double *mu, double *sigma_ex, double *sigma_ey, double *sigma_mx, double *sigma_my);

        // coefficients for electric field
        void coef_var_electric(int Nx_points, int Ny_points, const double *sigma_ex, const double *sigma_ey, const double *eps_x, double *eps_y, double *coef_ex1, double *coef_ex2, double *coef_ey1, double *coef_ey2);
        
        // coefficients for magnetic field (Διορθώθηκε η ονοματολογία σε hzx και hzy παντού)
        void coef_var_magnetic(int Nx_points, int Ny_points, const double *sigma_mx, const double *sigma_my, const double *mu, double *coef_hzx1, double *coef_hzx2, double *coef_hzy1, double *coef_hzy2);

        // main function for solver
        void FDTD(int Nx_points, int Ny_points, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
                  const double *coef_hzx1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, 
                  double* __restrict__ Ex_future, double* __restrict__ Ex_present, double* __restrict__ Ey_future, double* __restrict__ Ey_present,
                  double* __restrict__ hzx_future, double* __restrict__ hzy_future, double* __restrict__ hzx_present, double* __restrict__ hzy_present, 
                  double* __restrict__ Hz_future, double* __restrict__ Hz_present); 

        // function to create initial Gaussian pulse
        void initial_pulse(int Nx_points, int Ny_points, double *Ex_present, double *Ey_present, double *hzx_present, double *hzy_present, double *Hz_present);

        // function to create data snapshots during simulation
        void snapshots_FDTD(int Nx_points, int Ny_points, int steps_to_run, const double *coef_ex1, const double *coef_ex2, const double *coef_ey1, const double *coef_ey2,
                            const double *coef_hzx1, const double *coef_hzx2, const double *coef_hzy1, const double *coef_hzy2, 
                            double *Ex_future, double *Ex_present, double *Ey_future, double *Ey_present,
                            double *hzx_future, double *hzy_future, double *hzx_present, double *hzy_present, 
                            double *Hz_future, double *Hz_present);
};


#endif
