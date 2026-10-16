#ifndef WAVE_SOLVER_1D_HPP
#define WAVE_SOLVER_1D_HPP

#include<cmath>
#include<iostream>

class Wave_solver_1D{
    public:
        // function to compute analytical solution of 1D wave equation
        void create_grid(int N_points, double x_start, double x_end, double *X, double *T);

        // function to compute analytical solution of 1D wave equation
        void analytical_solution(int total_size, int N_terms, int N_sim, double *A, double lower_limit, double upper_limit, double (*func) (double, int), double *E, const double *X, const double *T, double c, double pi);

        // function for FDTD method for 1D wave function
        void FDTD_function(double T_total, double x_start, double x_end, int N_points, double *E_past, double *E_present, double *E_future, double *E_full, double *T_full, const double c, const double pi, const double mi, const double sigma);

    private:
        double simpons13(double lower_limit, double upper_limit, int N_sim, int n, double (*func) (double, int));

};

#endif