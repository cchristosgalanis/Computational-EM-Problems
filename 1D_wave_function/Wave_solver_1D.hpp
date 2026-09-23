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

    private:
        double simpons13(double lower_limit, double upper_limit, int N_sim, int n, double (*func) (double, int));

};

#endif