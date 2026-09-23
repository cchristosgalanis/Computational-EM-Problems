#include "Wave_solver_1D.hpp"
#include<iostream>
#include<cmath>

// file of function implementation

// simpons rule 1/3 
double Wave_solver_1D::simpons13(double lower_limit, double upper_limit, int N_sim, int n, double (*func) (double, int)){
    // N must be even
    if (N_sim % 2 != 0) {
        N_sim++;
    }

    double h = (upper_limit - lower_limit) / N_sim;
    double sum = func(lower_limit,n) + func(upper_limit,n);
    
    for (int i=1; i < N_sim; i++){
        double x = lower_limit + i * h;
        if (i % 2 == 0){
            sum += 2 * func(x,n); // even coefficients
        }
        else {
            sum += 4 * func(x,n); // odd coefficients
        }
    }

    return (h / 3.0) * sum;
}

// function to create space grid
void Wave_solver_1D::create_grid(int N_points, double x_start, double x_end, double *X, double *T){
    // steps
    double dx = (x_end - x_start) / (N_points - 1);
    double dt = (x_end - x_start) / (N_points - 1);

    // creating grid | using flatten: idx = x * N_points + t
    for (int i=0; i < N_points; i++){
        for (int j=0; j < N_points; j++){
            // compute 1D position in memory
            int idx = (i * N_points) + j;

            // X -> axon X
            X[idx] = i * dx;
            T[idx] = j * dt; 
        }
    }
}

// function to compute analytical solution
void Wave_solver_1D::analytical_solution(int total_size, int N_terms, int N_sim, double *A, double lower_limit, double upper_limit, double (*func) (double, int), double *E, const double *X, const double *T, double c, double pi){
    // using simpon's rule 1/3 to compute Fourier coefficients
    for (int i=0; i < N_sim; i++){
        A[i] = (2.0 / upper_limit) * simpons13(lower_limit, upper_limit, N_sim, i, func);
    }

    for (int i=0; i < total_size; i++){
        double E_val = 0.0;

        // Fourier terms
        for (int n=1; n < N_terms; n++){
            E_val += A[n] * std::cos((n * pi * c / upper_limit) * T[i]) * std::sin((n * pi / upper_limit) * X[i]);
        }

        E[i] = E_val;
    }
}