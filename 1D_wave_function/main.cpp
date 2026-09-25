#include"Wave_solver_1D.hpp"
#include"Config.hpp"
#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>

int main(){
    // parameters
    int N_points = Config::N_points;
    int total_size = N_points * N_points;
    int N_sim = Config::N_sim;
    int N_terms = Config::N_terms;
    double T_total = Config::T_total;

    std::cout << "Allocating memory for grid...\n";

    std::vector<double> X (total_size, 0.0); 
    std::vector<double> T(total_size, 0.0);
    std::vector<double> A(N_sim, 0.0);
    std::vector<double> E(total_size,0.0);
    std::vector<double> E_past(N_points, 0.0);
    std::vector<double> E_present(N_points, 0.0);
    std::vector<double> E_future(N_points, 0.0);

    // create object for functions
    Wave_solver_1D solver;

    // call function to create space
    solver.create_grid(N_points, Config::x_start, Config::x_end, X.data(), T.data());

    // call function analytical solution to get Electric field motion
    //solver.analytical_solution(total_size, N_terms,N_sim, A.data(), Config::x_start, Config::x_end, Config::func, E.data(), X.data(), T.data(), Config::c, Config::pi);

    // call function for FDTD method
    solver.FDTD_function(T_total, Config::x_start, Config::x_end, N_points, E_past.data(), E_present.data(), E_future.data(), Config::c, Config::pi, Config::mi, Config::sigma);

    for (int i=0; i < N_points; i++){
        std::cout << E_future[i] << std::endl;
    } 

    std::ofstream out_file("FDTD_wave_equation_results_100FourierTerms.csv");
    out_file << "x,E\n";

    double dx = (Config::x_end - Config::x_start) / (N_points - 1);

    for (int i = 0; i < N_points; i++) {
        double x_coord = Config::x_start + i * dx;
        out_file << x_coord << ","
                 << E_future[i] << "\n";    
    }
    
    out_file.close();

    std::cout << "\n[Success] Final FDTD snapshot successfully exported to 'FDTD_wave_equation_results_100FourierTerms.csv'\n";
}