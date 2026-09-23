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

    std::cout << "Allocating memory for grid...\n";

    std::vector<double> X (total_size, 0.0); 
    std::vector<double> T(total_size, 0.0);
    std::vector<double> A(N_sim, 0.0);
    std::vector<double> E(total_size,0.0);

    // create object for functions
    Wave_solver_1D solver;

    // call function to create space
    solver.create_grid(N_points, Config::x_start, Config::x_end, X.data(), T.data());

    // call function analytical solution to get Electric field motion
    solver.analytical_solution(total_size, N_terms,N_sim, A.data(), Config::x_start, Config::x_end, Config::func, E.data(), X.data(), T.data(), Config::c, Config::pi);

    // for (int i=0; i < total_size / 100; i++){
    //     std::cout << E[i] << std::endl;
    // } 

    std::ofstream out_file("data/analytical_wave_equation_results.csv");
    out_file << "x,t,E\n";

    for (int i = 0; i < total_size; i++) {
        out_file << X[i] << ","
                 << T[i] << ","
                 << E[i] << "\n";
    }
    
    out_file.close();

    std::cout << "\n[Success] Polynomial data successfully exported to 'analytical_wave_equation_results.csv'\n";
}