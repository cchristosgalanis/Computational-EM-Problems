#include"Wave_solver_1D.hpp"
#include"Config.hpp"
#include<iostream>
#include<cmath>
#include<vector>
#include<fstream>
#include<algorithm>

int main(){
    // parameters
    int N_points = Config::N_points;
    int total_size = N_points * N_points;
    int N_sim = Config::N_sim;
    int N_terms = Config::N_terms;
    double T_total = Config::T_total;

    double target_times[] = {0.4, 0.6, 1.0, 1.6, 2.0};
    int num_times = sizeof(target_times) / sizeof(target_times[0]);

    std::cout << "Allocating memory for grid...\n";

    // vectors 
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

    // Calculate spatial step dx
    double dx = (Config::x_end - Config::x_start) / (N_points - 1);

    // call function analytical solution to get Electric field motion
    // solver.analytical_solution(total_size, N_terms,N_sim, A.data(), Config::x_start, Config::x_end, Config::func, E.data(), X.data(), T.data(), Config::c, Config::pi);

    // call function for FDTD method
    solver.FDTD_function(T_total, Config::x_start, Config::x_end, N_points, E_past.data(), E_present.data(), E_future.data(), Config::c, Config::pi, Config::mi, Config::sigma);

    for (int i=0; i < N_points; i++){
        std::cout << E_future[i] << std::endl;
    } 

    // create output file 
    std::ofstream out_file("FDTD_wave_equation_results_100FourierTerms.csv");
    out_file << "x,E\n";

    for (int i = 0; i < N_points; i++) {
        double x_coord = Config::x_start + i * dx;
        out_file << x_coord << ","
                 << E_future[i] << "\n";    
    }
    
    out_file.close();

    std::cout << "\n[Success] Final FDTD snapshot successfully exported to 'FDTD_wave_equation_results_100FourierTerms.csv'\n";
    
  /* 
    code for getting .csv file for multiple time targets for FDTD method
  */

    // // Create output file and write headers (x, t, E) for Python compatibility
    // std::ofstream out_file("FDTD_wave_equation_results_100FourierTerms.csv");
    // out_file << "x,t,E\n";

    // std::cout << "Starting FDTD simulations...\n";

    // // Loop through all requested time snapshots
    // for (int k = 0; k < num_times; k++) {
    //     double current_t = target_times[k];
    //     std::cout << "Simulating for t = " << current_t << "...\n";

    //     std::fill(E_past.begin(), E_past.end(), 0.0);
    //     std::fill(E_present.begin(), E_present.end(), 0.0);
    //     std::fill(E_future.begin(), E_future.end(), 0.0);

    //     // execute FDTD for the current time window
    //     solver.FDTD_function(current_t, Config::x_start, Config::x_end, N_points, 
    //                          E_past.data(), E_present.data(), E_future.data(), 
    //                          Config::c, Config::pi, Config::mi, Config::sigma);

    //     // append results to the common CSV file
    //     for (int i = 0; i < N_points; i++) {
    //         double x_coord = Config::x_start + i * dx;
    //         out_file << x_coord << "," 
    //                  << current_t << "," 
    //                  << E_future[i] << "\n";    
    //     }
    // }
    
    // out_file.close();

    // std::cout << "\n FDTD snapshots successfully exported to 'FDTD_wave_equation_results_100FourierTerms.csv'\n";

    return 0;
}
    
