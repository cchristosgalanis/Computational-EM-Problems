#include"FDTD_solver.hpp"
#include"Config.hpp"
#include<cmath>
#include<iostream>
#include<vector>
#include<fstream>
#include<algorithm>

int main(){
    // getting parameters of problem
    int N_points = Config::Nx_points * Config::Ny_points;

    // vectors 
    std::vector<double> eps(N_points, 0.0); // dielectric values
    std::vector<double> mu(N_points, 0.0); // μ0 values
    std::vector<double> sigma_ex(N_points, 0.0); // cond value for electric field to x axe
    std::vector<double> sigma_ey(N_points, 0.0); // cond value for electric field to y axe
    std::vector<double> sigma_mx(N_points, 0.0); // cond value for magnetic field to x axe
    std::vector<double> sigma_my(N_points, 0.0); // cond value for magnetic field to y axe
    
    // vectors for coef variables (electric) for update equations
    std::vector<double> coef_ex1(N_points, 0.0);
    std::vector<double> coef_ex2(N_points, 0.0);
    std::vector<double> coef_ey1(N_points, 0.0);
    std::vector<double> coef_ey2(N_points, 0.0);

    // vectors for coef variables (magnetic) for update equations
    std::vector<double> coef_hxz1(N_points, 0.0);
    std::vector<double> coef_hxz2(N_points, 0.0);
    std::vector<double> coef_hzy1(N_points, 0.0);
    std::vector<double> coef_hzy2(N_points, 0.0);

    //vectors for electric and magnetic field
    std::vector<double> Ex_present(N_points, 0.0);
    std::vector<double> Ex_future(N_points, 0.0);
    std::vector<double> Ey_present(N_points, 0.0);
    std::vector<double> Ey_future(N_points, 0.0);
    std::vector<double> Hzx_present(N_points, 0.0);
    std::vector<double> Hzx_future(N_points, 0.0);
    std::vector<double> Hzy_present(N_points, 0.0);
    std::vector<double> Hzy_future(N_points, 0.0);
    std::vector<double> Hz_present(N_points, 0.0);
    std::vector<double> Hz_future(N_points, 0.0);

    // creating object for solver
    FDTD_solver solver;

    // creating grid
    solver.create_grid(Config::Nx_points, Config::Ny_points, Config::eps_0, Config::eps_r, eps.data(),mu.data(), sigma_ex.data(), sigma_ey.data(), sigma_mx.data(), sigma_my.data());

    // coef for electrif field update equations
    solver.coef_var_electric(Config::Nx_points, Config::Ny_points, sigma_ex.data(), sigma_ey.data(), eps.data(), coef_ex1.data(), coef_ex2.data(), coef_ey1.data(), coef_ey2.data());

    // coef for magnetic field update equations
    solver.coef_var_magnetic(Config::Nx_points, Config::Ny_points, sigma_mx.data(), sigma_my.data(), mu.data(), coef_hxz1.data(), coef_hxz2.data(), coef_hzy1.data(), coef_hzy2.data());
    
    // FDTD function 
    solver.FDTD(Config::Nx_points, Config::Ny_points, coef_ex1.data(), coef_ex2.data(), coef_ey1.data(), coef_ey2.data(), coef_hxz1.data(), coef_hxz2.data(),
    coef_hzy1.data(), coef_hzy2.data(), Ex_future.data(), Ex_present.data(), Ey_future.data(), Ey_present.data(), Hzx_future.data(), Hzy_future.data(), 
    Hzx_present.data(), Hzy_present.data(), Hz_future.data(), Hz_present.data());

    /*
        saving to .csv file Hz, |E| = (Ex^2+Ey^2)^(1/2), Poynting vector, W -> electromagnetic energy
    */

    std::ofstream file("FDTD_data.csv");
    file << "x,y,eps,Hz,Ex,Ey,E_mag,W,Sx,Sy,S_mag\n";

    for (int i=0; i<Config::Nx_points; i++){
        for (int j=0; j<Config::Ny_points; j++){
            int idx = i * Config::Ny_points + j;

            // coordinates
            double x = Config::x_start + i * Config::dx;
            double y = Config::y_start + j * Config::dy;
            double eps_val = eps[idx];
            double mu_val = mu[idx];

            // electric and magnetic fields
            double hz = Hz_future[idx];
            double ex = Ex_future[idx];
            double ey = Ey_future[idx];

            // computations
            double e_mag = std::sqrt((ex*ex)+(ey*ey));
            double W = 0.5 * eps_val * (e_mag*e_mag) + 0.5*mu_val*(hz*hz);

            // poynting vector
            double sx = ey*hz;
            double sy = -ex*hz;
            double s_mag = std::sqrt((sx*sx)+(sy*sy));

            file << x << "," << y << "," << eps_val << "," << 
            hz << "," << ex << "," << ey << "," << e_mag << "," << 
            W << "," << sx << "," << sy << "," << s_mag << "\n";
        }
    }
    file.close();
}