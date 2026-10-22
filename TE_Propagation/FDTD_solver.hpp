#ifndef FDTD_SOLVER_HPP
#define FDTD_SOLVER_HPP

#include"Config.hpp"

class FDTD_solver{
    public:
        void create_grid(int Nx_points, int Ny_points, double eps_0, double eps_r, double *eps);
};

#endif