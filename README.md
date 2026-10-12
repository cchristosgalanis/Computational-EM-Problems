
# 1D Wave Equation with Physics-Informed Neural Networks (PINNs)

## Overview
This project investigates the solution of the one-dimensional wave equation using Physics-Informed Neural Networks (PINNs).

The main objective is to explore how PINNs can approximate the solution of a time-dependent partial differential equation while incorporating the governing physical laws directly into the neural network training process.

## Mathematical Problem
The project focuses on the 1D wave equation:

\[ \begin{equation}
\frac{\partial^2 u}{\partial t^2}=c^2 \frac{\partial^2 u}{\partial x^2}
end{equation}
\]

where:
- \(u(x,t)\) represents the wave displacement.
- \(c\) is the wave propagation speed.
- \(x\) and \(t\) denote the spatial and temporal coordinates.

## PINN Approach
The neural network is trained by minimizing a loss function that incorporates:
- The physical residual of the wave equation.
- The initial conditions.
- The boundary conditions.

This approach allows the network to learn a solution that satisfies both the available constraints and the underlying physical model.

## Challenges Investigated
The project examines several difficulties that can affect PINN performance:

- **Spectral Bias:** Neural networks tend to learn low-frequency components before high-frequency ones.
- **Causality:** Training may fail to properly capture the temporal evolution of the solution.
- **Neural Tangent Kernel (NTK):** Different components of the loss function may converge at different rates.

## Proposed Approaches
Two approaches are discussed to address these limitations:

1. **Temporal Domain Decomposition:** Dividing the time domain into smaller subdomains to improve the learning of temporal dynamics.
2. **Dynamic Loss Balancing via NTK:** Adjusting the relative weights of the loss components based on their training behavior.

## Conclusion
This study highlights the limitations of standard PINNs when applied to wave propagation problems and discusses strategies that can improve their training and convergence.

## Author
**Christos Galanis**