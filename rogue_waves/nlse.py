import numpy as np
from numpy import sqrt, exp, cosh
from scipy.integrate import solve_ivp
import pde


def sech(x):
    return cosh(x)**(-1)

def analytic_solution(x, t):
    """Profile of a soliton solution for the NLSE."""
    return sqrt(2)*sech(x-t)*exp(1j*(0.5*x+0.75*t))

def nlse(t, u):
    q=1
    u_t = 1j*(pde.uxx_5c(u, dx) + (q*u)*np.square(np.abs(u)))
    return u_t

if __name__ == "__main__":

    # Set the size of the domain, and create the discretized grid.
    Nx = 301
    L0 = -10
    L1 = 30
    L=L1-L0
    x = np.linspace(L0, L1, Nx).reshape(-1,1)
    dx = L/(Nx-1)

    # Final time
    T = 14

    # Set the initial conditions.
    u0 = analytic_solution(x, 0).flatten() + analytic_solution(-x+15, 0).flatten()

    print("Computing the solution.")
    sol = solve_ivp(nlse, (0,T), u0, method='RK45')
    sol.y = np.square(np.abs(sol.y))

    pde.plot(x, sol)