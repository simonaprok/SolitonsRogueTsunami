import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


''' Code to solve (1+1) dimensional PDE by method of lines.
    We approximate the spacial derivatives by finite difference methods,
    giving a system of ODE in time, which are solved by scipy.integrate.solve_ivp().'''



def ux_5c(u, dx):
    ''' Approximates first derivative using five-pont
        central difference method, and periodic
        boundary conditions.'''

    n = len(u)

    def ux(i):
        return (u[(i-2) % n]
            - 8*u[(i-1) % n]
            + 8*u[(i+1) % n]
            -   u[(i+2) % n])/(12*dx)

    return np.fromfunction(ux, (n,), dtype=int)


def uxx_5c(u, dx):
    ''' Approximates second derivative using five-pont
        central difference method, and periodic
        boundary conditions.'''

    n = len(u)

    def uxx(i):
        return (-u[(i-2) % n]
            + 16*u[(i-1) % n]
            - 30*u[(i) % n]
            + 16*u[(i+1) % n]
            -    u[(i+2) % n])/(12*dx**2)

    return np.fromfunction(uxx, (n,), dtype=int)


def uxxx_7c(u, dx):
    ''' Approximates third derivative using seven-pont
        central difference method, and periodic
        boundary conditions.'''

    n = len(u)

    def uxxx(i):
        return (u[(i-3) % n]
            - 8*u[(i-2) % n]
            +13*u[(i-1) % n]
            -13*u[(i+1) % n]
            + 8*u[(i+2) % n]
            -   u[(i+3) % n])/(8*dx**3)

    return np.fromfunction(uxxx, (n,), dtype=int)

def plot(x, sol):

    fig = plt.figure(figsize=(10,7))
    gs = fig.add_gridspec(2,4)

    # plot 3d
    print("Plotting.")
    ax1 = fig.add_subplot(gs[0:2,0:3], projection='3d')
    surf = ax1.plot_surface(x.reshape(-1,1), sol.t, sol.y, antialiased=False, cmap='jet', cstride=6, rstride=6)
    ax1.minorticks_on()
    ax1.grid(True, which='both')

    #animation 
    print("Animating")
    sol_sq = sol.y.transpose()

    ax2 = fig.add_subplot(gs[0,-1])
    ax2.set_ylim(0, sol_sq.max())
    ax2.set_xlim(x[0], x[-1])
    line, = ax2.plot(x, sol_sq[0])
    time_template = 'time = %.1fs'
    time_text = ax2.text(0.05, 0.9, '', transform=ax2.transAxes)

    def init():  # only required for blitting to give a clean slate.
        line.set_ydata([np.nan] * len(x))
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_ydata(sol_sq[i])  # update the data.
        time_text.set_text(time_template % sol.t[i])
        return line, time_text

    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(sol_sq), interval=1e2*sol.t[-1]/len(sol_sq), blit=True)

    # check_invariants
    sol_sq = sol.y.transpose()
    a_0 = np.sum(sol_sq[0])
    inv = [np.sum(u)-a_0 for u in sol_sq]

    ax3 = fig.add_subplot(gs[1,-1])
    ax3.plot(sol.t, inv)
    plt.ylabel(f'{a_0} + dA')
    plt.title("Area")
    ax3.minorticks_on()
    ax3.grid(True, which='both')

    plt.tight_layout()
    plt.show()