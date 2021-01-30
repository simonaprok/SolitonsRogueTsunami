"""
Project title: Solitons, Rogue Waves and Tsunamis

This is the code which is used to calculate the free ocean surface displacement
due to a simplistic rectangular seafloor deformation for tsunami source in 2D.

Date: Jan, 2021
Authors: Simona Prokopovic
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.stats as stats
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad, dblquad
import matplotlib.colors as mcolors


# ocean bottom displacement from the block of the length 2a with a vertical displacement B
def displacement(x, a, B):
    step = [np.heaviside(i+a,a)-np.heaviside(i-a,a) for i in x]
    eta = B*np.array(step)
    return eta

# free surface displacement due to the block of the length 2a with a vertical displacement B
def I(x,H,B,a):

    xi, err = quad(lambda k: B*(2/np.pi)*np.cos(k*x)*np.sin(k*a)/(k*np.cosh(k*H)), 0, np.inf)
    return xi
    #return quad(lambda k: (k*np.cosh(k*H)), 0, np.inf)

H = 1000 #depth of the ocean
a = 2*H # 1/2 length of the block
B = 0.1*H # vertical displacement of the block


# span the grid in x direction
x_grid = np.linspace (-5*H, 5*H, 1000)

# find the displacement of the sea bottom
etas=displacement(x_grid, a, B)

# find the diplacement of the free surface
surf = [I(x,H,B,a) for x in x_grid]
surf = H+np.array(surf)

# The maximum displacement of the water surface
print("The maximum displacement is:", np.max(surf)-H, "meters")


#plot the 2D figure
plt.figure(figsize=(5, 6))
#plot the surface displacement
plt.stackplot(x_grid/H, surf/H, color='lightskyblue')
#plot the block
plt.stackplot(x_grid/H, etas/H, color='goldenrod')

plt.axhline(y=1, color='black', linestyle='--')
#plt.title('Simulation of ocean surface displacement due to instantaneous bottom deformation at depth H', loc='center', wrap=True, size=10)
plt.ylim(0,1.2)
plt.xlim(np.min(x_grid)/H,np.max(x_grid)/H)

# label the axes
plt.xlabel('X/H')
# scale with respect to the project paper
plt.ylabel('(Z-H)/H')

# annotations 
plt.text(-0.1, 0.12, "$\eta(x)$", size=20)
plt.text(-0.5, 1.12, r"$\xi(x)$", size=20)
plt.legend(['a=2H, $B_{0}$=0.1H'], handlelength=0, handletextpad=0, fancybox=True)

plt.tight_layout()

plt.savefig("test.pdf")
#plt.show()
