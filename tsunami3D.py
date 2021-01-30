"""
Project title: Solitons, Rogue Waves and Tsunamis

This is the code which is used to calculate the free ocean surface displacement
due to a simplistic rectangular seafloor deformation for tsunami source in 3D.

Date: Jan, 2021
Authors: Simona Prokopovic, Douglas Hull
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
from scipy.integrate import quad, dblquad, nquad
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad, dblquad
import matplotlib.colors as mcolors

# ocean bottom displacement from the block of the dimensions 2a x 2b with a vertical displacement B
def displacement(x, y, a, b, B):
    step = [[(np.heaviside(i+a,a)-np.heaviside(i-a,a))*(np.heaviside(j+b,b)-np.heaviside(j-b,b))for i in x]  for j in y]
    eta = B*np.array(step)
    return eta

# free surface displacement due to the block of the dimensions 2a x 2b with a vertical displacement B
def I(x,y,H,B,a,b):
    options={'limit':5}
    xi, err = nquad(lambda n,m: B*(4/np.pi**2)*np.cos(m*x)*np.cos(n*y)*np.sin(m*a)*np.sin(n*b)/(m*n*np.cosh(np.sqrt(n**2+m**2)*H)), [[0, np.inf], [0,np.inf]],opts=[options,options])

    # roughly track progress %
    xp = round((x/(5*H)+1)*50)
    yp = round((y/(5*H)+1)*50)
    print(f'y:{yp}%, x:{xp}%')

    #print(xi)
    return xi
    #return quad(lambda k: (k*np.cosh(k*H)), 0, np.inf)

H = 100 #depth of the ocean
a = 2*H # 1/2 length of the block
b = H # 1/2 width of the block
B = 0.9*H # vertical displacement of the block

# span the grid in x direction
x_grid = np.linspace (-5*H, 5*H, 50)
# span the grid in y direction
y_grid = np.linspace (-5*H, 5*H, 50)

# mesh the grid in 2D
X,Y = np.meshgrid(x_grid,y_grid)

#  find the displacement of the sea bottom
etas=displacement(x_grid, y_grid, a,b, B)

# Creates free surface with corner missing
surf = np.array([[I(x,y,H,B,a,b) if (x<0 or y>0)  else np.nan for x in x_grid] for y in y_grid])
surf = np.array(surf)+H

# make a figure
fig = plt.figure()
ax = fig.gca(projection='3d')

# plot sea bottom deformation with in units normalised by ocean height H
ax.plot_surface( X/H, Y/H, etas/H,
                        rstride=1, cstride=1,
                        alpha=1,
                        linewidth=0,
                        color='goldenrod',
                        antialiased=True
                      )

# plot ocean free surface deformation with in units normalised by ocean eight H
ax.plot_surface( X/H, Y/H, surf/H,
                        rstride=1, cstride=1,
                        alpha=1,
                        linewidth=0,
                        color='lightskyblue',
                        antialiased=True
                  )
# label the axes
ax.set_xlabel('X/H')
ax.set_ylabel('Y/H')
# scale with respect to the project paper
ax.set_zlabel('(Z-H)/H')
plt.savefig("testtest3D.pdf")
plt.show()
#sys.stdout.close()



#plt.show()
