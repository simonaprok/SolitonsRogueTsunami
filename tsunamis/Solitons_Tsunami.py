#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 11:51:54 2021
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.fftpack import diff as psdiff
from scipy.integrate import quad, dblquad

''' Code is based on Korteweg de Vries equation from the Scipy Cookbook'''
'''https://github.com/scipy/scipy-cookbook/blob/master/ipython/KdV.ipynb'''

def kdv_exact(x, v):
    ''' Exact Single Soliton Soltuion for the Korteweg–de Vries equation'''
    u = 0.5*v*np.cosh(0.5*np.sqrt(v)*x)**(-2)
    return u

def I(x,H,B,a):
    ''' Function to find the intial shape a Tsunami'''
    
    xi, err = quad(lambda k: B*(2/np.pi)*np.cos(k*x)*np.sin(k*a)/(k*np.cosh(k*H)), 0, np.inf)
    return xi

def tsunami_inital(x, H, Nx):
    
    '''Finding the inital Tsuanmi profile based on water depth'''
    
    H = 20 #depth of the ocean
    a = 2*H #length of the block
    B = 0.1*H #vertical displacement of the block
    L = (10*H)
    
    surf = [I(i,H,B,a) for i in x]

    return np.array(surf)
    

def kdv(u, t, L):
    ''' The Korteweg–de Vries equation, finding the change over time'''
    
    ux = psdiff(u, period=L, order = 1) # using psdiff which uses the fourier transform to calcutate the derivative
    uxxx = psdiff(u, period = L, order=3)
    
    # Different functions for B
    #B = 80.0/t
    B = 1
  
    dudt = -6*u*ux-B*uxxx 
    
    return dudt

def kdv_solution(u0, t, L):
    '''Solving the KdV equation using odeint '''
    sol = odeint(kdv, u0, t, args=(L,), mxstep=70000)
    return sol

# Title of Plot
Name = 'Tsuanmi'

T = 10.0 # Final Time
Nt = 25 # Number of timesteps
t = np.linspace(1, T, Nt) # time data for simulation

H = 10 # Water depth
f = 10 # Number of water depths for horizontal scale
Nx = 1000 # Number of position steps


L = 2*f*H # Total lenght of plot
x = np.linspace(-f*H, f*H, Nx)  # positions for simulation

# Intial tsuanmi shape
u0 = tsunami_inital(x, H, Nx)

#v = 1.0
#u0 = kdv_exact(x, v) # If you want to plot solitons

print("Computing the solution.")
sol = kdv_solution(u0, t, L)

print("Plotting.")

# Plotting all of the data
plt.figure(0)
plt.xlabel('Position')
plt.plot('Time')
plt.title(Name)
plt.imshow(sol)
plt.colorbar()


# Plotting the inital and final states
plt.figure(1)
plt.xlabel('Position')
plt.ylabel('Water Height')
plt.title(Name)

# For stackplot the level needs to be above 0
level = np.max(-sol)+2

plt.stackplot(x, sol[0]+level, color='blue')
plt.stackplot(x, sol[-1]+level, color='lightskyblue')

plt.plot(x, sol[-1]+level, c = 'lightskyblue', label = 'Final')
plt.plot(x, sol[0]+level, c = 'blue' , label = 'Inital')

plt.legend()


