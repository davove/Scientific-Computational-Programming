# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 21:19:06 2022

@author: David Ovetsky
Completed 10/3/22
Root Finding and Special Functions
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder > numba (compiler)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numba

@numba.jit(nopython = True)
def rhs2Body(t, y):
    #Just moon orbit, simple case to make sure things work fine
    #RHS Setup
    N = 2 # number of bodies
    Me = 5.9742e24
    M2 = 0.0123 * Me
    Marray = [Me, M2] #Masses
    G = 6.674e-11
    #For ease of access of all variables:
    xo = 0
    yo = N
    vxo = 2*N
    vyo = 3*N
    
    delx = [] #derivatives of all x_i
    dely = [] #derivatives of all y_i
    delvx = [] #derivatives of all vx_i
    delvy = [] #derivatives of all vy_i
    
    for i in range(N):
        delx.append(y[i+vxo]) #dx = vx
        dely.append(y[i+vyo]) #dy = vy
        sumx = 0 #Initializing ax_i
        sumy = 0 #Initializing ay_i
        for j in range(N):
            if(i == j): continue
            #F_i = G * sum(m_j * (x_jvec - x_ivec)/(x_j - x_i)^3 ) 
            dx = y[j + xo] - y[i+xo]
            dy = y[j+yo] - y[i + yo]
            dij = np.sqrt(dx*dx + dy*dy)
            sumx += Marray[j] * dx/(dij*dij*dij)
            sumy += Marray[j] * dy/(dij*dij*dij)
        delvx.append(G * sumx) 
        delvy.append(G * sumy)    
    return (np.array(delx+dely+delvx+delvy)) # total change is concatenation of all lists

@numba.jit(nopython = True)
def rhs3Body(t, y):
    #RHS Setup
    N = 3 # number of bodies
    Me = 5.9742e24
    M2 = 0.0123 * Me
    Marray = [Me, M2, 0.2 * M2] #Masses
    G = 6.674e-11
    #For ease of access of all variables:
    xo = 0
    yo = N
    vxo = 2*N
    vyo = 3*N
    
    delx = [] #derivatives of all x_i
    dely = [] #derivatives of all y_i
    delvx = [] #derivatives of all vx_i
    delvy = [] #derivatives of all vy_i
    
    for i in range(N):
        delx.append(y[i+vxo]) #dx = vx
        dely.append(y[i+vyo]) #dy = vy
        sumx = 0 #Initializing ax_i
        sumy = 0 #Initializing ay_i
        for j in range(N):
            if(i == j): continue
            #F_i = G * sum(m_j * (x_jvec - x_ivec)/(x_j - x_i)^3 ) 
            dx = y[j + xo] - y[i+xo]
            dy = y[j+yo] - y[i + yo]
            dij = np.sqrt(dx*dx + dy*dy)
            sumx += Marray[j] * dx/(dij*dij*dij)
            sumy += Marray[j] * dy/(dij*dij*dij)
        delvx.append(G * sumx) 
        delvy.append(G * sumy)    
    return (np.array(delx+dely+delvx+delvy)) # total change is concatenation of all lists
            
    

def firstTest():
    #setting second moon to be virtually non-existant
    
    tinit = 0.0
    tfinal = 28*24*3600 # One lunar month
    # initial values: x, y, vx, vy for each of N bodies
    xinit = [0.0,0.0]
    yinit = [0.0,3.84e8]
    vxinit = [-12.593,1019.0]
    vyinit = [0.0,0.0]
    y0 = xinit+yinit+vxinit+vyinit 
    soln = solve_ivp(rhs2Body, [tinit, tfinal], y0, rtol = 1.0e-6)
    plt.figure()
    plt.plot(soln.y[0,:], soln.y[2,:], 'b')
    plt.plot(soln.y[1,:], soln.y[3,:], 'r')
    plt.title('orbit of the moon around earth (28 days)')
    plt.axis('equal')
    
def secondTest():
    #full run
    tinit = 0.0
    tfinal = 200*24*3600
    # initial values: x, y, vx, vy for each of N bodies
    xinit = [0.0,0.0,-4.97e8]
    yinit = [0.0,3.84e8, 0.0]
    vxinit = [-12.593,1019.0,985.0]
    vyinit = [0.0,0.0,825.0]
    y0 = xinit+yinit+vxinit+vyinit 
    
    soln = solve_ivp(rhs3Body, [tinit, tfinal], y0, rtol = 1.0e-6)
    plt.figure()
    plt.plot(soln.y[0,:], soln.y[3,:], 'b')
    plt.plot(soln.y[1,:], soln.y[4,:], 'r')
    plt.plot(soln.y[2,:], soln.y[5,:], 'k')
    plt.title('3 Body Orbit')
    plt.axis('equal')
    
def optTest():
    #optional test of 3 body orbit
    tinit = 0.0
    tfinal = 40*24*3600
    # initial values: x, y, vx, vy for each of N bodies
    xinit = [0.0,0.0,0.0]
    yinit = [0.0,3.84e8, 4.1e8]
    vxinit = [-12.593,1019.0,1200.0]
    vyinit = [0.0,0.0,100.0]
    y0 = xinit+yinit+vxinit+vyinit 
    soln = solve_ivp(rhs3Body, [tinit, tfinal], y0, rtol = 1.0e-6)
    plt.figure()
    plt.plot(soln.y[0,:], soln.y[3,:], 'b')
    plt.plot(soln.y[1,:], soln.y[4,:], 'r')
    plt.plot(soln.y[2,:], soln.y[5,:], 'k')
    plt.title('3 Body Orbit, secondary moon orbiting our moon')
    
def chaosTest():
    #full run slightly varied to detect chaos
    tinit = 0.0
    tfinal = 200*24*3600
    # initial values: x, y, vx, vy for each of N bodies
    xinit = [0.0,0.0,-4.98e8]
    yinit = [0.0,3.84e8, 0.0]
    vxinit = [-12.593,1019.0,985.0]
    vyinit = [0.0,0.0,825.0]
    y0 = xinit+yinit+vxinit+vyinit 
    
    soln = solve_ivp(rhs3Body, [tinit, tfinal], y0, rtol = 1.0e-6)
    plt.figure()
    plt.plot(soln.y[0,:], soln.y[3,:], 'b')
    plt.plot(soln.y[1,:], soln.y[4,:], 'r')
    plt.plot(soln.y[2,:], soln.y[5,:], 'k')
    plt.title('3 Body Orbit, varied to observe chaotic nature')
    plt.axis('equal')

            
            
firstTest()
secondTest()
chaosTest()
optTest()
            
    
    
    
    
    
