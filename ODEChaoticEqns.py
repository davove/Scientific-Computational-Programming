# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 17:17:23 2022
Created on Fri Sep  9 13:16:22 2022

@author: David Ovetsky
Completed 9/23/22
Root Finding and Special Functions
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder > numba (compiler)
"""

import numpy as np
import matplotlib.pyplot as plt
import numba

@numba.jit(nopython=True)
def rk4(rhs, y, t, tend, nt, neqns):
    #4th order RK method to solve ODE defined by 
    #rhs. neqns is number of eqns, nt is number of
    #time steps. y is an array of results at all it, 
    #t is an empty array of size nt, tend is the 
    #final time. Assuming time starts at t=0
    
    #step and half-step size useful for finding all the k's
    h = tend/(nt-1)
    hh = h/2 
    
    #initialization of all k-vectors
    k1 = np.zeros(neqns)
    k2 = np.zeros(neqns)
    k3 = np.zeros(neqns)
    k4 = np.zeros(neqns)
    
    t[0] = 0.0
    for it in range(0, nt-1):
        #sets all k values accordingly
        rhs(y[it,:], t, k1) #sets k1
        rhs(y[it,:]+hh*k1, t[it]+hh, k2)
        rhs(y[it,:]+hh*k2, t[it]+hh, k3)
        rhs(y[it,:]+h*k3, t+h, k4)
        #sets next y and t
        y[it+1,:] = y[it,:] + h/6.0*(k1+2*k2+2*k3+k4)
        t[it+1] = t[it]+h
    
@numba.jit(nopython = True) 
def rhs(y,t,k):
    #Given input vector y and the time, calculates
    #a k according to ODE to be used, sets each of given
    #k's values accordingly. neqns is presumed to 
    #be defined by rhs.
    
    B = 1
    m = 1
    #input vector is [x, y, z, e, de/dt]
    k[0] = -y[1]
    k[1] = y[0] + y[3] * y[2]
    k[2] = -1 * y[3] * y[1]
    k[3] = y[4]
    k[4] = B * k[1] - m*m*y[3]
    
    

@numba.jit(nopython = True)
def rhsH0(y,t,k):
    #rhs describing simple harmonic oscillation.
    #y[0] is pos, y[1] is v
    a = 1 #not neg
    k[0] = y[1]
    k[1] = -(a)*y[0]
    
def SHMtest():
    nt = 400
    neqns = 2
    t = np.zeros(nt)
    tend = 20.0
    yy = np.zeros((nt, neqns))  
    yy2 = np.zeros((nt, neqns))
    yy[0,:] = [1, 0]#pos initial, vel initial
    yy2[0, :] = [0, 5]
    rk4(rhsH0,yy,t,tend,nt,neqns)
    rk4(rhsH0, yy2,t ,tend,nt,neqns)
    plt.figure(1)
    plt.plot(t,yy[:,0], 'b-')
    plt.plot(t,yy[:,1], 'b--')
# =============================================================================
#     plt.figure(2)
#     plt.plot(t,yy2[:,0], 'r-')
#     plt.plot(t,yy2[:,1], 'r--')
# =============================================================================
    plt.figure(3)
    plt.plot(yy[:,0],yy[:,1])

def finTest():
    
    nt= 16000
    neqns = 5
    t = np.zeros(nt)
    tend = 200.0
    yy = np.zeros((nt, neqns))
    yy1 = np.zeros((nt,neqns))
    yy[0,:] = [0,0,1,1.0e-6,0]
    yy1[0,:] = [0,0,1.01,1.0e-6,0]
    rk4(rhs, yy ,t, tend, nt, neqns)
    rk4(rhs, yy1 ,t, tend, nt, neqns)
    fig = plt.figure(1)
        plt.plot(t, yy[:,0],'k')
    plt.title('x(t) over 200 seconds')
    #plt.plot(t,yy1[:,0],'r')
    #plt.title('x2(t) over 200 seconds')
    #plt.plot(yy[:,1],yy[:,2], 'k')
    #plt.title('phase plot of y against z over 200 seconds')
    
# =============================================================================
#     fig = plt.figure(2)
#     ax = fig.add_subplot(111, projection='3d')
#     #ax.plot( yy[:,0], yy[:,1], yy[:,2], 'k', linewidth=1 )
#     ax.plot( yy1[:,0], yy1[:,1], yy1[:,2], 'r', linewidth=1 )
#     ax.view_init( -158, 135 )
#     ax.set_xlabel( 'x(t)')
#     ax.set_ylabel( 'y(t)' )
#     ax.set_zlabel( 'z(t)' )
#     plt.title( 'Jaynes-Cummings System')
#     plt.savefig( 'fig04f.eps' )
#     plt.show()
# =============================================================================
    

    
    
#SHMtest()   
finTest()
    
    

