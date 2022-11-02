# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 16:43:05 2022

@author: David Ovetsky
Completed 10/26/22
Root Finding and Special Functions
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder > numba (compiler)
"""
import matplotlib.pyplot as plt
import numba
import numpy as np
import pickle #used temporarily to quickly access Psi

@numba.jit(nopython=True)
def tridiag(a0,b0,c0,d0):
    #returns a phi array given all necessary values of the
    #tridiagonal matrix. This is general and not specifically for 
    #Crank-Nicholson.
    
    #a0,b0,c0, or d0 are mutable, making local copies
    a = a0.copy()
    b = b0.copy()
    c = c0.copy()
    d = d0.copy()
    
    dim = len(d0)
    psi = d0.copy() #our final psi we are solving for
    
    c[0] = c[0]/b[0]
    d[0] = d[0]/b[0]
    #Reduction into upper-diagonal form:
    for i in range(1,dim):
        den = b[i] - a[i]*c[i-1] # so as not to repeat calculations
        c[i] = c[i]/(den)
        d[i] = (d[i] - a[i]*d[i-1])/(den)
    #back-solving
    psi[dim-1] = d[dim-1]
    revRange = list(range(dim-1))[::-1]
    for j in revRange:
        psi[j] = d[j] - c[j]*psi[j+1]
    return psi
    

def CNschrod(nt = 101):
    #A solution to the schrodinger eqn for our given system. This isn't
    #a very general subroutine. Returns Psi for every t and x.
    
    #definition of intervals
    nx = 10001 #expecting nx > 10000 for proper dx 
    xfin = 1000
    tfin = 5e-14
    #deriving various geometry values
    dx = xfin/(nx-1)
    dt = tfin/(nt-1)
    x = np.linspace(0,xfin,nx)
    #initial values
    psi0 = np.empty(nx, complex)
    V = np.empty(nx)
    for i in range(nx):
        psi0[i] = np.exp(complex(-((x[i]-0.3*xfin)/20)**2,x[i]))
        V[i] = 3.90/(1+np.exp((0.5*xfin-x[i])/7))
    #Initialization of various useful values, also a,b,c arrays
    Psi = np.empty((nt,nx), complex)
    Psi[0,:] = psi0
    w = 2*dx*dx/dt
    h = 6.5821e-16
    h2term = 3.801
    a = np.empty(nx,complex)
    b = np.empty(nx,complex)
    c = np.empty(nx,complex)
    d = np.empty(nx,complex)
    for j in range(0,nx):
        a[j] = 1
        b[j] = complex(-2-dx*dx/h2term*V[j],h*w/h2term )
        c[j] = 1
    a[0] = 0
    c[nx-1] = 0
    for i in range(1,nt):
        d[0] = complex(2+dx*dx/h2term*V[0],h*w/h2term) * Psi[i-1,0] - Psi[i-1,1]
        d[nx-1] =-1*Psi[i-1,nx-2] + complex(2+dx*dx/h2term*V[nx-1],h*w/h2term) * Psi[i-1,nx-1]
        for j in range(1,nx-1):
            d[j] = -Psi[i-1,j-1]+ complex(2+dx*dx/h2term*V[j],h*w/h2term) * Psi[i-1,j] - Psi[i-1,j+1]
        Psi[i,:] = tridiag(a,b,c,d) 
    return Psi



def dtTest():
    nx = 10001
    xfin = 1000
    x = np.linspace(0,xfin,nx)
    Psiarr = []
    plt.figure()
    plt.title('magnitude of wavefunction squared at t = 5e-14')
    plt.xlabel('x (angstroms)')
    plt.ylabel('|ψ| squared (probability distribution)')
    for nt in [601,701,801,901, 1001]:
        tfin = 5e-14
        dt = tfin/(nt-1)
        
        Psitemp = CNschrod(nt = nt)[nt-1,:]
        Psisq = np.abs(Psitemp)**2
        Psiarr.append(Psitemp)
        plt.plot(x,Psisq,'--', label = f'dt = {dt}, nt = {nt}')
        print(f'dt = {dt}, nt = {nt}')
    plt.legend(loc = 'upper right')
    return Psiarr


    



def fullPlotRoutine(Psi, nt = 101):
    #general method that runs all the plot methods
    
    #plt.close('all')
    #definition of intervals
    nx = 10001 #expecting nx > 10000 for proper dx 
    xfin = 1000
    tfin = 5e-14
    #deriving various geometry values
    dx = xfin/(nx-1)
    dt = tfin/(nt-1)
    x = np.linspace(0,xfin,nx)
    
    tindexarr = np.empty(5,int) #used for later time plots
    for i in range(5):
        tindexarr[i] = round((i+1)*1e-14/dt)
    
    
    #V plot ------------------------------------
    V = np.empty(nx)
    for i in range(nx):
        V[i] = 3.90/(1+np.exp((0.5*xfin-x[i])/7))
        
    plt.figure()
    plt.plot(x,V,color = 'orange')
    plt.title('Voltage landscape')
    plt.xlabel('x (angstroms)')
    plt.ylabel('V (eV)')
    
    #t0 plots------------------------------------
    plt.figure()
    plt.plot(x,np.real(Psi[0,:]),'b')
    plt.title('Real part of wavefunction at t = 0')
    plt.xlabel('x (angstroms)')
    plt.ylabel('ψ (real)')
    plt.figure()
    plt.plot(x,np.imag(Psi[0,:]),'r')
    plt.title('Imaginary part of wavefunction at t = 0')
    plt.xlabel('x (angstroms)')
    plt.ylabel('ψ (imag)')
    plt.figure()
    plt.plot(x[2500:3500],np.real(Psi[0,2500:3500]),'b--', label = 'real')
    plt.title('Real part of wavefunction at t = 0')
    plt.xlabel('x (angstroms)')
    plt.ylabel('ψ (real)')
    plt.plot(x[2500:3500],np.imag(Psi[0,2500:3500]),'r--', label = 'imag')
    plt.title('Complex value of wavefunction at t = 0')
    plt.xlabel('x (angstroms)')
    plt.ylabel('ψ')
    plt.legend(loc = 'upper right')
    plt.figure()
    plt.plot(x,np.abs(Psi[0,:])**2,'g')
    plt.title('magnitude of wavefunction squared at t = 0')
    plt.xlabel('x (angstroms)')
    plt.ylabel('|ψ| squared (probability distribution)')
    #other t plots----------------------------------
    plt.figure()
    for i in range(5):
        plt.plot(x,np.abs(Psi[tindexarr[i],:])**2,label = f'{(i+1)*1e-14} seconds')
        print(np.sum(np.abs(Psi[tindexarr[i],:])**2))
    plt.title('magnitude of wavefunction squared at various times')
    plt.xlabel('x (angstroms)')
    plt.ylabel('|ψ| squared (probability distribution)')
    plt.plot(x,np.divide(V,8),'k--',label = 'V (eV, scaled by 1/80000)')
    plt.legend(loc = 'upper right')
    


Psi = CNschrod(nt = 901)
#with open('Psipickle', 'rb') as f:
#    Psi = pickle.load(f)
fullPlotRoutine(Psi, nt = 901)
Psiarr = dtTest()




# =============================================================================
##ANIMATION 
#
#nx = 10001 #expecting nx > 10000 for proper dx 
# xfin = 1000
# x = np.linspace(0,xfin,nx)
# 
# 
# V = np.empty(nx)
# for i in range(nx):
#     V[i] = 3.90/(1+np.exp((0.5*xfin-x[i])/7))/4
#     
# import matplotlib.animation as animation
# def animate(i):
#     ax.clear()
#     ax.set_ylim([0,4])
#     ax.plot(x,np.abs(Psi[i,:])**2,label='phi^2')
#     ax.plot(x,V,'k--',label='V (not on same y scale)')
#     ax.set_xlabel('x (Angstroms)')
#     ax.set_ylabel('psi^2')
#     ax.legend()
# fig_anim,ax=plt.subplots(1,1)
# ani=animation.FuncAnimation(fig_anim,animate,frames=range(0,851,5),interval=1,repeat=True)
# plt.show()
#     
# =============================================================================
    
    
    
    
    

