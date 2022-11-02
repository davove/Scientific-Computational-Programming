# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 22:29:14 2022

@author: David Ovetsky
Completed 9/6/22
Numerical Integration and Fresnel Diffraction Theory
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder
"""

import numpy as np
import matplotlib.pyplot as plt
   

def cEval(u0, rep):
    #Evaluate integral cos(pi/2 * u^2) from 0 to u0 for at least 'rep' intervals
    n = 1
    S = 0.5 * (np.cos(0) + np.cos(np.pi * u0**2 / 2))
    I = S * u0
    while (n<rep):
        n = 2 * n
        dx = u0/n
        for i in range(n):
            if i%2 != 0:
                S = S + np.cos(np.pi* (i*dx)**2 / 2)
        I = S * dx
    return I
    
    
def sEval(u0, rep):
    #Evaluate integral sin(pi/2 * u^2) from 0 to u0 for at least 'rep' intervals
    n = 1
    S = 0.5 * (np.sin(0) + np.sin(np.pi * u0**2 / 2))
    I = S * u0
    while (n<rep):
        n = 2 * n
        dx = u0/n
        for i in range(n):
            if i%2 != 0:
                S = S + np.sin(np.pi* (i*dx)**2 / 2)
        I = S * dx
    return I


def iEval(u0, rep):
    return 0.5 * ((cEval(u0,rep)+0.5)**2 + (sEval(u0,rep)+0.5)**2)


def result1():
    print('----- u0 = 0.5')
    for n in range (2,14):
        print(f"n: {2**n} || I/I_0 = {iEval(0.5 ,2**n)}")
    print('----- u0 = 3')
    for n in range (2,14):
        print(f"n: {2**n} || I/I_0 = {iEval(3 ,2**n)}")
    
#Automating search for 5 decimal-point tolerance.

def integral(func, x0, xf, tol):
    #evaluates the integral of a function from x0 to xf to a defined tolerance
    #uses trapezoid method
    n = 1
    S= 0.5*(func(x0) + func(xf))
    Inew = (xf-x0)*S
    Iold = 0
    dI= Inew
    rep = 1
    while (rep<=19 and (rep<=8 or dI>tol)):
        n = 2*n
        dx = (xf-x0)/n
        Iold = Inew
        for i in range(n):
            if(i%2 != 0):
                S = S + func(x0+i*dx)
        Inew = S*dx
        rep+=1
        dI = abs(Inew-Iold)
    return Inew


def iEvalTol(u0, tol):
    #Evaluation of integral with proper function
    return 0.5 * ((integral(lambda x : np.cos(np.pi/2*x**2),0,u0,tol)+0.5)**2 + (integral(lambda x:np.sin(np.pi/2*x**2), 0, u0,tol)+0.5)**2)

l = 0.5*10**-6
z = 10**-6

def result2():
    xSpace = np.linspace(-10**-6, 4*10**-6, 200)
    uSpace = xSpace * np.sqrt(2/(l*z))
    Iarr = []
    for u in uSpace:
        Iarr.append(iEvalTol(u,0.00001))
    plt.figure(1)
    plt.plot(xSpace,Iarr,'k-',label = 'I/I_0')
    plt.title('rel. intensity')
    plt.xlabel('distance (microns)')
    plt.ylabel('relative intensity')
    
   
result1()   
result2()




    

    
    

