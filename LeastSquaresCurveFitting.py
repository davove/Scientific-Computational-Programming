# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 17:26:24 2022

@author: David Ovetsky
Completed 10/31/22
Root Finding and Special Functions
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder > numba (compiler)

Python 3.9
"""

import numpy as np
import matplotlib.pyplot as plt

def GaussElim(Finp, b):
    #Given matrix F and vector b, carries out row reduction to simultaneously
    #solve and return inverse-F as well as the a-vector in the equation 
    #Fa = b. Assumes F and b are in a valid form for this problem.
    N = np.size(b)
    Finv = np.identity(N)
    F= Finp.copy()
    for i in range(N):
        #normalizing diagonal
        Fii = F[i][i]
        b[i] = b[i] / Fii
        for k in list(range(N))[::-1]:     
            F[i][k] =F[i][k]/Fii
            Finv[i][k] = Finv[i][k]/Fii
        for j in range(N):
            if j == i:
                continue
            Fji = F[j][i]
            b[j] = b[j]-(Fji * b[i])
            for k in list(range(N))[::-1]:
                F[j][k] = F[j][k] - Fji* F[i][k]
                Finv[j][k] = Finv[j][k] - Fji* Finv[i][k]
    return [b, Finv]


def GaussElimTest():
    #Just a test to make sure Gaussian elim implemented right
    A = np.identity(3)
    b = [1,2,3]
    ans = GaussElim(A,b)
    print(ans[0])
    print(ans[1])
    A = [[1,-2,8],[-2,3,11],[-1,2,0]]
    b = [1,2,-2]
    ans = GaussElim(A,b)
    print(ans[0])
    print(ans[1])
    A = [[-3/7,-1/7],[2/7,-4/7]]
    b = [1,2]
    ans = GaussElim(A,b)
    print(ans[0])
    print(ans[1])


def linSqRegCoeff(f, t, co2, sig):
    #A subroutine that finds a set of coefficients for a vector of functions
    #given the independent and dependent variable.
    m = len(f)
    a = np.zeros(m) #let the first 3 be quadratic, next 4 sinusoidal terms
    b = a.copy()
    F = np.zeros((m,m))
    for i in range(t.size):
        t_i = t[i]
        for l in range(m):
            b[l] = b[l] + f[l](t_i) * co2[i] / ((sig[i])**2)
            for k in range(m):
                F[l][k] = F[l][k] + f[l](t_i) * f[k](t_i) / ((sig[i])**2)
    temp = GaussElim(F,b)
    a = temp[0]
    Finv = temp[1]
    sig = np.zeros(m)
    for i in range(m):
        sig[i] = Finv[i][i]
    return [a,sig]

def linSqRegModel(co2, f, t, a, sig):
    #Given a set of functions and coefficients, builds up the actual model
    #function while also calculating the related reduced chi-squared value.
    m = len(f)
    co2Model = np.zeros(co2.size)
    chi = 0
    N = t.size    
    for i in range(N):
        t_i = t[i]
        tempSum = 0
        for k in range(m):
            tempSum = tempSum + f[k](t_i)*a[k]
        co2Model[i]= tempSum
        chi = chi + ((co2[i] - co2Model[i]) / (sig[i]))**2
    chi_r = chi/(N-m)
    return [co2Model, chi_r]
        
def main():   
    #The main function for running the code.
    
    #--- read CO2 data file and separate into t,CO2 data
    data = np.loadtxt( 'co2_mm_mlo.csv', delimiter=',', skiprows=53, \
    dtype='float' )
    t = data[:,2]
    co2 = data[:,3]
    
    sig = np.multiply(0.002, co2)
    # print statistics and plot to verify
    #print('t,co2 len= ',len(t),len(co2) )
    #print('t,co2 data range= ', min(t), max(t), min(co2), max(co2) )
    
    tReg = t.copy()
    t = np.subtract(t,1950)
    
    
    f = [lambda x : 1, lambda x : x, lambda x : x**2, lambda x : np.cos(2*np.pi*x), lambda x : np.sin(2*np.pi*x), lambda x : np.cos(2*2*np.pi*x), lambda x : np.sin(2*2*np.pi*x)]
    temp = linSqRegCoeff(f, t, co2, sig)
    a = temp[0]
    siga = temp[1]
    ans = linSqRegModel(co2, f, t, a, sig)
    co2Model = ans[0]
    chi_r = ans[1]
    ans = linSqRegModel(co2, f[:3],t,a[:3],sig)
    co2ModelSimp = ans[0]
    chi_rSimp = ans[1]
    ans = linSqRegModel(co2, f[:5],t,a[:5],sig)
    co2Model5Param = ans[0]
    chi_r5Param = ans[1]
    
    plt.close('all')
    
    plt.figure(1) # plot data vs. t
    plt.plot(tReg, co2, 'k-' )
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('original CO2 data')
    plt.show()
    
    
    plt.figure()
    plt.plot(tReg, co2, 'k-' , label = 'CO2 Data' )
    plt.plot(tReg,co2Model, label = 'Linear Regression')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data with lin. reg. line')
    plt.legend(loc = 'upper left')
    
    
    plt.figure()
    plt.plot(tReg[-100:], co2[-100:], 'k-', label = 'CO2 Data'  )
    plt.plot(tReg[-100:],co2Model[-100:], label = 'Linear Regression')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data with lin. reg. (last 100 values)')
    plt.legend(loc = 'upper left')
    
    
    plt.figure()
    plt.plot(tReg, co2, 'k-', label = 'CO2 Data' )
    plt.plot(tReg,co2ModelSimp,'r' ,label = 'Linear Regression (Simplified)')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data with purely quadratic lin. reg')
    plt.legend(loc = 'upper left')
    
    plt.figure()
    plt.plot(tReg,co2ModelSimp, 'r')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('purely quadratic lin. reg')
        
    Residual = co2-co2Model      
    uncertaintyPos = np.multiply(co2Model, 0.002)  
    uncertaintyNeg = np.multiply(co2Model, -0.002)  
    plt.figure()
    plt.scatter(tReg,Residual)  
    plt.plot(tReg, uncertaintyPos,'r--')
    plt.plot(tReg, uncertaintyNeg,'r--')
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('Residual with uncertainty delineated')
    
    plt.figure()
    plt.plot(tReg, co2, 'k-', label = 'CO2 Data' )
    plt.plot(tReg,co2Model5Param, label = 'Linear Regression (5 Parameters)')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data without 6-month harmonic')
    plt.legend(loc = 'upper left')
    
    print(f'the a vector is: {a}')
    print(f'the error for a is: {siga}')
    print(f'The reduced chi-squared value for the full model is {chi_r}')
    print(f'The reduced chi-squared value for the simplified model is {chi_rSimp}')
    print(f'The reduced chi-squared value for the 5 parameter model is {chi_r5Param}')
    
    g = [lambda x : 1, lambda x : x, lambda x : x**2, lambda x : np.cos(2*np.pi*x), lambda x : np.sin(2*np.pi*x), lambda x : np.cos(2*2*np.pi*x), lambda x : np.sin(2*2*np.pi*x), lambda x : np.cos(8*np.pi*x), lambda x : np.sin(8*np.pi*x)]
    
    temp = linSqRegCoeff(g, t, co2, sig)
    aComp = temp[0]
    sigaComp = temp[1]
    ans = linSqRegModel(co2, g, t, aComp, sig)
    co2ModelComp = ans[0]
    chi_rComp = ans[1]
    
    plt.figure()
    plt.plot(tReg, co2, 'k-' , label = 'CO2 Data' )
    plt.plot(tReg,co2ModelComp, label = 'Linear Regression')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data with lin. reg. line with more harmonics')
    plt.legend(loc = 'upper left')
    
    
    plt.figure()
    plt.plot(tReg[-100:], co2[-100:], 'k-', label = 'CO2 Data'  )
    plt.plot(tReg[-100:],co2ModelComp[-100:], label = 'Linear Regression')  
    plt.xlabel( 'time (year)')
    plt.ylabel( 'CO2 (in ppm)' )
    plt.title ('CO2 data with lin. reg. with more harmonics (last 100 values)')
    plt.legend(loc = 'upper left')

main()
    
    
    
#GaussElimTest()

    

