# -*- coding: utf-8 -*-
"""
        Created on Fri Sep  9 13:16:22 2022
        
        @author: David Ovetsky
        Completed 9/11/22
        Root Finding and Special Functions
        CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
        Anaconda > Spyder
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import *

def doPlot(func, x0, xf, linetype, funcname = ''):
    #Takes a function with one input and plots it from x0 to xf. Line is black, solid
    #by default. Funcname included for the purpose of labeling function in legend.
    funcOutput = []   
    xSpace = np.linspace(x0,xf,200)
    xSpaceArr = np.array(xSpace)
    output = func(xSpaceArr)
    plt.plot(xSpace,output,linetype, label = funcname)
    
def getBracketList(func, x0, xf, num):
    #Finds a bracket list which will contain in between the brackets the first
    #{num} roots, so the list is n+1 points. Searches through func in the range of x0 to xf
    h = (xf-x0)/1000
    brackList = []
    curSign = np.sign(func(x0))
    brackList.append(x0)
    x = x0
    while (len(brackList)<=num):
        #will throw an error if not proper number of roots reached
        x+=h
        tempSign = np.sign(func(x))
        if (tempSign != curSign):
            brackList.append(x)
            curSign = tempSign
    return brackList
        
        



def bisect(func, x1, x2, tol):
    #A function to find roots of function func by bisection, starting from the
    #bracket defined by x1 and x2. Raises an error if invalid bracket. Searches
    #for 0 with a tolerance of tol. Throws an error if max iterations exceeded.
    maxiter = 1000
    f1 = func(x1)
    f2 = func(x2)
    #Checking to see bracket is valid, saving sign as var for convenience:
    sf1 = np.sign(f1)
    sf2 = np.sign(f2)
    if sf1==sf2:
        raise Exception('invalid bracket selected')
    if (f1==0):
        return x1
    if (f2==0):
        return x2
    #Finding new point by bisection and then closing in bracket
    rep = 1
    xnew = 0.5*(x1+x2)
    fnew = func(xnew)
    while(abs(fnew) > tol):
        snew = np.sign(fnew)
        if (snew == sf1):
            x1 = xnew
            f1 = fnew
        else:
            x2 = xnew
            f2 = fnew
        xnew = 0.5*(x1+x2)
        fnew = func(xnew)
        rep+=1
        if (rep>=maxiter): raise Exception('too many iterations to find root')
    #returns the root, calculated value (for checking), and amount of reps
    return [xnew, fnew, rep]

def regFalsi(func, x1, x2, tol):
    #A function to find roots of function func by regula falsi, starting from the
    #bracket defined by x1 and x2. Raises an error if invalid bracket. Searches
    #for 0 with a tolerance of tol. Throws an error if max iterations exceeded.
    maxiter = 1000
    f1 = func(x1)
    f2 = func(x2)
    #Checking to see bracket is valid, saving sign as var for convenience:
    sf1 = np.sign(f1)
    sf2 = np.sign(f2)
    if sf1==sf2:
        raise Exception('invalid bracket selected')
    if (f1==0):
        return x1
    if (f2==0):
        return x2
    #Finding new x using line between points and closing in bracket
    rep = 1
    xnew = x1 - f1*(x2-x1)/(f2-f1)
    fnew = func(xnew)
    while(abs(fnew) > tol):
        snew = np.sign(fnew)
        if (snew == sf1):
            x1 = xnew
            f1 = fnew
        else:
            x2 = xnew
            f2 = fnew
        xnew = x1 - f1*(x2-x1)/(f2-f1)
        fnew = func(xnew)
        rep+=1
        if (rep>=maxiter): raise Exception('too many iterations to find root')
    #returns the root, calculated value (for checking), and amount of reps
    return [xnew, fnew, rep]


def result1(): 
    #plotting all the necessary Bessel functions
    plt.figure(1)
    plt.title('Bessel functions of the first kind')
    doPlot(j0, 0, 20, 'k-',funcname = 'J0')
    doPlot(j1,0, 20,'r-', funcname = 'J1')
    doPlot(lambda x : jn(2,x), 0, 20, 'b-', funcname = 'J2')
    plt.legend()
    plt.figure(2)
    plt.title('Bessel functions of the second kind')
    doPlot(y0, 0.75, 20, 'k-', funcname = 'Y0')
    doPlot(y1, 0.75, 20, 'r-', funcname = 'Y1')
    doPlot(lambda x : yn(2,x), 0.75, 20, 'b-', funcname = 'Y2')
    plt.legend()

def besselFuncCombo(x):
    return x*x*y0(x)*y1(x)**2 - j0(x)*j1(x)

def result2(tol):
    #Finding the first 5 roots of the above function using both methods
    bracketPoints = getBracketList(besselFuncCombo, 0.5,20,5)
    print(f'bisection method for tol {tol}:')
    for i in range(5):
        print(bisect(besselFuncCombo, bracketPoints[i], bracketPoints[i+1], tol))
    print(f'method of false position for tol {tol}:')
    for i in range(5):
        print(regFalsi(besselFuncCombo, bracketPoints[i], bracketPoints[i+1], tol))
       
    

result1()
#plotting besselFuncCombo
plt.figure(3)
plt.title('Bessel function combination')
doPlot(besselFuncCombo, 0.5, 20, 'k-')
plt.axhline(y=0, color = 'k')

result2(0.000001)
print('testing convergence speed')
result2(0.0000001)

