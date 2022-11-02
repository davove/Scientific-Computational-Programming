# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:13:03 2022

@author: David Ovetsky
Completed 10/18/22
Root Finding and Special Functions
CPU: AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx     2.10 GHz
Anaconda > Spyder > numba (compiler)
"""

import matplotlib.pyplot as plt
import numba
import numpy as np
import pickle

def defineBoundary(h):
    #A method for defining the specific geometry of our system. Assume
    #given h is less than or equal to 0.5 so all electrodes can
    #be dealt with normally. Could be substituted with a different 
    #function that defines: [nz,nr,bounds (electrodes),V (initial)]
    #
    if(h>0.5):
        raise Exception('please only input h<=0.5')
    Rmax = 20 #mm, large
    Zmax = 25 #mm, large
    nz = round(2*Zmax/h) #amount of z values, round used to ensure integer
    nr = round(Rmax/h) #amount of r values
    V = np.zeros((nz,nr),np.float32)
    bounds = np.zeros((nz,nr),np.bool8)
    #--- Setting up geometry ---
    bounds[0,:] = True #Edge bounds...
    bounds[-1,:] = True
    bounds[:,-1] = True
    R1 = round(3/h) #R values...
    R2 = round(5/h) 
    R3 = round(7/h)
    z0 = round(Zmax/h) # index for Z = 0
    Z1 = round(3.5/h) #Z values relative to z0...
    Z2 = round(8/h)
    Z3 = round(15/h)
    bounds[z0,R1:R3] = True #Electrodes...
    V[z0,R1:R3] = 1000
    bounds[z0-Z2:z0+Z2, R3] = True
    V[z0-Z2:z0+Z2, R3] = 1000
    bounds[z0+Z1:z0+Z3,R2] = True
    bounds[z0-Z3:z0-Z1,R1] = True
    return [bounds,V] #Needed for relax method to begin

 #most math done here - compiler needed
def relax(bounds,V,w,reverseFlag = False):
    #does one iteration of relaxation across the matrix V, using input bounds
    #for geometry. Returns the maximum measured potential difference
    #V is mutable so it will be changed for each iteration of the method
    #Note: due to asymmetry of propogation when traversal only occurs in one 
    #direction, reverseFlag allows for the traversal in the other.
    dim = bounds.shape
    zrange = list(range(dim[0]))
    rrange = list(range(dim[1]))
    if reverseFlag:
        zrange = zrange[::-1]
        rrange = rrange[::-1]
    delVmax = 0
    for i in zrange:
        for j in rrange:
            if (j==0 and not bounds[i,0]):
                Vnew = (1/6)*(4*V[i,1]+V[i+1,0]+V[i-1,0])
            elif(not bounds[i,j]):
                Vnew = (1/4)*(V[i,j+1]+V[i,j-1]+V[i+1,j]+V[i-1,j])+(1/(8*j))*(V[i,j+1]-V[i,j-1])
            else:
                Vnew = V[i,j]
            delV = abs(V[i,j] - Vnew)
            if(delV >= delVmax):
                delVmax = delV
            V[i,j] = (1-w)*V[i,j]+w*Vnew
    return delVmax


def wOptimize(begin,end):
    #A test run to attempt to find the optimum w by running relaxation using 
    #many different values for w.
    
    for w in np.linspace(begin,end,10):
        bounds,V = defineBoundary(0.1)
        delVmax = np.Inf
        flag = False
        rep = 0
        while(rep <=10000 and delVmax > 1):
            delVmax = relax(bounds,V,w, reverseFlag = flag)
            flag = not flag
            rep+=1
        print(str(w)+': '+str(rep))

def dVhOptimize(h,dVtol, Vinp=0):
    #A method that gives a V for a given h and Vtol.
    #In order to speed up calculation, an input V (Vinp) is allowed as a
    #starting point that is closer to the answer to reach convergence faster.
    #also returns V.
    
    bounds,V = defineBoundary(h)
    if (not Vinp.all() == 0):
        V = Vinp
    delVmax = np.Inf
    flag = False
    rep = 0
    w = 1.48
    while (rep <= 20000 and delVmax>dVtol):
        delVmax = relax(bounds,V,w, reverseFlag = flag)
        flag = not flag
        rep+=1
        #print(str(rep) + ' ' +str(delVmax) + ' - ' + str(Vrep))
    return V
            
def dVhTest():
    #A method for generating a table of Vrep values labeled with harr, Vtolarr 
    #values. Note that this took me a very long time to run for the last two 
    #harr values.
    Vtolarr = [1,0.5,0.1,0.05,0.01,0.005,0.001]
    harr = [0.5,0.25,0.1,0.05,0.03]

    for h in harr:
        print(str(h)+'-----')
        bounds,V = defineBoundary(h)
        for Vtol in Vtolarr:
            if(h==0.03 and Vtol==0.001):
                continue #this combination takes far too long, isn't necessary
            V = dVhOptimize(h, Vtol, V)
            Vrep = V[round(25/h),0]
            print(Vrep)
            
    #optimal result seems to be at h = 0.05, Vtol = 0.01
             
def calcV():
    #w = 1.48, h = 0.05, Vtol = 0.01, rep = 8175
    dVtol = 0.01
    h = 0.05
    
    bounds,V = defineBoundary(h)
    w = 1.48
    delVmax = np.Inf
    flag = False
    rep = 0
    while(rep <=10000 and delVmax > dVtol):
        delVmax = relax(bounds,V,w, reverseFlag = flag)
        flag = not flag
        rep+=1
        #print(str(rep) + ' ' +str(delVmax))
    return V

def saveV(V,filename):
    #Saves an inputed object V to a file. Greatly sped up process of debugging
    #since V could be loaded at will, much faster than recalculation.
    with open(str(filename), 'wb') as f:
        pickle.dump(V,f)
    
        
#From here I pickled V into the file Vload so that I could access it more readily

#V = calcV()
#saveV(V, 'Vload')


def VContour(Vinp):
    #The contour of a given Vinp, assuming h =0.05
    bounds,V = defineBoundary(0.05)
    V = Vinp
    
    dim = bounds.shape
    plt.contourf(np.multiply(list(range(dim[1])),0.05),np.multiply(list(range(dim[0])),0.05)-25, V, 20)
    rvals = [3,7]
    zvals = [0,0]
    plt.plot(rvals,zvals,'w--')
    rvals = [7,7]
    zvals = [-8,7.95]
    plt.plot(rvals,zvals,'w--')
    rvals = [3,3]
    zvals = [-15,-3.45]
    plt.plot(rvals,zvals,'k--')
    rvals = [5,5]
    zvals = [3.5,14.95]
    plt.plot(rvals,zvals,'k--')
    plt.title('Voltage distribution between electrodes')
    plt.xlabel('r in mm')
    plt.ylabel('z in mm')
  
@numba.jit(nopython=True)
def ErCal(V,h):
    #using forward difference to find E (V/m)
    dim = V.shape
    nz = dim[0]
    nr = dim[1]
    Er = np.zeros((nz,nr), np.float32)
    for r in range(1,nr-1):
        for z in range(1,nz-1):
            #derivatives not defined at boundaries 
            Er[z,r] = (V[z,r-1] - V[z,r+1])/(2*h /1000)
    return Er

@numba.jit(nopython=True)
def EzCal(V,h):
    #using forward difference to find E  (V/m)  
    dim = V.shape
    nz = dim[0]
    nr = dim[1]
    Ez = np.zeros((nz,nr), np.float32)
    for r in range(nr-1):
        for z in range(1, nz-1):
            #derivatives not defined at boundaries 
            Ez[z,r] = (V[z-1,r] - V[z+1,r])/(2*h / 1000)
    return Ez

def plotCenter(Vinp,Ez):
    bounds,V = defineBoundary(0.05)
    dim = bounds.shape
    zticks = np.multiply(list(range(250,dim[0]-249)),0.05)-25

    plt.figure()
    plt.plot(zticks,Vinp[250:-249,0])
    plt.title('Voltage at r = 0 from z = -10mm to z = 10mm')
    plt.ylabel('V in Volts')
    plt.xlabel('z in mm')
    plt.figure()
    plt.plot(zticks,Ez[250:-249,0], 'r')
    plt.title('E_z at r = 0 from z = -10mm to z = 10mm')
    plt.ylabel('E in Volts/m')
    plt.xlabel('z in mm')

def plotOffset(Vinp,Ez,Er):
    bounds,V = defineBoundary(0.05)
    dim = bounds.shape
    zticks = np.multiply(list(range(250,dim[0]-249)),0.05)-25
    plt.figure()
    plt.plot(zticks,Vinp[250:-249,50])
    plt.title('Voltage at r = 2.5mm from z = -10mm to z = 10mm')
    plt.ylabel('V in Volts')
    plt.xlabel('z in mm')
    plt.figure()
    plt.plot(zticks,Ez[250:-249,50], 'r')
    plt.plot(zticks,Er[250:-249,50], 'b')
    plt.title('E at r = 2.5mm from z = -10mm to z = 10mm')
    plt.ylabel('E in Volts/m')
    plt.xlabel('z in mm')

    
def plotExtra(Vinp,Ez,Er):
    bounds,V = defineBoundary(0.05)
    dim = bounds.shape
    zticks = np.multiply(list(range(dim[0])),0.05)-25
    plt.figure()
    plt.plot(zticks,Vinp[:,70])
    plt.title('Voltage at r = 3.0mm from z = -10mm to z = 10mm')
    plt.ylabel('V in Volts')
    plt.xlabel('z in mm')
    plt.figure()
    plt.plot(zticks,Ez[:,60], 'r')
    plt.plot(zticks,Er[:,60], 'b')
    plt.title('E at r = 3.0mm from z = -10mm to z = 10mm')
    plt.ylabel('E in Volts/m')
    plt.xlabel('z in mm')
    

@numba.jit(nopython=True)
def capCal(Er,Ez,h):
    epsilonNought = 8.8542 #[pF/m]
    V_c = 1000
    nz,nr = Er.shape
    Esq = np.zeros((nz,nr),np.float32)
    sum = 0
    for r in range(1,nr-1):
        for z in range(1,nz-1):
            Esq[z,r] = Er[z,r] * Er[z,r] + Ez[z,r] * Ez[z,r]
            sum+=Esq[z,r] * r /1000
    # expression equivalent to trapezoidal sum, noting that
    #all boundary values are 0 for neater equation
    EsqSum =sum * h * h * 2 * np.pi / (1000*1000)
    C = epsilonNought * EsqSum / (V_c * V_c)
    return C
 
def prelimRun():
    wOptimize(1.0,1.9) #find that 1.4-1.5 yields best performance
    wOptimize(1.4,1.49)#Select w = 1.48 as best value
    dVhTest()
    V = calcV()
    saveV(V, 'Vload')
   
def finalRun():
    
    #calcV()
    with open('Vload', 'rb') as f:
        V = pickle.load(f)
    
    #Above code should be altered if no Vload file exists. In that case 
    #uncomment calcV and comment out the loading.
    
    plt.figure()
    VContour(V) 
    Er = ErCal(V,0.05)
    Ez = EzCal(V,0.05)

    plotCenter(V,Ez)

    plotOffset(V,Ez,Er)
    plotExtra(V,Ez,Er)
    C = capCal(Er,Ez,0.05)
    print(C)

#prelimRun()
finalRun()
# =============================================================================
# bounds, V = defineBoundary(0.1)
# relax(bounds, V, 1.5)
# Vtemp = np.copy(V)
# relax(bounds, V, 1.5)
# =============================================================================



            
                    
    

            
