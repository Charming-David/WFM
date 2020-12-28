# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:14:51 2020
2d WMF by PLC
@author: lenovo
"""
import numpy as np
from matplotlib import pyplot as plt
from pylab import *

dx=0.11
dz=0.11
wavelength=1.55
k0=2*np.pi/wavelength
nmax=1.45+1.45*0.0075/2
nmin=1.45-1.45*0.0075/2
#nmin=1.45-1.45*0.0075/2

beta=k0*nmax*0.993#0.9925280199252802
betaa=0


lx=70
lz=50
phi=np.zeros((lx,lz),dtype=complex)
psi=np.zeros((lx,lz),dtype=complex)

n=np.zeros((lx,lz))
nn=np.zeros((lx,lz))
q=np.zeros((lx,lz))
n[:,:]=nmax

h=complex(0,0)
j=complex(0,1)
"""
def sinn(x):
    if x>1.1:
        return 0.9
    else:
        return 0.9
"""
    
alpha=0.05

#设定折射率的区域
for x in range(0,lx):
    for z in range(0,lz):
        if (z<30):#and(z<30):
            n[x,z]=nmin
        else:
            n[x,z]=nmax

for t in range(0,1):

    for x in range(0,lx-1):
        psi[x,0]=1*np.exp(-(x-lx/2)**2/(lx/20)**2)
        if psi[x,0]<0.001:
            psi[x,0]=0
        phi[x,0]=1*np.exp(-(x-lx*3/5)**2/(lx/20)**2)
        if phi[x,0]<0.001:
               phi[x,0]=0
    nn=np.fliplr(n)
    
    for z in range(0,lz-1):
        for x in range(1,lx-1):
            """
            beta=k0*n[x,z]*sinn(n[x,z])
            betaa=k0*nn[x,z]*sinn(nn[x,z])
            """
            if z!=0:
                phi[x,z+1]=-j*dz/(beta)*((2*phi[x,z]-phi[x+1,z]-phi[x-1,z])/dx/dx+((beta)**2-k0**2*nn[x,z]**2)*phi[x,z])+phi[x,z-1]
                psi[x,z+1]=-j*dz/(beta)*((2*psi[x,z]-psi[x+1,z]-psi[x-1,z])/dx/dx+((beta)**2-k0**2*n[x,z]**2)*psi[x,z])+psi[x,z-1]
            else:
                phi[x,z+1]=-0.5*j*dz/(beta)*((2*phi[x,z]-phi[x+1,z]-phi[x-1,z])/dx/dx+((beta)**2-k0**2*nn[x,z]**2)*phi[x,z])+phi[x,z]
                psi[x,z+1]=-0.5*j*dz/(beta)*((2*psi[x,z]-psi[x+1,z]-psi[x-1,z])/dx/dx+((beta)**2-k0**2*n[x,z]**2)*psi[x,z])+psi[x,z]
    phi=np.fliplr(phi)
"""
    for z in range(0,lz):
        for x in range(0,lx):
            h=psi[x,z]*np.conjugate(phi[x,z])
            q[x,z]=h.imag
            n[x,z]=n[x,z]-alpha*q[x,z]
            if (n[x,z]>1.5):
                n[x,z]=1.5
            elif(n[x,z]<1):
                n[x,z]=1
"""
plt.figure()
subplot(2,2,1)
plt.imshow(np.real(phi).T,cmap='gray',vmin=-1, vmax=1)
subplot(2,2,2)
plt.imshow(abs(phi).T,cmap='gray',vmin=0, vmax=1)
subplot(2,2,3)
plt.imshow(abs(psi).T,cmap='gray',vmin=0, vmax=1)
subplot(2,2,4)
plt.imshow(n.T,cmap='gray')
    