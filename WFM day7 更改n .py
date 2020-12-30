# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 16:17:56 2020

@author: David Lyu
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:58:14 2020
找到临界的dx dz；折射率的最大最小值的重新设定
@author: David Lyu
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:27:18 2020
beta的物理意义
@author: David Lyu
"""
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:58:46 2020
n格点=d**2e格点
@author: lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:01:05 2020
WFM n格点=4E格点
@author: lenovo
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:14:51 2020
2d WMF by PLC
@author: lenovo
"""
import numpy as np
from matplotlib import pyplot as plt

dx=0.5
dz=0.11
wavelength=1.55
k0=2*np.pi/wavelength
nmax=1.45+1.45*0.01/2
nmin=1.45-1.45*0.01/2

beta=k0*nmax*0.993#应大于0.9925280199252802保证导模

def sinn(x):
    if x>1.1:
        return 0.9
    else:
        return 0.85

nlx=30
nlz=100
d=5
lx=d*nlx
lz=d*nlz
phi=np.zeros((lx,lz),dtype=complex)
psi=np.zeros((lx,lz),dtype=complex)
n=np.zeros((lx,lz))
nn=np.zeros((lx,lz))
q=np.zeros((lx,lz))
n[:,:]=1.2
nn=np.fliplr(n)

h=complex(0,0)
j=complex(0,1)
alpha=0.1

"""#设定折射率的区域
for x in range(0,lx-1):
    for z in range(0,lz-1):
        if (z>30)or(z<26)or(x<40):
            n[x,z]=1
        else:
            n[x,z]=1.01
"""
for t in range(0,20):
    
    for x in range(0,lx-1):
        psi[x,0]=np.exp(-(x-lx*2/4)**2/(lx/24)**2)
        if psi[x,0]<0.001:
            psi[x,0]=0
        phi[x,0]=np.exp(-(x-lx*3/8)**2/(lx/24)**2)
        if phi[x,0]<0.001:
               phi[x,0]=0
            
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
                
    for z in range(0,int(lz/d)):
        for x in range(0,int(lx/d)):
            h=0
            for a in range(0,d-1):
                for b in range(0,d-1):
                    h=h+psi[d*x+a,d*z+b]*np.conjugate(phi[d*x+a,d*z+b])
                #h=psi[d*x,d*z]*np.conjugate(phi[d*x,d*z])+psi[d*x+d,d*z]*np.conjugate(phi[d*x+d-1,d*z])+psi[d*x,d*z+d-1]*np.conjugate(phi[d*x,d*z+d-1])+psi[d*x+d-1,d*z+d-1]*np.conjugate(phi[d*x+d-1,d*z+d-1])
            if abs(2*k0*h.imag/d/d)>1e-5:
                q[d*x:d*x+d,d*z:d*z+d]=2*k0*h.imag/d/d
            else:
                q[d*x:d*x+d,d*z:d*z+d]=0
            if  (np.sign(q[d*x,d*z])!=0):
                n[d*x:d*x+d,d*z:d*z+d]=1.2+alpha*np.sign(q[d*x,d*z])
            nn=np.fliplr(n)
    if t%1==0:
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(abs(psi),cmap='gray')
        plt.subplot(2,2,2)
        
        plt.imshow(abs(phi),cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(q,cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(n,cmap='gray')
    



