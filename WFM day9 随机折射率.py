# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:36:20 2021
折射率随机
@author: David Lyu
"""
import numpy as np
from matplotlib import pyplot as plt
import time
start_time = time.time()
dx=0.11
dz=0.11
wavelength=1.55
k0=2*np.pi/wavelength
nmid=1.45
nmax=nmid+nmid*0.0075/2
nmin=nmid-nmid*0.0075/2

beta=k0*nmax*0.993#应大于0.9925280199252802保证导模


nlx=60
nlz=40
d=5
lx=d*nlx
lz=d*nlz
phi=np.zeros((lx,lz),dtype=complex)
psi=np.zeros((lx,lz),dtype=complex)
n=np.zeros((lx,lz))
nn=np.zeros((lx,lz))
qq=np.zeros((lx,lz))
n[:,:]=nmid
"""
n0=np.random.randn(int(lx/d),int(lz/d))
n0=nmid+0.0075*nmid*np.sign(n0)
for z in range(0,int(lz/d)):
    for x in range(0,int(lx/d)):
        n[d*x:d*x+d,d*z:d*z+d]=n0[x,z];
"""
nn=np.fliplr(n)

a=np.zeros((lx),dtype=complex)
b=np.zeros((lx),dtype=complex)
c=np.zeros((lx),dtype=complex)

p=np.zeros((lx-2,lx-2),dtype=complex)
q=np.zeros((lx-2,lx-2),dtype=complex)
aa=np.zeros((lx-2,lx-2),dtype=complex)
aaa=np.zeros((lx-2,lx-2),dtype=complex)

h=complex(0,0)
j=complex(0,1)
alpha=0.0075*nmid/2

"""#设定折射率的区域
for x in range(0,lx-1):
    for z in range(0,lz-1):
        if (z>30)or(z<26)or(x<40):
            n[x,z]=1
        else:
            n[x,z]=1.01
"""
fom=np.zeros((10))
for t in range(0,10):
    
    for x in range(0,lx-1):
        psi[x,0]=np.exp(-(x-lx*2/4)**2/(lx/24)**2)
        if psi[x,0]<0.001:
            psi[x,0]=0
        phi[x,0]=np.exp(-(x-lx*3/8)**2/(lx/24)**2)
        if phi[x,0]<0.001:
               phi[x,0]=0
    p,q=0,0        
    for z in range(0,lz-1):
        b[:]=2*beta+j*dz/dx/dx-j*k0*k0*(n[:,z+1]**2-(beta/k0)**2)/2
        c[:]=2*beta-j*dz/dx/dx+j*k0*k0*(n[:,z]**2-(beta/k0)**2)/2
        a[:]=j*dz/dx/dx/2
        p=np.diag(b[1:lx-1])
        q=np.diag(c[1:lx-1])
        aa=np.diag(a[1:lx-1])
        aa=np.vstack((aa[1:,:],np.zeros((lx-2),dtype=complex)))
        p-=aa
        q+=aa
        aa=np.diag(a[1:lx-1])
        aa=np.hstack((aa[:,1:],np.zeros((lx-2,1),dtype=complex)))
        p-=aa
        q+=aa
        
        psi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(psi[1:lx-1,z]))
        
        b[:]=2*beta+j*dz/dx/dx-j*k0*k0*(nn[:,z+1]**2-(beta/k0)**2)/2
        c[:]=2*beta-j*dz/dx/dx+j*k0*k0*(nn[:,z]**2-(beta/k0)**2)/2
        a[:]=j*dz/dx/dx/2
        p=np.diag(b[1:lx-1])
        q=np.diag(c[1:lx-1])
        aa=np.diag(a[1:lx-1])
        aa=np.vstack((aa[1:,:],np.zeros((lx-2),dtype=complex)))
        p-=aa
        q+=aa
        aa=np.diag(a[1:lx-1])
        aa=np.hstack((aa[:,1:],np.zeros((lx-2,1),dtype=complex)))
        p-=aa
        q+=aa
        phi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(phi[1:lx-1,z]))
    phi=np.fliplr(phi)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(qq,cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(n,cmap='gray',vmin=nmin, vmax=nmax)
                
    for z in range(0,int(lz/d)):
        for x in range(0,int(lx/d)):
            h=0
            for aa in range(0,d-1):
                for bb in range(0,d-1):
                    h=h+np.conjugate(phi[d*x+aa,d*z+bb])*psi[d*x+aa,d*z+bb]
            #qq[d*x:d*x+d,d*z:d*z+d]=2*k0*h.imag/d/d
            
            if abs(2*k0*h.imag/d/d)>1e-5:
                qq[d*x:d*x+d,d*z:d*z+d]=2*k0*h.imag/d/d
            else:
                qq[d*x:d*x+d,d*z:d*z+d]=0
                
            n[d*x:d*x+d,d*z:d*z+d]=n[d*x:d*x+d,d*z:d*z+d]-alpha*np.sign(qq[d*x:d*x+d,d*z:d*z+d])
            if (n[d*x,d*z]>nmax):
                n[d*x:d*x+d,d*z:d*z+d]=nmax
            elif(n[d*x,d*z]<nmin):
                n[d*x:d*x+d,d*z:d*z+d]=nmin
            nn=np.fliplr(n)
    
    plt.subplot(2,2,3)
    plt.imshow(abs(psi),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(abs(phi),cmap='gray')
    
    fom[t]=np.sum(qq)/lx/lz
    end_time = time.time()
    print('running time: ',end_time-start_time)
plt.plot(fom)








