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
import time
start_time = time.time()
dx=0.11
dz=0.11
wavelength=1.55
k0=2*np.pi/wavelength
nmax=1.45+1.45*0.0075/2
nmin=1.45-1.45*0.0075/2

beta=k0*nmax*0.993#应大于0.9925280199252802保证导模


nlx=50
nlz=60
d=5
lx=d*nlx
lz=d*nlz
phi=np.zeros((lx,lz),dtype=complex)
psi=np.zeros((lx,lz),dtype=complex)
n=np.zeros((lx,lz))
nn=np.zeros((lx,lz))
qq=np.zeros((lx,lz))
n[:,:]=1.45
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
alpha=0.5

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
        b=2*beta+j*dz/dx/dx-j*k0*k0*(n[:,z+1]**2-(beta/k0)**2)/2
        c[:]=2*beta-j*dz/dx/dx+j*k0*k0*(n[:,z]**2-(beta/k0)**2)/2
        a[:]=j*dz/dx/dx/2
        p=np.diag(b[1:lx-1])
        aa=np.diag(a[1:lx-1])
        aa=np.vstack((aa[1:,:],np.zeros((lx-2),dtype=complex)))
        
        aaa=np.diag(a[1:lx-1])
        aaa=np.hstack((aaa[:,1:],np.zeros((lx-2,1),dtype=complex)))
        p=p-aa-aaa
        q=np.diag(c[1:lx-1])
        q=q+aa+aaa
        psi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(psi[1:lx-1,z]))
        
        b[:]=2*beta+j*dz/dx/dx-j*k0*k0*(nn[:,z+1]**2-(beta/k0)**2)/2
        c[:]=2*beta-j*dz/dx/dx+j*k0*k0*(nn[:,z]**2-(beta/k0)**2)/2
        a[:]=j*dz/dx/dx/2
        p[:,:]=np.diag(b[1:lx-1])
        aa[:,:]=np.diag(a[1:lx-1])
        aa[:,:]=np.vstack((aa[1:,:],np.zeros((lx-2),dtype=complex)))
        
        aaa[:,:]=np.diag(a[1:lx-1])
        aaa[:,:]=np.hstack((aaa[:,1:],np.zeros((lx-2,1),dtype=complex)))
        p=p-aa-aaa
        q[:,:]=np.diag(c[1:lx-1])
        q=q+aa+aaa
        phi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(phi[1:lx-1,z]))
    phi=np.fliplr(phi)
                
    for z in range(0,int(lz/d)):
        for x in range(0,int(lx/d)):
            h=0
            for aa in range(0,d-1):
                for bb in range(0,d-1):
                    h=h+psi[d*x+aa,d*z+bb]*np.conjugate(phi[d*x+aa,d*z+bb])
            qq[d*x:d*x+d,d*z:d*z+d]=2*k0*h.imag/d/d
            """
            if (t==1)and(abs(h)==0):
                n[d*x:d*x+d,d*z:d*z+d]=nmin
                print('1')
            """
            n[d*x:d*x+d,d*z:d*z+d]-=alpha*qq[d*x:d*x+d,d*z:d*z+d]
            if (n[d*x,d*z]>nmax):
                n[d*x:d*x+d,d*z:d*z+d]=nmax
            elif(n[d*x,d*z]<nmin):
                n[d*x:d*x+d,d*z:d*z+d]=nmin
            nn=np.fliplr(n)
    if t%1==0:
        
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(abs(psi),cmap='gray')
        plt.subplot(2,2,2)
        
        plt.imshow(abs(phi),cmap='gray')
        plt.subplot(2,2,3)
        plt.imshow(qq,cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(n,cmap='gray')
    fom[t]=np.sum(qq)/lx/lz
plt.plot(fom)
end_time = time.time()
print('running time: ',end_time-start_time)






