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
Created on Sun Dec 13 11:14:51 2020
2d WMF by PLC
@author: lenovo
"""
import numpy as np
from matplotlib import pyplot as plt
import time
start_time = time.time()
dx=0.11
dz=0.11         #光场的网格大小[μm]
wavelength=1.55     #波长[μm]
k0=2*np.pi/wavelength
nmid=1.45
nmax=nmid+nmid*0.0075/2     #折射率最大值最小值
nmin=nmid-nmid*0.0075/2

beta=k0*nmax*0.993#应大于0.9925280199252802保证导模


nlx=30          #折射率网格数目
nlz=30
d=9             #折射率网格尺寸与光场网格尺寸的倍数
lx=d*nlx
lz=d*nlz        #总网格个数
phi=np.zeros((lx,lz),dtype=complex)     #反向传播phi
psi=np.zeros((lx,lz),dtype=complex)     #正向传播psi
n=np.zeros((lx,lz))                     #折射率格子
nn=np.zeros((lx,lz))                    #折射率格子的左右镜像（为了计算反向传播场）
qq=np.zeros((lx,lz))                    #正反向光场的评价
n[:,:]=1.45
nn=np.fliplr(n)
#BPM计算需要的矩阵
a=np.zeros((lx),dtype=complex)
b=np.zeros((lx),dtype=complex)
c=np.zeros((lx),dtype=complex)

p=np.zeros((lx-2,lx-2),dtype=complex)
q=np.zeros((lx-2,lx-2),dtype=complex)
aa=np.zeros((lx-2,lx-2),dtype=complex)
aaa=np.zeros((lx-2,lx-2),dtype=complex)
h=complex(0,0)
#虚数单位
j=complex(0,1)

alpha=0.03      #迭代n的参数

"""#设定折射率的区域
for x in range(0,lx-1):
    for z in range(0,lz-1):
        if (z>30)or(z<26)or(x<40):
            n[x,z]=1
        else:
            n[x,z]=1.01
"""
fom=np.zeros((20))
for t in range(0,20):
    
    for x in range(0,lx-1):             #添加光源（高斯型，只在z=0处添加）
        psi[x,0]=np.exp(-(x-lx*2/4)**2/(lx/24)**2)
        if psi[x,0]<0.001:
            psi[x,0]=0
        phi[x,0]=np.exp(-(x-lx*4/8)**2/(lx/24)**2)
        if phi[x,0]<0.001:
               phi[x,0]=0
    p,q=0,0        
    for z in range(0,lz-1):             #计算一遍正向与反向传播的场
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
        
        psi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(psi[1:lx-1,z]))    #psi向前计算了一格
        
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
        phi[1:lx-1,z+1]=(np.linalg.inv(p)).dot(q.dot(phi[1:lx-1,z]))     #phi向前计算了一格
    phi=np.fliplr(phi)      #phi是反向传播场，需要反过来
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(qq,cmap='gray')
    plt.subplot(2,2,2)
    plt.imshow(n,cmap='gray',vmin=nmin, vmax=nmax)
                
    for z in range(0,int(lz/d)):
        for x in range(0,int(lx/d)):
            h=0
            for aa in range(0,d-1):     #这里计算一个折射率格子（9*9个光场）内的qq
                for bb in range(0,d-1):
                    h=h+psi[d*x+aa,d*z+bb]*np.conjugate(phi[d*x+aa,d*z+bb])
            qq[d*x:d*x+d,d*z:d*z+d]=2*k0*h.imag/d/d
            
            
            n[d*x:d*x+d,d*z:d*z+d]-=alpha*qq[d*x:d*x+d,d*z:d*z+d]       #折射率的一次迭代（连续折射率）
            if (n[d*x,d*z]>nmax):       #对折射率的上下界有要求   
                n[d*x:d*x+d,d*z:d*z+d]=nmax
            elif(n[d*x,d*z]<nmin):
                n[d*x:d*x+d,d*z:d*z+d]=nmin
            nn=np.fliplr(n)
    
    plt.subplot(2,2,3)
    plt.imshow(abs(psi),cmap='gray')
    plt.subplot(2,2,4)
    plt.imshow(abs(phi),cmap='gray')
    
    fom[t]=np.sum(qq)/lx/lz         #记录下这一次qq矩阵的平均值
    end_time = time.time()
    print('running time: ',end_time-start_time)
plt.plot(fom)







