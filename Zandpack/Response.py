#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:26:43 2024

@author: aleksander
"""

from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simpson,trapezoid
trapz = trapezoid


@njit
def phi(Ti, T,i,ti):
    t = T-Ti
    assert ((ti[1:] - ti[:-1])>0).all()
    assert len(ti)>i
    dt0 = ti[1]  - ti[0]
    dtn = ti[-1] - ti[-2]
    
    if i == 0 and t>=ti[0] and t<ti[1]:
        return 1-(t - ti[i])/dt0
    if i == 0 and t>=ti[0]-dt0 and t<ti[0]:
        return (t - (ti[0] - dt0) )/dt0
    
    if i == len(ti)-1 and t>=ti[-2] and t<ti[-1]:
        return (t - ti[-2])/(ti[-1] - ti[-2])
    if i == len(ti)-1 and t>=ti[-1] and t<ti[-1] + dtn:
        return 1-(t - (ti[-1]) ) / dtn
    
    if t<ti[i-1] or t>ti[i+1]:
        pass 
    else:
        if t<ti[i]:
            return     (t - ti[i-1])/(ti[i] - ti[i-1])
        else:
            return 1 - (t - ti[i])/(ti[i+1] - ti[i])
    
    return 0.0

# @njit
# def phiv(t,i,ti):
#     out = np.zeros(len(t))
#     for k in range(len(t)):
#         out[k] = phi(t[k], i, ti)
#     return out

#ti = np.linspace(0,20, 5)
#ti = np.sort(np.random.random(20)*20)
#ti = np.array([2.0, 5.0, 10.0, 11.0, 14.0, 19.0, 26.0, 28.0]) * 0.1
#ti = np.array([1.,2,2.5, 3,3.5, 4, 4.5, 6., 7.2, 7.8, 8., 8.3, 8.4, 10, 12.])
# ti = np.linspace(0, 24, 35) + (np.random.random(35)- .5)*0.25
# fi = np.cos(1.0*ti)
# Lv = np.zeros(len(ti))
# tc = .1

# xn = np.linspace(-5, 35,500)
# Li = np.zeros(len(ti))
# F  = np.zeros(len(xn))
# it = 0
# for xi in xn:
#     for i in range(len(ti)):
#         Li[i] = quad(phi, 0, tc, args=(xi, i, ti))[0]/tc#np.interp(xi, t2v, L[i,:])
#     F[it] = Li.dot(fi)
#     it   += 1

#tv = np.linspace(-5,30,1000)


# t2v = np.linspace(-5,40,150)
# L   = np.zeros((len(ti), len(t2v)))
# L2  = np.zeros((len(ti), len(t2v)))
# E   = L.copy()
#samp = np.linspace(0,tc, 10000)
#for it in range(len(t2v)):
#    for i in range(len(ti)):
#        def phi2(tp):
#            return phiv(t2v[it]-tp, i, ti)
        # def phi3(tp):
        #     return phi(t2v[it]-tp, i, ti)
        # L2[i,it], E[i,it] = quad(phi3, 0, tc)
#        L[i,it] = simpson(phi2(samp), x=samp)/tc


class AverageDM:
    def __init__(self,  tc,  dm0, t0, RKN, spacing = None, Update_counter=0):
        n = 7
        self.dm0   = dm0.copy()#np.zeros(dm.shape + (n,))
        self.ti    = [ti for ti in np.linspace(t0-2*tc, t0, n)]
        self.dmi   = [dm0.copy() for i in range(n)]
        self.tc    = tc
        self.RKN         = RKN
        self.spacing     = tc/10 if spacing is None else spacing
            
        self.Lhist       = None
        dt = tc/5
        self.shorthis_t  = np.array([t0-4.0*dt, t0-3*dt, t0-2.0*dt, t0-1.0*dt, t0])
        self.shorthis_dm = dm0[None,:,:,:][[0,0,0,0,0]].copy()
        self.Update_counter = Update_counter + 0 + 0
        
    
    def get_weights(self,t):
        ti_now = np.array(self.ti + [t])
        nt = len(ti_now)
        Lhist = np.zeros(nt)
        for i in range(nt):
            err = 1.0
            ns  = 2
            while err>1e-7:
                r1 = 0.0
                ls1 = np.linspace(0,self.tc, ns)
                for k in range(ns-1):
                    t1,t2 = ls1[k],ls1[k+1]
                    r1 += quad(phi, t1, t2, args=(t, i, ti_now))[0]/self.tc
                r2 = .0
                ls2 = np.linspace(0,self.tc, ns+1)
                for k in range(ns):
                    t1,t2 = ls2[k],ls2[k+1]
                    r2 += quad(phi, t1, t2, args=(t, i, ti_now))[0]/self.tc
                err = abs(r2-r1)
                ns += 1
            Lhist[i] =  r2
        self.Lhist = Lhist
        
    def get_average(self, t, dm):
        nt = len(self.ti)
        out = np.zeros(self.dm0.shape,dtype=self.dm0.dtype)
        self.get_weights(t)
        for i in range(nt):
            out += self.dmi[i] * self.Lhist[i]
        out += dm * self.Lhist[-1]
        return out
    
    def trim_history(self, t):
        ti_array = np.array(self.ti)
        dt = np.diff(ti_array)
        dt_max = dt.max()
        delta = self.tc + dt.max()
        #print(delta)
        idx = np.where(np.abs(t-ti_array ) > delta )[0]
        idx = np.sort(idx)[::-1]
        for i in idx:
            del self.ti[i]
            del self.dmi[i]
    
    def update_short_history(self, t,dm):
        self.shorthis_t = np.roll(self.shorthis_t, -1)
        self.shorthis_t[-1] = t
        self.shorthis_dm = np.roll(self.shorthis_dm, -1, axis=0)
        self.shorthis_dm[-1] = dm
    
    def update_history(self):
        ti_array = np.array(self.ti)
        if ti_array.max() > (self.shorthis_t.min() - self.spacing):
            return
        for i in range(4):
            # Make sure RK algo hasnt made step shorter
            if self.shorthis_t[i+1]>self.shorthis_t[i]:
                self.dmi.append(self.shorthis_dm[i].copy())
                self.ti.append(self.shorthis_t[i] )
                return
    
    def Update(self, t, dm):
        if np.mod(self.Update_counter, self.RKN) == 0:
            self.update_short_history(t, dm)
            self.update_history()
            self.trim_history(t)
        self.Update_counter += 1
        
        
        
# t0 =  -80.0
# dt = 0.0025
# dm0 = np.zeros((1,2,2))
# dm0[:,0,0] = np.cos(t0)
# dm0[:,1,1] = np.sin(t0)
# dm0[:,1,0] = np.sin(t0) * np.cos(t0)
# dm0[:,0,1] = np.sin(2 * t0) * np.cos(t0)
# Init = AverageDM(2.0, dm0, -80, 6, )
# t0 += dt
# res1 = []
# res2 = []
# tv   = []
# for i in range(500):
#     dm0[:,0,0] = np.cos(t0)
#     dm0[:,1,1] = np.sin(t0)
#     dm0[:,1,0] = np.sin(t0) * np.cos(t0)
#     dm0[:,0,1] = np.sin(2 * t0) * np.cos(t0) + (1 / ((t0 + 70)**2  + .1) )
    
#     Init.Update(t0, dm0)
#     res1.append(dm0.copy())
#     res2.append(Init.get_average(t0, dm0))
#     tv .append(t0)
#     t0 += dt * (1 - 0.9 * np.random.random())
    
# plt.plot(tv,np.array(res1)[:,0,1,0], color='g')
# plt.plot(tv,np.array(res2)[:,0,1,0], color='g', linestyle='dashed')

# plt.plot(tv,np.array(res1)[:,0,0,1], color='k')
# plt.plot(tv,np.array(res2)[:,0,0,1], color='k', linestyle='dashed')





