#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:35:34 2026

@author: aleks
"""

import numpy as np
from numba import njit, prange
from Zandpack import k0nfig
from scipy.integrate import solve_ivp as ode
from  Zandpack.td_constants import hbar, electron_charge
import matplotlib.pyplot as plt
from time import time

if k0nfig.GPU:
    import cupy as cp

##############################################################
############### COMPILED FUNCTIONS ###########################
##############################################################


if k0nfig.NUMBA:
    @njit(parallel = k0nfig.NUMBA_PARALLEL)
    def PI_opti(psi_a, ixi, psi_a_idx):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc = psi_a.shape[2]
        res = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            for c in prange(nc):
                for x in range(nx):
                    if psi_a_idx[x,c]>=0:
                        jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], res[k,:,:] )
        return res/hbar
    
    @njit(cache = k0nfig.CACHE)
    def dot_3d(A,B):
        n = A.shape[0]
        res = np.zeros((n, A.shape[1], B.shape[2]), dtype = np.complex128)
        for i in range(n):
            res[i] = A[i].dot(B[i])
        return res
    
    @njit(cache = k0nfig.CACHE)
    def jit_outer(a,b,res):
        na = len(a)
        nb = len(b)
        for i in range(na):
            res[i,:] += a[i] * b
        return res
    
    @njit(cache = k0nfig.CACHE, parallel = k0nfig.PARALLEL_PI)
    def PI_nb(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc = psi_a.shape[3]
        res = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            reduc_k = np.zeros((nx,nc,nc),dtype = res.dtype)
            for x in prange(nx):
                for c in range(nc_top):
                   jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
            res[k] = reduc_k.sum(axis=0)
        return res/hbar
    
    
    @njit(cache = k0nfig.CACHE, fastmath=k0nfig.FASTMATH)
    def _Q_jit_outer(a,b,res):
        na   = len(a)
        nb   = len(b)
        ac   = np.conj(a)
        bc   = np.conj(b)
        
        for i in range(na):
            ai  = a[i]
            bci = bc[i]
            for j in range(i,nb):
                res[i,j] += ai * b[j] + ac[j]*bci
    
    @njit(cache = k0nfig.CACHE, fastmath=k0nfig.FASTMATH)
    def _Q_jit_outer_v2(a,b,res):
        na   = len(a)
        nb   = len(b)
        ac   = np.conj(a)
        bc   = np.conj(b)
        idx  = np.where(np.abs(b)>1e-13)[0]
        bmin = idx.min()
        bmax = idx.max()+1
        
        for i in range(0, bmin):
            ai  = a[i]
            bci = bc[i]
            for j in range(bmin,bmax):
                res[i,j] += ai * b[j] + ac[j]*bci
        
        for i in range(bmin, bmax):
            ai  = a[i]
            bci = bc[i]
            for j in range(i,nb):
                res[i,j] += ai * b[j] + ac[j]*bci
    
    @njit(cache = k0nfig.CACHE, fastmath = k0nfig.FASTMATH)
    def _Q_make_hermitian(Q):
        na,nb = Q.shape
        for i in range(na):
            for j in range(0,i):
                Q[i,j] = np.conj(Q[j,i])
    
    @njit(cache = k0nfig.CACHE, parallel = k0nfig.PARALLEL_PI, fastmath = k0nfig.FASTMATH)
    def Q_nb(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk     = psi_a.shape[0]
        nx     = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc     = psi_a.shape[3]
        res    = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            reduc_k = np.zeros((nx,nc,nc),dtype = res.dtype)
            for x in prange(nx):
                for c in range(nc_top):
                   ###_Q_jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
                   _Q_jit_outer_v2(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
            res[k] = reduc_k.sum(axis=0)
            _Q_make_hermitian(res[k])
        return res/hbar
    
    def Q_np(psi_a, ixi,nonzero_idx = None):
        """
                
            Returns (Pi + Pi^dagger )/hbar
            Fastest version written so far and is pretty simple.
            
            Both psi_a and ixi is (nk,nx,noT,no) arrays
            
        """
        assert len(psi_a.shape) == 4
        nk,nx,nc_top,nc     = psi_a.shape
        if nonzero_idx is None:
            res    = np.matmul(psi_a.reshape(nk,nx*nc_top,nc).transpose(0,2,1),
                               ixi[:,:,:nc_top,:].reshape(nk,nx*nc_top,nc))
        else:
            res    = np.zeros((nk,nc,nc), dtype=np.complex128)
            subres = np.matmul(psi_a.reshape(nk,nx*nc_top,nc).transpose(0,2,1),
                               ixi[:,:,:nc_top,nonzero_idx
                                   ].reshape(nk,nx*nc_top,len(nonzero_idx)), 
                              )
            res[:,:,nonzero_idx] = subres[:,:,:]
        res   += res.transpose(0,2,1).conj()
        return res/hbar
#######  Here ends the used functions (there are also a couple above)
#######  The stuff below is leftovers of the development.
  



    
###### UNSUPPORTED / DEPRICATED ATM
if k0nfig.GPU:
    def PI_gpu(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc = psi_a.shape[3]
        nax = cp.newaxis
        res = cp.matmul(psi_a[... , nax] , ixi[:,:,0:nc_top, nax, :]).sum(axis = (1,2))
        cp.cuda.Stream.null.synchronize()
        return res/hbar
    
    def Jk_gpu(PI_a):
        return (2*electron_charge/hbar) * cp.trace(PI_a, axis1 = 1, axis2 = 2).real

# Python  +  NumPy functions
def PI_np(psi_a,ixi):
    assert len(psi_a.shape) == 4
    nc_top = psi_a.shape[2]
    nax = np.newaxis
    res = (psi_a[... , nax] @ ixi[:,:,0:nc_top, nax, :]).sum(axis = (1,2))
    return res/hbar

if k0nfig.PI_VERSION == 'NUMBA':
    PI = PI_nb
elif k0nfig.PI_VERSION =='NUMPY':
    PI = PI_np
elif k0nfig.GPU and k0nfig.PI_VERSION=='GPU':
    PI = PI_gpu

def J(PI_a):
    return (2*electron_charge/hbar) * np.trace(PI_a, axis1 = 1, axis2 = 2).sum(axis = 0).real

def Jk(PI_a):
    return (2*electron_charge/hbar) * np.trace(PI_a, axis1 = 1, axis2 = 2).real


def AdaptiveRK4(f, sig0, psi0, omega0, eps, t0, t1,
                dH, delta_func, Ixi,
                h_guess = None, dH_given = True,
                print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
                write_func = None, print_step = 50, plot = False, use_GPU = False,
                atol = None, rtol = None,
                elec_names = ['left', 'right']):
    
    
    # Adaptive timestep RK4
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
                  [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
                  [439/216  , -8         , 3680/513  , -845/4104, np.nan],
                  [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
                  ]
                  )
    
    C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
    CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
    CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])
    
    if k0nfig.GPU and use_GPU:
        xp  = cp
        Cur = Jk_gpu
        PI_x = PI_gpu
        save_array = cp.asnumpy
    else:
        xp         = np
        Cur        = Jk
        PI_x       = PI
        save_array = np.asarray
    if atol is None:
        def TERR(k1,k2,k3,k4,k5,k6):
            res = 0.0
            for i in range(3):
                res_I = xp.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
                                       CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] ))**2  )
                res+=res_I
            return xp.sqrt(res)
    else:
        def TERR(k1,k2,k3,k4,k5,k6):
            e_sig = xp.abs(CT[0]*k1[0] + CT[1]*k2[0] + CT[2]*k3[0] +
                           CT[3]*k4[0] + CT[4]*k5[0] + CT[5]*k6[0] )
            
            e_psi = xp.abs(CT[0]*k1[1] + CT[1]*k2[1] + CT[2]*k3[1] +
                           CT[3]*k4[1] + CT[4]*k5[1] + CT[5]*k6[1] )
            
            e_omg = xp.abs(CT[0]*k1[2] + CT[1]*k2[2] + CT[2]*k3[2] +
                           CT[3]*k4[2] + CT[4]*k5[2] + CT[5]*k6[2] )
            
            f1 = ((e_sig - atol[0])/rtol[0] - xp.abs(state_sig)).max()
            f2 = ((e_psi - atol[1])/rtol[1] - xp.abs(state_psi)).max()
            f3 = ((e_omg - atol[2])/rtol[2] - xp.abs(state_omg)).max()
            error = 1**(max((f1,f2,f3)))
            return error
    
    def step_fourth(y_pre, k1,k2,k3,k4,k5,k6):
        res  =(
               y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0],
               y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1],
               y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]
              )
        return res    
    
    def hnew(h, eps, TE):
        return 0.9 * h * (eps / TE) ** (1 / 5)
    
    def scalar_mult(Arr,number):
        Arr*=number

    current_left   = []
    current_right  = []
    density_matrix = []
    times          = []
    
    
    if sig0 is None or psi0 is None or omega0 is None:
        state_sig = xp.load(name+'_last_sig.npy')
        state_psi = xp.load(name+'_last_psi.npy')
        state_omg = xp.load(name+'_last_omega.npy')
        if t0 is None:
            t0        = float(xp.load(name+'_last_time.npy'))
    else:
        state_sig =  sig0.copy()
        state_psi =  psi0.copy()
        state_omg =  omega0.copy()
    
    if h_guess is None:
        h = (t1-t0)/1000
    else:
        h  = 0.0
        h += h_guess
    
    step =  0
    T0   =  0
    T0  += t0
    
    with open(name+'.txt','w') as file:
        file.write('\n\n\n\n\nStart (Wait for compilation)\n')
    time_start= time()
    data = dict()
    for e in elec_names:
        data.update({'current_'+e:[]})
    
    #SIGMAS = np.zeros((6,)+state_sig.shape)
    
    
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            #print(t0)
            dt = A * h
            
            k1 =     f(t0+dt[0], state_sig, state_psi, state_omg, 
                       dH, delta_func, dH_given = dH_given )
            #k1 = tuple([h*v for v in k1])
            [scalar_mult(v,h) for v in k1]
            
            k2 =     f(t0+dt[1], state_sig + B[1,0]*k1[0], 
                                 state_psi + B[1,0]*k1[1], 
                                 state_omg + B[1,0]*k1[2],
                       dH, delta_func, dH_given = dH_given )
            #k2 = tuple([h*v for v in k2])
            [scalar_mult(v,h) for v in k2]
            
            k3 =     f(t0+dt[2], state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
                                 state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
                                 state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k3 = tuple([h*v for v in k3])
            [scalar_mult(v,h) for v in k3]
            
            
            k4 =     f(t0+dt[3], state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
                                 state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
                                 state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k4 = tuple([h*v for v in k4])
            [scalar_mult(v,h) for v in k4]
            
            k5 =     f(t0+dt[4], state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
                                 state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
                                 state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k5 = tuple([h*v for v in k5])
            [scalar_mult(v,h) for v in k5]
            
            k6 =     f(t0+dt[5], state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
                                 state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
                                 state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            # k6 = tuple([h*v for v in k6])
            [scalar_mult(v,h) for v in k6]
            
            if fixed_mode:
                break
            else:
                TE  =  TERR(k1, k2, k3, k4, k5, k6)
                h   =  hnew(h, eps, TE)
        
        density_matrix += [save_array(state_sig[:,:,:])]
        for _ie, e in enumerate(elec_names):
            data['current_'+e]+=[save_array(Cur(PI_x(state_psi[:, _ie],Ixi[:, _ie]) ) ) ]
        
        state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6)
        times  +=  [t0]
        t0     +=   h
        
        if np.mod(step, print_step) == 0:
            with open(name + '.txt', 'a') as file:
                file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
                file.write('current timestep: '+str(h) +'fs\n')
                file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
                if write_func is not None:
                    write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
                if plot==True:
                    plt.show()
                    for _ie, e in enumerate(elec_names):
                        plt.plot(xp.array(times), xp.array(data['current_'+e]), label = str(_ie))
                    plt.xlabel('Time [fs]', size = 20)
                    plt.savefig('Current(t)',dpi =300)
                    plt.show()
                    plt.pause(0.05)
            
            xp.save(name + '_last_sig', state_sig)
            xp.save(name + '_last_psi', state_psi)
            xp.save(name + '_last_omega', state_omg)
            xp.save(name + '_last_time', xp.array(times[-1]))
            xp.save('_times', xp.array(times))
            current_keys = [k for k in data.keys() if 'current' in k]
            for ck in current_keys:
                xp.save('_'+ck, save_array(data[ck]))
            
            xp.save('_#electrons_device',xp.trace(save_array(density_matrix),axis1 = 2, axis2=3 ))
            
        step+=1
    
    data.update({'density matrix': save_array(density_matrix)})
    
    runtime = time()-time_start
    
    with open(name + '.txt', 'a') as file:
        file.write('100%\n')
        file.write('Runtime: ' + str(runtime) + 'seconds')
    
    return np.array(times), data

def AdaptiveDOP(f, sig0, psi0, omega0, eps, t0, t1,
                dH, delta_func, Ixi,
                h_guess = None, dH_given = True,
                print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
                write_func = None, print_step = 50, plot = False, use_GPU = False,
                atol = None, rtol = None,
                elec_names = ['left', 'right']):
    
    
    # Adaptive timestep RK4
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    from DOP54 import A as AA
    from DOP54 import b4,b5
    CT = b5-b4
    A = AA[:,0]
    B = AA[:,1:]
    CH = b5
    
    
    if k0nfig.GPU and use_GPU:
        xp  = cp
        Cur = Jk_gpu
        PI_x = PI_gpu
        save_array = cp.asnumpy
    else:
        xp         = np
        Cur        = Jk
        PI_x       = PI
        save_array = np.asarray
    if atol is None:
        def TERR(k1,k2,k3,k4,k5,k6,k7):
            res = 0.0
            for i in range(3):
                res_I = xp.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
                                       CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] + CT[6] * k7[i]))**2  )
                res+=res_I
            return xp.sqrt(res)
    else:
        def TERR(k1,k2,k3,k4,k5,k6):
            e_sig = xp.abs(CT[0]*k1[0] + CT[1]*k2[0] + CT[2]*k3[0] +
                           CT[3]*k4[0] + CT[4]*k5[0] + CT[5]*k6[0] )
            
            e_psi = xp.abs(CT[0]*k1[1] + CT[1]*k2[1] + CT[2]*k3[1] +
                           CT[3]*k4[1] + CT[4]*k5[1] + CT[5]*k6[1] )
            
            e_omg = xp.abs(CT[0]*k1[2] + CT[1]*k2[2] + CT[2]*k3[2] +
                           CT[3]*k4[2] + CT[4]*k5[2] + CT[5]*k6[2] )
            
            f1 = ((e_sig - atol[0])/rtol[0] - xp.abs(state_sig)).max()
            f2 = ((e_psi - atol[1])/rtol[1] - xp.abs(state_psi)).max()
            f3 = ((e_omg - atol[2])/rtol[2] - xp.abs(state_omg)).max()
            error = 1**(max((f1,f2,f3)))
            return error
    
    def step_fourth(y_pre, k1,k2,k3,k4,k5,k6, k7):
        res  =(
               y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0]+CH[6]*k7[0],
               y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1]+CH[6]*k7[1],
               y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]+CH[6]*k7[2]
              )
        return res
    
    def hnew(h, eps, TE):
        return 0.9 * h * (eps / TE) ** (1 / 5)
    
    current_left   = []
    current_right  = []
    density_matrix = []
    times          = []
    
    
    if sig0 is None or psi0 is None or omega0 is None:
        state_sig = xp.load(name+'_last_sig.npy')
        state_psi = xp.load(name+'_last_psi.npy')
        state_omg = xp.load(name+'_last_omega.npy')
        if t0 is None:
            t0        = xp.load(name+'_last_time.npy')[0]
    else:
        state_sig =  sig0.copy()
        state_psi =  psi0.copy()
        state_omg =  omega0.copy()
    
    if h_guess is None:
        h = (t1-t0)/1000
    else:
        h  = 0.0
        h += h_guess
    
    step =  0
    T0   =  0
    T0  += t0
    
    with open(name+'.txt','w') as file:
        file.write('\n\n\n\n\nStart (Wait for compilation)\n')
    time_start= time()
    data = dict()
    for e in elec_names:
        data.update({'current_'+e:[]})
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            #print(t0)
            dt = A * h
            
            k1 =     f(t0+dt[0], state_sig, state_psi, state_omg, 
                       dH, delta_func, dH_given = dH_given )
            k1 = tuple([h*v for v in k1])
            
            k2 =     f(t0+dt[1], state_sig + B[1,0]*k1[0], 
                                 state_psi + B[1,0]*k1[1], 
                                 state_omg + B[1,0]*k1[2],
                       dH, delta_func, dH_given = dH_given )
            k2 = tuple([h*v for v in k2])
            
            k3 =     f(t0+dt[2], state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
                                 state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
                                 state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k3 = tuple([h*v for v in k3])
            
            k4 =     f(t0+dt[3], state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
                                 state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
                                 state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k4 = tuple([h*v for v in k4])
            
            k5 =     f(t0+dt[4], state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
                                 state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
                                 state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k5 = tuple([h*v for v in k5])
            
            k6 =     f(t0+dt[5], state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
                                 state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
                                 state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            k6 = tuple([h*v for v in k6])
            
            k7 =     f(t0+dt[6], state_sig + B[6,5]*k6[0] + B[6,4]*k5[0] + B[6,3]*k4[0] + B[6,2]*k3[0] + B[6,1]*k2[0] + B[6,0]*k1[0],
                                 state_psi + B[6,5]*k6[1] + B[6,4]*k5[1] + B[6,3]*k4[1] + B[6,2]*k3[1] + B[6,1]*k2[1] + B[6,0]*k1[1],
                                 state_omg + B[6,5]*k6[2] + B[6,4]*k5[2] + B[6,3]*k4[2] + B[6,2]*k3[2] + B[6,1]*k2[2] + B[6,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            k7 = tuple([h*v for v in k6])
            
            
            
            if fixed_mode:
                break
            else:
                TE  =  TERR(k1, k2, k3, k4, k5, k6, k7)
                h   =  hnew(h, eps, TE)
        
        density_matrix += [save_array(state_sig[:,:,:])]
        for _ie, e in enumerate(elec_names):
            data['current_'+e]+=[save_array(Cur(PI_x(state_psi[:, _ie],Ixi[:, _ie]) ) ) ]
        
        state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6,k7)
        times  +=  [t0]
        t0     +=   h
        
        if np.mod(step, print_step) == 0:
            with open(name + '.txt', 'a') as file:
                file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
                file.write('current timestep: '+str(h) +'fs\n')
                file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
                if write_func is not None:
                    write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
                if plot==True:
                    plt.show()
                    for _ie, e in enumerate(elec_names):
                        plt.plot(xp.array(times), xp.array(data['current_'+e]), label = str(_ie))
                    plt.xlabel('Time [fs]', size = 20)
                    plt.savefig('Current(t)',dpi =300)
                    plt.show()
                    plt.pause(0.05)
            
            xp.save(name + '_last_sig', state_sig)
            xp.save(name + '_last_psi', state_psi)
            xp.save(name + '_last_omega', state_omg)
            xp.save(name + '_last_time', xp.array(times[-1]))
            xp.save('_times', xp.array(times))
            xp.save('_JL',    save_array(current_left))
            xp.save('_JR',    save_array(current_right))
            xp.save('_#electrons_device',xp.trace(save_array(density_matrix),axis1 = 2, axis2=3 ))
        step+=1
    
    data.update({'density matrix': save_array(density_matrix)})
    
    runtime = time()-time_start
    
    with open(name + '.txt', 'a') as file:
        file.write('100%\n')
        file.write('Runtime: ' + str(runtime) + 'seconds')
    
    return np.array(times), data


def three2one(sig, psi, omg):
    nk    = sig.shape[0]
    no    = sig.shape[1]
    noT   = psi.shape[3]
    nlead = psi.shape[1]
    Nm    = psi.shape[2]
    # it lives in device 
    N_sig = no**2 
    #lead, mode, eigen, orbital
    N_psi = nlead * Nm * noT * no
    nz    = omg.shape[-1]#nlead * Nm * noT * nlead * Nm * no
    
    
    return np.hstack((sig[:,:,:].    reshape(nk * N_sig),\
                      psi[:,:,:,:,:].reshape(nk * N_psi),\
                      omg[:,:].      reshape(nk * nz)))



def one2three(y, nk, no, noT, nlead, Nm, nz):
    N_sig = nk * no**2
    N_psi = nk * nlead * Nm * noT * no
    N_omg = nk * nz
    
    return y[0:N_sig].reshape((nk, no, no)), \
           y[N_sig : (N_sig + N_psi)].reshape((nk, nlead, Nm, noT, no)), \
           y[N_sig+N_psi:N_sig+N_psi+N_omg].reshape((nk, nz))

def propagate(g, y0, t_span, atol, rtol, method = 'RK45'):
    t_eval = np.array([t_span[1] ])
    sol = ode(g, t_span, y0, rtol = rtol, atol = atol, t_eval = t_eval)
    return sol.y

def scipy_ode(f, sig0, psi0, omg0, t0, t_eval, dH, delta_variant, Ixi, 
              dH_given = True, method = 'RK45', dt_guess = None, 
              atol = 1e-6, rtol = 1e-4):
    nk    = sig0.shape[0]
    no    = sig0.shape[1]
    noT   = psi0.shape[3]
    nlead = psi0.shape[1]
    Nm    = psi0.shape[2]
    nz    = omg0.shape[-1]#nlead*Nm*noT*nlead*Nm*no
    assert (t0<t_eval).all()
    n_eval = len(t_eval)
    
    def F(t,y):
        y1,y2,y3 = one2three(y, nk, no, noT, nlead, Nm, nz)
        D_sig, D_psi, D_omg = f(t, y1, y2, y3, dH, delta_variant, dH_given = dH_given)
        return three2one(D_sig,D_psi,D_omg)
    
    state_sig =  sig0.copy()
    state_psi =  psi0.copy()
    state_omg =  omg0.copy()
    
    t_prev  = 0
    t_prev += t0
    
    cl =  []
    cr =  []
    rho=  []    
    
    for i in range(n_eval):
        ts = (t_prev , t_eval[i])
        print(ts)
        
        dt =  t_eval[i] - t_prev
        state_sig, state_psi, state_omg = one2three(
                                                      propagate(F, three2one(state_sig, state_psi, state_omg),
                                                                ts, atol, rtol, method = method),
                                                     
                                                      nk, no, noT, nlead, Nm, nz
                                                    )
        
        rho += [state_sig[0]]
        cl  += [Jk(
                    PI(
                        state_psi[:, 0], 
                        Ixi[:, 0]
                        )
                    )
                ]
        t_prev += dt
    
    return t_eval, np.array(rho), np.array(cl), #state_sig, state_psi, state_omg

####################################
##Code Graveyard ## Rest in peace  # 
####################################
#       #        ##       #        #  
#       #        ##       #        #
#    #######     ##    #######     #
#       #        ##       #        #
#       #        ##       #        #
#       #        ##       #        #
#       #        ##       #        #
####################################

# @njit
# def other_psi(psi, xi, Ixi):
#     nk,na,nx,nc = psi.shape[0:4]
#     new_psi     = np.zeros(psi.shape, dtype = np.complex128)
#     xi_conj = xi.conj()
#     norms = (xi * xi_conj).sum(axis = 4)
#     for k in range(nk):
#         for a in range(na):
#             for x in range(nx):
#                 for c in range(nc):
#                     v = psi[k,a,x,c,:].dot(xi_conj[k,a,x,c,:]) / norms[k,a,x,c]
#                     new_psi[k,a,x,c,:] = np.conj(v) * Ixi[k,a,x,c,:]
#     return new_psi




# def AdaptiveRK4_Htnojit(f, sig0, psi0, omega0, eps, t0, t1,
#                         Ht_func, delta_func, Ixi,
#                         h_guess = None,
#                         print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
#                         write_func = None, print_step = 10, plot = False):
        
#     from time import time
#     # Adaptive timestep RK4
#     # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
#     A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
#     B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
#                   [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
#                   [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
#                   [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
#                   [439/216  , -8         , 3680/513  , -845/4104, np.nan],
#                   [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
#                   ]
#                   )
    
#     C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
#     CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
#     CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])
    
#     def TERR(k1,k2,k3,k4,k5,k6):
#         res = 0.0
        
#         for i in range(3):
            
#             res_I = np.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
#                                   CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] ))**2  )
#             res+=res_I
#             #print(res_I)
#         return np.sqrt(res)
    
#     def step_fourth(y_pre, k1,k2,k3,k4,k5,k6):
#         res  =(
#                y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0],
#                y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1],
#                y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]
#               )
#         return res    
    
#     def hnew(h, eps, TE):
#         return 0.9 * h * (eps / TE) ** (1 / 5)
    
#     current_left  = []
#     current_right = []
#     density_matrix = []
#     times = []
#     if h_guess is None:
#         h = (t1-t0)/100
#     else:
#         h  = 0
#         h += h_guess
    
#     state_sig =  sig0.copy()
#     state_psi =  psi0.copy()
#     state_omg =  omega0.copy()
    
#     step = 0
#     T0 = 0
#     T0+= t0
#     with open(name+'.txt','w') as file:
#         file.write('\n\n\n\n\nStart (Wait for compilation)\n')
#     time_start= time()
#     while t0 <= t1:
#         TE = 10 * eps
        
#         Ht = Ht_func(t0, state_sig, state_psi, state_omg)
        
#         while TE > eps:
#             dt = A * h
            
#             k1 =     f(t0+dt[0], Ht, state_sig, state_psi, state_omg, 
#                        delta_func)
#             k1 = tuple([h*v for v in k1])
            
#             k2 =     f(t0+dt[1], Ht,
#                                  state_sig + B[1,0]*k1[0], 
#                                  state_psi + B[1,0]*k1[1], 
#                                  state_omg + B[1,0]*k1[2],
#                        delta_func)
#             k2 = tuple([h*v for v in k2])
            
#             k3 =     f(t0+dt[2], Ht,
#                                  state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
#                                  state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
#                                  state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
#                        delta_func)
#             k3 = tuple([h*v for v in k3])
            
#             k4 =     f(t0+dt[3], Ht,
#                                  state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
#                                  state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
#                                  state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
#                        delta_func)
#             k4 = tuple([h*v for v in k4])
            
#             k5 =     f(t0+dt[4], Ht,
#                                  state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
#                                  state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
#                                  state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
#                        delta_func)
#             k5 = tuple([h*v for v in k5])
            
            
#             k6 =     f(t0+dt[5], Ht,
#                                  state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
#                                  state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
#                                  state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
#                        delta_func)
#             k6 = tuple([h*v for v in k6])
#             if fixed_mode:
#                 break
#             else:
#                 TE = TERR(k1, k2, k3, k4, k5, k6)
#                 h  = hnew(h, eps, TE)
        
#         density_matrix += [state_sig[0,:,:]]
        
#         current_left  += [J(
#                     PI(
#                         state_psi[:, 0], 
#                         Ixi[:, 0]
#                         )
#                     )
#                   ]
#         current_right  += [J(
#                                 PI(
#                                     state_psi[:, 1], 
#                                     Ixi[:, 1]
#                                     )
#                             )
#                           ]
        
#         #print(np.abs(state_sig).sum() , np.abs(state_psi).sum(), np.abs(state_omg).sum())
#         state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6)
#         times += [t0]
#         t0 += h
#         if np.mod(step, print_step) == 0:
#             with open(name + '.txt', 'a') as file:
#                 file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
#                 file.write('current timestep: '+str(h) +'fs\n')
#                 file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
#                 if write_func is not None:
#                     write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
#                 if plot==True:
#                     plt.plot(np.array(times), np.array(current_left))
            
#             #print('\n '+ str(h))
#         step+=1
    
#     data = {'current left':  np.array(current_left),
#             'current right': np.array(current_right),
#             'density matrix': np.array(density_matrix),
#             'last sigma': state_sig,
#             'last psi':   state_psi,
#             'last omega': state_omg}
#     runtime = time()-time_start
    
#     with open(name + '.txt', 'a') as file:
#         file.write('100%\n')
#         file.write('Runtime: ' + str(runtime) + 'seconds')
    
#     return np.array(times), data





# def F(t,x):
#     return np.array(f(t, x[0], x[1],x[2], 0,0))

    
# def f(t,x,y,z, a,b, dH_given = True):
#     if t<5:
#         return (1*x + -2*y) * np.sin(t*5), (2*z+x), (-y-z-x)
#     else:
#         return (1*x + -2*y), (2*z+x), (-y-z-x)

# # def dH(t):
# #     return 0


# def delta(a,t):
#     return 0

# xi = np.random.random(1)

# a,b,c = np.array([1.0]),np.array([2.0]),np.array([3.0])

# t,d = AdaptiveRK4(f, a, b, c, 1e-10, -10, 0,
#                   dH, delta, xi,
#                   h_guess = None, dH_given = True,
#                   print_to_file= True, fixed_mode = False)



    
    
    
    
    # def make_f_safe(self):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy() 
    #     GG_M   = self.GG_M.copy() 
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.sampling_idx.shape[1] + nf
    #     omega_idx  = self.omega_idx.copy()
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,3,4)
    #                              , 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)), 
    #                              axis = 2)
        
    #     #xi_inv = 
        
    #     xi_conj = xi.conj()
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
        
        
        
    #     self.diff_GGM_GLM = diff_GGM_GLM.copy()
    #     self.diff_GGP_GLP = diff_GGP_GLP.copy()
    #     self.xi = xi
        
    #     @njit
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
            
    #         dt = np.complex128
            
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
            
    #         # For equations (4), (20) & # (21) Croy & Popescu 2016
    #         nk = old_sig.shape[0]
            
    #         # For use in psi eom and omega eom
    #         psi_conj = old_psi.conj()
            
    #         # Lead energy shift at time t´
    #         delta_t = np.zeros(nl)
            
    #         for a in range(nl):
    #             delta_t[a] = delta_variant(t, a)
            
    #         # Hamiltonian at time t
    #         #if dH_given:
    #         Ht = H + dH(t)
    #         #else:
    #         #    Ht = dH(t)
    #         #print(delta_t)
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         #print(D_sig)
    #         for a in range(nl):
    #             pi_a   = PI(old_psi[:, a], xi_conj[:, a]) #### Transpose removed, MAYBE NOT?????
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
            
    #         #####
            
    #         ##### Psi vector EOM:
    #         hbar_sq = hbar**2
    #         for k in range(nk):
    #             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
    #             old_omega_k = old_omega[k]
                
    #             for a in range(nl):
    #                 for x in range(Ntot):
    #                     for c in range(no):
    #                         Hchi = Ht[k,:,:].copy()
    #                         idx  = omega_idx[a,x,c,:,:,:].copy().reshape((nl*Ntot*no))
    #                         bOOl = ( idx >= 0 )
    #                         xi_sub    = xi_k[:,bOOl]
    #                         idx       = idx[bOOl]
    #                         omega_sub = old_omega_k[idx]
    #                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                            
    #                         for diag in range(no):
    #                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                            
    #                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(old_psi[k,a,x,c,:] ) / hbar                     +\
    #                                                   GL_P[k,a,x,c]*xi[k,a,x,c,:]                              +\
    #                                                   diff_GGP_GLP[k,a,x,c]* old_sig[k,:,:].dot(xi[k,a,x,c,:])  +\
    #                                                   omega_sum/hbar_sq 
    #                                                   )
                            
            
    #         for k in range(nk):
    #             for a1 in range(nl):
    #                 for x1 in range(Ntot):
    #                     for c1 in range(no):
    #                         psi_axc_1 = old_psi[k,a1,x1,c1,:]
    #                         xi_axc_1  =      xi[k,a1,x1,c1,:]
    #                         xp_axc_1  =     Xpp[k,a1,x1,c1] + delta_t[a1]
                            
    #                         diff_ggp_glp = diff_GGP_GLP[k,a1,x1,c1]
                            
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = omega_idx[a1,x1,c1,a2,x2,c2]
    #                                     if idx < 0:
    #                                         pass
    #                                     else:
    #                                         psi_axc_2    = psi_conj[k,a2,x2,c2,:]
    #                                         xi_axc_2     =  xi_conj[k,a2,x2,c2,:]
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            
    #                                         D_omega[k, idx] = ( -1j*(xm_axc_2  - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
    #                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
    #                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
    #                                                           )
    #         return D_sig, D_psi, D_omega
        
    #     return f
    
    
    # def make_f_exp(self, parallel = True, fastmath = True):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy() 
    #     GG_M   = self.GG_M.copy() 
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.sampling_idx.shape[1] + nf
    #     omega_idx  = self.omega_idx.copy()
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,3,4)
    #                              , 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)), 
    #                              axis = 2)
        
    #     xi_conj = xi.conj()
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
        
    #     #self.diff_GGM_GLM = diff_GGM_GLM.copy()
    #     #self.diff_GGP_GLP = diff_GGP_GLP.copy()
    #     #self.xi = xi
        
    #     @njit(parallel = parallel, fastmath = fastmath)
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
            
    #         dt = np.complex128
            
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
            
    #         # For equations (4), (20) & # (21) Croy & Popescu 2016
    #         nk = old_sig.shape[0]
            
    #         # For use in psi eom and omega eom
    #         psi_conj = old_psi.conj()
            
            
            
    #         # Lead energy shift at time t´
    #         delta_t = np.zeros(nl)
            
    #         for a in range(nl):
    #             delta_t[a] = delta_variant(t, a)
            
    #         # Hamiltonian at time t
    #         #if dH_given:
    #         Ht = H + dH(t)
    #         #else:
    #         #    Ht = dH(t)
    #         #print(delta_t)
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         #print(D_sig)
    #         for a in range(nl):
    #             pi_a   = PI(old_psi[:, a], xi_conj[:, a]) #### Transpose removed, MAYBE NOT?????
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
            
    #         #####
            
    #         ##### Psi vector EOM:
    #         hbar_sq = hbar**2
    #         for k in range(nk):
    #             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
    #             old_omega_k = old_omega[k]
                
    #             for a in range(nl):
    #                 for x in prange(Ntot):
    #                     for c in range(no):
    #                         Hchi = Ht[k,:,:].copy()
    #                         idx  = omega_idx[a,x,c,:,:,:].copy().reshape((nl*Ntot*no))
    #                         bOOl = ( idx >= 0 )
    #                         xi_sub    = xi_k[:,bOOl]
    #                         idx       = idx[bOOl]
    #                         omega_sub = old_omega_k[idx]
    #                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                            
    #                         for diag in range(no):
    #                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                            
    #                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(old_psi[k,a,x,c,:] ) / hbar                     +\
    #                                                   GL_P[k,a,x,c]*xi[k,a,x,c,:]                              +\
    #                                                   diff_GGP_GLP[k,a,x,c]* old_sig[k,:,:].dot(xi[k,a,x,c,:])  +\
    #                                                   omega_sum/hbar_sq 
    #                                                   )
                            
                            
    #                         psi_axc_1 = old_psi[k,a,x,c,:]
    #                         xi_axc_1  =      xi[k,a,x,c,:]
    #                         xp_axc_1  =     Xpp[k,a,x,c] + delta_t[a]
                            
    #                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                            
                            
                            
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = omega_idx[a,x,c,a2,x2,c2]
    #                                     if idx < 0:
    #                                         pass
    #                                     else:
    #                                         psi_axc_2    = psi_conj[k,a2,x2,c2,:]
    #                                         xi_axc_2     =  xi_conj[k,a2,x2,c2,:]
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            
    #                                         D_omega[k, idx] = ( -1j*(xm_axc_2  - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
    #                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
    #                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
    #                                                           )
    #         return D_sig, D_psi, D_omega
        
    #     return f
##############################################################################
            # self.broadenings.append(gel.gamma.copy())
            # self.Lorentzian_centers.append(gel.ei.copy())
            
            # gep2  = self.Ortho_Gammas[e].get_e_subset(idx_fermi_poles)
            # gep   = gel.evaluate_Lorentzian_basis(self.Contour[idx_fermi_poles])
            # gep  = Blocksparse2Numpy(gep,  SLICES)
            # gel  = Blocksparse2Numpy(gel,  SLICES)
            # gep2 = Blocksparse2Numpy(gep2, SLICES)
            
            # self._gp2_matrices += [ gep2 ]
            # EIG = np.linalg.eig
            # el, vl = Sorted_Eig(gel)#
            # assert np.allclose(gel, gel.transpose(0,1,3,2) )
            # ep, vp = Sorted_Eig(gep)#
            # assert np.allclose(gep, gep.transpose(0,1,3,2) )
            # Gl_eig.append(el)
            # Gl_vec.append(vl)
            # Gp_eig.append(ep)
            # Gp_vec.append(vp)
            # Gl.append(gel)
            # Gp.append(gep)
        
        # self.Gl_eig = np.array(Gl_eig) # indices: lead, k, x, state
        # self.Gl_vec = np.array(Gl_vec) # indices: lead, k, x, state, :
        # self.Gp_eig = np.array(Gp_eig)
        # self.Gp_vec = np.array(Gp_vec)
        # assert self.Gl_eig.shape[2] == NumL
        # self._Inv_Gl_vec = np.linalg.inv(self.Gl_vec)
        # self.Inv_Gp_vec  = np.linalg.inv(self.Gp_vec)
        # print('Maximum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.max(),6)))
        # print('Minimum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.min(),6)))

        


# def make_f_general_V3(self, parallel = False, fastmath = False):
#     # The second index on H was for being able to multiply with energy-resolved quantities,
#     # which is not needed anymore
#     H      = self.Hdense[:,0,:,:].copy()
#     Xpp    = self.Xpp.copy()
#     Xpm    = self.Xpm.copy()
#     GG_P   = self.GG_P.copy()
#     GG_M   = self.GG_M.copy()
#     GL_P   = self.GL_P.copy()
#     GL_M   = self.GL_M.copy()
#     nl     = self.num_leads
#     nf     = self.num_poles
#     no     = H.shape[2]
#     nk     = H.shape[0]
#     Ntot   = self.NumL + nf
#     omega_idx  = self.omega_idx.copy()
#     psi_idx    = self.psi_idx.copy()
    
#     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
#     xi     = np.concatenate((
#                               self.Gl_vec.transpose(1,0,2,4,3)
#                               , 
#                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
#                              ), 
#                             axis = 2
#                             )
    
#     xi = np.ascontiguousarray(xi)
    
#     Ixi    = np.concatenate(
#                             (
#                               self.Gl_vec.transpose(1,0,2,4,3).conj()
#                               ,
#                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
#                             ),
#                            axis = 2
#                            )
    
#     Ixi = np.ascontiguousarray(Ixi)
    
#     diff_GGP_GLP = GG_P - GL_P
#     diff_GGM_GLM = GG_M - GL_M
#     self.diff_ggp_glp = diff_GGP_GLP
#     self.diff_ggm_glm = diff_GGM_GLM
    
#     self.xi  = xi
#     self.Ixi = Ixi
    
#     for a in range(nl):
#         for k in range(nk):
#             for xl in range(self.Gl_vec.shape[2]):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 =   self.Gl_vec[a,k,xl,:,c]
#                     vec2 =   self.Gl_vec[a,k,xl,:,c].conj()
#                     O    =  np.multiply.outer(vec1, vec2)
#                     m   +=   self.Gl_eig[a,k,xl,c] * O
#                 assert np.allclose(m, self._gl_matrices[a][k,xl,:,:])
#             for xf in range(2 * nf):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 = self.Gp_vec[a,k,xf,:,c]
#                     vec2 = self.Inv_Gp_vec[a,k,xf,c,:]
#                     O    = np.multiply.outer(vec1, vec2)
#                     m+=self.Gp_eig[a,k,xf,c] * O
#                 assert np.allclose(m, self._gp_matrices[a][k,xf,:,:])
    
#     @njit(parallel = parallel, fastmath = fastmath)
#     def f(t, 
#           old_sig, old_psi, old_omega,
#           dH, delta_variant, dH_given = True):
#         #Create the arrays needed
#         dt = np.complex128
#         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
#         D_omega =  np.zeros(old_omega.shape , dtype = dt)
#         nk = old_sig.shape[0]
        
#         # psi_dagger is needed:
#         psi_tilde  = old_psi.conj()
#         # Store bias at time t
#         delta_t = np.zeros(nl)
#         for a in range(nl): delta_t[a] = delta_variant(t, a)
#         # Get Hamiltonian
#         if dH_given: Ht =  H + dH(t, old_sig)
#         else:        Ht =      dH(t, old_sig)
        
#         ##### Density matrix EOM:
#         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
#         for a in range(nl):
#             pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
#             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
#         D_sig *= 1/hbar
        
#         ##### Psi & Omega EOM:
#         hbar_sq = hbar**2
#         for k in range(nk):
#             # Order the xi-vectors in columns (.T)
#             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
#             old_omega_k = old_omega[k]
#             old_sig_k   = old_sig[k,:,:].copy()
#             for x in prange(Ntot):
#                 x_omega_idx = omega_idx[:,x,:,:,:,:]
#                 psi_kx = old_psi[k,:,x,:,:]
#                 xi_kx  = xi[k,:,x]
#                 Xpp_kx = Xpp[k,:,x]
                
#                 for a in range(nl):
#                     xa_omega_idx = x_omega_idx[a]
#                     psi_kax      = psi_kx[a]
#                     xi_kax       = xi_kx [a]
#                     Xpp_kax      = Xpp_kx[a]
                    
#                     for c in range(no):
#                         # We also use axc_idx etc. later
#                         #axc_idx   = omega_idx[a,x,c,:,:,:].copy()
#                         axc_idx    = xa_omega_idx[c,:,:,:].copy()
                        
#                         #psi_axc_1 = old_psi[k,a,x,c,:]
#                         psi_axc_1  = psi_kax[c,:]
                        
#                         #xi_axc_1  = xi[k,a,x,c,:]
#                         xi_axc_1   = xi_kax[c,:]
                        
#                         #xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
#                         xp_axc_1   = Xpp_kax[c] + delta_t[a]
                        
#                         #We non-zero psi's are has an index that is greater than or equal to zero.
                        
#                         ###
#                         if psi_idx[a,x,c]>=0:
#                             idx       = axc_idx.reshape((nl*Ntot*no))
#                             # Use only non-zero Omegas
#                             bOOl      =(idx >= 0 )
#                             xi_sub    = xi_k[:,bOOl]
#                             idx       = idx[bOOl]
#                             omega_sub = old_omega_k[idx]
#                             # Do sum in the last term of the EOM
#                             omega_sum = (omega_sub * xi_sub).sum(axis = 1)
#                             # Make the part involving Hamiltonian
#                             Hchi      = Ht[k,:,:].copy()
#                             for diag in range(no):
#                                 Hchi[diag,diag] +=  - xp_axc_1
#                             #Calculate
#                             D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
#                                                       GL_P[k,a,x,c] * xi_axc_1                            +\
#                                                       diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
#                                                       omega_sum/hbar_sq
#                                                     )
#                         ###
                        
#                         # Differences between the Lambdas:
#                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
#                         for a2 in range(nl):
#                             psi_a_2         = psi_tilde[k,a2]
#                             xi_a_2          = Ixi[k,a2]
#                             diff_ggm_glm_a_2= diff_GGM_GLM[k,a2]
#                             xm_a_2          = Xpm[k,a2] + delta_t[a2]
                            
#                             for x2 in range(Ntot):
#                                 psi_ax_2         = psi_a_2[x2]
#                                 xi_ax_2          = xi_a_2[x2]
#                                 diff_ggm_glm_ax_2= diff_ggm_glm_a_2[x2]
#                                 xm_ax_2          = xm_a_2[x2]
                                
#                                 for c2 in range(no):
#                                     idx = axc_idx[a2,x2,c2]
#                                     # Only non-zero terms calculated:
#                                     if idx < 0:
#                                         pass
#                                     else:
#                                         psi_axc_2    = psi_ax_2[c2]          #psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
#                                         xi_axc_2     = xi_ax_2[c2]           #Ixi[k,a2,x2,c2,:]       # 
#                                         diff_ggm_glm = diff_ggm_glm_ax_2[c2] #diff_GGM_GLM[k,a2,x2,c2]
#                                         xm_axc_2     = xm_ax_2[c2]            #Xpm[k,a2,x2,c2] + delta_t[a2]
#                                         #Calculate
#                                         D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
#                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)                +\
#                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
#                                                           )
        
#         return D_sig, D_psi, D_omega
    
    
#     return f
#### This method is expendable, always make the optimization 
#### in "make_f_general" and then just add in down here
# def make_f_general_v2(self, parallel = False, fastmath = False):
#     # The second index on H was for being able to multiply with energy-resolved quantities,
#     # which is not needed anymore
#     H      = self.Hdense[:,0,:,:].copy()
#     Xpp    = self.Xpp.copy()
#     Xpm    = self.Xpm.copy()
#     GG_P   = self.GG_P.copy()
#     GG_M   = self.GG_M.copy()
#     GL_P   = self.GL_P.copy()
#     GL_M   = self.GL_M.copy()
#     nl     = self.num_leads
#     nf     = self.num_poles
#     no     = H.shape[2]
#     nk     = H.shape[0]
#     Ntot   = self.NumL + nf
#     omega_idx  = self.omega_idx.copy()
    
#     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
#     xi     = np.concatenate((
#                               self.Gl_vec.transpose(1,0,2,4,3)
#                               , 
#                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
#                              ), 
#                             axis = 2
#                             )
    
#     xi = np.ascontiguousarray(xi)
    
#     Ixi    = np.concatenate(
#                             (
#                               self.Gl_vec.transpose(1,0,2,4,3).conj()
#                               ,
#                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
#                             ),
#                            axis = 2
#                            )
    
#     Ixi = np.ascontiguousarray(Ixi)
    
#     diff_GGP_GLP = GG_P - GL_P
#     diff_GGM_GLM = GG_M - GL_M
#     self.diff_ggp_glp = diff_GGP_GLP
#     self.diff_ggm_glm = diff_GGM_GLM
    
#     self.xi  = xi
#     self.Ixi = Ixi
    
#     for a in range(nl):
#         for k in range(nk):
#             for xl in range(self.Gl_vec.shape[2]):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 =   self.Gl_vec[a,k,xl,:,c]
#                     vec2 =   self.Gl_vec[a,k,xl,:,c].conj()
#                     O    =  np.multiply.outer(vec1, vec2)
#                     m   +=   self.Gl_eig[a,k,xl,c] * O
#                 assert np.allclose(m, self._gl_matrices[a][k,xl,:,:])
#             for xf in range(2 * nf):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 = self.Gp_vec[a,k,xf,:,c]
#                     vec2 = self.Inv_Gp_vec[a,k,xf,c,:]
#                     O    = np.multiply.outer(vec1, vec2)
#                     m+=self.Gp_eig[a,k,xf,c] * O
#                 assert np.allclose(m, self._gp_matrices[a][k,xf,:,:])
#     ####    VERSION 2    #####
#     @njit(parallel = parallel, fastmath = fastmath)
#     def f(t, Ht,
#           old_sig, old_psi, old_omega,
#           delta_variant):
        
#         dt = np.complex128
#         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
#         D_omega =  np.zeros(old_omega.shape , dtype = dt)
#         nk = old_sig.shape[0]
#         psi_tilde  = old_psi.conj()
        
#         delta_t = np.zeros(nl)
#         for a in range(nl): delta_t[a] = delta_variant(t, a)
#         #if dH_given: Ht =  H + dH(t, old_sig)
#         #else:        Ht =      dH(t, old_sig)
        
#         ##### Density matrix EOM:
#         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
#         for a in range(nl):
#             pi_a   = PI(old_psi[:, a], Ixi[:, a]) 
#             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
#         D_sig *= 1/hbar
        
#         ##### Psi & Omega EOM:
#         hbar_sq = hbar**2
#         for k in range(nk):
#             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
#             old_omega_k = old_omega[k]
#             old_sig_k   = old_sig[k,:,:].copy()
#             for x in prange(Ntot):
#                 for a in range(nl):
#                     for c in range(no):
#                         # We also use axc_idx later
#                         axc_idx = omega_idx[a,x,c,:,:,:].copy()
#                         idx     = axc_idx.reshape((nl*Ntot*no))
                        
#                         bOOl      =(idx >= 0 )
#                         xi_sub    = xi_k[:,bOOl]
#                         idx       = idx[bOOl]
#                         omega_sub = old_omega_k[idx]
#                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
#                         psi_axc_1 = old_psi[k,a,x,c,:]
#                         xi_axc_1  = xi[k,a,x,c,:]
#                         xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
#                         Hchi      = Ht[k,:,:].copy()
                        
#                         for diag in range(no):
#                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                        
#                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
#                                                   GL_P[k,a,x,c] * xi_axc_1                            +\
#                                                   diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
#                                                   omega_sum/hbar_sq
#                                                 )
                        
#                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                        
#                         for a2 in range(nl):
#                             for x2 in range(Ntot):
#                                 for c2 in range(no):
#                                     idx = axc_idx[a2,x2,c2]
#                                     if idx < 0:
#                                         pass
#                                     else:
#                                         psi_axc_2    = psi_tilde[k,a2,x2,c2,:]
#                                         xi_axc_2     = Ixi[k,a2,x2,c2,:]
#                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
#                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                        
#                                         D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
#                                                                  diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
#                                                                  diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
#                                                           )
        
#         return D_sig, D_psi, D_omega
    
#     return f


    # def make_f_most_experimental(self, parallel = False, fastmath = False, nogil = False):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy()
    #     GG_M   = self.GG_M.copy()
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.NumL + nf
    #     omega_idx  = self.omega_idx.copy()
    #     psi_idx    = self.psi_idx.copy()
    #     no_top = 0; no_top += self.max_orbital_idx + 1
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
    #                             ), axis = 2)
        
    #     xi = np.ascontiguousarray(xi)
        
    #     Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
    #                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
    #                             ),axis = 2 )
    #     Ixi = np.ascontiguousarray(Ixi)
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
    #     self.diff_ggp_glp = diff_GGP_GLP
    #     self.diff_ggm_glm = diff_GGM_GLM
        
    #     self.xi  = xi
    #     self.Ixi = Ixi
        
    #     @njit(parallel = parallel, fastmath = fastmath, nogil = nogil)
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
    #         #Create the arrays needed
    #         dt = np.complex128
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
    #         nk = old_sig.shape[0]
            
    #         # psi_dagger is needed:
    #         psi_tilde  = old_psi.conj()
    #         # Store bias at time t
    #         delta_t = np.zeros(nl)
    #         for a in range(nl): delta_t[a] = delta_variant(t, a)
    #         # Get Hamiltonian
    #         if dH_given: Ht =  H + dH(t, old_sig)
    #         else:        Ht =      dH(t, old_sig)
            
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         for a in range(nl):
    #             pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
    #         D_sig *= 1/hbar
            
    #         ##### Psi & Omega EOM:
    #         hbar_sq  = hbar**2
    #         _1ohbar  = 1/hbar
    #         _1johbar = 1j /hbar
    #         for x in prange(Ntot):
    #             for k in range(nk):
    #                 for a in range(nl):
    #                     for c in range(no_top):
    #                         # We also use axc_idx etc. later
    #                         axc_idx   = omega_idx[a,x,c,:,:,:]
    #                         psi_axc_1 = old_psi[k,a,x,c,:]
    #                         xi_axc_1  = xi[k,a,x,c,:]
    #                         xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
    #                         # We non-zero psi's are has an index that is greater than or equal to zero.
    #                         # if psi_idx[a,x,c]>=0:
    #                         # Calculate
    #                         D_psi[k,a,x,c,:]=  ( (Ht[k].dot(psi_axc_1 )  - (Xpp[k,a,x,c] + delta_t[a] ) * psi_axc_1) * _1ohbar
    #                                                       + GL_P[k,a,x,c] * xi_axc_1
    #                                                       + diff_GGP_GLP[k,a,x,c]* old_sig[k].dot(xi_axc_1)
    #                                                )
                            
    #                         ###
    #                         #  Differences between the Lambdas:
    #                         omega_sum = np.zeros(no, dtype = np.complex128)
    #                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = axc_idx[a2,x2,c2]
    #                                     # Only non-zero terms calculated:
    #                                     if idx >= 0:
    #                                         psi_axc_2    = psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
    #                                         xi_axc_2     = Ixi[k,a2,x2,c2,:]       # 
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
    #                                         # Calculate
    #                                         D_omega[k, idx] = ( -(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] *_1johbar 
    #                                                           +    diff_ggm_glm * np.sum(xi_axc_2  * psi_axc_1)
    #                                                           +    diff_ggp_glp * np.sum(psi_axc_2 * xi_axc_1 )
    #                                                            )
                                            
    #                                         #if psi_idx[a,x,c]>=0:
    #                                         omega_sum += old_omega[k, idx] * xi[k,a2,x2,c2]
                            
    #                         #if psi_idx[a,x,c]>=0:
    #                         D_psi[k,a,x,c,:] += (omega_sum/hbar_sq)
                            
            
    #         return D_sig, -1j * D_psi, D_omega
        
        
    #     return f
    
# def Inspect_SE_lorentzian_fit(self, lead,I,J,i,j,Emin = -3.0, Emax = 3.0, ik = 0,size = 2):
#     if len(self.fitted_self_energies)==0:
#         print('no fitted self energies!')
#         return
    
#     m1 = self.self_energies[lead]
#     m2 = self.fitted_self_energies[lead]
#     E  = np.linspace(Emin,Emax,1000)
#     plt.show()
#     sidx = self.sampling_idx[lead]
#     plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].real,label = r'Re[Sampled $\Sigma$]',marker = '*',s = size)
#     plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].imag,label = r'Im[Sampled $\Sigma$]',marker = '*',s = size)
    
#     plt.plot(E, m2.evaluate_Lorentzian_basis(E).Block(I,J)[ik,:,i,j].real,label = 'Re[Lorentz fit]',linestyle = 'dashed')
#     plt.plot(E, m2.evaluate_Lorentzian_basis(E).Block(I,J)[ik,:,i,j].imag,label = 'Im[Lorentz fit]',linestyle = 'dashed')
#     plt.legend()

    
    
    # def make_f_purenp_opti(self):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:]
    #     Xpp    = self.Xpp
    #     Xpm    = self.Xpm
    #     GG_P   = self.GG_P
    #     GG_M   = self.GG_M
    #     GL_P   = self.GL_P
    #     GL_M   = self.GL_M
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     Ntot   = self.NumL + nf
    #     Nlr    = self.NumL
        
    #     #omega_idx  = self.omega_idx
    #     psi_idx       = self.psi_idx
    #     noT = 0; noT += self.max_orbital_idx + 1
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
    #                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
    #                             ), axis = 2)
        
    #     xi = np.ascontiguousarray(xi)
        
    #     Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
    #                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
    #                             ),axis = 2 )
    #     Ixi = np.ascontiguousarray(Ixi)
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
    #     self.diff_ggp_glp = diff_GGP_GLP
    #     self.diff_ggm_glm = diff_GGM_GLM
        
    #     self.xi  = xi
    #     self.Ixi = Ixi
        
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
    #         #Create the arrays needed
    #         dt = np.complex128
    #         nk = old_sig.shape[0]
            
    #         # psi_dagger is needed:
    #         psi_tilde  = old_psi.conj()
    #         # Store bias at time t
    #         delta_t = np.zeros(nl)
    #         for a in range(nl): delta_t[a] = delta_variant(t, a)
    #         # Get Hamiltonian
    #         if dH_given: Ht =  H + dH(t, old_sig)
    #         else:        Ht =      dH(t, old_sig)
            
    #         ##### Density matrix EOM:
    #         D_sig =  - 1j * (Ht@old_sig - old_sig@Ht)
    #         for a in range(nl):
    #             pi_a   = PI_np(old_psi[:, a], Ixi[:, a]) 
    #             #print(pi_a.shape)
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
    #         ##### Psi & Omega EOM:
    #         nax = np.newaxis
    #         Xpp_delta  =  Xpp + delta_t[nax,:, nax, nax]
    #         Xpm_delta  =  Xpm + delta_t[nax,:, nax, nax]
            
    #         D_psi  = old_psi @ np.expand_dims(Ht.transpose((0, 2, 1)), (1, 2)) / hbar
    #         D_psi -= np.expand_dims(Xpp_delta[:,:,:,:noT] / hbar, 4) * old_psi [:,:,:,:noT]
    #         D_psi += np.expand_dims(GL_P[:,:,:,:noT], 4) * xi[:,:,:,:noT]
    #         D_psi += np.expand_dims(diff_GGP_GLP[:,:,:,:noT], 4) * (
    #                                 xi[:,:,:,:noT] @ np.expand_dims(old_sig.transpose((0, 2, 1)), (1, 2))
    #                                 )
            
    #         om_shape = (nk, nl*Ntot*noT, nl*Ntot*no)
    #         xi_shape = (nk, nl*Ntot*no , -1        )
            
    #         D_psi   += (old_omega.reshape(om_shape) @ xi.reshape(xi_shape)).reshape(D_psi.shape) / (hbar ** 2)
            
            
            
    #         s_ll = (nk, nl, Nlr, noT, nl, Nlr, no)
    #         s_lf = (nk, nl, Nlr, noT, nl, nf , no)
    #         s_fl = (nk, nl, nf , noT, nl, Nlr, no)
            
    #         D_omega  = np.zeros(old_omega.shape,dtype = dt)
            
    #         D_omega[:,:,0:Nlr,:,:,0:Nlr, :]  +=  (old_psi[:, :, 0:Nlr, :noT, :].reshape(nk, nl*Nlr*noT, no) @
    #                                             ((diff_GGM_GLM[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, 1)
    #                                             * Ixi[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, no)).transpose((0, 2, 1)))).reshape(s_ll)
    #         D_omega[:,:,0:Nlr,:,:,Nlr:Ntot, :]  +=  (old_psi[:,:,0:Nlr,:noT,:].reshape(nk, nl*Nlr*noT, no) @
    #                                               ((diff_GGM_GLM[:,:,Nlr:Ntot,:].reshape(nk, nl*nf*no, 1)
    #                                               * Ixi[:,:,Nlr:Ntot,:].reshape(nk, nl*nf*no, no)).transpose((0, 2, 1)))).reshape(s_lf)
    #         D_omega[:,:,Nlr:Ntot,:,:,0:Nlr, :]  +=  (old_psi[:,:,Nlr:Ntot,:noT,:].reshape(nk, nl*nf*noT, no) @
    #                                               ((diff_GGM_GLM[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, 1)
    #                                               * Ixi[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, no)).transpose((0, 2, 1)))).reshape(s_fl)
            
    #         D_omega[:,:,:,0:noT, :, :, 0:noT] += ((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1)
    #                                               * xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)) @
    #                                                 old_psi.conj().reshape(nk, nl*Ntot*noT, no)
    #                                                 .transpose((0, 2, 1))
    #                                                 ).reshape(nk, nl, Ntot, noT, nl, Ntot, noT)
            
    #         for k in range(nk):
    #             D_omega[k]  += (np.subtract.outer(Xpp_delta[k,:,:,:noT], Xpm_delta[k])*old_omega[k])*(1j/hbar)
            
    #         return D_sig, -1j * D_psi, D_omega
        
        
    #     return f





# def run_curvefit_SE(self,lead,I,J,i,j,ik=0, use_dense_grid=True, fix_L_idx = []):
#     sidx = self.sampling_idx[lead]
#     bij  = self.NO_fitted_lorentzians[lead].Block(I,J)
    
#     if hasattr(self, '_old_sampling_idx') and use_dense_grid:
#         sidx = self._old_sampling_idx[lead]
#     m1   = self.self_energies[lead].Block(I,J)[ik,sidx,i,j]
    
#     if bij is not None:
#         g0   = bij[ik,:,i,j].copy()
#     else:
#         g0 = np.zeros(len(self.NO_fitted_lorentzians[lead].ei[0,:]))
#         return None, None
    
#     line   = self.Contour[sidx].real
#     ei, wi = self.NO_fitted_lorentzians[lead].ei[ik].copy(),self.fitted_lorentzians[lead].gamma[ik].copy()
    
#     idx0 = np.array(fix_L_idx, dtype=int)
#     idx1 = np.array([i for i in range(len(ei)) if i not in idx0])
#     G1   = g0[idx0] # fixed vals
#     G0   = g0[idx1] # variable vals
#     N    = len(g0)
#     def func(x, *p):
#         _p = np.zeros((3, N))
#         _p[0] = wi
#         _p[1] = ei
#         _p[2,idx1] = np.array(p).real
#         _p[2,idx0] = G1.real
#         return (KK_L_sum(x, _p)/2-L_sum(x, _p)/2).real
    
#     poptr, pcov = curve_fit(func, line, m1.real, p0=G0.real)
#     def func(x, *p):
#         _p = np.zeros((3, N))
#         _p[0] = wi
#         _p[1] = ei
#         _p[2,idx1] = np.array(p).real
#         _p[2,idx0] = G1.imag
#         return (KK_L_sum(x, _p)/2-L_sum(x, _p)/2).real
#     popti, pcov = curve_fit(func, line, m1.imag, p0=G0.imag)
#     popt = poptr + 1j * popti
#     POPT = np.zeros(N, dtype=complex)
#     POPT[idx0] = G1
#     POPT[idx1] = popt
#     return POPT, g0
    
    
    
