#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 12:09:35 2026

@author: aleks
"""
import numpy as np
from numba import njit, prange
import numba
import k0nfig as config

fm       = False# config.FASTMATH
parallel = False# config.NUMBA_PARALLEL
inline   = 'never'
@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_sig_forward_nb(Sig_v, b, out, idx):
    n1,n2,n3,n4 = Sig_v.shape
    _b = np.ascontiguousarray(b)
    assert Sig_v.ndim == 4
    assert _b.ndim == 1
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                tmp = 0.0 + 0.0j
                for i in idx:
                    tmp += Sig_v[i,j,k,l] * _b[i]
                out[j,k,l] = tmp
@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_sig_forward_nb_last_DONTUSE(Sig_v, b):
    idx = np.arange(len(b))
    assert len(b) == Sig_v.shape[0]
    step_sig_forward_nb(Sig_v, b, Sig_v[len(b)-1], idx)

#FURTHER OPTIMIZATIONS POSSIBLE HERE LIKE THIS FOR THE
#COMPUTATION OF THE ERROR AND LAST STEP (LAST STEP TURNS OUT TO BE FASTER WITH
# THE MATMUL ALREADY USED)

@njit(fastmath=fm, parallel=parallel,inline=inline)
def TERR2_sig_nb(Sig_v, CH):
    n1,n2,n3,n4 = Sig_v.shape
    _CH = np.ascontiguousarray(CH)
    assert Sig_v.ndim == 4
    assert _CH.ndim == 1
    err = 0.0
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                val = 0.0 + 0.0j
                for i in range(n1):
                    val += (Sig_v[i,j,k,l] * _CH[i])
                err += val.real ** 2 + val.imag ** 2
    return err

@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_psi_forward_nb(Psi_v, b, out, idx):
    n1,n2,n3,n4,n5,n6 = Psi_v.shape
    _b = np.ascontiguousarray(b)
    assert Psi_v.ndim == 6
    assert _b.ndim == 1
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                for m in range(n5):
                    for n in range(n6):
                        tmp = 0.0 + 0.0j
                        for i in idx:
                            tmp += Psi_v[i,j,k,l,m,n] * _b[i]
                        out[j,k,l,m,n] = tmp
@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_psi_forward_nb_last_DONTUSE(Psi_v, b):
    idx = np.arange(len(b))
    assert len(b) == Psi_v.shape[0]
    step_psi_forward_nb(Psi_v, b, Psi_v[len(b)-1], idx)
@njit(fastmath=fm, parallel=parallel,inline=inline)
def TERR2_psi_nb(Psi_v, b):
    n1,n2,n3,n4,n5,n6 = Psi_v.shape
    _b = np.ascontiguousarray(b)
    assert Psi_v.ndim == 6
    assert _b.ndim == 1
    err = 0.0
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                for m in range(n5):
                    for n in range(n6):
                        val = 0.0 + 0.0j
                        for i in range(n1):
                            val += Psi_v[i,j,k,l,m,n] * _b[i]
                        err += val.real ** 2 + val.imag ** 2
    return err

@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_omg_forward_nb(Omg_v, b, out, idx):
    n1,n2,n3,n4,n5,n6,n7,n8 = Omg_v.shape
    _b = np.ascontiguousarray(b)
    assert Omg_v.ndim == 8
    assert _b.ndim == 1
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                for m in range(n5):
                    for n in range(n6):
                        for y in range(n7):
                            for x in range(n8):
                                tmp = 0.0 + 0.0j
                                for i in idx:
                                    tmp += Omg_v[i,j,k,l,m,n,y,x] * _b[i]
                                out[j,k,l,m,n,y,x] = tmp


@njit(fastmath=fm, parallel=parallel,inline=inline)
def step_omg_forward_nb_fermi_opti(Omg_v, b, out, idx, fp_loc, fp_all):
    # RK, k, a, x, c ,a, x,c
    # OPTIMIZED VERSION OF the function above,
    # Doesnt modify the zeros of omega
    n1,n2,n3,n4,n5,n6,n7,n8 = Omg_v.shape
    _b = np.ascontiguousarray(b)
    assert Omg_v.ndim == 8
    assert _b.ndim == 1
    assert len(fp_loc) == n4
    assert len(fp_all) == n7
    
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                is_fp_loc = fp_loc[l]
                for m in range(n5):
                    for n in range(n6):
                        for y in range(n7):
                            is_fp_all = fp_all[y]
                            if is_fp_loc and is_fp_all:
                                pass
                            else:
                                for x in range(n8):
                                    tmp = 0.0 + 0.0j
                                    for i in idx:
                                        tmp += Omg_v[i,j,k,l,m,n,y,x] * _b[i]
                                    out[j,k,l,m,n,y,x] = tmp

@njit(fastmath=fm, parallel=parallel,inline=inline)
def TERR2_omg_nb(Omg_v, b):
    n1,n2,n3,n4,n5,n6,n7,n8 = Omg_v.shape
    _b = np.ascontiguousarray(b)
    assert Omg_v.ndim == 8
    assert _b.ndim == 1
    err = 0.0
    for j in range(n2):
        for k in prange(n3):
            for l in range(n4):
                for m in range(n5):
                    for n in range(n6):
                        for y in range(n7):
                            for x in range(n8):
                                tmp = 0.0 + 0.0j
                                for i in range(n1):
                                    tmp += Omg_v[i,j,k,l,m,n,y,x] * _b[i]
                                err += tmp.real**2 + tmp.imag**2
    return err

### TESTS ###
DO_TEST=False
if DO_TEST:
    Sig_t = np.random.random((6,1,150, 400))+1.0j*np.random.random((6,1,150, 400))
    Psi_t = np.random.random((6,1,2,60, 25, 200))+1.0j*np.random.random((6,1,2,60, 25, 200))
    Omg_t = np.random.random((6,1,2,60, 25,2,60,25))+1.0j*np.random.random((6,1,2,60, 25,2,60,25))
    
    IDX = np.array([0,3,1,5])
    v = np.random.random(6)
    out_sig1 = np.zeros(Sig_t.shape[1:])+0.0j
    out_sig2 = np.zeros(Sig_t.shape[1:])+0.0j
    
    out_psi1 = np.zeros(Psi_t.shape[1:])+0.0j
    out_psi2 = np.zeros(Psi_t.shape[1:])+0.0j
    
    out_omg1 = np.zeros(Omg_t.shape[1:])+0.0j
    out_omg2 = np.zeros(Omg_t.shape[1:])+0.0j
    
    
    step_sig_forward_nb(Sig_t, v, out_sig1, IDX)
    np.matmul(Sig_t[IDX].transpose(1,2,3,0), v[IDX].copy(), out=out_sig2)
    assert np.allclose(out_sig1, out_sig2)

    
    step_psi_forward_nb(Psi_t, v, out_psi1, IDX)
    np.matmul(Psi_t[IDX].transpose(1,2,3,4,5,0), v[IDX].copy(), out=out_psi2)
    assert np.allclose(out_psi1, out_psi2)
    
    step_omg_forward_nb(Omg_t, v, out_omg1, IDX)
    np.matmul(Omg_t[IDX].transpose(1,2,3,4,5,6,7,0), v[IDX].copy(), out=out_omg2)
    assert np.allclose(out_omg1, out_omg2)




    @numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
    def abs2(x):
        return x.real**2 + x.imag**2
        
    def TERR2_sig(y1, CT):
        return abs2(y1.transpose(1, 2, 3,        0)@CT).sum()
    def TERR2_psi(y2, CT):
        return abs2(y2.transpose(1, 2, 3, 4, 5,    0)@CT).sum()
    def TERR2_omg(y3, CT):
        return abs2(y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT).sum()
    
    def step_fourth_sig(Y1, CH):
        MM(Y1.transpose(1, 2, 3,             0), CH, Y1[-1])
    def step_fourth_psi(Y2, CH):
        MM(Y2.transpose(1, 2, 3, 4, 5,       0), CH, Y2[-1])
    def step_fourth_omg(Y3, CH):
        MM(Y3.transpose(1, 2, 3, 4, 5, 6, 7, 0), CH, Y3[-1])
    _MatMul = np.matmul
    def MM(A, B, OUT):
        _MatMul(A, B, out = OUT)
    err1 =  TERR2_sig_nb(Sig_t, v)
    err2 =  TERR2_sig(Sig_t, v)
    assert np.allclose(err1,err2)
    print(err1 - err2)
    err1 =  TERR2_psi_nb(Psi_t, v)
    err2 =  TERR2_psi(Psi_t, v)
    assert np.allclose(err1,err2)
    print(err1 - err2)
    err1 =  TERR2_omg_nb(Omg_t, v)
    err2 =  TERR2_omg(Omg_t, v)
    assert np.allclose(err1,err2)
    print(err1 - err2)
    SO = Sig_t[-1].copy()
    step_sig_forward_nb_last_DONTUSE(Sig_t, v)
    res1 = Sig_t[-1].copy()
    Sig_t[-1] = SO
    step_fourth_sig(Sig_t, v)
    res2 = Sig_t[-1].copy()
    assert np.allclose(res1,res2)
    SO = Psi_t[-1].copy()
    step_psi_forward_nb_last_DONTUSE(Psi_t, v)
    res1 = Psi_t[-1].copy()
    Psi_t[-1] = SO
    step_fourth_psi(Psi_t, v)
    res2 = Psi_t[-1].copy()
    assert np.allclose(res1,res2)
    
    
    
    
    