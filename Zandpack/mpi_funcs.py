#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:54:51 2022

@author: aleksander
"""

import numpy as np
import numba
import k0nfig as config

njit   = numba.njit
prange = numba.prange

assert config.NUMBA

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

# Generic operations 
_MatMul = np.matmul
def MM(A, B, OUT):
    _MatMul(A, B, out = OUT)

_multiply = np.multiply
#

def step_fourth(Y1, Y2, Y3, CH):
    MM(Y1.transpose(1, 2, 3,             0), CH, Y1[-1])
    MM(Y2.transpose(1, 2, 3, 4, 5,       0), CH, Y2[-1])
    MM(Y3.transpose(1, 2, 3, 4, 5, 6, 7, 0), CH, Y3[-1])

def TERR(y1, y2, y3, CT):
    M1 = y1.transpose(1, 2, 3,        0)@CT
    M2 = y2.transpose(1, 2, 3, 4, 5,    0)@CT
    M3 = y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT
    res = (abs2(M1).sum() + abs2(M2).sum()+abs2(M3).sum())**0.5
    return res

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL)
def OuterSubtraction(A,B,C ,out):
    nk,na,nx,noT = A.shape
    NA,NX,NO = B.shape[1:4]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            for cc in range(NO):
                                out[k,a,x,c,aa,xx,cc] += (A[k,a,x,c] - B[k,aa,xx,cc]) * C[k,a,x,c,aa,xx,cc]

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL)
def OuterSubtraction_and_hmult(A,B,C,out,h):
    nk,na,nx,noT = A.shape
    NA,NX,NO = B.shape[1:4]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            for cc in range(NO):
                                out[k,a,x,c,aa,xx,cc] += (A[k,a,x,c] - B[k,aa,xx,cc]) * C[k,a,x,c,aa,xx,cc]
                                out[k,a,x,c,aa,xx,cc] *= h

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL)
def OuterSubtraction_and_hmult_hard_opti(A,B,C,out,h):
    nk,na,nx,noT = A.shape
    NA,NX,NO = B.shape[1:4]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            val = A[k,a,x,0] - B[k,aa,xx,0]
                            for cc in range(NO):
                                out[k,a,x,c,aa,xx,cc] += val * C[k,a,x,c,aa,xx,cc]
                                out[k,a,x,c,aa,xx,cc] *= h

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL, fastmath=config.FASTMATH)
def OuterSubtraction_hard_opti(A,B,C,out):
    nk,na,nx,noT = A.shape
    NA,NX,NO     = B.shape[1:4]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                outer_kax = (A[k,a,x,0] - B[k,:,:,0])
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            out[k,a,x,c,aa,xx,:]+= outer_kax[aa,xx]*C[k,a,x,c,aa,xx,:]

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL, fastmath=config.FASTMATH)
def OuterSubtraction_hard_opti_V2(A,B,C,out):
    nk,na,nx,noT = A.shape
    NA,NX,NO     = B.shape[1:4]
    NO2          = C.shape[6]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                outer_kax = (A[k,a,x,0] - B[k,:,:,0])
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            tmpval = outer_kax[aa,xx]
                            for cc in range(NO2):
                                out[k,a,x,c,aa,xx,cc] += tmpval*C[k,a,x,c,aa,xx,cc]

###################

@njit(parallel = config.NUMBA_OUTER_SUBTRACTION_PARALLEL)
def OuterSubtractionAssign(A,B,C ,out):
    nk,na,nx,noT = A.shape
    
    NA,NX,NO = B.shape[1:4]
    for x in prange(nx):
        for k in range(nk):
            for a in range(na):
                for c in range(noT):
                    for aa in range(NA):
                        for xx in range(NX):
                            for cc in range(NO):
                                out[k,a,x,c,aa,xx,cc] = (A[k,a,x,c] - B[k,aa,xx,cc]) * C[k,a,x,c,aa,xx,cc]

###################

# @njit(fastmath=config.FASTMATH)
# def idxaddpsi(psi, val, idx):
#     count = 0
#     for i in idx:
#         psi[:,:,:,:,idx] += val[:,:,:,:, count]
#         count += 1
    

##################

@njit(fastmath =config.FASTMATH)
def hermitian_kaij2ravel(mat, out):
    no      = mat.shape[2]
    assert out.shape[2] == no*(no+1)
    count   = 0
    for i in range(no):
        for j in range(i,no):
            out[:,:,count] = mat[:,:,i,j]
            count+=1

@njit(fastmath =config.FASTMATH)
def add_ravelled_hermitian(ravel, out):
    no     = out.shape[2]
    count  = 0
    assert ravel.shape[2] == no*(no+1)
    for i in range(no):
        out[:,:,i,i] += ravel[:,:,count]
        count += 1
        for j in range(i+1,no):
            out[:,:,i,j] += ravel[:,:,count]
            out[:,:,j,i] += np.conj(ravel[:,:,count])
            count += 1





# for reworked version

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






# @njit
# def opt_MM_for_omg(A, B, nl,nf, out):
#     nk = A.shape[0]
#     n1 = A.shape[1]
#     n2 = B.shape[2]
#     for ik in range(nk):
#         for i in range(n1):
#             for j in range(n2):
#                 out[ik,i,j] = np.dot(A[ik,i], B[ik,j])
    




# @njit(parallel = True)
# def multiply_v2(A,B,C):
#     n1 = len(A)
#     for i in prange(n1):
#         np.multiply( A[i] , B[i], out = C[i])
        # C[i] = A[i] * B[i]
# nax = np.newaxis
# def OuterSubtraction(a,b,c,out):
#     nk = a.shape[0]
#     noT = out.shape[3]
#     for k in range(nk):
#         out[k] += np.subtract.outer(a[k,:,:,:noT], b[k]) * c[k]

# def OuterSubtraction_v3(a,b,c,out):
#     noT = out.shape[3]
#     out += (a[:, :  , :  ,:noT,  nax, nax, nax] - b[:, nax, nax, nax,  :  , :  , :  ])*c

# def CR(s):
#     return np.random.random(s).astype(np.complex128) + 1j *np.random.random(s).astype(np.complex128)

# sig  = CR((7,3,200,200))
# psi  = CR((7,3,2,20,25,200))
# omg  = CR((7,3,2,20,25,2,20,200))
# Xpp  = CR((3,2,20,200))

# def TERR_old(y1, y2, y3, CT):
#     res  = np.sum(np.abs((y1.transpose(1, 2, 3,        0)@CT))**2)
#     res += np.sum(np.abs((y2.transpose(1, 2, 3, 4, 5,    0)@CT))**2)
#     res += np.sum(np.abs((y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT))**2)
#     return np.sqrt(res)
