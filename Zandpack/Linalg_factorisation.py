#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:02:44 2023

@author: aleksander
"""

import numpy as np
from scipy import linalg as LA
from itertools import product


def SVD(M, hermitian=False, equality_check = True):
    """
    Returns svd decomposition in format
    such that
    for i in range(num_sing): 
    M[...,:,:] += (U[...,i,:,None]@V[...,i,None,:]) * s[...,i][...,None,None]
    """
    U, s, V = np.linalg.svd(M, hermitian = hermitian)
    perm = np.arange(len(U.shape))
    perm[-2] = len(perm)-1
    perm[-1] = len(perm)-2
    U = U.transpose(perm).copy()
    if equality_check:
        acc = np.zeros(M.shape,dtype=complex)
        for i in range(s.shape[-1]):
            acc[...,:,:] += (U[...,i,:,None]@V[...,i,None,:]) * s[...,i][...,None,None]
        assert np.allclose(acc,M)
    return U,s,V

def TAKAGI(M, hermitian = False, zero_tol = 1e-14, equality_check = True):
    """
    Paper:

    Singular value decomposition for the Takagi factorization
    of symmetric matrices
    Alexander M. Chebotarev a,⇑,1, Alexander E. Teretenkov


    Computes Takagi factorisation on giving a 
    matrix on the form A = UDU^T
    """
    
    U,s,V = np.linalg.svd(M, hermitian=hermitian)
    
    perm = np.arange(len(M.shape))
    perm[-2] = len(perm)-1
    perm[-1] = len(perm)-2
    def dag(M):
        return np.conj(M.transpose(perm))
    
    Z      = dag(U)@(dag(V).conj())
    rng = [range(idx) for idx in M.shape[0:-2]]
    
    if len(rng)==1: ITER = product(rng[0])
    if len(rng)==2: ITER = product(rng[0],rng[1] )
    if len(rng)==3: ITER = product(rng[0],rng[1], rng[2])
    if len(rng)==4: ITER = product(rng[0],rng[1], rng[2], rng[3])
    if len(rng)==5: ITER = product(rng[0],rng[1], rng[2], rng[3], rng[4])
    
    matpow = LA.fractional_matrix_power
    Uz     = np.zeros(M.shape,dtype=complex)
    for ij in ITER:
        zi = Z[ij]
        ui = U[ij]
        uzi= Uz[ij]
        uzi[:,:] = ui@(matpow(zi, 0.5))
    
    vecs = dag(Uz).conj().copy()
    lambs= s.copy()
    if equality_check:
        acc = np.zeros(M.shape,dtype=complex)
        for i in range(s.shape[-1]):
            acc[...,:,:] += (vecs[...,i,:,None]@vecs[...,i,None,:]) * lambs[...,i][...,None,None]
        assert np.allclose(acc,M)
    return vecs, lambs, vecs.copy()


def LDL(M, hermitian = False, equality_check= True):
    perm = np.arange(len(M.shape))
    perm[-2] = len(perm)-1
    perm[-1] = len(perm)-2
    def dag(M):
        return np.conj(M.transpose(perm))
    
    rng = [range(idx) for idx in M.shape[0:-2]]
    if len(rng)==1: ITER = product(rng[0])
    if len(rng)==2: ITER = product(rng[0],rng[1] )
    if len(rng)==3: ITER = product(rng[0],rng[1], rng[2])
    if len(rng)==4: ITER = product(rng[0],rng[1], rng[2], rng[3])
    if len(rng)==5: ITER = product(rng[0],rng[1], rng[2], rng[3], rng[4])
    L     = np.zeros(M.shape,dtype=complex)
    d     = np.zeros(M.shape[:-1],dtype=complex)
    for ij in ITER:
        _l, _d, _ = LA.ldl(M[ij], hermitian = hermitian, lower=True)
        L[ij] = _l
        d[ij] = np.diag(_d)
        assert np.allclose(np.diag(np.diag(_d)),_d)
    
    vecl = dag(L).conj()
    vecr = vecl.conj()
    
    # np.allclose(L@d@(dag(L).conj()), M)
    if equality_check:
        acc = np.zeros(M.shape,dtype=complex)
        for i in range(d.shape[-1]):
            acc[...,:,:] += (vecl[...,i,:,None]@vecr[...,i,None,:]) * d[...,i][...,None,None]
        assert np.allclose(acc,M)
    return vecl, d, vecr

def QR(M, hermitian = False, equality_check= True):
    perm = np.arange(len(M.shape))
    perm[-2] = len(perm)-1
    perm[-1] = len(perm)-2
    def dag(M):
        return np.conj(M.transpose(perm))
    
    rng = [range(idx) for idx in M.shape[0:-2]]
    if len(rng)==1: ITER = product(rng[0])
    if len(rng)==2: ITER = product(rng[0],rng[1] )
    if len(rng)==3: ITER = product(rng[0],rng[1], rng[2])
    if len(rng)==4: ITER = product(rng[0],rng[1], rng[2], rng[3])
    if len(rng)==5: ITER = product(rng[0],rng[1], rng[2], rng[3], rng[4])
    Q     = np.zeros(M.shape,dtype=complex)
    R     = np.zeros(M.shape,dtype=complex)
    lambs = np.zeros(M.shape[:-1],dtype=complex)
    for ij in ITER:
        q,r = LA.qr(M[ij])
        Q[ij][:,:]   = q
        R[ij][:,:]   = r
        lambs[ij][:] =1.0
    vecl = dag(Q).conj()
    vecr = R
    
    if equality_check:
        acc = np.zeros(M.shape,dtype=complex)
        for i in range(lambs.shape[-1]):
            acc[...,:,:] += (vecl[...,i,:,None]@vecr[...,i,None,:]) * lambs[...,i][...,None,None]
        assert np.allclose(acc,M)
    return vecl, lambs, vecr

def EIG(M, Hermitian = False, equality_check=True):
    perm = np.arange(len(M.shape))
    perm[-2] = len(perm)-1
    perm[-1] = len(perm)-2
    rng = [range(idx) for idx in M.shape[0:-2]]
    if len(rng)==1: ITER = product(rng[0])
    if len(rng)==2: ITER = product(rng[0],rng[1] )
    if len(rng)==3: ITER = product(rng[0],rng[1], rng[2])
    if len(rng)==4: ITER = product(rng[0],rng[1], rng[2], rng[3])
    if len(rng)==5: ITER = product(rng[0],rng[1], rng[2], rng[3], rng[4])
    
    def dag(M):
        return np.conj(M.transpose(perm))
    s, _vecf1    = np.linalg.eig(M)
    no           = M.shape[-1]
    _ivecf1      = np.linalg.inv(_vecf1)
    _vecf1       = dag(_vecf1).conj()
    
    vecf1  = _ivecf1#dag(_ivecf1)#.conj()
    ivecf1 = _vecf1#dag(vecf1)#.conj()
    
    
    if equality_check:
        acc = np.zeros(M.shape,dtype=complex)
        for i in range(s.shape[-1]):
            acc[...,:,:] += (vecf1[...,i,:,None]@ivecf1[...,i,None,:]) * s[...,i][...,None,None]
        assert np.allclose(acc,M)
    return vecf1, s, ivecf1

M  = np.random.random((20,50,50))  + 1j * np.random.random((20,50,50))
M += M.transpose(0,2,1)
# TAKAGI(M)
#LDL(M)
# QR(M)
EIG(M)


        
    
    
