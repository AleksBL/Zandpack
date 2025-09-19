#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:30:53 2023

@author: aleksander
"""

from numba import float64, complex128, int32, int8,njit  
from numba.experimental import jitclass
import numpy as np


@njit
def dag(v):     return v.conj().T
@njit
def outer(a,b): return np.outer(a,dag(b))
@njit
def INNER(M1,M2):
    return (M1 * M2.conj()).sum()

spec = [
    ('H', complex128[:,:]), 
    ('eig_e',  float64[:]),
    ('eig_v',  complex128[:,:]),
    ('shape',  int32[:]),
    ('Leig_v', complex128[:,:,:,:]),
    ('Leig_v_calc', int8)
]

@jitclass(spec)
class Louivillian:
    def __init__(self, H):
        self.H = H
        no = H.shape[0]
        self.eig_e, self.eig_v = np.linalg.eigh(H)
        self.shape  = np.array((no,no),dtype=np.int32)
        self.Leig_v = np.zeros((no,no,no,no), dtype=np.complex128)
        self.Leig_v_calc = 0
        self.calculate_Leigv()
    
    def eigv(self,i,j):
        if self.Leig_v_calc == 0:
            o = outer(self.eig_v[:,i], self.eig_v[:,j])
            return o/np.sqrt(INNER(o,o))
        else:
            return self.Leig_v[i,j]

    def calculate_Leigv(self):
        for i in range(self.H.shape[0]):
            for j in range(self.H.shape[1]):
                self.Leig_v[i,j] = self.eigv(i,j)
        self.Leig_v_calc = 1
    
    def proj_span(self,vec, return_proj=1):
        proj = np.zeros(vec.shape, dtype=vec.dtype)
        coef = np.zeros(vec.shape,dtype=np.complex128)
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if i==j:
                    pass
                else:
                    spanvec   = self.eigv(i,j)
                    c         = INNER(vec, spanvec)
                    coef[i,j] = c
                    proj      += c*spanvec
        if return_proj==1:
            return proj
        else:
            return coef
        
    
    def proj_null(self,vec, return_proj):
        proj = np.zeros(vec.shape, dtype=vec.dtype)
        coef = np.zeros(vec.shape,dtype=np.complex128)
        for i in range(self.shape[0]):
            nullvec   = self.eigv(i,i)
            c         = INNER(vec, nullvec)
            proj      += c*nullvec
            coef[i,i] = c
        if return_proj==1:
            return proj
        else:
            return coef
        




















