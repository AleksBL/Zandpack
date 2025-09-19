#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 09:17:21 2023

@author: aleksander
"""

import numpy as np
from numba import njit, prange, vectorize, float64, complex128, float32, complex64
from Zandpack.TimedependentTransport import _Q_jit_outer_v2, _Q_make_hermitian
from mpi_funcs import MM
from PadeDecomp import Hu_poles
from Block_matrices.Croy import L as Lorentzian
import k0nfig as config

def Fermi(E, mu, kT):
    return 1/(1+np.exp((E-mu)/kT))
def Fermib(E,mu,kT):
    return 1.0-Fermi(E,mu, kT)

_MatMul = np.matmul
def MM(A, B, OUT):
    _MatMul(A, B, out = OUT)
_multiply = np.multiply

def CZ(s):
    return np.zeros(s, dtype=np.complex128)

@vectorize([float64(complex128),float32(complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

@njit(fastmath = True, parallel=True)
def PI4(psi, xic, Qout):
    """
    EQ.  17
    """
    nk,na,no,nl,nf,_ = psi.shape
    psi_acc        = np.zeros( (na, no), dtype=np.complex128)
    for ik in range(nk):
        for ia in prange(na):
            for io in range(no):
                for iL in range(nl):
                    psi_acc[ia,:] = 0.0
                    for iF in range(nf):
                        psi_acc[ia] += psi[ik,ia,io,iL,iF,:]
                    _Q_jit_outer_v2(psi_acc[ia], xic[ik,ia,io,iL], Qout[ik,ia])
            _Q_make_hermitian(Qout[ik,ia])

def get_scalars(NK, NA, NO, NL, NF, 
                mu_i, kT_i, L_poles,
                Gl_eig):
    """
    Get the scalar quantitiess for the propagation scheme.
    See text elsewhere
    """
    
    Xpp    = CZ((NK,NA,NO,NL,NF+1))
    Xpm    = CZ((NK,NA,NO,NL,NF+1))
    GGP    = CZ((NK,NA,NO,NL,NF+1))
    GGM    = CZ((NK,NA,NO,NL,NF+1))
    GLP    = CZ((NK,NA,NO,NL,NF+1))
    GLM    = CZ((NK,NA,NO,NL,NF+1))

    _zi,_R = Hu_poles(NF)
    for ik in range(NK):
        for ia in range(NA):
            mu = mu_i[ia]
            kT = kT_i[ia]
            for io in range(NO):
                for iL in range(NL):
                    for iF in range(NF+1):
                        Lp     = L_poles[ik,ia,iL]
                        W_al   = Lp.imag
                        E_al   = Lp.real
                        eigval = Gl_eig[ik,ia,iL,io]
                        
                        if iF == 0:
                            pole = Lp
                        else:
                            pole = mu + _zi[iF-1]*kT
                        
                        if iF == 0:
                            Xpp[ik,ia,io, iL, iF] = pole
                            Xpm[ik,ia,io, iL, iF] = pole.conj()
                            GGP[ik,ia, io, iL, iF] = -1j/2*eigval*W_al* Fermib(pole,        mu, kT)
                            GGM[ik,ia, io, iL, iF] = -1j/2*eigval*W_al* Fermib(pole.conj(), mu, kT)
                            GLP[ik,ia, io, iL, iF] = +1j/2*eigval*W_al* Fermi(pole,        mu, kT)
                            GLM[ik,ia, io, iL, iF] = +1j/2*eigval*W_al* Fermi(pole.conj(), mu, kT)                    
                        else:
                            Xpp[ik,ia, io, iL, iF] = pole
                            Xpm[ik,ia, io, iL, iF] = pole.conj()
                        
                            GGP[ik,ia, io, iL, iF] = + _R[iF-1] * kT*eigval*Lorentzian(pole,        W_al, E_al)
                            GGM[ik,ia, io, iL, iF] = - _R[iF-1] * kT*eigval*Lorentzian(pole.conj(), W_al, E_al)
                            GLP[ik,ia, io, iL, iF] = + _R[iF-1] * kT*eigval*Lorentzian(pole,        W_al, E_al)
                            GLM[ik,ia, io, iL, iF] = - _R[iF-1] * kT*eigval*Lorentzian(pole.conj(), W_al, E_al)
    
    del ik,ia,io,iL,iF                    
    assert (Xpp.imag>0).all()
    assert (Xpm.imag<0).all()
    
    return Xpp, Xpm, GGP, GGM, GLP, GLM

#### NEEDS ADAPTATION
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

@njit(fastmath = True, parallel=config.NUMBA_PARALLEL)
def add_omg2_2_omg1(omg1, omg2):
    nl = omg1.shape[3]
    for iL1 in range(nl):
        sub1 = omg2[:,:,:,iL1]
        for iF2 in range(omg2.shape[-1]):
            omg1[:,:,:,nl,0] += sub1[..., iF2]
    
    
    
def TERR2_sig(y1, CT):
    return abs2(y1.transpose(1, 2, 3,        0)@CT).sum()
def TERR2_psi(y2, CT):
    return abs2(y2.transpose(1, 2, 3, 4, 5,    0)@CT).sum()
def TERR2_omg(y3, CT):
    return abs2(y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT).sum()

def step_fourth_sig(Y1, CH):
    MM(Y1.transpose(1, 2, 3,             0), CH, Y1[-1])
def step_fourth_psi(Y2, CH):
    MM(Y2.transpose(1, 2, 3, 4, 5, 6,       0), CH, Y2[-1])
def step_fourth_omg(Y3, CH):
    MM(Y3.transpose(1, 2, 3, 4, 5, 6, 7, 8, 9, 0), CH, Y3[-1])


















