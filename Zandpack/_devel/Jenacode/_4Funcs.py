#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:10:56 2023

@author: aleksander
"""
import numpy as np
from numba import njit,vectorize,prange
import numba
from Zandpack.TimedependentTransport import _Q_jit_outer_v2, _Q_make_hermitian
from Block_matrices.Croy import L as Lorentzian
from numba.typed import List

@vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2

#@njit
def Fermi(E, mu, kT):
    return 1/(1+np.exp((E-mu)/kT))

#@njit
def Fermib(E,mu,kT):
    return 1.0-Fermi(E,mu, kT)

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

def MM(a,b,c):
    np.matmul(a,b,out=c)

@njit
def get_nonzero_omg(deltaP, deltaM, tol = 1e-14):
    #NUMBA LIST INITIALISED WITH SOME ARBITRARY ARRAYS 
    
    BLOCKS = []; BLOCKS.append(np.zeros((2,5,), dtype=np.complex128))
    INDICES= []; INDICES.append(np.zeros((5,6), dtype=np.int64))
    ######
    
    nk,na,no,nl,nf = deltaP.shape
    count_nz = np.zeros((na,na), dtype=np.int64)
    ij2i      = np.zeros((na,na),dtype=np.int8)
    i2ij      = np.zeros((na**2,2),dtype=np.int8)
    
    for ia in range(na):
        for ja in range(na):
            for io in range(no):
                for iL in range(nl):
                    for iF in range(nf):
                        ele_i = deltaP[:, ia,io,iL,iF]
                        for jo in range(no):
                            for jL in range(nl):
                                for jF in range(nf):
                                    ele_j = deltaM[:,ja, jo, jL, jF]
                                    if (np.abs(ele_i)<tol).all() and (np.abs(ele_j)<tol).all():
                                        pass
                                    else:
                                        count_nz[ia, ja] += 1
    bit = 0
    for ia in range(na):
        for ja in range(na):
            BLOCKS.append(np.zeros((nk, count_nz[ia,ja]), dtype=np.complex128))
            INDICES.append(np.zeros((count_nz[ia,ja], 6), dtype=np.int64))
            it = 0
            for io in range(no):
                for iL in range(nl):
                    for iF in range(nf):
                        ele_i = deltaP[:, ia,io,iL,iF]
                        for jo in range(no):
                            for jL in range(nl):
                                for jF in range(nf):
                                    ele_j = deltaM[:,ja, jo, jL, jF]
                                    if (np.abs(ele_i)<tol).all() and (np.abs(ele_j)<tol).all():
                                        pass
                                    else:
                                        INDICES[bit+1][it] = (io,iL,iF,jo,jL,jF)
                                        it += 1
            i2ij[bit] = (ia,ja)
            ij2i[ia,ja]= bit
            bit+=1
    
    
    return BLOCKS[1:], INDICES[1:], ij2i,i2ij

def ODE_OMG(BLOCKS, INDICES, tmp_psi, xi, Xpp_t, Xpm_t, deltaP,
            deltaM, OMGS, i, B_ij2i,
            h):
    na = tmp_psi.shape[1]
    for ia in range(na):
        for ja in range(na):
            I = B_ij2i[ia,ja]
            ODE_OMG_ij(BLOCKS[I], INDICES[I], tmp_psi, xi, Xpp_t, Xpm_t, 
                       deltaP, deltaM, OMGS[I][i], h,ia,ja)

@njit(fastmath = True, parallel = True)
def ODE_OMG_ij(omgs, inds, psi, xi, Xpp, Xpm, 
               deltaP, deltaM, outblock,h,ia,ja):
    nk = psi.shape[0]
    na = psi.shape[1]
    for ik in range(nk):
        nomg     = omgs.shape[1]
        xi_ja    = xi[ik,ja].conj()
        xi_ia    = xi[ik,ia]
        psi_ia   = psi[ik,ia]
        psi_ja   = psi[ik,ja].conj()
        Io,IL,IF = inds[:,0:3].T
        Jo,JL,JF = inds[:,3:6].T
        dP       = deltaP[ik, ia,:,:,:]
        dM       = deltaM[ik, ja,:,:,:]
        for it in prange(nomg):
            outblock[ik,it]  = -1j*( Xpm[ik, ja, Jo[it], JL[it], JF[it]]\
                                     -Xpp[ik, ia, Io[it], IL[it], IF[it]]\
                                   ) * omgs[ik,it]
            outblock[ik,it] += dM[Jo[it], JL[it], JF[it]]*\
                                  np.dot( xi_ja[Jo[it], JL[it],         :],
                                        psi_ia[Io[it], IL[it], IF[it], :]
                                        )
            outblock[ik,it] += dP[Io[it], IL[it], IF[it]]*\
                                  np.dot(psi_ja[Jo[it], JL[it], JF[it], :],
                                          xi_ia[Io[it], IL[it],         :] 
                                        )
            outblock[ik,it] *= h


@njit(fastmath=True, parallel = True)
def sum_over_omg(psiout, omgs, inds, xi, ia, ja):
    na = psiout.shape[1]
    nomg = omgs.shape[1]
    nk = psiout.shape[0]
    xi_ja = xi[:,ja]
    Io,IL,IF = inds[:,0:3].T
    Jo,JL,JF = inds[:,3:6].T
    for ik in range(nk):
        for it in prange(nomg):
            psiout[ik, ia, Io[it],IL[it],IF[it],:] \
                    +=\
                   omgs[ik,it] * xi[ik,ja, Jo[it], JL[it],:]





def ODE_PSI(psi, H,Xpp, sigma, xi, GLP, DeltaP, 
            OMG_blocks, OMG_inds, B_ij2i, out):
    #          k,a,c,l,f, vector   k,a,c,l,f,   vector 
    # inital overwrite
    na   = psi.shape[1]
    
    out[:,:,:,:,:] = GLP[...,None]*xi[...,None,:]
    DMxi           = (sigma[:,None,None,None] @   xi.reshape(xi.shape  + (1,)))[..., 0]
    out[:,:,:,:,:]+= DeltaP[:,:,:,:,:,None]*DMxi[...,None,:] 
    out[:,:,:,:,:]+= (    H[:,None,None,None] @ (psi.reshape(psi.shape + (1,))))[..., 0]
    out[:,:,:,:,:]-=    Xpp[:,:,:,:, :, None] *  psi
    
    for ia in range(na):
        for ja in range(na):
            sum_over_omg(out, 
                         OMG_blocks[B_ij2i[ia,ja]], 
                         OMG_inds  [B_ij2i[ia,ja]],
                         xi,ia,ja)
            
            
            
            
            
            


    

def TERR2_sig(y1, CT):
    return abs2(y1.transpose(1, 2, 3, 0)@CT).sum()
def TERR2_psi(y2, CT):
    return abs2(y2.transpose(1, 2, 3, 4, 5, 6, 0)@CT).sum()
def TERR2_omg(y3, CT):
    return abs2(y3.transpose(1,2,0)@CT).sum()

def step_fourth_sig(Y1, CH):
    MM(Y1.transpose(1, 2, 3,             0), CH, Y1[-1])
def step_fourth_psi(Y2, CH):
    MM(Y2.transpose(1, 2, 3, 4, 5, 6,    0), CH, Y2[-1])
def step_fourth_omg(Y3, CH):
    MM(Y3.transpose(1, 2, 0),                CH, Y3[-1])


#def TERR2_omg(y3, CT):
#    return abs2(y3.transpose(1, 2, 3, 4, 5, 6, 7, 0)@CT).sum()
  

    














