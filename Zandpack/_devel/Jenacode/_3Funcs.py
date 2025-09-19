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


@njit(fastmath = True, parallel=True)
def PI3(psi, Ixi, Qout):
    """
    EQ.  17
    """
    nk,na,nx,noT,_ = psi.shape
    for ik in range(nk):
        for ia in prange(na):
            for ix in range(nx):
                for io in range(noT):
                    _Q_jit_outer_v2(psi[ik,ia,ix,io], Ixi[ik,ia,ix,io], Qout[ik,ia])
            _Q_make_hermitian(Qout[ik,ia])
            

def MM(a,b,c):
    np.matmul(a,b,out=c)

#s@njit
def get_nonzero_omg(deltaP, deltaM, tol = 1e-14):
    #NUMBA LIST INITIALISED WITH SOME ARBITRARY ARRAYS 
    
    BLOCKS = []; BLOCKS.append(np.zeros((2,5,), dtype=np.complex128))
    INDICES= []; INDICES.append(np.zeros((5,4), dtype=np.int64))
    ######
    
    nk,na,nx,no = deltaP.shape
    count_nz    = np.zeros((na,na), dtype=np.int64)
    ij2i        = np.zeros((na,na),dtype=np.int8)
    i2ij        = np.zeros((na**2,2),dtype=np.int8)
    
    for ia in range(na):
        for ja in range(na):
            for ix in range(nx):
                for io in range(no):
                    ele_i = deltaP[:, ia,ix,io]
                    for jx in range(nx):
                        for jo in range(no):
                            ele_j = deltaM[:,ja, jx, jo]
                            if (np.abs(ele_i)<tol).all() and (np.abs(ele_j)<tol).all():
                                pass
                            else:
                                count_nz[ia, ja] += 1
    bit = 0
    for ia in range(na):
        for ja in range(na):
            BLOCKS.append(np.zeros((nk, count_nz[ia,ja]), dtype=np.complex128))
            INDICES.append(np.zeros((count_nz[ia,ja], 4), dtype=np.int64))
            it = 0
            for ix in range(nx):
                for io in range(no):
                    ele_i = deltaP[:, ia,ix,io]
                    for jx in range(nx):
                        for jo in range(no):
                            ele_j = deltaM[:,ja, jx, jo]
                            if (np.abs(ele_i)<tol).all() and (np.abs(ele_j)<tol).all():
                                pass
                            else:
                                INDICES[bit+1][it] = (ix,io,jx,jo)
                                it += 1
            i2ij[bit] = (ia,ja)
            ij2i[ia,ja]= bit
            bit+=1
    
    return BLOCKS[1:], INDICES[1:], ij2i,i2ij

def ODE_OMG(BLOCKS, INDICES, tmp_psi, xi,Ixi, Xpp_t, Xpm_t, deltaP,
            deltaM, OMGS, i, B_ij2i,
            h):
    na = tmp_psi.shape[1]
    for ia in range(na):
        for ja in range(na):
            I = B_ij2i[ia,ja]
            ODE_OMG_ij(BLOCKS[I], INDICES[I], tmp_psi, xi,Ixi, Xpp_t, Xpm_t, 
                       deltaP, deltaM, OMGS[I][i], h,ia,ja)

@njit(fastmath = True, parallel = True)
def ODE_OMG_ij(omgs, inds, psi, xi,Ixi, Xpp, Xpm, 
               deltaP, deltaM, outblock,h,ia,ja):
    nk = psi.shape[0]
    for ik in range(nk):
        nomg     = omgs.shape[1]
        xi_ja    = Ixi[ik,ja]
        xi_ia    = xi[ik,ia]
        psi_ia   = psi[ik,ia]
        psi_ja   = psi[ik,ja].conj()
        Ix,Io    = inds[:,0:2].T
        Jx,Jo    = inds[:,2:4].T
        dP       = deltaP[ik, ia,:,:]
        dM       = deltaM[ik, ja,:,:]
        for it in prange(nomg):
            outblock[ik,it]  = -1j*( Xpm[ik, ja, Jx[it], Jo[it]]\
                                    -Xpp[ik, ia, Ix[it], Io[it]]\
                                   ) * omgs[ik,it]
            outblock[ik,it] += dM[Jx[it], Jo[it]]*\
                                  np.dot( xi_ja[Jx[it], Jo[it],   :],
                                         psi_ia[Ix[it], Io[it],    :]
                                        )
            outblock[ik,it] += dP[Ix[it], Io[it]]*\
                                  np.dot(psi_ja[Jx[it], Jo[it], :],
                                          xi_ia[Ix[it], Io[it], :] 
                                        )
            outblock[ik,it] *= h
            

@njit(fastmath=True, parallel = True)
def sum_over_omg(psiout, omgs, inds, xi, ia, ja):
    na = psiout.shape[1]
    nomg = omgs.shape[1]
    nk = psiout.shape[0]
    xi_ja = xi[:,ja]
    Ix,Io = inds[:,0:2].T
    Jx,Jo = inds[:,2:4].T
    for ik in range(nk):
        for it in prange(nomg):
            psiout[ik, ia, Ix[it],Io[it],:] \
                    +=\
                   omgs[ik,it] * xi[ik,ja, Jx[it], Jo[it],:]

def ODE_PSI(psi, H,Xpp, sigma, xi, GLP, DeltaP, 
            OMG_blocks, OMG_inds, B_ij2i, out):
    #          k,a,c,l,f, vector   k,a,c,l,f,   vector 
    # inital overwrite
    na   = psi.shape[1]
    
    out[:,:,:,:,:] = GLP[...,None]*xi
    DMxi           = (sigma[:,None,None] @   xi.reshape(xi.shape  + (1,)))[..., 0]
    out[:,:,:,:,:]+= DeltaP[:,:,:,:, None]*DMxi 
    out[:,:,:,:,:]+= (    H[:,None,None] @ (psi.reshape(psi.shape + (1,))))[..., 0]
    out[:,:,:,:,:]-=    Xpp[:,:,:,:, None] *  psi
    
    for ia in range(na):
        for ja in range(na):
            sum_over_omg(out, 
                         OMG_blocks[B_ij2i[ia,ja]], 
                         OMG_inds  [B_ij2i[ia,ja]],
                         xi,ia,ja)



@njit(fastmath = True)
def Av3(OPSI, psi, psic, xi, ixi, xpp, xpm, _deltaM, deltaP, h0, ):
    na,nx,noT,no = psi.shape[0:4]
    compound = na*nx*noT
    X_sum               = xi.reshape(compound,no).T
    I = np.eye(no)
    for a in range(na):
        for x in range(nx):
            ig = I * xpp[a,x,0]-h0
            FRAC =  (1j/(  xpm - xpp[a, x, 0])).reshape(compound,1)
            ARR1 = (FRAC*_deltaM) * ixi.reshape(compound,no)
            ARR2 = FRAC.reshape(compound,1) * psic.reshape(compound,no)
            for c in range(noT):
                SUM  = np.dot(X_sum, np.dot(ARR1, psi[a,x,c]))
                SUM += np.dot(X_sum, np.dot(ARR2, deltaP[a,x,c]*xi[a,x,c]))#* includefrac
                SUM += np.dot(ig, psi[a,x,c])
                OPSI[a,x,c,:]  += SUM
    


def Memoryless_ODE_PSI(psi, H,Xpp, Xpm, sigma, xi,ixi, GLP, DeltaP, DeltaM,
                       out):
    #          k,a,c,l,f, vector   k,a,c,l,f,   vector 
    # inital overwrite
    
    out[:,:,:,:,:] = GLP[...,None]*xi
    DMxi           = (sigma[:,None,None] @   xi.reshape(xi.shape  + (1,)))[..., 0]
    out[:,:,:,:,:]+= DeltaP[:,:,:,:, None]*DMxi 
    out[:,:,:,:,:]+= (    H[:,None,None] @ (psi.reshape(psi.shape + (1,))))[..., 0]
    out[:,:,:,:,:]-=    Xpp[:,:,:,:, None] *  psi
    nk = psi.shape[0]
    psic = np.conj(psi)
    na,nx,noT,no = psi.shape[1:5]
    for ik in range(nk):
        Av3(out[ik], psi[ik], psic[ik], xi[ik], ixi[ik], 
            Xpp[ik], Xpm[ik],DeltaM[ik].reshape(na*nx*noT,1),DeltaP[ik], H[ik])
        
    
    
    
    





def TERR2_sig(y1, CT):
    return abs2(y1.transpose(1, 2, 3, 0)@CT).sum()
def TERR2_psi(y2, CT):
    return abs2(y2.transpose(1, 2, 3, 4, 5, 0)@CT).sum()
def TERR2_omg(y3, CT):
    return abs2(y3.transpose(1,2,0)@CT).sum()
def step_fourth_sig(Y1, CH):
    MM(Y1.transpose(1, 2, 3,       0), CH, Y1[-1])
def step_fourth_psi(Y2, CH):
    MM(Y2.transpose(1, 2, 3, 4, 5, 0), CH, Y2[-1])
def step_fourth_omg(Y3, CH):
    MM(Y3.transpose(1, 2,          0), CH, Y3[-1])


    














