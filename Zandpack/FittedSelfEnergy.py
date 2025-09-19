#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:00:13 2022

@author: aleksander
"""
import os
import numpy as np
import numba as nb
from Block_matrices.Croy import (evaluate_Lorentz_basis_matrix, 
                                 evaluate_KK_matrix, 
                                 evaluate_Lorentz_basis_matrix_hermitian, 
                                 evaluate_KK_matrix_hermitian)
from Block_matrices.Block_matrices import Blocksparse2Numpy
import sys
sys.path.append(__file__[:-19])
from Loader import flexload

class Lorentzian_SE:
    def __init__(self,Ei, Gi, Coeffs):
        """
           Ei:    Possitions of lorentzians on real axis (np.ndarray, (#nk, #Centres))
           Gi:    Broadenings of lorentzians on real axis (np.ndarray, (#nk, #Centres))
           Coeffs: Fitting Coefficients of the Broadening Matrix.
           
           Serves to give the self-energy from evaulating the 
        """
        self.Coeffs = Coeffs.copy()
        self.Ei     = Ei.copy()
        self.Gi     = Gi.copy()
    
    def evaluate(self, E, bias = 0.0, tol = 1e-14, ik = 0, hermitian_parts=False):
        """E:    Complex (n,) np.ndarray
           bias: float, shifting E
        """
        Ci, Ei, Gi = self.Coeffs    , self.Ei    , self.Gi
        if hermitian_parts:
            im = -0.5j * evaluate_Lorentz_basis_matrix_hermitian(Ci, E-bias, Ei, Gi , tol  = tol)
            re =  0.5  * evaluate_KK_matrix_hermitian(           Ci, E-bias, Ei, Gi , tol  = tol)
        else:
            im = -0.5j * evaluate_Lorentz_basis_matrix(Ci, E-bias, Ei, Gi , tol  = tol)
            re =  0.5  * evaluate_KK_matrix(           Ci, E-bias, Ei, Gi , tol  = tol)
        return re+im
    
    def evaluate_gamma(self,E, bias = 0.0, tol=1e-15, force_hermitian=False):
        if force_hermitian==False:
            return evaluate_Lorentz_basis_matrix(self.Coeffs, E-bias, self.Ei, self.Gi , tol  = tol)
        else:
            return evaluate_Lorentz_basis_matrix_hermitian(self.Coeffs, E-bias, self.Ei, self.Gi , tol  = tol)

def from_blockmatrix(BM, slices):
    """
    BM:     Block matrix with lorentzian flag
    slices: slices for translating block matrix to normal numpy array.
    This can be taken from either the TimedependentTransport class or the system BTD BM
    """
    ei,gi = BM.ei, BM.gamma
    Coeffs= Blocksparse2Numpy(BM, slices)
    return Lorentzian_SE(ei, gi, Coeffs)

@nb.njit(fastmath = True)
def from_eigendecomp(vals, vecs, ivecs, out):
    no = len(vals)
    for io in range(no):
        out += vals[io] * vecs[io,:].reshape(-1, 1) * ivecs[io,:].reshape(1,-1)
    

def from_saved_file(directory, ik = None):
    """
        directory: A saved directory containing the needed quantities for the
        propagation scheme, plus the eigenvalues and eigenvectors of the 
        coefficient matrix. See the code TimedepedentTransport.Writer for 
        the needed quantities.
        
        returns: instances of the class Lorentzian_SE for each lead.
        
    """
    try:
        np.load(directory+'/Superconductor.npy')
        mode = 'direct'
    except:
        mode = 'fromeig'
    
    if mode == 'fromeig':
        Nl        =  np.load(directory + '/num_lorentzians.npy')
        #Nf        =  np.load(directory + '/num_poles_fermi.npy')
        Nlead     =  np.load(directory + '/num_leads.npy'   )
        xi        =  flexload(directory + '/xi.npy')
        Ixi       =  flexload(directory + '/Ixi.npy')
        EigVal_Gl =  np.load(directory + '/_Gl_Eigenvalues.npy')
        #EigVal_Gp =  np.load(directory + '/_Gp_Eigenvalues.npy')
        SEs       =  []
        nk        =  xi.shape[0]
        if ik is None: kidx = np.arange(nk)
        else:          kidx = np.array([ik])
        for i in range(Nlead):
            Ei      =  np.load(directory+'/Centres_Lorentzian_'   +str(i)+'.npy')
            Gi      =  np.load(directory+'/Broadening_Lorentzian_'+str(i)+'.npy')
            eigval  =  EigVal_Gl[i,kidx,:, :]
            eigvec  =  xi [kidx,i,0:Nl, :, :]
            ieigvec =  Ixi[kidx,i,0:Nl, :, :]
            no      =  xi.shape[-1]
            Coeffs  =  np.zeros((len(kidx), Nl, no, no), dtype = np.complex128)
            for jk in range(len(kidx)):
                for jl in range(Nl):
                    from_eigendecomp(eigval [jk,jl], # no eigenvalues 
                                     eigvec [jk,jl], # (no,no) eigenvectors
                                     ieigvec[jk,jl], # (no,no) duals of eigenvectors
                                     Coeffs [jk,jl]) # output here
            SEs    += [Lorentzian_SE(Ei, Gi, Coeffs)]
        return SEs
    elif mode =='direct':
        Nlead     =  np.load(directory + '/num_leads.npy'   )
        SEs       =  []
        for i in range(Nlead):
            Ei      =  np.load(directory+'/Centres_Lorentzian_'   +str(i)+'.npy')
            Gi      =  np.load(directory+'/Broadening_Lorentzian_'+str(i)+'.npy')
            Coeffs  =  np.load(directory+'/Gamma_Coeffs.npz')['arr_0'][i]
            SEs    += [Lorentzian_SE(Ei, Gi, Coeffs)]
        return SEs
    

