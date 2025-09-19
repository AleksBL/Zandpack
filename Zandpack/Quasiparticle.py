#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:48:43 2022

@author: aleksander

SteadyState: Simple calculator object for calculation of steadystate properties

Berry Phase functions in the bottom are for calculating berry phases.

"""
import numpy as np
from numba import njit
from scipy.integrate import quad_vec,quad
from scipy.optimize import minimize
from time import time



class SteadyState:
    def __init__(self, Ham, SEs):
        """
            Ham: Hamiltonian, square np.ndarray
            SEs: list of Lorentzian SE objects or a list of anythin that has 
                 a an evaluate method that returns something that can be added
                 to Ham
            only works for one k-point, so this should be picked beforehand
        """
        self.Ham = Ham.copy()
        self.SEs = SEs
        self._no = Ham.shape[-1]
    
    def solve_QP_equation(self,
                          E_guess,
                          its, 
                          Biases,
                          dHam,
                          Hermitian = True,
                          conv_tol = 1e-5,
                          ):
        
        """
           E_guess: float, starting energy 
           its    : int,   number of iterations to make
           eta    : add broadening to the self energy evaluation
           Hermititan: use or dont use np.linalg.eigh
           biases: defaults to 0.0 for all electrodes, but can be specified as a list/np.array
        """
        out1 = np.zeros(its,dtype = np.complex128)
        out2 = np.zeros(its,dtype = np.complex128)
        E_eval = E_guess + 0.0
        if Hermitian: Eig = np.linalg.eigh
        else:         Eig = np.linalg.eig
        
        for count in range(its):
            SE  = sum([self_energy.evaluate(np.array([E_eval]), bias = Biases[i])[0,0] 
                       for i,self_energy in enumerate(self.SEs) ])
            e,v = Eig(self.Ham+dHam + (SE + SE.T.conj())/2)
            iSE = (SE - SE.T.conj())/2
            
            dE     = np.abs(e - E_eval)
            idx    = np.where(dE == dE.min())[0][0]
            E_eval = e[idx]
            vec    = v[:,idx]
            out1[count] = E_eval
            out2[count] = (vec.conj()).dot(iSE).dot(vec)
            if abs(out1[count] - out1[count-1])<conv_tol:
                break
        out1 = out1[0:count+1]
        out2 = out2[0:count+1]
        return E_eval, vec, out1, out2
    
    def transmission_and_dos(self, Egrid, eta, Biases, dH):
        SE   = [self_energy.evaluate(np.array(Egrid), bias = Biases[i])[0] 
                for i,self_energy in enumerate(self.SEs) ]
        Nlead= len(self.SEs)
        I    =  np.identity(self._no)
        idx  =  np.arange(self._no)
        HamV = self.Ham + dH
        assert len(self.Ham.shape) == 2
        assert len(dH.shape) == 2
        
        G    =  np.linalg.inv((Egrid+1j*eta)[:,None,None]*I - HamV - sum(SE))
        Gams =  [1j*(se - se.conj().transpose(0,2,1)) for se in SE]
        Tij  =  np.zeros((Nlead, Nlead, len(Egrid)),dtype = np.complex128)
        DOS  = -np.imag(G[:,idx,idx]).sum(axis = 1)/np.pi
        M1   = np.zeros(G.shape,dtype = np.complex128)
        M2   = np.zeros(G.shape,dtype = np.complex128)
        for i in range(Nlead):
            for j in range(Nlead):
                np.matmul(Gams[i], G,                         out = M1)
                np.matmul(Gams[j], G.conj().transpose(0,2,1), out = M2)
                Tij[i,j,:] = (M1 * (M2.transpose(0,2,1))).sum(axis=(1,2))
        
        return Tij, DOS
    
    def eq_density_matrix(self, eta, dH = None, kT = 0.025, 
                          mu=0.0, 
                          dE_bottom = -40, 
                          dE_top = 1.0, n_workers = 1,
                          epsabs = 1e-4,
                          epsrel = 1e-3
                          ):
        I = np.identity(self._no)
        def G(E):
            SE  = sum([self_energy.evaluate(np.array([E]), bias = mu)[0,0] 
                       for i,self_energy in enumerate(self.SEs) ])
            if dH is None:
                iG = E * I - (self.Ham     ) - SE
            else:
                iG = E * I - (self.Ham + dH) - SE
                
            return np.linalg.inv(iG)
        
        global _glob_f
        def _glob_f(E):
            fd = 1/(1+np.exp((E-mu)/kT))
            Gf = G(E)
            return (fd*.5j/np.pi) * (Gf - Gf.conj().T)
        
        M1,e1 = quad_vec(_glob_f, -np.inf,         mu + dE_bottom, workers = n_workers, epsrel=epsrel, epsabs = epsabs)
        print('Got One')
        
        M2,e2 = quad_vec(_glob_f,  mu + dE_bottom, mu + dE_top   , workers = n_workers, epsrel=epsrel, epsabs = epsabs)
        print('Got One')
        
        M3,e3 = quad_vec(_glob_f,  mu + dE_top,    np.inf        , workers = n_workers, epsrel=epsrel, epsabs = epsabs)
        print('Got One')
        
        del _glob_f
        return M1+M2+M3,e1+e2+e3
    
    def get_shift(self, N_target, dH = None, 
                  eta = 0.001, 
                  kT = 0.025, 
                  mu = 0.0,
                  dE_bottom = -40,
                  dE_top = 1,
                  n_workers = 1,
                  method = 'nelder-mead',
                  x0 = 0.0,
                  epsabs = 1e-4,
                  epsrel = 1e-3
                  ):
        I = np.identity(self._no)
        if dH is None:
            def Ham(shift):
                return self.Ham + I*shift
        else:
            def Ham(shift):
                return self.Ham + dH + I*shift
        def error(shift):
            DM, error = self.eq_density_matrix(eta, dH,kT,mu,dE_bottom,dE_top, 
                                               n_workers, epsrel =epsrel, epsabs=epsabs)
            print('error:', error)
            return abs(np.trace(DM) - N_target)
        return minimize(error, x0, method =method)
    
    def Current(self, eta, dH = None,
                          mu_i = [.0, .0, ],
                          kT_i = [.025, .025],
                          dE_bottom = -40, 
                          dE_top = 40.0, 
                          epsabs = 1e-8,
                          epsrel = 1e-10
                          ):
        if dH is None:
            dH = np.zeros((self._no, self._no))
        I = np.eye(self._no)
        
        def Gr(E):
            SE  = sum([self_energy.evaluate(np.array([E]), bias = mu_i[i])[0,0] 
                       for i,self_energy in enumerate(self.SEs) ])
            return np.linalg.inv(E * I - (self.Ham + dH) - SE)
        
        def Sig_l(E,lead):
            fd = 1/(1+np.exp((E-mu_i[lead])/kT_i[lead]))
            Gam = self.SEs[lead].evaluate_gamma(np.array([E]), bias = mu_i[lead])[0,0]
            return 1j * fd * Gam
        def Sig_g(E,lead):
            fd = 1/(1+np.exp((E-mu_i[lead])/kT_i[lead]))
            Gam = self.SEs[lead].evaluate_gamma(np.array([E]), bias = mu_i[lead])[0,0]
            return -1j *(1- fd) * Gam
        def G_l(E):
            SE_l_tot = sum([Sig_l(E, i) for i in range(len(self.SEs))])
            gr = Gr(E)
            return gr@SE_l_tot@(gr.conj().T)
        
        def G_g(E):
            SE_g_tot = sum([Sig_g(E, i) for i in range(len(self.SEs))])
            gr = Gr(E)
            return gr@SE_g_tot@(gr.conj().T)
        
        def _J(E, lead):
            Gg = G_g(E)
            Gl = G_l(E)
            Sg = Sig_g(E, lead)
            Sl = Sig_l(E, lead)
            return np.sum(Gg*(Sl.T)) - np.sum(Gl*(Sg.T))
        
        J = np.zeros(len(self.SEs),dtype=complex)
        E = np.zeros(len(self.SEs))
        for i in range(len(self.SEs)):
            def f(E):
                return _J(E, i)
            r1,e1 = quad(f, -np.inf, mu_i[i] + dE_bottom           , epsrel=epsrel, epsabs = epsabs)
            r2,e2 = quad(f,  mu_i[i] + dE_bottom, mu_i[i] + dE_top , epsrel=epsrel, epsabs = epsabs)
            r3,e3 = quad(f,  mu_i[i] + dE_top  , np.inf            , epsrel=epsrel, epsabs = epsabs)
            J[i] = r1+r2+r3
            E[i] = e1+e2+e3
        
        return J,E
    
    #def get_pivot(self):
    #    
    
    
    
    def Glesser(self, eta, dH = None,
                mu_i = [.0, .0, ],
                kT_i = [.025, .025],
                dE_bottom = -40, 
                dE_top = 40.0, 
                epsabs = 1e-6,
                epsrel = 1e-5,
                N_poles= None,
                use_FP_theorem = True,
                n_workers      =   4):
        
        from Zandpack.PadeDecomp import Hu_poles, FD_expanded, FD
        if N_poles is not None:
            zi,R  = Hu_poles(N_poles) 
        
        if N_poles is None:
            def fdf(E,mu, kT):
                return 1/(1+np.exp((E - mu)/kT))
        else:
            def fdf(E,mu,kT):
                _E = np.array([E])
                return FD_expanded(_E, zi, 1/kT , mu = mu, coeffs = R)
        
        if dH is None:
            dH = np.zeros((self._no, self._no))
        I = np.eye(self._no)
        
        def Gr(E):
            SE  = sum([self_energy.evaluate(np.array([E]), bias = mu_i[i])[0,0] 
                       for i,self_energy in enumerate(self.SEs) ])
            return np.linalg.inv(E * I - (self.Ham + dH) - SE)
        
        def Sig_l(E,lead):
            fd = fdf(E, mu_i[lead], kT_i[lead])
            Gam = self.SEs[lead].evaluate_gamma(np.array([E]), bias = mu_i[lead])[0,0]
            return 1j * fd * Gam
        
        def G_l(E):
            SE_l_tot = sum([Sig_l(E, i) for i in range(len(self.SEs))])
            gr = Gr(E)
            return 1j*gr@SE_l_tot@(gr.conj().T)/(2*np.pi)
        
        def DM(E):
            fd = fdf(E, mu_i[0], kT_i[0])
            gr = Gr(E)
            return 1j/(2*np.pi) * fd * (gr  - gr.conj().T)
        
        global _dont_use_global_f
        if use_FP_theorem:
            def _dont_use_global_f(E):
                return DM(E)
        else:
            def _dont_use_global_f(E):
                return G_l(E)
        
        t1 = time()
        r1,e1 = quad_vec(_dont_use_global_f, -np.inf, mu_i[0] + dE_bottom           , epsrel=epsrel, epsabs = epsabs, workers = n_workers)
        t2 = time()
        r2,e2 = quad_vec(_dont_use_global_f,  mu_i[0] + dE_bottom, mu_i[0] + dE_top , epsrel=epsrel, epsabs = epsabs, workers = n_workers)
        t3 = time()
        r3,e3 = quad_vec(_dont_use_global_f,  mu_i[0] + dE_top  , np.inf            , epsrel=epsrel, epsabs = epsabs, workers = n_workers)
        t4 = time()
        print('NEGF DM calculation:')
        print('Lower: ', t2-t1)
        print('Middle: ', t3-t2)
        print('Upper: ', t4-t3)
        
        del _dont_use_global_f
        
        return r1+r2+r3,e1+e2+e3
    

    
    
            
            
    
    

@njit
def BerryPhase_1state(v):
    # v: (ns, N)
    # Resta 2000 paper
    # returns: 1: Berry phase, 
    #          2: individual overlaps over the loop
    #          3: cumulative phase over the loop
    ns   = v.shape[0]
    dots = np.zeros((ns), dtype = np.complex128)
    for i in range(ns-1):
        dots[i] = np.conj(v[i]).dot(v[i+1])
    dots[ns-1] = np.conj(v[ns-1]).dot(v[0])
    P = np.prod(dots)
    cumulative = np.zeros((ns), dtype = np.complex128)
    for i in range(ns):
        cumulative[i] = np.prod(dots[:i])
    
    return -np.log(P).imag, dots, -np.log(cumulative).imag


@njit
def BerryPhase_Nstate(V):
    v = V
    #v : (ns, N, N) array
    # see https://www.physics.rutgers.edu/pythtb/formalism.html
    ns,N = v[:,:,0].shape
    dots = np.zeros((ns), dtype = np.complex128)
    M  = np.zeros((N,N),dtype = np.complex128)
    Mv = np.zeros((ns, N, N), dtype = np.complex128)
    C  = np.zeros((ns, N, N), dtype = np.complex128)
    
    for i in range(ns):
        if i<ns-1:
            for I in range(N):
                for J in range(N):
                    M[I,J] = np.conj(v[i,I,:]).dot(v[i+1,J,:])
        else:
            for I in range(N):
                for J in range(N):
                    M[I,J] = np.conj(v[i,I,:]).dot(v[0,J,:])
        Mv[i,:,:] = M[:,:]
        dots[i] = np.linalg.det(M)
    # C[0] = Mv[0]
    # for i in range(1,ns):
    #     C[i] = np.eye(N)
    #     for j in range(i):
    #         np.dot(C[i], Mv[j], out=C[i])
    P = np.prod(dots)
    return -np.log(P).imag, Mv




