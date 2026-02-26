import numpy as np
from numba import njit
# The functions found here are either from the Croy 2009 paper
# or the Jie Hu paper from 2011. Read these to understand these functions.
# They simply return the poles and weights of an expansion of the fermi function 
# in terms of simple poles

def Pade_poles_and_coeffs(N_F):
    x = Pade_Poles(N_F)
    return x, np.ones(len(x))

def Pade_Poles(N_F):
    Z = np.zeros((N_F, N_F), dtype = np.complex128)
    for i in range(N_F):
        for j in range(N_F): 
            I = i+1
            if j == i+1:
                Z[i,j] = 2 * I * ( 2 * I -1)
            if i == N_F - 1:
                Z[i,j] = -2 * N_F * (2 * N_F -1)
    
    eig, v = np.linalg.eig(Z)
    x = 2 * np.sqrt(eig)
    xr = x.real
    xi = x.imag
    x[xi<0] *= -1
    return x
###
### 16.01.2026 
### This Fermi function below is formally wrong,
### and should be considered errornous, but for the poles obtained 
### from the JieHu2011 method, it gives the correct result, 
### as the poles are symmetrically distributed x=0
### *_v2 is the correct way to write the Fermi function
###
def FD_expanded(E, xp, beta , mu = 0.0, coeffs = None):
    Xpp = mu  + xp/beta
    Xpm = mu  - xp/beta
    if coeffs is None:
        coeffs = np.ones(len(xp))
    diffs =  (1 / beta) * (1 / np.subtract.outer(E , Xpp)  + 1 / np.subtract.outer(E , Xpm)) * coeffs
    return 1 / 2 - diffs.sum(axis = 1)

def FD_expanded_v2(E, xp, beta , mu = 0.0, coeffs = None):
    Xpp = mu + xp / beta
    Xpm = mu + np.conj(xp) / beta
    # Xpm = mu + xp.conj() / beta
    if coeffs is None:
        coeffs = np.ones(len(xp))
    diff1   = (1 / np.subtract.outer(E , Xpp)) * coeffs
    diff1  += (1 / np.subtract.outer(E , Xpm)) * np.conj(coeffs)
    diff1  *= (1 / beta)
    return 1 / 2 - diff1.sum(axis = 1)
@njit
def FD_expanded_v2_opt(E, xp, beta, mu, coeffs, out):
    Xpp = mu + xp / beta
    Xpm = mu + np.conj(xp) / beta
    ne = len(E)
    npol = len(xp)
    assert len(out) == ne
    out[:] = 0.0
    coeffB = coeffs / beta
    coeffB_C = np.conj(coeffB)
    for i in range(ne):
        Ei = E[i]
        for j in range(npol):
            xjp = Xpp[j]
            xjm = Xpm[j]
            Rjp = coeffB[j] 
            Rjm = coeffB_C[j]
            out[i] -= (Rjp/(Ei - xjp) + Rjm/(Ei - xjm)) 
    # return out





def FD(E, beta, mu = 0.0):
    return 1 / (1 + np.exp((E - mu) * beta))

def diff(E, xp, beta, mu = 0.0):
    return FD(E, beta, mu = mu) - FD_expanded(E, xp, beta, mu = mu)

def Hu_b(m):
    return 2 * m -1

def Hu_RN(N):
    return 1/(4 * (N+1) *Hu_b(N+1))

def Hu_Gamma(M):
    Mat = np.zeros((M,M),dtype=complex)
    for i in range(M):
        for j in range(M):
            I = i+1
            J = j+1
            if  i == j+1 or i == j-1:
                Mat[i,j] = 1/np.sqrt(Hu_b(I) * Hu_b(J))
    assert np.allclose(Mat,Mat.conj().T)
    return Mat

def Hu_roots_Q(N):
    M = 2 * N
    e,v = np.linalg.eigh(Hu_Gamma(M))
    e = np.sort(e[e>1e-15])[::-1]
    return 2/e

def Hu_roots_P(N):
    M = 2 * N
    e,v = np.linalg.eigh(Hu_Gamma(M)[1:, 1:])
    e = e[e>1e-15]
    e = 2/e
    return e

def Hu_coeffs(N, _old_behavior = False):
    if N>40 and _old_behavior == False:
        return Hu_coeffs_symbolic(N)
    Const =  N * Hu_b(N+1)/2
    Qx = Hu_roots_Q(N)
    Px = Hu_roots_P(N)
    coeffs = []
    for i in range(N):
        p1 = Qx**2 - Qx[i]**2
        p1[np.abs(p1)<1e-15] = 1.0
        p1 = np.prod(p1)
        p2 = np.prod(Px ** 2 - Qx[i]**2 )
        coeffs += [Const*p2/p1]
    return np.array(coeffs)
# 18.01 Added sympy version of the Hu_coeffs rutine
# This allows for getting the residuals for Nl>42
# 
def Hu_coeffs_symbolic(N):
    import sympy as sp
    Const =  N * Hu_b(N+1)/2
    Qx = Hu_roots_Q(N)
    Px = Hu_roots_P(N)
    Qx = sp.MutableDenseNDimArray(Qx)
    Px = sp.MutableDenseNDimArray(Px)
    coeffs = []
    for i in range(N):
        p1 = Qx.applyfunc(lambda x: x**2) - sp.Array(np.ones(len(Qx))) * Qx[i]**2
        for j in range(len(p1)):
            if abs(p1[j])<1e-15:
                p1[j] = 1
        p1 = sp.prod(p1)
        p2 = sp.prod(Px.applyfunc(lambda x: x**2) -  sp.Array(np.ones(len(Px))) * Qx[i]**2 )
        coeffs += [Const*p2/p1]
    return np.array(coeffs).astype(np.float64)
    


def Hu_poles(N, _old_behavior=False):
    return 1j * Hu_roots_Q(N), Hu_coeffs(N, _old_behavior=_old_behavior)

### AAA_algorithm, 
#"The AAA Algorithm for Rational Approximation"
# by Yuji Nakatsukasa, Olivier Sète, and Lloyd N. Trefethen, 
# SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522. (doi)
#from aaa import aaa
#def AAA_poles(Emin, Emax, tol, kT):
#     Eg = np.arange(Emin/kT, Emax/kT, step = 0.01)
#     F  = FD(Eg, 1.0, mu = 0.0)-1/2
#     r  = aaa(F, Eg, tol = tol, )
#     pol, res = r.polres()
#     idx = pol.imag>=0
#     return  pol[idx], res[idx]
     #return  pol, res


# def Hu_coeffs_testing(N):
#     Const =  N * Hu_b(N+1)/2
#     Qx = np.array(Hu_roots_Q(N),dtype=np.float128)
#     Px = np.array(Hu_roots_P(N),dtype=np.float128)
#     coeffs = []
#     for i in range(N):
#         p1 = Qx**2 - Qx[i]**2
#         #p1[np.abs(p1)<1e-15] = 1.0
#         p1 = np.prod(p1, where=(np.abs(p1)>1e-15), dtype=np.float128)
#         p2 = Px ** 2 - Qx[i]**2
#         p2 = np.prod(p2,  where=(np.abs(p2)>1e-15), dtype=np.float128)
#         coeffs += [Const*p2/p1]
#     return np.array(coeffs)

