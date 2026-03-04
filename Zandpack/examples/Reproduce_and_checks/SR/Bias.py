from numba import njit as C
import numpy as np
S=np.sin
@C
def bias(t,a):
    ex=np.exp(-(t- 4*np.pi))
    dec=min((ex,1))
    f = (1/(2.0+S(t+S(t)**2)))*dec-0.5
    if a == 0: return +0.05/((t-75)**2 + 0.025)
    else:return -0.05/((t-75)**2 + 0.025)
@C
def dH(t,sig):
    o  = np.zeros(sig.shape, dtype=np.complex128)
    o[0,4,4]+=sig[0,4,4]-sig[0,3,3]/3
    o[0,3,3]-=sig[0,3,3]+sig[0,4,4]/3
    return o
def dissipator(t,s): return .0
