import numpy as np
from numba import njit

@njit
def dH(t,sigma):
    ## Note that in order to make this function work in the nozand code
    ## One has to add a transformation in the end with a function like this:
    ## def HamO2NO(Hlike):  return iL @ Hlike  @ iL
    ## and make dH return HamO2NO(out)
    ##
    ## L should be loaded like:
    ## from Zandpack.Help import TDHelper
    ## Hlp = TDHelper("SRC")
    ## "L   = Hlp.Lowdin; iL = Hlp.invLowdin
    out = np.zeros(sigma.shape, dtype = sigma.dtype)
    out[0, 0, 0] = np.sin(t/10)
    out[0, 5, 5] = np.sin(t/6)
    out[2,3,3] = np.sin(t/2)
    out[1,9,9] = np.sin(t/2)
    return out

@njit
def delta(t,a):
    if a == 0:
        return +np.sin(t/20)
    return -np.sin(t/20)
