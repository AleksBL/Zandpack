import numpy as np
from scipy.interpolate import CubicSpline
from numba import njit

def make_spline(x,y):
    cs = CubicSpline(x, y,axis = 0)
    k = 3
    x,c = cs.x.copy(), cs.c.copy()
    
    @njit
    def spline(xi):
        if xi>=x.max() or xi<x.min():
            assert 1 == 0
        
        dx = xi - x
        dx[dx<0] = 10**6
        i = np.where(dx == dx.min())[0][0]
        res = np.zeros(y.shape[1:],dtype = y.dtype)
        for m in range(k+1):
            res += c[m, i] * (xi - x[i])**(k-m)
        return res
    
    
    return spline







