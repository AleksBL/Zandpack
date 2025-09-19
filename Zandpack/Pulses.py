import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
import os
#import GETPATH

D = __file__[:-10]
P1 = np.load(D+'/ExpPulses/air_photonics_pulse.npy')
t1,p1 = P1.T
t1 = t1 * 1000
t1 = t1[10:60]
p1 = p1[10:60]
p1 = p1 / np.abs(p1).max()
t1 = t1 - t1.min()
P2 = np.load(D+'/ExpPulses/toptica_pulse.npy')
t2,p2 = P2.T
t2 = t2* 1000
t2 = t2[90:170]
p2 = p2[90:170]
p2 = p2 / np.abs(p2).max()
t2 = t2 - t2.min()

cs1 = CubicSpline(t1,p1)
cs2 = CubicSpline(t2,p2)
x1,c1 = cs1.x, cs1.c
x2,c2 = cs2.x, cs2.c

k = 3
ts = 15
ts2 = 300
@njit
def air_photonics_pulse(t):
    dt = t - t1
    dt[dt<0] = 10**5
    i = np.where(dt == dt.min())[0][0]
    res = 0.0
    if t<t1.min() or t>t1.max():
        return 0.0
    for m in range(k+1):
        res += c1[m, i] * (t - x1[i])**(k-m)
    res  = res * (np.tanh((t-100) / ts) - np.tanh((t - 550) / ts)) / 2
    return res

@njit
def toptica_pulse(t):
    dt = t - t2
    dt[dt<0] = 10**5
    i = np.where(dt == dt.min())[0][0]
    res = 0.0
    if t<t2.min() or t>t2.max():
        return 0.0
    
    for m in range(k+1):
        res += c2[m, i] * (t - x2[i])**(k-m)
    res  = res * (np.tanh((t-300) / ts2) - np.tanh((t - 4000) / ts2)) / 2
    return res

@njit
def env1(x):
    return np.exp(- x**2)
    
@njit
def generic_pulse(t, s, f, phase, delay, envelope):
    """Args: 
        t: time 
        s: abruptness of turning on
        f: frequency in where it should conform with the time being in femtoseconds (fixed by hbar)
        phase: phase-delay of sine factor
        delay: delay for max value of pulse
        envelope: envelope function, eg the function env, but customizable
        
        """
    x = (t-delay)/s
    return envelope(x) * np.sin(2 * np.pi * f * t - phase)

@njit
def zero_bias(t, a):
    return 0.0

@njit
def zero_dH(t, sigma):
    A = np.zeros(sigma.shape, dtype=np.complex128)
    return A

@njit
def box_pulse(t, tp, ts, V):
    return V * (np.tanh(t / ts) - np.tanh((t - tp) / ts)) / 2

@njit
def step(x, mu, s):
    return 1 / (1 + np.exp((x - mu) / s))

def make_constant_bias(const):
    @njit
    def constant_bias(t,a):
        if a == 0:
            return  const/2
        if a == 1:
            return -const/2
    return constant_bias

@njit
def stairs(x,stair_height, stair_width,n, softness=0.1):
    res = 0.0
    for i in range(1,n):
        res = res + (1-step(x, i * stair_width, softness))*stair_height
    return res

@njit
def pumpprobe(t, fpump, fprobe, Trep, t1,t2, tstep = 5.0):
    val = fpump(t)
    
    if t2>t and t>t1:
        dt = np.mod(t-t1, Trep)
        val += fprobe(dt)*box_pulse(dt, Trep, tstep, 1.0)
        val += fprobe(dt-Trep)*box_pulse(dt-Trep, Trep, tstep, 1.0)
        val += fprobe(dt+Trep)*box_pulse(dt+Trep, Trep, tstep, 1.0)
    return val

        





#print(__file__)


