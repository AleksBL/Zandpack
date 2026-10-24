import numpy as np
from Zandpack.Pulses import zero_dH, zero_bias, stairs

def bias(t,a):
    if a == 0:
        return  stairs(t, 0.2, 20, 20)
    if a == 1:
        return -stairs(t, 0.2, 20, 20)

dH = zero_dH
def dissipator(t,s): return 0.0
