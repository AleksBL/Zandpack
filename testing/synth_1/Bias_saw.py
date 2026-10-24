from shared import dH_saw as dH, delta_saw as delta

# The MPI codes import dH, bias and dissipator from Bias.py / Bias_NO.py.
bias = delta

def dissipator(t, s):
    return 0.0
