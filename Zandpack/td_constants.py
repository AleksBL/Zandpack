from numpy import pi
import os
# The three constants that are needed in the NEGF timepropagation scheme
hbar            = 6.582119569*10**-1 # eV * fs
try:
    hbar = float(os.environ["ZANDPACK_HBAR"])
except:
    pass
plancks_const   = 2 * pi * hbar      # Unused?
electron_charge = 1.0                # Everything comes out in units of the electron charge
