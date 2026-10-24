#!/usr/bin/env python3
"""
Bias and dH for the nonorthogonal (nozand) run.

This device is a real SIESTA calculation (S != I) AND its `dH` reads the
density matrix, so -- like synth_dm -- **both** sides of `dH` have to be
transformed:

    argument:   sigma_NO -> sigma_O = iL @ sigma_NO @ iL   (iL = invLowdin)
    result:     H_O      -> H_NO    = iL @ H_O @ iL

Note: nozand imports this file under the module name "Bias", so it must not
import from Bias.py itself (circular self-import).
"""

from Zandpack.Help import TDHelper
from shared import dH as dH_orthogonal, bias, dissipator

Hlp = TDHelper('TDZGNR')
iL  = Hlp.invLowdin

def dH(t, sigma_NO):
    sigma_O = iL @ sigma_NO @ iL
    return iL @ dH_orthogonal(t, sigma_O) @ iL
