"""Bias and dH for the orthogonal (zand) run.

`dH` lives in `shared.py` because `nozand` imports its bias file under the
module name `Bias` -- if `Bias_NO.py` imported `Bias.py` that would be a
circular self-import.
"""

from shared import dH, delta

bias = delta

def dissipator(t, s):
    return 0.0
