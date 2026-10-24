#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias and dH for the nonorthogonal (nozand) run.

shared.dH is written in the orthogonal (Lowdin) basis, which is what the zand
code uses. nozand propagates in the nonorthogonal basis, so dH has to be
transformed into it:

    H_NO = S^(1/2) @ H_O @ S^(1/2) = iL @ H_O @ iL,   iL = invLowdin

The two other Bias inputs (bias, dissipator) are basis independent.
"""

from Zandpack.Help import TDHelper
from shared import dH as dH_orthogonal, delta

Hlp = TDHelper('TDT_chain')
iL  = Hlp.invLowdin


def HamO2NO(Hlike):
    return iL @ Hlike @ iL


def dH(t, sigma):
    return HamO2NO(dH_orthogonal(t, sigma))


def bias(t, a):
    return delta(t, a)


def dissipator(t, s):
    return 0.0
