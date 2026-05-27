#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:22:38 2023

@author: aleksander
"""

import matplotlib.pyplot as plt
import sys
import os
fp = __file__[:-7]
sys.path.append(fp)

if ('ZANDPACK_MPL' not in os.environ) or (os.environ['ZANDPACK_MPL']=='LATEX'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
        })
    try:
        plt.style.use(os.environ['ZANDPACK_PLOTSTYLE'])
    except KeyError:
        plt.style.use(fp+'/plot_style.txt')
else:
    pass

from mpi_tools import combine_currents as J
from mpi_tools import combine_dm as DM
from mpi_tools import combine_pi as PI
from mpi_tools import occupation_number as N
# from mpi_tools import compute_neumann_entropy as S
from mpi_tools import galperin_entropy as gS
from mpi_tools import mutual_information as MI
from mpi_tools import interp_and_fft_complex, interp_and_fft 
from mpi_tools import rolling_minmax, read_all_currents,eval_all
from mpi_tools import rectified_charge as Q
