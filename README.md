# Introduction
Zandpack is an open-source code, which can carry out timedependent quantum transport calculations utilizing the auxiliary mode expansion (AME) method. The technique builts in non-equilibrium Greens function (NEGF) theory and allows one to simulate an open system, with a device coupled to electrodes, evolving under different timedependent biases and fields. The code allows for incorporating dynamic electronic effects in the device by interfacing to e.g SIESTA or DFTB+ simulating electronic dynamics in the device region. Furthermore, any other LCAO-based DFT code can in be used if the user is willing to put in the effort to create a callable interface to the code. A standardized way that utilizes SIESTA, TranSIESTA and TBtrans is already in place and is demonstrated in the available tutorials. 

## Features 
 - Easy handling of the steady-state TranSIESTA and TBtrans calculations using the siesta_python code
 - The AME method utilizes a level-width function expanded in Lorentzian functions. The fitting procedure converting a level-width function sampled on a energy-grid to one expanded in the Lorentzian functions is done with userinput by the Zandpack code. For this purpose algorithms with ensure a positive semidefinite fitted level-width function. 
 -  Command line tools for determininng both steady-state steady state and the system evolving under external fields:
       - The SCF tool for obtaining the steady-state density matrix.
       - The psinought tool for obtaining the steady state auxiliary mode wave-vectors.
       - The zand code to propagate the full initial state under timedependent bias and fields. This part is implemented using mpi4py and can scale to many compute nodes. 

## Installation
To install Zandpack, navigate to the Zandpack folder containing the setup.py file in a terminal and execute
```console
    python3 -m pip install -e .
```
Additionally, add the these two folders to your PATH environment variable: 
```console
   export PATH="/YOUR/PATH/TO/Zandpack/Zandpack/cmdtools:$PATH"
   export PATH="/YOUR/PATH/TO/Zandpack/Zandpack/mpi:$PATH"
```
Zandpack depends on the following packages: 
    - numpy
    - numba
    - scipy
    - sisl
    - matplotlib
    - siesta_python
    - Block_matrices
    - Gf_Module
The "Block_matrices" and "Gf_Module" codes can be found on GitHub https://github.com/AleksBL

## Documentation
A html file can be found in the docs/_build/html directory. 

## Support 
Users can get support by submitting an issue on the Zandpack Github page. Requests can also be directed at aleksander.bach@dipc.org.
## Licence
Zandpack is released under the Mozilla Public License v2.0. See the LICENCE file in the Zandpack directory.
## Directory Structure
The most important parts of the code are listed below.

── Zandpack
    ├── cmdtools
        ├── SCF  
        ├── psinought  
        ├── Adiabatic  
        ├── td_info  
        ├── N-SC-N  
        ├── add_spin_component  
        └── modify_occupations  
    ├── mpi  
        └── zand  
    ├── svg
        └── logo3.svg
    ├── ExpPulses
        ├── air_photonics_pulse.npy
        └──toptica_pulse.npy
    ├── _junk
    ├── _devel
    ├── <u>__</u>init<u>__</u>.py  
    ├── TimedependentTransport.py  
    ├── docstrings.py  
    ├── FittedSelfEnergy.py  
    ├── FittingTools.py  
    ├── Help.py  
    ├── GETPATH.py  
    ├── Interpolation.py  
    ├── LanczosAlg.py  
    ├── Linalg_factorisation.py  
    ├── Louivillian.py  
    ├── Loader.py  
    ├── LICENCE  
    ├── mpi_funcs.py  
    ├── mpi_RK4pars_dev.py  
    ├── mpi_splitter.py  
    ├── mpi_timer.py  
    ├── PadeDecomp.py  
    ├── plot.py  
    ├── plot_style.txt  
    ├── k0nfig.py  
    ├── Response.py  
    ├── Pulses.py  
    ├── Quasiparticle.py  
    ├── td_constants.py  
    ├── params2latex.py  
    └── Writer.py  
