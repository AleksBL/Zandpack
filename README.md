(Repo setup in progress. Expect more tutorials and polish over the next week  (26.02.2026))
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://www.mozilla.org/en-US/MPL/2.0/)
[![DOI for citation]()](https://doi.org/10.1016/j.cpc.2026.110087)
# Introduction

Zandpack is an open-source Python package for performing time-dependent quantum transport calculations using the Auxiliary Mode Expansion (AME) method. Built on Non-Equilibrium Green’s Function (NEGF) theory, Zandpack enables simulations of open quantum systems (e.g., devices coupled to electrodes) evolving under time-dependent biases and fields. The code is designed to interface with SIESTA, DFTB+, or any LCAO-based DFT code, allowing for dynamic electronic effects in the device region.


## Features 
 - Easy handling of the steady-state TranSIESTA and TBtrans calculations using the siesta_python code
 - Auxiliary Mode Expansion (AME): Uses a level-width function expanded in Lorentzian functions, with algorithms ensuring positive semidefinite fits.
 -  Command line tools for determininng both steady-state steady state and the system evolving under external fields:
       - The SCF tool for obtaining the steady-state density matrix.
       - The psinought tool for obtaining the steady state auxiliary mode wave-vectors.
       - The zand code to propagate the full initial state under timedependent bias and fields. This part is implemented using mpi4py and can scale to many compute nodes. 
 - Extensible: Supports custom interfaces for other LCAO-based DFT codes.

## Installation
To install Zandpack, download the code as a zip file, unpack it and navigate to the Zandpack folder containing the setup.py file in a terminal. Now execute
```console
    python3 -m pip install -e .
```
and you will have an editable install of the code. 

Additionally, add the these two folders to your PATH environment variable: 
```console
   export PATH="/YOUR/PATH/TO/Zandpack/Zandpack/cmdtools:$PATH"
   export PATH="/YOUR/PATH/TO/Zandpack/Zandpack/mpi:$PATH"
   export PATH="/YOUR/PATH/TO/Zandpack/Zandpack/mpitest:$PATH"
   
```
Zandpack depends on the following packages: 
- numpy
- numba
- scipy
- sisl
- psutil
- joblib 
- matplotlib
- siesta_python
- Block_matrices
- Gf_Module

The "Block_matrices", "Gf_Module" and siesta_python codes can be found on GitHub https://github.com/AleksBL and are installed analogously to this code (download and pip install...).

## Documentation
A html file can be found in the docs/_build/html directory. 

## Tutorials
Tutorials are available as introductory notebooks. Navigate to into the second Zandpack folder and copy the notebooks from there into a directory where you wish to run your calculations (Desktop or other). Other example calculations will also be made public here at some point. 

## Support 
Users can get support by submitting an issue on the Zandpack Github page. Requests can also be directed at aleksander.bl@proton.me.
## Licence
Zandpack is released under the Mozilla Public License v2.0. See the LICENCE file in the Zandpack directory.

