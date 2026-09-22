Timedependent propagation code.
nozand: works in the nonorthogonal basis (user responsibility to adjust bias file for this)
zand: works in the orthogonal basis 

Executing the code is as easy as 
```bash
   export OMP_NUM_THREADS=1   (or as many as you want)
   export NUMBA_NUM_THREADS=1 (as many as OMP_NUM_THREADS)
   mpirun -np 8 zand Dir=$PWD  > zand.out   (Prefer nozand, zand is outdated.)
   # or nonorthogonal version (faster because of more sparsity in matrices)
   mpirun -np 8 nozand Dir=$PWD [ initial_file=Initial.py bias_file=Bias.py 
                                  outdir=None sigma_start=None psi_start=None
                                  omg_start=None ]  > nozand.out
   # here you should adjust the number of processors to your system
```
You should have an Initial.py and a Bias.py file in the same directory as your zand/nozand calculation
The Zandpack.wrapper module provides several ways to call this from python.
Input files are written by running the SCF and psinought codes. If these are not used beforehand,
you will start the time-propagation from the isolated device DM and auxillary mode wavectors that are zero.
This is far from the steady state solution and you will end up with large oscillations.
However, if you propagate the system for long enough, it should reach the steady state (given constant driving bias during this).

nozand supports writing "nozand --help " to get the available inputs.
