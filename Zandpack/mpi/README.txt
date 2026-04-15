Timedependent propagation code.
nozand: works in the nonorthogonal basis
zand: works in the orthogonal basis

Executing the code is as easy as 
```bash
   export OMP_NUM_THREADS=1   (or as many as you want)
   export NUMBA_NUM_THREADS=1 (as many as OMP_NUM_THREADS)
   mpirun -np 16 zand Dir=working/directory  > RUN.out
   # or nonorthogonal version (faster because of more sparsity in matrices)
   mpirun -np 16 nozand Dir=working/directory  > RUN.out
   # here you should adjust the number of processors to your system
```
You should have an Initial.py and a Bias.py file in the same directory as your zand/nozand calculation
The Zandpack.wrapper module provides several ways to call this from python
