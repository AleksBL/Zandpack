Timedependent Transport code.

Refer to pdf for how to use this

See the "boilerplatecode" folder (../boilerplatecode relative to this file) for the basic setup of the Bias.py as Initial.py files.

Executing the code is as easy as 
   export OMP_NUM_THREADS=1   (or as many as you want)
   export NUMBA_NUM_THREADS=1 (as many as OMP_NUM_THREADS)
   mpirun -np 16 Zand Dir=$PWD > RUN.out
   
