export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

cp Initial1.py Initial.py
mpirun -np 4 Zand Dir=$PWD > RUN1.out
cp -r propfile_save propfile_save_run1
cp Initial2.py Initial.py
mpirun -np 4 Zand Dir=$PWD > RUN2.out
