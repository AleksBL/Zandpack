export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python croy2009.py
# No steady state initialzation here
mpirun -np 3 nozand Dir=$PWD
mv TDT_Croy2009_save TDT_Croy2009_save_nozand
mpirun -np 3 zand Dir=$PWD
mv TDT_Croy2009_save TDT_Croy2009_save_zand

python compare.py
