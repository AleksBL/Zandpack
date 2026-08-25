export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

python Honeychurch2018.py
# No steady state initialzation here
mpirun -np 3 nozand Dir=$PWD
mv TDT_save TDT_hc2018_save_nozand
mpirun -np 3 zand Dir=$PWD
mv TDT_save TDT_TDT_hc2018_save
python compare.py
