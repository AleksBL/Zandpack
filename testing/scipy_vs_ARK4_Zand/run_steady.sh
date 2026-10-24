export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

# Separate pipeline: the same test, but both sides start from a steady state
# (device SCF + psinought) instead of the default isolated-device density
# matrix.  Kept out of run.sh on purpose -- SCF rewrites propfile/Arrays/, so it
# changes the starting state of whatever runs after it.  See handoff.md.
python make_device.py
python make_fit.py
sh get_steady_state.sh

python scipyode.py

# clear both the save directory the code writes into and the one we move it to
rm -rf propfile_save propfile_save_zand
mpirun -np 4 zand Dir=$PWD
mv propfile_save propfile_save_zand

rm -rf propfile_save propfile_save_nozand
mpirun -np 4 nozand Dir=$PWD bias_file=Bias_NO.py
mv propfile_save propfile_save_nozand

python Compare.py
