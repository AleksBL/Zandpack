export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

python make_device.py
python make_fit.py

# Optional steady-state initialisation (device SCF + auxiliary modes), see
# get_steady_state.sh.  It refines the arrays in propfile/Arrays/, which is
# what the MPI codes start from -- so it must be a separate mode, otherwise
# the two sides no longer start from the same (sigma, psi, omega).
# sh get_steady_state.sh

python scipyode.py

# clear both the save directory the code writes into and the one we move it to:
# the MPI codes *append* current/DM chunks, so a leftover directory would be
# stitched into the new result by J()/N() and silently corrupt the comparison.
rm -rf propfile_save propfile_save_zand
mpirun -np 4 zand Dir=$PWD
mv propfile_save propfile_save_zand

rm -rf propfile_save propfile_save_nozand
mpirun -np 4 nozand Dir=$PWD bias_file=Bias_NO.py
mv propfile_save propfile_save_nozand

python Compare.py
