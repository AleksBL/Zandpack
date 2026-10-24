export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

# Separate pipeline: the same test, but both sides start from a steady state.
#
#   device.py          builds the device and writes synth/Arrays/
#   get_steady_state   SCF + psinought rewrite those arrays
#   both propagators   read the rewritten arrays, so they still start from the
#                      same (sigma, xi, Ixi) -- see run_scipy_ode.py
python device.py
sh get_steady_state.sh

python run_scipy_ode.py

# clear the save directory before each run: the MPI codes append chunks, so a
# leftover directory would be stitched into the new result by J()
rm -rf synth_save synth_save_zand
mpirun -np 3 zand Dir=$PWD
mv synth_save synth_save_zand

rm -rf synth_save synth_save_nozand
mpirun -np 3 nozand Dir=$PWD bias_file=Bias_NO.py
mv synth_save synth_save_nozand

python compare.py
