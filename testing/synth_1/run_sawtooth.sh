export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

# Same device as run.sh, driven by a sawtooth pulse instead of the smooth one.
# Everything else (device, fit, tolerance, time window) is identical, so the
# two runs can be compared directly.
python run_scipy_ode.py saw

# Recover from an interrupted previous run that left the bias swapped in: the
# backup is written before the swap, so restoring it here is always safe.
[ -f .Bias_backup.py ] && mv -f .Bias_backup.py Bias.py

# zand takes only "Dir=..." -- it does not understand bias_file=, so the bias
# file has to be swapped in for the run (nozand does take bias_file=).
# Put it back even if this script dies, so the *smooth* run in run.sh can never
# pick up the sawtooth drive by accident.
cp Bias.py .Bias_backup.py
trap 'mv -f .Bias_backup.py Bias.py 2>/dev/null' EXIT INT TERM
cp Bias_saw.py Bias.py

# clear both the save directory the code writes into and the one we move it to:
# the MPI codes *append* current/DM chunks, so a leftover directory would be
# stitched into the new result by J()/N() and silently corrupt the comparison.
rm -rf synth_save synth_save_zand_saw
mpirun -np 3 zand Dir=$PWD
mv synth_save synth_save_zand_saw
mv -f .Bias_backup.py Bias.py

rm -rf synth_save synth_save_nozand_saw
mpirun -np 3 nozand Dir=$PWD bias_file=Bias_NO_saw.py
mv synth_save synth_save_nozand_saw

python compare.py saw
