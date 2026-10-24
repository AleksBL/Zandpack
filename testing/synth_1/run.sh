export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

python device.py
python run_scipy_ode.py

# recover from an interrupted sawtooth run that left the bias swapped in
[ -f .Bias_backup.py ] && mv -f .Bias_backup.py Bias.py

# clear both the save directory the code writes into and the one we move it to:
# the MPI codes *append* current/DM chunks, so a leftover directory would be
# stitched into the new result by J()/N() and silently corrupt the comparison.
rm -rf synth_save synth_save_zand
mpirun -np 3 zand Dir=$PWD
mv synth_save synth_save_zand

rm -rf synth_save synth_save_nozand
mpirun -np 3 nozand Dir=$PWD bias_file=Bias_NO.py
mv synth_save synth_save_nozand

python compare.py

# the same device and settings, additionally driven by a sawtooth pulse
sh run_sawtooth.sh
