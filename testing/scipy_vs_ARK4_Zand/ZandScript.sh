export ZANDPACK_HBAR=1.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MPLBACKEND=Agg
export RK_method=RK45

# Restart / stitching check, run for both codes: propagate 0 -> 12, keep the
# state, then restart from it and run 11 -> 20.  Initial.py is swapped out for
# the duration and restored at the end, so the main configuration is untouched.
run_code () {                    # $1 = executable, $2 = extra keyword (may be empty)
    cp Initial1.py Initial.py
    mpirun -np 4 $1 Dir=$PWD $2 > RUN1_$1.out
    cp -r propfile_save propfile_save_run1_$1
    cp Initial2.py Initial.py
    mpirun -np 4 $1 Dir=$PWD $2 > RUN2_$1.out
}

cp Initial.py .Initial_backup.py

run_code zand   ""
run_code nozand "bias_file=Bias_NO.py"

mv .Initial_backup.py Initial.py
