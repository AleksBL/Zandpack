export filename=TDT2T_G2
rm *.npy 
rm *.npz


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export linearize=False
python3 DFTCalculator.py &
sleep 15s

SCF Dir=$PWD file=$filename > scf1.out
export linearize=True
python3 Bias.py
SCF Dir=$PWD file=$filename DM_randomness=0.0 > scf2.out

kill %1

