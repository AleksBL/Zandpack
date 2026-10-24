# Optional steady-state pipeline for the synthetic test: the device SCF, then
# the auxiliary modes.  Both rewrite synth/Arrays/, which is what zand and
# nozand start from -- and run_scipy_ode.py now reads its sigma/xi/Ixi from the
# same arrays, so the two sides keep starting from the same state.
#
#   sh get_steady_state.sh && sh run.sh
#
SCF Dir=$PWD file=synth DM_randomness=0.0
psinought Dir=$PWD file=synth
