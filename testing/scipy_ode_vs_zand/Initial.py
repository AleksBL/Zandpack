from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
import numpy as np
name="SRC"
t0=0.0
t1=20.0
eps=1e-10
usesave=False
LoadFromFull=False
checkpoints=[-43.75, -12.5, 18.75, 50.0]
save_checkpoints=False
print_timings=False
stepsize=0.1
n_dm_compress=1
save_PI=False
saveevery=50
Adir = name + '/Arrays/'
Arrs = load_dictionary(Adir)
compress_mat=True
dm_triu_only=True
dm_occ_only=False

# fast mode: shorten the window from the environment (run_zandpack_tests --fast)
import os as _os
t0 = float(_os.environ.get("ZP_T0", t0))
t1 = float(_os.environ.get("ZP_T1", t1))
saveevery = int(_os.environ.get("ZP_SAVEEVERY", saveevery))
