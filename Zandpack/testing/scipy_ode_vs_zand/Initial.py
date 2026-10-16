from Zandpack.td_constants import hbar
from Zandpack.Loader import load_dictionary
import numpy as np
name="SRC"
t0=0.0
t1=50.0
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
