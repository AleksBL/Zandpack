import numpy as np
from siesta_python.siesta_python import SiP
import sisl
import matplotlib.pyplot as plt
from pickle import load
from params import oDir, oSiP

D0 = load(open(oSiP,'rb'))
basis = 'SZ'
Dev = SiP(D0.lat, D0.pos_real_space, D0.s,
          directory_name =  'Device',
          mesh_cutoff    =  150.0, 
          kp             =  [1,1,1],
          pp_path        =  '../../pp_psf',
          dictionary = {'DEFAULT HSetupOnly': 'True',
                        'DEFAULT SaveHS':     'True',
                        'DEFAULT DirectPhi':  'True'},
          mpi      = '',
          basis    = basis,
          reuse_dm = True, )

Dev.fdf()
Dev.run_siesta_in_dir()
Dev.write_more_fdf(['User.Basis.NetCDF True'], name = 'DEFAULT')
Dev.Visualise()
plt.close()

