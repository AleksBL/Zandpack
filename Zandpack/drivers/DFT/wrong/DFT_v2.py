import numpy as np
from siesta_python.siesta_python import SiP
import sisl

devg  =  sisl.get_sile('Device.xyz').read_geometry()

Dev   = None

basis = 'SZ'


Dev = SiP(devg.cell, devg.xyz, devg.toASE().numbers,
          directory_name =  'Device',
          mesh_cutoff    =  100.0,
          kp             =  [1,1,1],
          pp_path        =  '../../pp',
          dictionary = {'DEFAULT HSetupOnly': 'True',
                        'DEFAULT SaveHS': 'True',
                        'DEFAULT DirectPhi':'True'},
          mpi      = '',
          basis    = basis,
          reuse_dm = True, )

Dev.fdf()
Dev.run_siesta_in_dir()
Dev.write_more_fdf(['User.Basis.NetCDF True'], name = 'DEFAULT')
Dev.Visualise()

