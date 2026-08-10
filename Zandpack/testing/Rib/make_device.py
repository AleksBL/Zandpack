import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['SISL_NUM_PROCS']='1'

from siesta_python.siesta_python import SiP
from siesta_python.funcs import zigzag_g
import sisl
import matplotlib.pyplot as plt
import numpy as np
from Zandpack.TimedependentTransport import TD_Transport as TDT

RUN = True
dk  = 1000.0

g = zigzag_g()
tx, ty, td = 2,10,30

gem   = g.tile(tx,0)
gep   = g.tile(tx,0).move(g.cell[0] * (td- tx))
slab  = g.tile(td,0).tile(ty,1)
emmax = gem.xyz[:,0].max()+.1
epmin = gep.xyz[:,0].min()-.1

#y1,y2,y3,y4,y5,y6 = 2.0, 7.0, 6.0, 15.0, 15,32
b11,b12 = 2,6
b21,b22 = 12, 23
b31,b32 = 25, 31
x1,x2   = 22, 40
N_E     = 121
N_F     = 20
def rcond(r):
    r0,r1,r2 = r
    if r0<emmax or r0>epmin:
        return False
    if b11<r1 and b12>r1:
        return False
    if b21<r1 and b22>r1:
        return False
    if b31<r1 and b32>r1:
        return False
    if x1<r0 and x2 > r0 and r1>b11 and r1<b32:
        return False
    return True

dev = slab.remove([i for i in range(slab.na) if rcond(slab.xyz[i])])

e1  = SiP(gem.cell, gem.xyz, gem.to.ase().numbers,
         directory_name ='E1', sl = 'E1', sm = 'E1',
         semi_inf       = '-a1',
         elec_RSSE      = True,
         elec_SurRSSE   = True,
         )

e2  = SiP(gep.cell, gep.xyz, gep.to.ase().numbers,
         directory_name='E2', sl = 'E2', sm = 'E2',
         semi_inf      = '+a1',
         elec_RSSE     = True,
         elec_SurRSSE  = True,
         )

Dev = SiP(dev.cell, dev.xyz, dev.to.ase().numbers,
          directory_name = 'Device',
          elecs = [e1,e2], Chem_Pot = [0.0, 0.0],
          kp_tbtrans = [1,1,1],
          save_SE = True,
          )

vals = [[0.1, 1.5], [0.0, -2.7]]

Hem  = sisl.Hamiltonian(gem); Hem.construct(vals)
Hep  = sisl.Hamiltonian(gep); Hep.construct(vals)
Hd   = sisl.Hamiltonian(dev); Hd. construct(vals)

e1.manual_H(Hem)
e2.manual_H(Hep)

R    = TDT ([e1.to_sisl(), e2.to_sisl()],  Dev.to_sisl(), kT_i = [0.025, 0.025])
line = np.linspace(-8,8,N_E)+1j*1e-2
line = np.vstack([line]*2)
R.Make_Contour(line, N_F, pole_mode = 'JieHu2011')
Dev.custom_tbtrans_contour = R.Contour

if RUN == False:
    assert 1 == 0

e1.Real_space_SI(1, (1,ty,1), 0.0, R.Contour,(1,3,1),
                 parallel_E = True,num_procs = 4,
                 dk = dk)

e2.Real_space_SI(1, (1,ty,1), 0.0, R.Contour,(1,3,1),
                 parallel_E = True,num_procs = 4,
                 dk = dk)
Dev.find_elec_inds()
Dev.fdf(eta = 1e-3)
Dev.manual_H(Hd.sub(Dev._rearange_indices))
Dev.run_tbtrans_in_dir(DOS_GF = True)

R.Device = Dev
R.read_data()
R.pickle('1Rib')
