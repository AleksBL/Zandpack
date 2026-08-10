import sisl
import numpy as np
import matplotlib.pyplot as plt
from TimedependentTransport.TimedependentTransport import TD_Transport as TDT
from siesta_python.siesta_python import SiP

g = sisl.geom.graphene(orthogonal = True)

ty = 3
tx = 5

e1 = g.tile(ty,1)
e2 = g.tile(ty,1).move(g.cell[0] * (tx-1))
e3 = sisl.geom.sc(1.42, sisl.Atom('H', R = 1.5))
e3.set_nsc((1,1,3))
dev= g.tile(ty,1).tile(tx,0).remove([25])

diC= np.linalg.norm(dev.xyz - dev.center(),axis = 1)
idxC= np.where(diC == diC.min())[0][0]
e3 = e3.move(dev.xyz[idxC] - e3.xyz[0])

R  = TDT([e1,
          e2,
          e3
          ],
         dev, 
         kT_i=[0.025,
               0.025, 
               0.025
               ], 
         mu_i = [0.0, 0.0,
                 0.0
                 ])

line = np.vstack([np.linspace(-4, 4,150) + 1j * 0.02]*3)

R.Make_Contour(line, 15, pole_mode = 'JieHu2011')

hem  = sisl.Hamiltonian(e1, orthogonal = False)
hep  = sisl.Hamiltonian(e2, orthogonal = False)
heo  = sisl.Hamiltonian(e3, orthogonal = False)
Hd   = sisl.Hamiltonian(dev,orthogonal = False)

E1 = SiP(e1.cell, e1.xyz, e1.toASE().numbers,
         directory_name = 'E1',sl = 'E1', sm = 'E1',
         semi_inf = '-a1',
         )

E2 = SiP(e2.cell, e2.xyz, e2.toASE().numbers,
         directory_name = 'E2',sl = 'E2', sm = 'E2',
         semi_inf = '+a1',
         )

E3 = SiP(e3.cell, e3.xyz, e3.toASE().numbers,
        directory_name = 'E3',sl = 'E3', sm = 'E3',
        semi_inf = '+a3',
          )

Dev= SiP(dev.cell, dev.xyz, dev.toASE().numbers,
         directory_name='Device', 
         elecs = [E1,
                  E2,
                  E3
                  ], 
         Chem_Pot = [.0, 
                     .0, 
                     .0
                     ],
         custom_tbtrans_contour = R.Contour,
         kp_tbtrans = [1,5,1],
         save_SE = True
         )

hem.construct([[0.1, 1.45],[[0.0, 1.0], [-2.7,0.2]]])
hep.construct([[0.1, 1.45],[[0.0, 1.0], [-2.7,0.2]]])
heo.construct([[0.1, 1.45],[[0.0, 1.0], [-2.7,0.2]]])
Hd .construct([[0.1, 1.45],[[0.0, 1.0], [-2.7,0.2]]])

E1.fois_gras(hem)
E2.fois_gras(hep)
E3.fois_gras(heo)
Dev.find_elec_inds()
Hd = Hd.sub(Dev._rearange_indices)
Dev.fois_gras(Hd)
Dev.fdf()
#Dev.write_more_fdf(['TBT.Symmetry.Timereversal False'], name = 'TS_TBT')
Dev.run_tbtrans_in_dir()
R.Device = Dev
R.read_data()
R.Inspect_Transmission(1, 0)
R.pickle('Dev1')













