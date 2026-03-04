import sisl
import numpy as np
from Zandpack.TimedependentTransport import TD_Transport
from siesta_python.siesta_python import SiP
import matplotlib.pyplot as plt


t1,t2 = 2,2
graphene = sisl.geom.graphene(orthogonal=True)
slab = graphene.tile(t1,0).tile(t2,1)

STM = sisl.Geometry([0, 0, 1], atoms=sisl.Atom('Au', R=1.0001), sc=sisl.SuperCell([10, 10, 1], nsc=[1, 1, 3]))
STM = STM.tile(3,2).move(slab.center() + np.array([0,-0.5, -1]))
tip = STM.tile(1,2)

geom_dev = slab.add(tip)
geom_dev = geom_dev.add_vacuum(40,2)
R  = TD_Transport ([graphene, tip],  geom_dev, kT_i = [0.1, 0.1])
line = np.linspace(-5,5,151)+1j*1e-2 + 1e-5
line = np.vstack([line]*2)
R.Make_Contour(line, 10, pole_mode = 'JieHu2011')

C = R.Contour
# Setup Containers

#Realspace electrode object, more or less a wrapper for some sisl calls
elec_RS = SiP(graphene.cell,graphene.xyz, graphene.atoms.Z,
             pp_path = '../pp',
             semi_inf='ab',
             mpi = '',
             directory_name = 'C', sl = 'C', sm = 'C',
             elec_RSSE = True)

# The vertical electrode
stm    = SiP(STM.cell, STM.xyz, STM.atoms.Z,
             pp_path = '../pp', mpi = '',
             semi_inf = '+a3', 
             directory_name = 'Au_tip', 
             sl = 'Au_tip', sm = 'Au_tip',
             )

# The combined Transport object
Dev    = SiP(geom_dev.cell, geom_dev.xyz, geom_dev.atoms.Z,
             solution_method = 'transiesta',
             pp_path = '../pp',mpi = '',
             directory_name = 'Device', 
             elecs = [elec_RS, stm], # note the electrodes goes into it here
             Chem_Pot = [0.0, 0.0],
             kp_tbtrans = [1,1,1],
             save_SE=True, 
             custom_tbtrans_contour=C)

H_grph = sisl.Hamiltonian(graphene)
H_grph.construct([[0.0, 1.45],[0.0, -2.7]])
H_stm = sisl.Hamiltonian(STM)
H_stm.construct([[0.0, 1.45],[0.0, -2.7]])

elec_RS.manual_H(H_grph)
stm    .manual_H(H_stm)
elec_RS.fdf()
stm    .fdf()
elec_RS.Real_space_SE(0,1,(t1,t2,1), 0.0, -0, 0, 2/50, Contour = C, parallel_E=True)
Dev.find_elec_inds() # Get electrode indices, and reorder
                     # the coordinates 

H_dev = sisl.Hamiltonian(Dev.to_sisl()) # Device Hamiltonian
H_dev.construct([[0.0, 1.45],[0.0, -2.7]])
Dev.manual_H(H_dev)
Dev.fdf()
Dev.run_tbtrans_in_dir()
# Dev2 = Dev.projection( -100.0, 100.0)

R.Device = Dev # Manually assign the Device attribute to read from
R.read_data()  # read all the output from TBtrans
R.pickle('C1')
