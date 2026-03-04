import sisl
import matplotlib.pyplot as plt
import numpy as np
from ase.visualize import view
from ase.build import molecule
from siesta_python.siesta_python import SiP
from siesta_python.funcs import numpy_inds_to_string, listinds_to_string
def fromASE(g):
    return sisl.Geometry(g.positions, atoms = g.numbers, sc = np.array(g.cell))

slako = '/home/aleksander/Desktop/slako/matsci/matsci-0-3/'

angmom  = {'C':'p',
           'H':'s',
           'N':'p'}
basis = 'SZ'
vx1,vy1 = -15.4, -2.6
vx2,vy2 =  15.4, 2.6
tx1     = 10
tx2     = 10
benzene = fromASE(molecule("C6H6")).sub([0,1,2,3,4,5,7,9,10])
NH2     = fromASE(molecule("NH3")).sub([0,1,2]).rotate(60,[0,0,1]).move([0,2.8,0])
mol     = benzene.add(NH2).rotate(-120,[0,0,1])
C       = fromASE(molecule('C'))

e1 = sisl.geom.zgnr(4)#.tile(2,0)
e1 = e1.sub([i for i in range(e1.na) if e1.xyz[i,1]>12])
e2 = e1.copy()

r_idx_1 = [56,57,58,59,53,50,2]
r_idx_2 = [0,1,2,3,6,9,57]

M1 = e1.tile(tx1,0).move([vx1,vy1,0]).sub([i for i in range(e1.na*tx1) if i not in r_idx_1])
M2 = e2.tile(tx2,0).move([vx2,vy2,0]).sub([i for i in range(e1.na*tx1) if i not in r_idx_2])

dev = M1.add(M2)
T = dev.center(what = 'cell')-dev.center() + np.array([20,0,0])
idx_11 = np.where(dev.xyz[:,0] == dev.xyz[:,0].min())[0][0]
idx_12 = 92#np.where(dev.xyz[:,0] == dev.xyz[:,0].max())[0][0]
idx_21 = np.where(e1.xyz [:,0]  == e1.xyz[:,0].min())[0][0]
idx_22 = np.where(e2.xyz [:,1]  == e2.xyz[:,1].min())[0][0]

dev = dev.add_vacuum(40,0)
dev = dev.move(dev.center(what = 'cell')-dev.center(), )
dev = dev.add(mol.move(dev.center(what = 'cell')))

e1 = e1.tile(2,0).move(dev.xyz[idx_11] - e1.xyz[idx_21] + 3*e1.cell[0])
e2 = e2.tile(2,0).move(dev.xyz[idx_12] - e2.xyz[idx_22] - 2*e2.cell[0])

R1  = 2.8
R2  = 4.3
def Rot(r,Theta):
    theta = np.pi / 180 * Theta
    cos = np.cos; sin = np.sin
    rot = np.array([[cos(theta), -sin(theta),0], [sin(theta), cos(theta),0],[0,0,1]])
    return rot.dot(r)

vec = np.array([0,1,0])

dev = dev.add(C.move(dev.center(what="cell") + R1 * Rot(vec,120)))
dev = dev.add(C.move(dev.center(what="cell") + R2 * Rot(vec,120)))
dev = dev.add(C.move(dev.center(what="cell") + R1 * Rot(vec,-60 )))
dev = dev.add(C.move(dev.center(what="cell") + R2 * Rot(vec,-60 )))
C = dev.center()

E1 = SiP(e1.cell, e1.xyz, e1.to.ase().numbers,
         directory_name='E1',sl = 'E1', sm = 'E1',
         semi_inf = '-a1', kp = [50,1,1],
         basis = basis,
         pp_path  = slako,
         )

E2 = SiP(e2.cell, e2.xyz, e2.to.ase().numbers,
         directory_name='E2',sl = 'E2', sm = 'E2',
         semi_inf = '+a1', kp = [50,1,1],
         basis = basis,
         pp_path  = slako,
         )

D = SiP(dev.cell, dev.xyz, dev.to.ase().numbers, 
        directory_name='PohlRelax', elecs = [E1, E2],
        basis = basis,solution_method = 'transiesta',
        Chem_Pot = [.0, .0], kp=[1,1,1],
        print_mulliken=False,
        print_console=True,
        pp_path  = slako,
        kp_tbtrans = [1,1,1]
        )

E1.write_and_run_dftb_in_dir(angmom,)
E2.write_and_run_dftb_in_dir(angmom,)
hem  = E1.dftb2sisl()
hep  = E2.dftb2sisl()
hem.set_nsc((3,1,1))
hep.set_nsc((3,1,1))
bzm  = sisl.MonkhorstPack(hem, [50,1,1])
bzp  = sisl.MonkhorstPack(hep, [50,1,1])
mum  = hem.fermi_level(bz = bzm, q = E1.ideal_cell_charge( {'C':4.0, 'H':1.0} ))
mup  = hep.fermi_level(bz = bzp, q = E2.ideal_cell_charge( {'C':4.0, 'H':1.0} ))
hem.shift(-mum)
hep.shift(-mup)




H = molecule('H')
def PCond(r):
    if np.linalg.norm(r - C)<5:
        return False
    return True

D.Passivate_with_molecule(H, 1.1, 
                          cond_filling = True, 
                          cond = PCond)
E1.Passivate_with_molecule(H, 1.1, 
                           cond_filling = True, 
                           cond = PCond)
E2.Passivate_with_molecule(H, 1.1, 
                           cond_filling = True, 
                           cond = PCond)
D.find_elec_inds()
def Buffer(r):
    if r[0]<E1.pos_real_space[:,0].min() - 0.1 or r[0]>E2.pos_real_space[:,0].max() + 0.1:
        return True
    else:
        return False

D.set_buffer_atoms(Buffer)
D.write_and_run_dftb_in_dir(angmom,)
Hd  = D.dftb2sisl()
Hd.set_nsc((1,1,1))
mud = Hd.fermi_level(bz = sisl.MonkhorstPack(Hd, [1,1,1]), q = E2.ideal_cell_charge( {'C':4.0, 'H':1.0} ))

dic = {'mum':mum,
       'mup':mup, }

# idx1 = [i+1 for i in range(len(D.s)) if abs(D.pos_real_space[i,0] - C[0])<8]
# idx2 = [i+1 for i in range(len(D.s)) if abs(D.pos_real_space[i,0] - C[0])>8]
# s1   = 'clear ' + listinds_to_string(numpy_inds_to_string(np.array(idx1)))
# s2   = 'atom  '+listinds_to_string(numpy_inds_to_string(np.array(idx2))) + ' 0 0 0'
# D.Visualise()
# E1.fdf(); E1.run_siesta_electrode_in_dir()
# E2.fdf(); E2.run_siesta_electrode_in_dir()
# D.fdf(eta = 1e-3)  
# D.run_siesta_in_dir()
