import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['SISL_NUM_PROCS']='1'

from siesta_python.siesta_python import SiP
from siesta_python.funcs import zigzag_g
import sisl
from Zandpack.plot import plt
from Zandpack.TimedependentTransport import TD_Transport as TDT
from Zandpack.FittingTools import piecewise_linspace 
import numpy as np
from ase.build import molecule


data = {}

RUN = True
dk  = 1000.0
g   = zigzag_g().move([1e-4, 1e-4,0])
tx, ty, td = 4,6,16
H     = molecule('H')
vacx  = np.diag([20,0,0])
gem   = g.tile(tx,0)
gep   = g.tile(tx,0).move(g.cell[0] * (td- tx))
gep.xyz[[0,3]]+= gep.cell[0] #+ g.cell[0]/2
slab  = g.tile(td+1,0).tile(ty,1)
slab  = slab.add_vacuum(10,0)
emmax = gem.xyz[:,0].max()+.1
emmin = gem.xyz[:,0].min()-.1
epmin = gep.xyz[:,0].min()-.1
epmax = gep.xyz[:,0].max()+.1

#y1,y2,y3,y4,y5,y6 = 2.0, 7.0, 6.0, 15.0, 15,32
b11,b12 =  6,  18
b21,b22 =  100, 230
b31,b32 =  250, 310
x1,x2   =  26, 22 # 22, 26

def plot(_H, nsc = (1,1,1),smax = 3.0, smin = 1.0, long_only = False):
    Hd3 = _H.copy()
    Hd3.set_nsc(nsc)
    xyz = Hd3.xyz
    H   = Hd3.Hk().toarray()
    na  = len(xyz)
    for ia in range(na):
        plt.scatter(xyz[ia,0],xyz[ia,1])
        ri = xyz[ia]
        ja = np.where((np.abs(H[ia,:])<smax)*(np.abs(H[ia,:])>smin))[0]
        rj = xyz[ja][ja!=ia]
        rij= rj - ri
        for dr in rij:
            if long_only:
                if np.linalg.norm(dr)>1.5 :
                    plt.arrow(ri[0],ri[1], dr[0],dr[1],#s=20.0
                              )
            else:
                plt.arrow(ri[0],ri[1], dr[0],dr[1],#s=20.0
                          )
    plt.axis('equal')

def plot2(Hl):
    for h in Hl:
        hp = h.sub_orbital([i for i in range(h.na) if h.atoms.Z[i]==6], [2])
        os = [hp[i,i,0] for i in range(hp.na)]
        plt.scatter(hp.xyz[:,0], os)
    
    


def rcond(r):
    r0,r1,r2 = r
    if (r0<emmax and r0>emmin):
        return False
    if (r0>epmin and r0<epmax):
        return False
    if r0>epmax:
        return True
    if b11<r1 and b12>r1:
        return False
    if b21<r1 and b22>r1:
        return False
    if b31<r1 and b32>r1:
        return False
    if x1<r0 and x2 > r0 and r1>b11 and r1<b32:
        return False
    return True

dev   = slab.remove([i for i in range(slab.na) if rcond(slab.xyz[i])])
#dev   = dev.remove([175, 40])
slako = '/home/aleksander/Desktop/slako/3ob/3ob-3-1/'
#slako = '/home/aleksander/Desktop/slako/mio/mio-1-1/'
angmom= {'C':'p',
         'H':'s'
         }
def c1(r):
    if r[0,0]<es1.pos_real_space[:,0].min():
        return False
    return True
def c2(r):
    if r[0,0]>es2.pos_real_space[:,0].max():
        return False
    return True
e1 = SiP(gem.cell, gem.xyz, gem.atoms.Z,
         directory_name ='E1', sl = 'E1', sm = 'E1',
         semi_inf       = '-a1',
         elec_RSSE      = True,
         elec_SurRSSE   = True,
         pp_path = slako,
         kp = [24,24,1]
         )

es1 = SiP(gem.cell + vacx,gem.xyz, gem.atoms.Z,
          directory_name ='ES1', sl = 'ES1', sm = 'ES1',
          kp = [1,24,1],
          pp_path = slako,
          )

es1.Passivate_with_molecule(H, 1.05, cond_filling=True, cond= c1)

e2 = SiP(gep.cell, gep.xyz, gep.atoms.Z,
         directory_name='E2', sl = 'E2', sm = 'E2',
         semi_inf      = '+a1',
         elec_RSSE     = True,
         elec_SurRSSE  = True,
         pp_path = slako,
         kp = [24,24,1]
         )
es2 = SiP(gep.cell + vacx,gep.xyz, gep.atoms.Z,
          directory_name ='ES2', sl = 'ES2', sm = 'ES2',
          kp = [1,24,1],
          pp_path = slako,
          )
es2.Passivate_with_molecule(H, 1.05, cond_filling=True, cond= c2)
es2.reararange(np.roll(np.arange(es2.s.shape[0]), 2))

dev.cell[0]-= g.cell[0]*1.5
Dev = SiP(dev.cell, dev.xyz, dev.atoms.Z,
          directory_name = 'Device',
          elecs = [e1,e2], Chem_Pot = [0.0, 0.0],
          
          kp_tbtrans = [1,1,1],
          save_SE = True,
          pp_path = slako,
          )

def cd(r):
    if r[0,0]<Dev.pos_real_space[:,0].min() or r[0,0]>Dev.pos_real_space[:,0].max():
        return False
    return True

Dev.Passivate_with_molecule(H, 1.05, cond_filling=True, cond = cd)#
#assert 1 == 0
e1.write_and_run_dftb_in_dir(angmom)
e2.write_and_run_dftb_in_dir(angmom)
es1.write_and_run_dftb_in_dir(angmom)
es2.write_and_run_dftb_in_dir(angmom)
Hs1   = es1.dftb2sisl()
Hs2   = es2.dftb2sisl()
Hs1.set_nsc((1,5,1))
Hs2.set_nsc((1,5,1))

Hem   = e1.dftb2sisl()
Hep   = e2.dftb2sisl()
bzm   = sisl.MonkhorstPack(Hem, [24,24,1])
bzp   = sisl.MonkhorstPack(Hep, [24,24,1])
bz1   = sisl.MonkhorstPack(Hs1, [1,24,1])
bz2   = sisl.MonkhorstPack(Hs2, [1,24,1])
mum   = Hem.fermi_level(bz = bzm, q = e1.ideal_cell_charge( {'C':4.0}))
mup   = Hep.fermi_level(bz = bzp, q = e2.ideal_cell_charge( {'C':4.0}))
mus1  = Hem.fermi_level(bz = bz1, q = es1.ideal_cell_charge({'C':4.0, 'H':1.0}))
mus2  = Hep.fermi_level(bz = bz2, q = es2.ideal_cell_charge({'C':4.0, 'H':1.0}))

for T in range(-(Hs1.nsc[1]//2), (Hs1.nsc[1]//2)+1):
    UC = (0,T,0)
    for ia in range(Hem.na):
        dij = np.linalg.norm(Hem.xyz[ia] - Hs1.xyz, axis=1)
        ia2 = np.where(dij == dij.min())[0][0]
        for io in range(Hem.atoms[ia].no):
            for ja in range(Hem.na):
                dij = np.linalg.norm(Hem.xyz[ja] - Hs1.xyz, axis=1)
                ja2 = np.where(dij == dij.min())[0][0]
                for jo in range(Hem.atoms[ja].no):
                    Hs1[Hs1.a2o(ia2) + io, Hs1.a2o(ja2) + jo, UC] = Hem[Hem.a2o(ia) + io, Hem.a2o(ja) + jo,UC]

for T in range(-(Hs2.nsc[1]//2), (Hs2.nsc[1]//2)+1):
    UC = (0,T,0)
    for ia in range(Hep.na):
        dij = np.linalg.norm(Hep.xyz[ia] - Hs2.xyz, axis=1)
        ia2 = np.where(dij == dij.min())[0][0]
        for io in range(Hep.atoms[ia].no):
            for ja in range(Hep.na):
                dij = np.linalg.norm(Hep.xyz[ja] - Hs2.xyz, axis=1)
                ja2 = np.where(dij == dij.min())[0][0]
                for jo in range(Hep.atoms[ja].no):
                    Hs2[Hs2.a2o(ia2) + io, Hs2.a2o(ja2) + jo, UC] = Hep[Hep.a2o(ia) + io, Hep.a2o(ja) + jo,UC]

Hem.shift(-mum)
Hep.shift(-mup)
Hs1.shift(-mum)
Hs2.shift(-mup)

#assert 1 == 0

e1.manual_H(Hem)
e2.manual_H(Hep)

R  = TDT ([e1.to_sisl(), e2.to_sisl()],  Dev.to_sisl(), kT_i = [0.025, 0.025])
line  = piecewise_linspace([-8, -3, 0, 8], [31,150,80])
#line = np.linspace(0, 0.5, 50)
line = np.vstack([line]*2)
R.Make_Contour(line, 20, pole_mode = 'JieHu2011')
Dev.custom_tbtrans_contour = R.Contour
if RUN == False:
    assert 1 == 0

e1.Real_space_SI(1, (1,ty,1), 0.0, R.Contour,(1,7,1),
                 parallel_E = True,num_procs = 4,
                 dk = dk,
                 only_couplings = True,
                 Hsurf_in = Hs1,
                 )

e2.Real_space_SI(1, (1,ty,1), 0.0, R.Contour,(1,7,1),
                  parallel_E = True,num_procs = 4,
                  dk = dk, 
                  only_couplings = True,
                  Hsurf_in = Hs2,
                  )

Dev.find_elec_inds()
Dev.write_and_run_dftb_in_dir(angmom)
# In[]
Hd = Dev.dftb2sisl()
bzd  = sisl.MonkhorstPack(Hd, [1,1,1])
mud  = Hd.fermi_level(bz = bzd, q = Dev.ideal_cell_charge({'C':4.0,'H':1.0}))

shft = -mum + 0.25
Hd.shift(-mum+0.25)
np.savez('Stst', mud = shft)
Hd.set_nsc((1,1,1))
odx_e1 =  np.hstack([np.arange(Hd.a2o(i), Hd.a2o(i) + Hd.atoms[i].no) for i in Dev.elec_inds[0]])
odx_e2 =  np.hstack([np.arange(Hd.a2o(i), Hd.a2o(i) + Hd.atoms[i].no) for i in Dev.elec_inds[1]])
#assert 1 == 0

#Ht1, _ =  e1.to_sisl('TSHS')
#Ht2, _ =  e2.to_sisl('TSHS')
# for i,I in enumerate(odx_e1):
#     for j,J in enumerate(odx_e1):
#         Hd[I,J] = Ht1[i,j] #+ 0.1
# for i,I in enumerate(odx_e2):
#     for j,J in enumerate(odx_e2):
        # Hd[I,J] = Ht2[i,j]
# for i in range(Hd.no):
#     Hd[i,i,0] += 0.1

#print((Hd.eigh()))
Dev.manual_H(Hd)
Dev.fdf()
Dev.pickle('Dev')
#Dev.run_tbtrans_in_dir(DOS_GF = True)

#R.Device = Dev
#orbs     = [4*i+2 for i in range(len(Dev.s)) if Dev.s[i]==6]
#R.read_data()
#R.less_memory()
#R.pickle('GrRibGr2')
