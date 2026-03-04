from siesta_python.siesta_python import SiP
import sisl
# Make geometry
tx_e,tx_d,kp = 2,12,[30,1,1]
def run(A):
    A.Passivate(1,1.1)
    if len(A.elecs)!=0: A.find_elec_inds()
    A.fdf(); A.run_siesta_in_dir()
g=sisl.geom.zgnr(3); gem=g.tile(tx_e,0)
gep=gem.move(g.cell[0]*(tx_d - tx_e))
gd =g.tile(tx_d,0).remove([35,38,41])
# Make Calculator objects, do H-passivation
EM = SiP(gem.cell, gem.xyz, gem.atoms.Z,mpi='',
         directory_name='EM', sl ='EM', kp=kp,
         semi_inf = '-a1', basis = 'SZ')
EP = SiP(gep.cell, gep.xyz, gep.atoms.Z,mpi='',
         directory_name='EP', sl ='EP', kp=kp,
         semi_inf = '+a1', basis = 'SZ')
Dev = SiP(gd.cell, gd.xyz, gd.atoms.Z,
          directory_name='Dev',elecs = [EM,EP],
          basis ='SZ',Chem_Pot=[0,0],mpi='',
          solution_method='transiesta',
          print_console=True, save_SE=True)
# DFT calculation like in Listing %\textcolor{green}{\ref{lst:T2}}%
run(EM); run(EP); run(Dev)
# Proceces roughly Fig. %\textcolor{green}{\ref{fig:T5_structure}}% +more
Dev.figures(custom_bp=[([[0,0,0],[.5,0,0]],
                         [r'$\Gamma$',r'$X$'])]*2)
# Save to file
Dev.pickle('Dev')
