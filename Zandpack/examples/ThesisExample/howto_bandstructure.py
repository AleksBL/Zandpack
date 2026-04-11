from siesta_python.siesta_python import SiP
import sisl; import numpy as np
# try: Au2Cl2, C2, TaS2, NbSe2, Ti3C2H2O2
# and compare with Ref. %\textcolor{green}{\cite{c2dbweb}}%
name = 'Ti3C2H2O2'; P = plt.plot
g=sisl.get_sile('c2db/'+name+'.xyz'
                ).read_geometry()
clc = SiP(g.cell,g.xyz,g.atoms.Z,
          mpi='',mesh_cutoff=500,
          kp = [9,9,1],basis='SZP',
          xc='GGA', TwoDim=True)
clc.fdf();clc.run_siesta_in_dir()
H,S=clc.to_sisl('TSHS')
p,n=clc.make_bandpath()
n = [r'$'+ni+'$' if 'GAMMA' not in ni
     else r'$\Gamma$' for ni in n]
band=sisl.BandStructure(H, p, 2000,n)
e   =np.zeros((2000,H.no))
for i,k in enumerate(band.k):
    e[i]=H.eigh(k = k) #<- Bandstructure
lband=band.lineark()
from Zandpack.plot import plt
for i in range(e.shape[1]):
    o,uo=e[:, i].copy(),e[:, i].copy()
    o[o>0]=np.nan; uo[uo<0]=np.nan
    P(lband,o,color='darkblue')
    P(lband,uo,color='royalblue', alpha=0.5)
lk,kt,kl  = band.lineark(True)
plt.xticks(kt, kl); plt.ylim(-4,4)
plt.xlim(0,lband.max())
plt.ylabel(r'$\epsilon_{\mathbf{k}}$',
           size=20)
plt.tight_layout()
plt.savefig(name+'_Bands.svg')
