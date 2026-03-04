from Initial import name, np
from Zandpack.Pulses import box_pulse as BP
from Zandpack.plot import plt
from sisl import get_sile
Rep,K1,K2 = 2.0, 23, 24
pref=name+'/Arrays/'
L=np.load(pref+'S^(-0.5).npy')
p=np.load(pref+'/pivot.npy')
Hs=get_sile('Dev/siesta.TSHS').read_hamiltonian()
r=np.load(pref+'Positions.npy')
r=r[[Hs.o2a(i) for i in p]]
Rmp = (r[:,0] - np.average(r[:,0]))
Rmp/= (r[:,0].max() - r[:,0].min())
def dH(t,sigma):
    V   =(bias(t,1)-bias(t,0))*0.5
    Hext=np.diag(Rmp*V); dh  =L@Hext@L
    dh[0,K1,K1]+=Rep*sigma[0,K1,K1]
    dh[0,K2,K2]+=Rep*sigma[0,K2,K2]
    return dh
def bias(t,a):
    return np.sign(a-.5)*(1-BP(t,50.,3.,1))*.5
def dissipator(t,sig): return 0.0

#P1,P2 = r[K1],r[K2]
#plt.scatter(r[:,0], r[:,1])
#plt.scatter(P1[0],P1[1])
#plt.scatter(P2[0],P2[1])
#plt.axis('equal')
#plt.text(P1[0],P1[1]-0.9, r'$K_{1}$',size=16)
#plt.text(P2[0],P2[1]+0.5, r'$K_{2}$',size=16)
#plt.tight_layout()
#plt.xlim(r[:,0].min(), r[:,0].max())
#plt.ylim(5,18)
#plt.savefig('Setup.svg')


