import numpy as np
import os
import re
import numba as nb
from tqdm import tqdm
from Zandpack.Loader import flexload

ld = os.listdir

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

def combine_currents(dirs, n=2):
    """ 
        dirs :  the *_save directory/ies where a Zand calculation has been done
        n    : number of electrodes
        
    """
    C = [[] for i in range(n)]
    T = []
    count = 0
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'current_' in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if 'times' in v]
        tf.sort(key=natural_keys)
        for ff in f:
            vals = ff.split('_')
            if len(vals)==2:
                pass
            else:
                L     = int(vals[1])
                _Jt = np.load(d+'/'+ff)
                C[L] += [_Jt]
        
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
        count += 1
    CC  =  [np.vstack(C[i]) for i in range(n)]
    T   =   np.hstack(T)[:-count]
    return T, CC

def combine_dm(dirs, times_label = 'DMt'):
    """
    like combine_currents, but for the density matrix instead.
    """

    DM = []
    T  = []
    count = 0
    
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'DM' in v and 'DMt' not in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if times_label in v]
        tf.sort(key=natural_keys)
        for ff in f:
            _dm = np.load(d+'/'+ff)
            DM += [_dm]
        
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
        count += 1
    DM  =  np.vstack(DM)
    T   =   np.hstack(T)#[:-count]
    return T, DM

def combine_pi(dirs, times_label = 'DMt'):
    """
    like combine_currents, but for the PI matrix instead.
    """

    DM = []
    T  = []
    count = 0
    
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'PIa' in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if times_label in v]
        tf.sort(key=natural_keys)
        for ff in f:
            _dm = np.load(d+'/'+ff)
            DM += [_dm]
        
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
        count += 1
    DM  =   np.vstack(DM)
    T   =   np.hstack(T)
    return T, DM

def closest_dm(times, Dir, dt_avg, times_label = 'DMt'):
    DM = []
    T  = []
    f = ld(Dir)
    f = [v for v in f if 'DM' in v and 'DMt' not in v]
    f.sort(key=natural_keys)
    tf = [v for v in ld(Dir) if times_label in v]
    
    tf.sort(key=natural_keys)
    bidx = []
    
    for i,ff in enumerate(tf):
        _t  = np.load(Dir+'/'+ff)
        bidx += [i]*len(_t)
        T    += [_t]
    
    indices = []
    globT   = np.hstack(T)
    dTt     = np.array([np.abs(globT - ti).min() for ti in times])
    for j,_t in enumerate(times):
        indices += [ [ [], [] ] ]
        for i, td in enumerate(T):
            dt  = np.abs(td - _t)
            if dt_avg > 0:
                idx = np.where(dt < dt_avg)[0]
            else:
                idx = np.where(dt == dTt[j])[0]
            if len(idx) > 0:
                indices[j][0] += [i]
                indices[j][1] += [idx]
                
    for info in indices:
        _DM = []
        for ib, b in enumerate(info[0]):
            _dm = np.load(Dir + '/' + f[b])[info[1][ib]]
            _DM += [_dm]
        DM += [np.average(np.vstack(_DM), axis = 0)]
    DM = np.array(DM)
    return DM

def occupation_number(dirs, times_label = 'DMt'):
    from tqdm import tqdm
    N = []
    T = []
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'DM' in v and 'DMt' not in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if times_label in v]
        tf.sort(key=natural_keys)
        for ff in tqdm(f):
            _dm = np.load(d+'/'+ff)
            N  += [np.trace(_dm,axis1 = 2,axis2=3)]
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
    N   =  np.vstack(N)
    T   =   np.hstack(T)
    return T, N
def partial_charges(dirs, times_label = 'DMt', Transform=None):
    from tqdm import tqdm
    Ni= []
    T = []
    M = Transform
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'DM' in v and 'DMt' not in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if times_label in v]
        tf.sort(key=natural_keys)
        
        for ff in tqdm(f):
            _dm = np.load(d+'/'+ff)
            idx = np.arange(_dm.shape[-1])
            if M is not None:
                Ni  += [(M@_dm@M)[..., idx, idx]]
            else:
                Ni  += [_dm[..., idx, idx]]
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
    Ni  =  np.vstack(Ni)
    T   =  np.hstack(T)
    return T, Ni



#def compute_neumann_entropy(dirs,times_label = 'DMt'):
#    S = []
#    T = []
#    for d in dirs:
#        f = ld(d)
#        f = [v for v in f if 'DM' in v and 'DMt' not in v]
#        f.sort(key=natural_keys)
#        tf = [v for v in ld(d) if times_label in v]
#        tf.sort(key=natural_keys)
#        for ff in tqdm(f):
#            _dm = np.load(d+'/'+ff)
#            e,v = np.linalg.eigh(_dm)
#            S += [-(e*np.log(e)).sum(axis=2)]
#        
#        for ff in tf:
#            T+=[np.load(d+'/'+ff)]
#    S = np.vstack(S)
#    T = np.hstack(T)
#    return T,S


def galperin_entropy_old(dm, eigtol=-1e-4):
    e1,v1 = np.linalg.eigh(dm)
    #filter
    if e1.min()<0:
        print('Warning: DM has <0 eigenvalues')
        if e1.min()<eigtol:
            print('Error: DM eigenvalues too negative!')
            assert 1==0
        e1[e1<1e-15]=1e-15
    sig  = v1@(e1[..., None]*v1.transpose(0,1,3,2).conj())
    sigG = np.eye(sig.shape[-1]) - sig
    e2   =  np.linalg.eigvalsh(sigG)
    S = -(e1*np.log(e1)).sum(axis=-1)-(e2*np.log(e2)).sum(axis=-1)
    return S

def galperin_entropy(dm, eigtol=-1e-4):
    e1 = np.linalg.eigvalsh(dm)
    # Filter
    if e1.min()<0:
        print('Warning: DM has <0 eigenvalues')
        if e1.min()<eigtol:
            print('Error: DM eigenvalues too negative!')
            assert 1==0
    e1[e1<1e-15]         = 1e-15
    e1[e1>(1.0 - 1e-7)] = 1.0 - 1e-7
    e2 = 1-e1
    S = -(e1*np.log(e1)).sum(axis=-1)-(e2*np.log(e2)).sum(axis=-1)
    #print(e1.min(), e1.max())
    #print(e2.min(), e2.max())
    return S

def mutual_information(dm, parts, eigtol = -1e-4, return_S=False):
    Sf = galperin_entropy(dm, eigtol=eigtol)
    MI = np.zeros(Sf.shape)
    for p in parts:
        inds = np.array(p)
        MI += galperin_entropy(dm[..., inds[:, None], inds[None,:]], eigtol=eigtol)
    MI -= Sf
    if return_S:
        return MI, Sf
    return MI


def compute_neumann_entropy(dirs,times_label = 'DMt'):
    S = []
    T = []
    for d in dirs:
        f = ld(d)
        f = [v for v in f if 'DM' in v and 'DMt' not in v]
        f.sort(key=natural_keys)
        tf = [v for v in ld(d) if times_label in v]
        tf.sort(key=natural_keys)
        for ff in tqdm(f):
            _dm = np.load(d+'/'+ff)
            e,v = np.linalg.eigh(_dm)
            S += [-(e*np.log(e)).sum(axis=2)]
        
        for ff in tf:
            T+=[np.load(d+'/'+ff)]
    S = np.vstack(S)
    T = np.hstack(T)
    return T,S
def trim_DM_dir(Dir, n, base_len = 50, dtype = np.complex64):
    files = os.listdir(Dir)
    for f in files:
        if 'DM' in f:
            dm_load = np.load(Dir+'/'+f)
            if len(dm_load)<base_len:
                pass
            else:
                np.save(Dir+'/'+f, dm_load[::n].astype(dtype))

def Project_DM_on_QPfile(DMdir, QPf, ik = 0, n = 1, times_label = 'DMt'):
    EQP, tQP, vQP, iSE = filter_duplicates(QPf)
    f = ld(DMdir)
    f = [v for v in f if 'DM' in v and 'DMt' not in v]
    f.sort(key=natural_keys)
    tf = [v for v in ld(DMdir) if times_label in v]
    tf.sort(key=natural_keys)
    
    Container = []
    
    for i in range(len(f)):
        DMt = np.load(DMdir+'/'+ f[i])[:,ik].copy().astype(np.complex128)
        t   = np.load(DMdir+'/'+tf[i])[::n]
        if len(DMt)!=len(t): t = t[:-1]
        Container += subrutine_Project_DM(t, DMt, tQP, vQP, EQP, iSE)
        
    return np.array(Container)

@nb.njit
def subrutine_Project_DM(t, DM, tQP, vecs, Ev, iSE):
    #dt = np.abs(t - tQP)
    container = [np.zeros(4)] # Container list object of real[(3,)]
    arr3      = np.zeros(4)   # arr3 is gonna take the values and get copied
    
    for i in range(len(t)):
        ti,dm_i = t[i], DM[i]
        dt      = np.abs(ti - tQP)            #Find closest QP states
        J       = np.where(dt == dt.min())[0]
        for j in J:
            e,v,ise = Ev[j], vecs[j], iSE[j]
            arr3[0] = ti.real
            arr3[1] = e.real
            arr3[2] = np.real(v.conj().dot(dm_i).dot(v))
            arr3[3] = np.abs(ise)
            container += [arr3.copy()]
    
    return container[1:]



def load_QP(qpfile):
    if '.npz' not in qpfile:
        qpfile = qpfile+'.npz'
    QP = np.load(qpfile)
    return QP

def QP_Bands(qpfile):
    QP = load_QP(qpfile)
    Es = get_last_QP_energy(QP['Evec'])
    Es = Es[QP['Conv']>0]
    ts = QP['t'][QP['Conv']>0]
    return ts, Es


def filter_duplicates(qpfile, etol = 5e-5, dottol=0.99):
    QP = load_QP(qpfile)
    idx= QP['Conv']>0
    Es =  get_last_QP_energy(QP['Evec'][idx])
    ts  = QP['t'][idx]
    vecs= QP['vec'][idx]
    iSE = get_last_QP_energy(QP['iSE'][idx])
    
    return _Filter_duplicates(Es, ts, vecs, iSE, etol, dottol)

@nb.njit
def get_last_QP_energy(Evec):
    ncalc, itmax = Evec.shape
    out   = np.zeros((ncalc), dtype = np.complex128)
    for i in range(ncalc):
        for j in range(itmax-1,-1,-1):
            if np.isnan(Evec[i,j]) == False:
                out[i] = Evec[i,j]
                break
    return out

@nb.njit
def _Filter_duplicates(Es, ts, vecs, iSE, etol, dottol):
    dup = np.zeros(len(Es))
    ns  = len(Es)
    for i in range(ns):
        Ei = Es[i]
        ti = ts[i]
        vic= vecs[i].conj()
        for j in range(i+1,ns):
            Ej = Es[j]
            tj = ts[j]
            vj = vecs[j]
            if abs(ti - tj)<1e-10:
                if abs(Ei - Ej)<etol and abs((vic.dot(vj)))>dottol:
                    dup[j] = 1
    return Es[dup<1],ts[dup<1], vecs[dup<1], iSE[dup<1]



# See siesta_python, cmdtools
# @nb.njit
# def CoulombPot(ri, rj, frontfac, lower = 0.1):
#     # 1/r potential
#     dij = np.linalg.norm(ri-rj)
#     if dij>lower:
#         return frontfac / dij
#     else:
#         return Pot_singhandle((ri+rj)/2, 0.2)

# @nb.njit
# def Pot_singhandle(Ri,dR):
#     # Dummy singularity handler
#     return 4 * np.pi * dR

# @nb.njit
# def Coulomb_integrals(S, R, Pot, stol=1e-3, frontfac = 0.25):
#     IJKL = [np.array([-1,-1,-1,-1])]
#     V  = [0.0]
#     no = S.shape[0]
#     for i in range(no):
#         for l in range(no):
#             if abs(S[i,l])>stol:
#                 Ril = (R[i] + R[l])/2
#                 for j in range(no):
#                     for k in range(no):
#                         if abs(S[j,k])>stol:
#                             Rjk = (R[j] + R[k])/2
#                             IJKL += [np.array([i,j,k,l])]
#                             V    += [Pot(Ril, Rjk, frontfac)]
#     return IJKL, V





