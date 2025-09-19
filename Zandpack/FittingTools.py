import numpy as np
import sys
sys.path.append(__file__[:-15])
from plot import plt


def rattle_lorentzians(L, mag_E = 0.1, mag_G = 0.1, ik = None, il = None, seed = None):
    """
        Random nudge of the Lorentzian L
        L: Lorentzian Block-matrix
        mag_E: positive number
        mag_G: postive number
        ik:   number or numpy-indices for the k-points to rattle
        il:   indices of the particular lorentzian linewidth function to rattle
        seed: int for numpy.random.seed
    """
    s = L.ei.shape
    if seed is not None:
        np.random.seed(seed)
    if ik is None:
        idx_k = np.arange(s[0])
    else:
        idx_k = np.array(ik)
    if il is None:
        idx_l = np.arange(s[1])
    else:
        idx_l = np.array(il)
    rnd_ei = (np.random.random((len(idx_k), len(idx_l)))-0.5)*mag_E
    rnd_gi = (np.random.random((len(idx_k), len(idx_l)))-0.5)*mag_G
    L.ei   [idx_k[:, None], idx_l[None, :]] += rnd_ei
    L.gamma[idx_k[:, None], idx_l[None, :]] += rnd_gi

def f_space(f, x0, x1, dx):
    xl = [0+x0]
    while xl[-1] < x1:
        xl+=[(xl[-1] + dx/f(xl[-1]))]
    return np.array(xl)

class FitLog:
    def __init__(self):
        self.Pair_list = []
    def save(self, TDT):
        Pairs = []
        for Ls in TDT.NO_fitted_lorentzians:
            Pairs += [(Ls.ei.copy(), Ls.gamma.copy())]
        self.Pair_list.append(Pairs)
    def get_previous_fit(self, i = -1):
        return self.Pair_list[i]

def file_from_lorentzian(Ei, Gi, name = 'L_file', decimals = 8, min_tol =  None):
    ei  = np.round(Ei.copy(),decimals)
    gi  = np.round(Gi.copy(), decimals)
    if min_tol is None:
        min_tol = np.zeros(ei.shape)
    nk = len(ei)
    nl = len(ei[0])
    arr = np.zeros((4 * nk, nl))
    for i in range(nk):
        arr[4*i  ,:] = ei[i,:]
        arr[4*i+1,:] = gi[i,:]
        arr[4*i+2,:] = min_tol[i,:]
        arr[4*i+3,:] = np.nan
    np.savetxt(name+".csv", np.round(arr.T,decimals), delimiter=",  ", fmt = '%.'+str(decimals)+'f')

def lorentzian_from_file(fname):
    _fname = ""+fname
    if '.csv' not in fname:
        _fname = fname+'.csv'
    res = np.genfromtxt(_fname, delimiter = ',  ').T
    
    nl = len(res[0,:])
    nk = res.shape[0]//4
    ei = np.zeros((nk,nl))
    gi = np.zeros((nk,nl))
    mt = np.zeros((nk,nl))
    for i in range(nk):
        ei[i] = res[4*i  , :]
        gi[i] = res[4*i+1, :]
        mt[i] = res[4*i+2, :]
    return ei, gi, mt
    
    
def find_correction(R,Emin = -10000.0, Emax = 10000.0, linear_part = False, Ei = 0.0,Wi = 3.0):
    # from Block_matrices.Croy import KK_L_sum
    from scipy.optimize import curve_fit
    
    # R:  Fitted TDT Object
    # out A matrix with shape (nk,1, no,no)
    idx = R.sampling_idx[0]
    if hasattr(R, '_old_sampling_idx'):
        idx = R._old_sampling_idx[0]
    
    idx = idx[np.where((R.Contour[idx]>Emin)*R.Contour[idx]<Emax)]
    Eg  = R.Contour[idx].real
    L = R.bs2np(R.Lowdin[0])
    m1  = sum([R.bs2np(_se)[:,idx]
              for _se in R.self_energies])
    m1  = (m1 + m1.transpose(0,1,3,2).conj())/2
    
    SEs        =  R.SE_from_lorentzian_fit(R.Contour[idx], NO = True)
    #m2  =  R.bs2np(SEs[0]) + R.bs2np(SEs[1])
    m2         = sum([R.bs2np(ses) for ses in SEs])
    m2         = (m2 + m2.transpose(0,1,3,2).conj())/2
    mf         = m1-m2
    correction = np.average(mf, axis = 1)
    if linear_part == False:
        return L@correction[:, None]@L
    
    C = np.zeros(correction.shape, dtype = complex)
    X = np.zeros(correction.shape, dtype = complex)
    Bol = (np.abs(mf)>1e-10).any(axis=(0,1))
    I,J = np.where(Bol)
    
    for ik in range(C.shape[0]):
        for c in range(len(I)):
            i,j = I[c], J[c]
            def func(x, A):#,B):
                pars = np.zeros((3,1))
                pars[0] = Wi
                pars[1] = Ei
                #pars[2] = B
                return (A )#+ KK_L_sum(x, pars)).real
            poptr, pcov = curve_fit(func, Eg, mf[ik,:,i,j].real, p0=[C[ik,i,j].real]#,0]
                                    )
            
            def func(x, A):#,B):
                pars = np.zeros((3,1))
                pars[0] = Wi
                pars[1] = Ei
                #pars[2] = B
                return (A )#+ KK_L_sum(x, pars)).real
            popti, pcov = curve_fit(func, Eg, mf[ik,:,i,j].imag, p0=[C[ik,i,j].imag]#,0]
                                    )
            # if (i,j) == (8,8):
            #     x = np.linspace(-1,1,1000)
            #     plt.plot(Eg.real, m1[ik,:,i,j].real, label = 'sampled')
            #     plt.plot(Eg.real, 7*m2[ik,:,i,j].real, label = 'fit')
            #     plt.plot(Eg.real, mf[ik,:,i,j].real, label = 'difference')
            #     plt.plot(x, func(x,poptr[0], poptr[1]), label = 'fit diff')
            #     plt.legend()
            #     plt.xlim(-1,1)
            #     plt.show()
                
            C[ik,i,j] = poptr[0] + 1j *popti[0]
            #X[ik,i,j] = poptr[1] + 1j *popti[1]
    
    return L@C@L#,L@X@L


    
    





def piecewise_linspace(Ev, n):
    assert len(n)==len(Ev)-1
    N = len(n)
    ints = []
    for i in range(N):
        dE = Ev[i+1]-Ev[i]
        ints += [np.linspace(Ev[i],Ev[i+1]-dE/n[i], n[i])]
    return np.hstack(ints)









