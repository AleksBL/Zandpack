import numpy as np
from numba import njit
Norm = np.linalg.norm


@njit
def proj(u,v):
    return (prod(u,v) / prod(u,u)) * u
@njit
def prod(u,v):
    return u.conj().dot(v)
@njit
def _GrSch_last_and_add(Vecs, new):
    """
    Vecs: N, no
    new : no
    out: N+1, no
    """
    N, no = Vecs.shape
    s     = (N+1, no)
    out   = np.zeros(s, dtype = Vecs.dtype)
    out[0:N, :] = Vecs[:,:]
    NEW = normalise1(new)
    for i in range(N):
        NEW -= proj(Vecs[i],NEW)
    out[i+1, :] = normalise1(NEW)
    return out

def GrSch_last_and_add(Vecs, new):
    """
    Vecs: (N, no) or (nk, N, no)
    new : (no,) or (nk, no)
    out:  (N+1, no) or (nk, N+1, no), Vecs, but with an orthonormalised "new" added
    """
    
    if len(Vecs.shape) == 2:
        assert len(new.shape) == 1
        return _GrSch_last_and_add(Vecs, new)
    if len(Vecs.shape) == 3:
        assert len(new.shape) == 2
        assert new.shape[0] == Vecs.shape[0]
        s   = Vecs.shape
        Rep = np.zeros((s[0], s[1]+1, s[2]),dtype = Vecs.dtype)
        for i in range(s[0]):
            Rep[i] = _GrSch_last_and_add(Vecs[i], new[i])
        return Rep
@njit
def normalise1(v):
    return v/Norm(v)
@njit
def normalise2(v):
    out = np.zeros(v.shape,dtype=v.dtype)
    for i in range(v.shape[0]):
        out[i] = normalise1(v[i])
    return out
@njit
def Norm2(v):
    out = np.zeros(v.shape[0], dtype = v.dtype)
    for i in range(v.shape[0]):
        out[i] = np.sqrt(v[i].conj().dot(v[i]))
    return out
@njit
def Av(M,v):
    assert len(v.shape) == 2
    assert len(M.shape) == 3
    out = np.zeros(v.shape, dtype = M.dtype)
    for i in range(M.shape[0]):
        out[i,:] = M[i].dot(v[i])
    return out
@njit
def dag(v):
    if len(v.shape) == 1:
        return np.conj(v)
    if len(v.shape) == 2:
        return np.conj(v.T)

def Lanczos(M: np.ndarray, N: int, v_init = None, seed = None):
    """
    M: (nk, no,no) or (no,no)
    N: how many Lanczos vectors
    v
    returns : diagonal ai, coupling bi and orthogonal basis
    
    """
    
    if len(M.shape) == 2: _M = M[None,:,:]
    else: _M = M
    if not np.allclose(_M, _M.transpose(0,2,1).conj()):
        print('\n\n Lanczos: Input not Hermitian, algorithm does not make sense for this case!\n\n')
    
    no = _M.shape[-1]
    nk = _M.shape[0]
    if N>no:
        print('\n\n Lanczos: Requested number of Lanczos vectors exceeds matrix dimension\n\n')
        assert 1 == 0
    
    if not _M.shape[-2] == _M.shape[-1]:
        print('\n\n Lanczos: Not square matrix\n\n')
    
    if v_init is not None: 
        if len(v_init.shape) == 1: v_init = v_init[None,:]
        elif len(v_init.shape) == 2: pass
        else:
            print('\n\n Lanczos: Wrong shape of initial vector \n\n')
            assert 1 == 0
    vout = np.zeros((nk , 1, no ), dtype = _M.dtype)
    alp  = np.zeros((nk,  N,    ), dtype = _M.dtype)
    bet  = np.zeros((nk,  N-1,  ), dtype = _M.dtype)
    
    if seed is not None:   
        np.random.seed(seed)
    
    if v_init is None: vout[:,0,:] = normalise2(np.random.random((nk, no)) )
    else:              vout[:,0,:] = normalise2(v_init)
    
    wp1      = Av(_M, vout[:,0,:])
    #print(wp1.shape)
    a1       = (wp1.conj() * vout[:,0,:]).sum(axis=1)
    #print(a1.shape, vout.shape)
    w1       =  wp1 - a1[:,None]*vout[:,0,:]
    alp[:,0] = a1
    
    for m in range(1,N):
        b1   = Norm2(w1)
        #print(b1.shape)
        v1   = w1 / b1[:,None]
        vout = GrSch_last_and_add(vout, v1)
        wp1  = Av(_M, vout[:,-1,:])
        a1   = (wp1.conj()*vout[:,-1]).sum(axis=1)
        #print(wp1.shape, a1.shape, vout.shape, b1.shape)
        w1   = wp1 - a1[:,None]*vout[:,-1,:] - b1[:,None]*vout[:,-2,:]
        alp[:,m]   = a1
        bet[:,m-1] = b1
    
    # keep consistent shapes
    if len(M.shape) == 2:
        return alp[0], bet[0], vout[0]
    else:
        return alp, bet, vout

#@njit
def Lanczos_eigv(M, N):
    a, b, v = Lanczos(M, N)
    A       = np.diag(b.conj(), -1) + np.diag(a, 0) + np.diag(b, 1)
    return np.linalg.eigvalsh(A).min()







# vi = np.random.random((1, 100)) + 1j * np.random.random((1, 100))
# M = np.random.random((1,100,100)) + 1j*np.random.random((1,100,100))
# M = M + M.transpose(0,2,1).conj()
# a,b,v   = Lanczos(M, 20, v_init = vi)
# eig,vec = np.linalg.eigh(M)
# M2 = np.diag(a[0])
# for i in range(len(M2)-1):
#     M2[i,i+1] = b[0,i]
#     M2[i+1,i] = b[0,i]
# eig2,vec2 = np.linalg.eigh(M2)
# S = v @ v.conj().transpose(0,2,1)
# for i in range(25):
#     a = GrSch_last_and_add(a, np.random.random((5,200)))



 
