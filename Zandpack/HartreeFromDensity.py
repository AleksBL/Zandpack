import numpy as np
from numba import njit, objmode, jit, prange
from numba.typed import List
from Zandpack import k0nfig 
from scipy.fftpack import fftn,ifftn
FT = fftn
iFT= ifftn
from time import time


@njit
def numba_fft3(input):
    with objmode(out='complex64[:,:,:]'):
        out = FT(input)#np.fft.fftn(input)
    return out

@njit
def numba_ifft3(input):
    with objmode(out='complex64[:,:,:]'):
        out = np.fft.ifftshift(iFT(input))
    return out

@njit
def fftZP(A):
    #Nx=A.shape[0]; Ny=A.shape[1]; Nz=A.shape[2]; 
    Nx, Ny, Nz = A.shape
    t1=0; t2=0; t3=0
    if np.mod(Nx,2)==0: t1=1
    if np.mod(Ny,2)==0: t2=1
    if np.mod(Nz,2)==0: t3=1
    p00=Nx//2-t1; p01=Nx//2
    p10=Ny//2-t2; p11=Ny//2
    p20=Nz//2-t3; p21=Nz//2
    
    with objmode(out='complex64[:,:,:]'):
        out = np.pad(A,((p00,p01),(p10,p11),(p20,p21)))
    
    return out

@njit
def Pil_ud(A,Nx,Ny,Nz):
    dim=A.shape
    t1=0; t2=0; t3=0
    if np.mod(Nx,2)==0: t1=1
    if np.mod(Ny,2)==0: t2=1
    if np.mod(Nz,2)==0: t3=1
    p00=Nx//2-t1; p01=Nx//2
    p10=Ny//2-t2; p11=Ny//2
    p20=Nz//2-t3; p21=Nz//2
    return A[p00:dim[0]-p01,p10:dim[1]-p11,p20:dim[2]-p21]    

@njit
def gdv(dr,k):
    n   = np.sqrt(np.sum(dr**2,axis=-1)) + 1e-10
    res = np.exp(1j*k*n)/(4*np.pi*n)
    return res

@njit
def get_g(shape, dx):
    nx,ny,nz = shape
    x = np.arange(-nx+1, nx) * dx
    y = np.arange(-ny+1, ny) * dx
    z = np.arange(-nz+1, nz) * dx
    g = np.zeros((2 * nx - 1, 2 * ny -1, 2 * nz -1))
    r_sing = ((3/(4*np.pi))*dx**3)**(1/3)
    
    for i in range(2 * nx-1):
        for j in range(2 * ny - 1):
            for k in range(2 * nz -1):
                if i == nx-1 and j == ny-1 and k == nz -1:
                    pass
                    #g[i,j,k] = r_sing**2/dx**3
                else:
                    g[i,j,k] = 1 /(x[i]**2 + y[j]**2 + z[k]**2)
    return (g**0.5) * dx**3 /(4*np.pi) 


@njit
def Poisson(rho, g_fft):
    nx,ny,nz = rho.shape
    return  - Pil_ud( numba_ifft3(numba_fft3(fftZP(rho)) * g_fft) ,nx,ny,nz)

@njit(fastmath = True)
def intersecting_inds(a,b):
    I1 = List()
    I2 = List()
    where = List()
    for i1,v in enumerate(a):
        for i2,vv in enumerate(b):
            if vv == v:
                where.append(v)
                I1.append(i1)
                I2.append(i2)
    return I1,I2, where

@jit(parallel = False, fastmath = False)
def make_density(list_orbitals, density_matrix, orb_kind, orb_pos, dx,
                 density, static_density, tol = 1e-10, add_static = False, return_Flist = False, Flist = None,
                 Sij = None):
    pos  = (orb_pos / dx).astype(np.int32)
    rem  =  orb_pos/dx - pos
    if add_static:
        density += static_density
    
    n_orb    = len(orb_kind)
    NX,NY,NZ = density.shape
    if Flist is None:
        print('Calculating Flist')
        Fliist = List()
        #for i in range(n_orb):
        #    Flist.append(simple_interpolate(list_orbitals[orb_kind[i]], rem[i]))
        Flist = [simple_interpolate(list_orbitals[orb_kind[i]], rem[i]) for i in range(n_orb)]
    if return_Flist:
        return Flist
    
    if Sij is None: Sij = np.ones((n_orb, n_orb))
    
    for i in prange(n_orb):
        x1,x2,x3 = pos[i]
        fi       = Flist[i]
        nx,ny,nz = fi.shape
        Ix = np.arange(x1-nx//2, x1 + nx//2 +1); Ix = Ix[(Ix<NX)*(Ix>-1)]
        Jx = np.arange(x2-ny//2, x2 + ny//2 +1); Jx = Jx[(Jx<NY)*(Jx>-1)]
        Kx = np.arange(x3-nz//2, x3 + nz//2 +1); Kx = Kx[(Kx<NZ)*(Kx>-1)]
        for j in range(i, n_orb):
            y1,y2,y3 = pos[j]
            ry1,ry2,ry3 = rem[j]
            fj       = Flist[j]
            mx,my,mz = fj.shape
            
            Iy = np.arange(y1-mx//2, y1 + mx//2 +1); Iy = Iy[(Iy<NX)*(Iy>-1)]
            Jy = np.arange(y2-my//2, y2 + my//2 +1); Jy = Jy[(Jy<NY)*(Jy>-1)]
            Ky = np.arange(y3-mz//2, y3 + mz//2 +1); Ky = Ky[(Ky<NZ)*(Ky>-1)]
            
            dij = density_matrix[i,j].real
            sij = Sij[i,j]
            #print(dij.shape,sij.shape)
            if abs(dij) > tol and abs(sij)>tol:
                IIx, IIy, Wi = intersecting_inds(Ix, Iy)
                JJx, JJy, Wj = intersecting_inds(Jx, Jy)
                KKx, KKy, Wk = intersecting_inds(Kx, Ky)
                if len(Wi)>0 and len(Wj)>0 and len(Wk)>0:
                    subfi = fi[min(IIx):max(IIx)+1,min(JJx):max(JJx)+1,min(KKx):max(KKx)+1]
                    subfj = fj[min(IIy):max(IIy)+1,min(JJy):max(JJy)+1,min(KKy):max(KKy)+1]
                    if i==j:
                        fact = dij
                    else:
                        fact = 2 * dij
                    prod = subfi * subfj * fact
                    #print(density[min(Wi):max(Wi)+1, min(Wj):max(Wj)+1, min(Wk):max(Wk)+1].shape, prod.shape)
                    density[min(Wi):max(Wi)+1, min(Wj):max(Wj)+1, min(Wk):max(Wk)+1] += prod 

@jit(parallel = True, fastmath = True)
def matrixelementsoffield(list_orbitals,  orb_kind, orb_pos, dx,
                          Field, Out, Flist = None,
                          Sij = None, tol = 1e-10):
    pos  = (orb_pos / dx).astype(np.int32)
    rem  =  orb_pos/dx - pos
    
    n_orb    = len(orb_kind)
    NX,NY,NZ = Field.shape
    if Flist is None:
        print('Calculating Flist')
        Flist = [simple_interpolate(list_orbitals[orb_kind[i]], rem[i]) for i in range(n_orb)]
    
    if Sij is None: Sij = np.ones((n_orb, n_orb))
    
    for i in prange(n_orb):
        x1,x2,x3 = pos[i]
        fi       = Flist[i]
        nx,ny,nz = fi.shape
        Ix = np.arange(x1-nx//2, x1 + nx//2 +1); Ix = Ix[(Ix<NX)*(Ix>-1)]
        Jx = np.arange(x2-ny//2, x2 + ny//2 +1); Jx = Jx[(Jx<NY)*(Jx>-1)]
        Kx = np.arange(x3-nz//2, x3 + nz//2 +1); Kx = Kx[(Kx<NZ)*(Kx>-1)]
        for j in range(i, n_orb):
            y1,y2,y3 = pos[j]
            ry1,ry2,ry3 = rem[j]
            fj       = Flist[j]
            mx,my,mz = fj.shape
            
            Iy = np.arange(y1-mx//2, y1 + mx//2 +1); Iy = Iy[(Iy<NX)*(Iy>-1)]
            Jy = np.arange(y2-my//2, y2 + my//2 +1); Jy = Jy[(Jy<NY)*(Jy>-1)]
            Ky = np.arange(y3-mz//2, y3 + mz//2 +1); Ky = Ky[(Ky<NZ)*(Ky>-1)]
            
            sij = Sij[i,j]
            if  abs(sij)>tol:
                IIx, IIy, Wi = intersecting_inds(Ix, Iy)
                JJx, JJy, Wj = intersecting_inds(Jx, Jy)
                KKx, KKy, Wk = intersecting_inds(Kx, Ky)
                if len(Wi)>0 and len(Wj)>0 and len(Wk)>0:
                    subfi = fi[min(IIx):max(IIx)+1,min(JJx):max(JJx)+1,min(KKx):max(KKx)+1]
                    subfj = fj[min(IIy):max(IIy)+1,min(JJy):max(JJy)+1,min(KKy):max(KKy)+1]
                    
                    prod = subfi * subfj 
                    Fieldij  = Field[min(Wi):max(Wi)+1, min(Wj):max(Wj)+1, min(Wk):max(Wk)+1]
                    Out[i,j] = np.sum(Fieldij * prod) * dx**3
                    Out[j,i] = Out[i,j]



# @jit#(parallel = False, fastmath = False)
# def make_density_jit(list_orbitals, density_matrix, orb_kind, orb_pos, dx,
#                  density, static_density,Flist, tol = 1e-10, add_static = False):
#     pos  = (orb_pos / dx).astype(np.int32)
#     rem  =  orb_pos/dx - pos
#     if add_static:
#         density += static_density
    
#     n_orb    = len(orb_kind)
#     NX,NY,NZ = density.shape
    
#     for i in range(n_orb):
#         x1,x2,x3 = pos[i]
#         fi       = Flist[i]
#         nx,ny,nz = fi.shape
#         Ix = np.arange(x1-nx//2, x1 + nx//2 +1); Ix = Ix[(Ix<NX)*(Ix>-1)]
#         Jx = np.arange(x2-ny//2, x2 + ny//2 +1); Jx = Jx[(Jx<NY)*(Jx>-1)]
#         Kx = np.arange(x3-nz//2, x3 + nz//2 +1); Kx = Kx[(Kx<NZ)*(Kx>-1)]
#         for j in range(i, n_orb):
#             y1,y2,y3 = pos[j]
#             ry1,ry2,ry3 = rem[j]
#             fj       = Flist[j]
#             mx,my,mz = fj.shape
            
#             Iy = np.arange(y1-mx//2, y1 + mx//2 +1); Iy = Iy[(Iy<NX)*(Iy>-1)]
#             Jy = np.arange(y2-my//2, y2 + my//2 +1); Jy = Jy[(Jy<NY)*(Jy>-1)]
#             Ky = np.arange(y3-mz//2, y3 + mz//2 +1); Ky = Ky[(Ky<NZ)*(Ky>-1)]
            
#             dij = density_matrix[i,j].real
            
#             if abs(dij) > tol:
#                 IIx, IIy, Wi = intersecting_inds(Ix, Iy)
#                 JJx, JJy, Wj = intersecting_inds(Jx, Jy)
#                 KKx, KKy, Wk = intersecting_inds(Kx, Ky)
#                 if len(Wi)>0 and len(Wj)>0 and len(Wk)>0:
#                     subfi = fi[min(IIx):max(IIx)+1,min(JJx):max(JJx)+1,min(KKx):max(KKx)+1]
#                     subfj = fj[min(IIy):max(IIy)+1,min(JJy):max(JJy)+1,min(KKy):max(KKy)+1]
#                     prod = subfi * subfj
#                     _t = 1
#                     print(prod.shape, len(Wi), len(Wj), len(Wk))
#                     if i==j:
#                         density[min(Wi):max(Wi)+_t, min(Wj):max(Wj)+_t, min(Wk):max(Wk)+_t] += prod * dij
#                     if i!=j:
#                         density[min(Wi):max(Wi)+_t, min(Wj):max(Wj)+_t, min(Wk):max(Wk)+_t] += prod * 2 * dij
                    
                    

@njit
def test(l):
    return min(l)

@njit
def simple_interpolate(f, fracs):
    w1 = fracs
    return intrp(intrp(intrp(f, w1[0], 0), w1[1], 1), w1[2], 2)

@njit
def roll1(f,axis):
    res = np.zeros(f.shape)
    if axis == 0:
        res [1:,:,:] = f[:-1,:,:]
    if axis == 1:
        res [:,1:,:] = f[:,:-1,:]
    if axis == 2:
        res [:,:,1:] = f[:,:,:-1]
    return res

@njit
def intrp(f,frc, axis):
    return f*(1-frc) + roll1(f, axis) * frc

@njit(cache = k0nfig.CACHE)
def wavg1(A, w, idx):
    return A[idx,:,:]*(1-w) +A[idx+1,:,:]*w 
@njit(cache = k0nfig.CACHE)
def wavg2(A, w, idx):
    return A[:,idx,:]*(1-w) +A[:,idx+1,:]*w 
@njit(cache = k0nfig.CACHE)
def wavg3(A, w, idx):
    return A[:,:,idx]*(1-w) +A[:,:,idx+1]*w 
@njit(cache = k0nfig.CACHE)
def cube_interp(A,w3, idxs):
    return wavg3(wavg2(wavg1(A, w3[0], idxs[0]), w3[1], idxs[1]),w3[2], idxs[2])

@njit(cache = k0nfig.CACHE)
def Hartree(dsig, LO, OK,L, iL, S_ij, xyz, dx,dens, static_density, g_fft, reset_density = True,add_static=True):
    NO = len(dsig)
    r_xyz = (xyz/dx).astype(np.int32)
    frac_xyz = xyz/dx - r_xyz
    dsig_no = iL @ dsig @ iL
    #dens = np.zeros(static_density.shape, dtype = np.complex128)
    if reset_density:
        dens[:,:,:]=0.0
    make_density(LO, dsig_no, OK, xyz, dx,dens, static_density, add_static=add_static)
    V    = Poisson(dens, g_fft)
    out = np.zeros(dsig.shape, dtype = np.complex128)
    
    for i in range(NO):
        rix, riy,riz = r_xyz[i]
        out[i,i] = cube_interp(V,frac_xyz[i] , r_xyz[i])
        for j in range(i+1,NO):
            rjx,rjy,rjz = r_xyz[j]
            rijx = (rix + rjx)//2
            rijy = (riy + rjy)//2
            rijz = (riz + rjz)//2
            
            out[i,j] = V[rijx, rijy,rijz] * S_ij[i,j]
            out[j,i] = np.conj(out[i,j])
    
    return iL @ out @ iL



