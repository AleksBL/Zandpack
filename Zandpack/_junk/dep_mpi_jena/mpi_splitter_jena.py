#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 12:06:22 2022

@author: aleksander
"""
import numpy as np
from k0nfig import Check_partition_scheme as _Check
from k0nfig import Supress_parallel_K as SupK


def get_sources_jena_scheme(nw):
    assert nw > 1
    return [i for i in range(nw)][1:]

def find_max_orbital_idx_per_lead_jena(GG_P, GL_P, zero_tol, fast=False):
    """
    Returns the max orbital index where all following eigenvalues are 
    below the given tolerance.
    Used in the Zand code
    
    """
    nk, nlead, nc, nl, nf =  GG_P.shape
    psi_idx               = -np.ones((nlead, nc, nl, nf),dtype = np.int64)
    it_2                  = 0
    
    for a1 in range(nlead):
        for ic in range(nc):
            for iL in range(nl):
                for iF in range(nf):
                    if np.abs(GG_P[:,a1,ic,iL,iF]).max()<zero_tol and np.abs(GL_P[:,a1,ic,iL,iF]).max()<zero_tol:
                        if fast:
                            continue
                        else:
                            pass
                    else:
                        psi_idx[a1,ic,iL,iF] = it_2
                        it_2 += 1
    l = []
    for i in range(nlead):
        l += [np.where(psi_idx[i]>-1)[0].max()+1]
    noT_lead = np.array(l)
    return noT_lead

def _helper_partition_scheme(nw, nk, na, nl, nf, noT_lead):
    """
    Splits the leads and poles and eigenindices 
    into chukcs to be assigned to the worker nodes in
    the Zand code.
    
    
    """
    assert nw>na
    NW = nw - 1
    def to_int(f):
        if int(f)==0:
            return int(f)+1
        else:
            return int(f)
    
    lead_procs = [to_int(NW * noT_lead[_i]/noT_lead.sum()) for _i in range(len(noT_lead))]
    tot_procs  = sum(lead_procs)
    diff       = NW - tot_procs
    lead_procs[lead_procs.index(max(lead_procs))]+= diff
    
    cumsum     = np.cumsum(lead_procs)
    lead_pool  = [np.arange(0,lead_procs[0])]
    for i in range(len(lead_procs)-1):
        lead_pool += [np.arange(cumsum[i], cumsum[i+1])]
    
    pole_split_leads = [np.array_split(np.arange(0,nl), len(lead_workers)) 
                        for lead_workers in lead_pool]
    
    count = 0
    out   = []
    fpoles = np.arange(0,nf)
    for ia, p in enumerate(pole_split_leads):
        for pp in p:
            count += 1
            out   += [[ia, np.arange(0, noT_lead[ia]), pp, fpoles,]]
    # Sort according to computational load.
    # The worker with the fewest orbitals and poles
    # will be done first. This worker should be first in
    # line to send its data to the master node
    priority  = np.argsort([len(o[1]) * len(o[3]) for o in out])
    out2      = [out[i] for i in priority]
    return out2

def partition_jena_scheme(i,nw, nk, na, nl, nf, noT_lead, eigsplit=1):
    out = _helper_partition_scheme(nw, nk, na, nl, nf, noT_lead)
    M   = ['Sigma', 'Worker']
    K   = np.arange(nk).astype(int)
    L   = np.arange(na).astype(int)
    X   = np.arange(nl).astype(int)
    F   = np.arange(nf).astype(int)
    orb = np.arange(0, noT_lead.max()+1)
    
    if i == 0: 
        return [M[0],          K, L, X,F,orb ]
    else:
        out = out[i-1]
        return [M[1]+str(i-1), K, np.array([out[0]]), out[1], out[2], out[3]]



# # Check that runs every time the code is called 
# # and fails if the partition scheme does not work
# if _Check:
#     k_checks  = [1,2,3,5]
#     l_checks  = [1,2,3,4]
#     x_checks  = [3,10,11,32,49,52,61,215]
#     nw_checks = [2,3,4,7,8,12,16,20,24,28,32,40,48,56,112]
#     nw_checks2= [7,8,12,16,20,24,28,32,40,48,56,112]
    
#     from numba import njit
#     @njit
#     def ZCount(z, ki, li, xi):
#         for k in ki:
#             for l in li:
#                 for x in xi:
#                     z[k,l,x] +=1
    
#     for iw in nw_checks:
#         for ik in k_checks:
#             for il in l_checks:
#                 for ix in x_checks:
#                     zeros = np.zeros((ik,il,ix), dtype = int)
#                     ident_check = []
#                     info = [partition_scheme2 (jj, iw, ik, il, ix) for jj in range(1, iw)]
#                     for v in info:
#                         _kidx = v[1]
#                         _lidx = v[2]
#                         _xidx = v[3]
#                         ZCount(zeros,_kidx, _lidx, _xidx)
                    
#                     if not (zeros==1).all():
#                         print('Partitioning failed with (iw, ik, il, ix)',iw,ik,il,ix)
#                         assert 1 == 0
                    
#                     if iw in nw_checks2:
#                        # print(iw, il,ix)
#                         zeros    = np.zeros((ik,il,ix), dtype = int)
#                         noT_lead = np.random.randint(1,45, (il, ))
#                         info     = [partition_scheme3 (jj, iw, ik, il, ix, noT_lead) for jj in range(1, iw)]
                        
#                         for v in info:
#                             _kidx = v[1]
#                             _lidx = v[2]
#                             _xidx = v[3]
#                             ZCount(zeros,_kidx, _lidx, _xidx)
                        
#                         if not (zeros==1).all():
#                             print('Partitioning failed with (iw, ik, il, ix)',iw,ik,il,ix)
#                             assert 1 == 0
                    
                    
                    
    
#     print('Partitions Checked!')
    

    
    
    
    
    
    
    
    

# def partition(i, nw, nk, nl, nx):
#     M = ['MASTER','PI', 'DPSI', 'DOMG']
    
#     K = np.arange(nk).astype(int)
#     L = np.arange(nl).astype(int)
#     X = np.arange(nx).astype(int)
    
#     if nw==4:
#         if i==0: return [M[0]    , K,L,X]
#         if i==1: return [M[1]+'0', K,L,X]
#         if i==2: return [M[2]+'0', K,L,X]
#         if i==3: return [M[3]+'0', K,L,X]
#     if nw==7:
#         if i==0: return [M[0], K,L,                X          ] # MASTER/ODE
        
#         if i==1: return [M[1]+'0', K,L[0:nl//2],   X          ] # PI_1
#         if i==2: return [M[1]+'1', K,L[nl//2: ],   X          ] # PI_2
        
#         if i==3: return [M[2]+'0', K,L,            X[0:nx//2] ] # DPSI_1
#         if i==4: return [M[2]+'1', K,L,            X[nx//2: ] ] # DPSI_2
        
#         if i==5: return [M[3]+'0', K,L,            X[0:nx//2] ] # DOMG_1
#         if i==6: return [M[3]+'1', K,L,            X[nx//2: ] ] # DOMG_2
    
#     if nw==13:
        
#         if i==0: return  [M[0],     K,         L,            X          ] # MASTER/ODE
        
#         if i==1: return  [M[1]+'0', K,L[0:nl//2],   X[0 :nx//2]          ] # PI_1
#         if i==2: return  [M[1]+'1', K,L[nl//2: ],   X[0 :nx//2]          ] # PI_2
#         if i==3: return  [M[1]+'2', K,L[0:nl//2],   X[nx//2:nx]          ] # PI_3
#         if i==4: return  [M[1]+'3', K,L[nl//2: ],   X[nx//2:nx]          ] # PI_4
        
#         if i==5: return  [M[2]+'0', K,         L[0:nl//2],   X[0:nx//2] ] # DPSI_1
#         if i==6: return  [M[2]+'1', K,         L[0:nl//2],   X[nx//2: ] ] # DPSI_2
#         if i==7: return  [M[2]+'2', K,         L[nl//2: ],   X[0:nx//2] ] # DPSI_3
#         if i==8: return  [M[2]+'3', K,         L[nl//2: ],   X[nx//2: ] ] # DPSI_4
        
#         if i==9 : return [M[3]+'0', K,         L[0:nl//2],   X[0:nx//2] ] # DOMG_1
#         if i==10: return [M[3]+'1', K,         L[0:nl//2],   X[nx//2: ] ] # DOMG_2
#         if i==11: return [M[3]+'2', K,         L[nl//2: ],   X[0:nx//2] ] # DOMG_3
#         if i==12: return [M[3]+'3', K,         L[nl//2: ],   X[nx//2: ] ] # DOMG_4
    
#     # If no match is found we throw an obscure message + error to think about
#     print('No Profile for the wanted number of processes. Make your own profile in the splitter file!')
#     assert 1 == 0

    
    
