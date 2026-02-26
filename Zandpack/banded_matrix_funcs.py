#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 13:20:19 2026

@author: aleks
"""
import numpy as np
from scipy.linalg import bandwidth

def bandedmatvec(M, v, out, min_nb, skip_bw=False):
    _bl,_ = bandwidth(M)
    _bl = _bl.max()
    NO  = M.shape[-1]
    nblocks = NO // _bl
    if nblocks < min_nb:
        np.matmul(v, np.expand_dims(M.transpose((0,2,1)),(1, 2)), out=out)
        return
    idx = np.array_split(np.arange(NO), nblocks)
    for ihp in range(nblocks):
        _sd  = idx[ihp]
        comb_idx = idx[ihp].copy()
        if ihp != 0:
            comb_idx = np.hstack((idx[ihp - 1], comb_idx))
        if ihp != (nblocks-1):
            comb_idx = np.hstack((comb_idx, idx[ihp + 1]))
        if len(comb_idx)<1 or len(_sd)<1:
            assert 1 == 0
        Mat = M[:, _sd[:,None], comb_idx[None,:]]
        out[..., _sd] = np.matvec(Mat, v[..., comb_idx])

def bandedmatmat(M, K, out, min_nb, skip_bw):
    """ If the matrix has a narrow bandwidth, this function will be more 
    efficient  when computing the product M @ K. As a rule of thumb,
    the nblocks (see code body , (NO // bw)) should be greater than 5 
    for this function to be more efficient. The value min_nb determines when 
    the function switches to the normal numpy.matmul function instead of 
    picking out slices of the matrix M.
    """
    if skip_bw:
        np.matmul(M, K, out=out)
        return
    _bl,_ = bandwidth(M)
    _bl = _bl.max()
    NO  = M.shape[-1]
    nblocks = NO // _bl
    if nblocks < min_nb:
        np.matmul(M, K, out=out)
        return
    idx = np.array_split(np.arange(NO), nblocks)
    for ihp in range(nblocks):
        _sd  = idx[ihp]
        comb_idx = idx[ihp].copy()
        if ihp != 0:
            comb_idx = np.hstack((idx[ihp - 1], comb_idx))
        if ihp != (nblocks-1):
            comb_idx = np.hstack((comb_idx, idx[ihp + 1]))
        if len(comb_idx)<1 or len(_sd)<1:
            assert 1 == 0
        Mat = M[... , _sd[:,None], comb_idx[None,:]]
        T2  = K[... , comb_idx, :]
        out[... , _sd, :] = np.matmul(Mat, T2)



I_WANT_TO_RUN_TESTS=False
#Tests for errors
if I_WANT_TO_RUN_TESTS:
    from time import time
    import matplotlib.pyplot as plt
    
    L_params   = []
    L_Mv1  = []
    L_Mv2  = []
    L_bMv = []
    L_MM = []
    L_bMM = []
    min_NB = 4
    for irun in range(200):
        print(irun)
        NO = np.random.randint(50,2000)
        BW = np.random.randint(20,500)
        # if BW>NO:
        #     BW = NO - 2
        
        psi     = np.random.random((1,2, 55, 7, NO)) + 0.0j
        psiout1 = np.zeros((1,2, 55, 7, NO)) + 0.0j
        psiout2 = np.zeros((1,2, 55, 7, NO)) + 0.0j
        Ht  = np.random.random((1, NO, NO))*10.0 + 0.0j
        DM  = np.random.random((1, NO, NO)) + 0.0j
        for i in range(NO):
            diff = np.abs(np.arange(NO) - i)
            idx  = np.where(diff>BW + np.random.randint(0,5))
            Ht[:, i, idx] = 0.0
        _bl = bandwidth(Ht)[0].max()
        nblocks = NO // _bl
        tv1 = time()
        np.matmul(psi, np.expand_dims(Ht.transpose((0,2,1)),(1, 2)), out=psiout1)
        tv2 = time()
        np.matvec(Ht, psi, out=psiout2)
        tv3 = time()
        assert np.allclose(psiout1, psiout2)
        tmp_psi_summer_2 = np.zeros((1,2, 55, 7, NO)) + 0.0j
        tv4 = time()
        bandedmatvec(Ht, psi, tmp_psi_summer_2, min_NB, False)
        tv5 = time()
        assert np.allclose(psiout1, tmp_psi_summer_2)
        res2= np.zeros(Ht.shape, dtype=complex)
        tm1 = time()
        res1= Ht @ DM
        tm2 = time()
        bandedmatmat(Ht, DM, res2, min_NB, False)
        tm3 = time()
        assert np.allclose(res1, res2)
        
        L_params.append((nblocks, BW, NO))
        L_Mv1.append(tv2-tv1)
        L_Mv2.append(tv3-tv2)
        L_bMv.append(tv5-tv4)
        L_MM.append(tm2-tm1)
        L_bMM.append(tm3-tm2)
        if np.mod(irun, 10) == 0:
            plt.scatter(np.array(L_params)[:,0], 
                        np.log10(np.array(L_MM)/np.array(L_bMM)), 
                        label = 'Matmul speedup', color='r')
            plt.scatter(np.array(L_params)[:,0], 
                        np.log10(np.array(L_Mv1)/np.array(L_bMv)), 
                        label = 'Matvec speedup', color='g')
            plt.hlines(0.0, 0, np.array(L_params)[:,0].max())
            plt.xlabel('# blocks')
            plt.ylabel('Log Speedup')
            plt.legend()
            plt.show()

    L_params = np.array(L_params)
    L_Mv1 = np.array(L_Mv1)
    L_Mv2 = np.array(L_Mv2)
    L_bMv = np.array(L_bMv)
    L_MM  = np.array(L_MM)
    L_bMM = np.array(L_bMM)