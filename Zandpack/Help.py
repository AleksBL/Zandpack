#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:21:17 2024

@author: investigator
"""
import numpy as np
from Zandpack.Loader import flexload
import os
from time import sleep
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def get_SigCorr(ArrayDir):
    """ArrayDir: String with a path to an Arrays/ subfolder 
       created by Zandpack
    """
    num_leads = int(np.load(ArrayDir+'/num_leads.npy'))
    files = os.listdir(ArrayDir)
    Sig0_O = []
    Sig1_O = []
    Sig0_NO = []
    Sig1_NO = []
    hcorr_name = 'Hamiltonian_renormalisation_correction.npy'
    for ie in range(num_leads):
        Sig0_O  += [flexload(ArrayDir+'/Sig0_'+str(ie)+'.npy')]
        Sig1_O  += [flexload(ArrayDir+'/Sig1_'+str(ie)+'.npy')]
        Sig0_NO += [flexload(ArrayDir+'/Sig0_NO_'+str(ie)+'.npy')]
        Sig1_NO += [flexload(ArrayDir+'/Sig1_NO_'+str(ie)+'.npy')]
    if hcorr_name in files:
        hcorr = flexload(ArrayDir + '/'+hcorr_name)
    else:
        hcorr = np.zeros(Sig0_O[0].shape,
                         dtype = np.complex128)
    return Sig0_O, Sig1_O, Sig0_NO, Sig1_NO, hcorr

class TDHelper:
    def __init__(self, Dir, orthogonal = True, valid_ranks = [0]):
        if rank not in valid_ranks:
            self.status = 'uninitialized'
            self.num_leads=None
            self.invLowdin=None
            self.Lowdin=None
            self.orb_pos=None
            return
        self.status = 'initialized'
        self.dir=Dir + '/Arrays'
        A1, A2, A3, A4, A5  = get_SigCorr(self.dir)
        self.Sig0_O  = np.array(A1)
        self.Sig1_O  = np.array(A2)
        self.Sig0_NO = np.array(A3)
        self.Sig1_NO = np.array(A4)
        self.Hcorr   = A5
        self.H0        = flexload(self.dir+'/H_Ortho.npy')
        self.DM0       = flexload(self.dir+'/DM_Ortho.npy')
        self.no        = self.H0.shape[-1]
        self.orthog    = orthogonal
        self.num_leads = len(self.Sig0_O)
        self.Lowdin    = flexload(self.dir+'/S^(-0.5).npy')
        self.invLowdin = np.linalg.inv(self.Lowdin)
        self.S         = self.invLowdin @ self.invLowdin
        self.piv       = flexload(self.dir+'/pivot.npy')
        self.positions = flexload(self.dir+'/Positions.npy')
        try:
            o2a = flexload(self.dir+'/pivot_o2a.npy')
            self.orb_pos   = self.positions[o2a]
            self.orb_elecs = [np.where((np.abs((self.Sig0_NO[ia]) + np.abs(self.Sig1_NO[ia]))>1e-7).any(axis=(0,1)))[0]
                              for ia in range(self.num_leads)]
            self.pos_elecorbs = [self.orb_pos[idx] for idx in self.orb_elecs]
            
        except:
            self.orb_pos = None
            try:
                self.orb_elecs = [np.load("read_coupling_inds_"+str(ia)+".npy")
                                  for ia in range(self.num_leads)]
            except:
                print("Didnt suceeed in reading any electrode positions... Giving random positions")
                self.orb_elecs = [np.array([0]), np.array([self.no-1])]
    
    def approxfield2mat(self, t, field, custom_o2a = None, orthogonal=True):
        if self.orb_pos is None and custom_o2a is None:
            if not hasattr(self, "orb_pos_warning"):
                print("No orbital positions have been given, cant calculate the approximate matrix of your field! :( ")
                print("Try to specify the custom_o2a in the TDHelper class...")
                self.orb_pos_warning = "warning"    
            return np.zeros(self.H0.shape, dtype=complex)
        else:
            no  = self.H0.shape[-1]
            fiv = np.zeros(no,dtype=complex)
            for i in range(no):
                fiv[i] = field(self.orb_pos[i], t) * 0.5
            out = (fiv[None,:,None]+fiv[None,None,:]) * self.S
            if orthogonal:
                return self.Lowdin @ out @ self.Lowdin
            else:
                return out
    
            
    def lowdin_transform(self, A):
        """ returns S^(-1/2) @ A @ S^(-1/2)
        """
        return self.Lowdin @ A @ self.Lowdin
    def inv_lowdin_transform(self, A):
        """ returns S^(1/2) @ A @ S^(1/2)
        """
        return self.invLowdin @ A @ self.invLowdin
    def bare_H0(self, orthogonal = False):
        """ The Hamiltonian without the lead-device orthogonalisation terms and
            the correction to H (from using the Renormalise_H function in the TD_Transport class).
            The Hamiltonian that comes out of this function 
            with orthogonal=False is directly comparible with the device-part
            of the Hamiltonian from a DFT-calculation. If a linear expansion
            of the Hamiltonian is done, this is useful.
            
            orthogonal indicates if the the output is represented in the 
            orthogonal basis or not (True or False).
        """
        if orthogonal:
            return self.H0 - self.Sig0_O.sum(axis=0) - self.Hcorr
        else:
            return  self.inv_lowdin_transform(self.H0) \
                  - self.Sig0_NO.sum(axis=0) \
                  - self.inv_lowdin_transform(self.Hcorr)
        
    def lead_dev_dyncorr(self, DeltaList = None, orthogonal = True):
        if DeltaList is None:
            DeltaList = [0.0, ]* self.num_leads
        DeltaList = np.array(DeltaList)
        if orthogonal:
            return np.dot(self.Sig1_O.transpose(1,2,3,0),  -DeltaList)
        else:
            return np.dot(self.Sig1_NO.transpose(1,2,3,0), -DeltaList)
    
    def __str__(self):
        specific_info = '------------------------------------\n'
        specific_info+= '------------------------------------\n'
        if self.status == 'initialized':
            specific_info+= 'directory = '+str(self.dir)
            specific_info+= '\nnum electrodes = '+str(self.num_leads)
            specific_info+= '\northogonal: '+str(self.orthog) + '\n'
        specific_info+= '------------------------------------\n'
        specific_info+= '------------------------------------\n'
        specific_info+= helpstring
        specific_info+= '------------------------------------\n'
        specific_info+= '------------------------------------\n'
        return specific_info
    
    def print_help(self):
        print(helpstring)
        
helpstring = \
"""
    This class is intended for helping with:
    ** The transformation between the orthogonal and non-orthogonal 
       basis of the device.
    ** The dynamical correction stemming from the transformation that is used 
       to orthogonalise the device basis from the electrode (/lead) bases
       
    ** Implementation of DFT.
       
    Common transformations are: 
        * Transformation from the non-orthogonal Hamiltonian to the Hamiltonian 
          in the orthogonal basis. This happens as
              H^O = S^(-1/2) @ H^NO @ S^(-1/2)
        * Transformation from the non-orthogonal density matrix to the orthogonal
          density matrix. The density matrix transforms the opposite way of the 
          Hamiltonian, as
              Sig^O = S^(1/2) @ Sig^NO @ S^(1/2)
"""



## Simple set of functions that communicates 
## through text. 

#DFT Obj side:
newDM_file = 'NEWDM.txt'
newH_file  = 'NEWH.txt'
stop_file  = 'STOP.txt'
sleep_time_1 = 1e-2
sleep_time_2 = 1e-2

def check_for_new_dm():
    files = os.listdir()
    if newDM_file in files:
        sleep(sleep_time_1)
        os.remove(newDM_file)
        return True
    return False
def signal_new_H():
    with open(newH_file,'w') as f:
        f.write('some text')
        f.close()
    print('New H!')
def check_for_stop():
    files = os.listdir()
    if stop_file in files:
        return True
    return False
# Bias.py / Calculator side

def signal_new_dm():
    with open(newDM_file,'w') as f:
        f.write('NEW')

def wait_for_new_H():
    cond = True
    while cond:
        f = os.listdir()
        sleep(0.01)
        if newH_file in f:
            cond = False # break loop
    sleep(0.01)
    return

def remove_NEWH():
    os.remove('NEWH.txt')


