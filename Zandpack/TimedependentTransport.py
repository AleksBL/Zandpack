import os
import sys
sys.path.append(__file__[:-26])
import numpy as np
from siesta_python.siesta_python import SiP
import sisl
from Gf_Module.Gf import read_SE_from_tbtrans, read_overlap_data, pivot_and_sub
from Gf_Module.Gf import Greens_function_olead as gfunc
from Block_matrices.Block_matrices import Blocksparse2Numpy, block_sparse
from Block_matrices.Block_matrices import multisort_eigval as Sorted_Eig
from time import time
from scipy.integrate import solve_ivp as ode
import k0nfig
from td_constants import hbar, plancks_const, electron_charge
from PadeDecomp import Pade_poles_and_coeffs, Hu_poles, FD_expanded 
from Block_matrices.Croy import evaluate_Lorentz_basis_matrix
from scipy.signal import find_peaks
from plot import plt
from scipy.optimize import curve_fit
from Block_matrices.Croy import L_sum,KK_L_sum
from tqdm import tqdm
from docstrings import CiteString

if k0nfig.GPU:
    import cupy as cp

if k0nfig.NUMBA:
    from numba import njit, prange

# This file contains the main class for the timedependent transport.

class TD_Transport:
    # cite Croy & Saalman & Popescu papers!
    def __init__(self, elec_geoms, device_geom, mu_i = [0.0, 0.0], kT_i =[0.1, 0.1]):
        """
        elec_geoms  : list of sisl geometries for electrodes
        device_geom : sisl geometry of device region
        mu_i        : list of chemical potentials
        kT_i        : list of tempeatures in meV 
        """
        self.num_leads   = len(elec_geoms)
        self.elec_geoms  = [sisl_replica(eg) for eg in elec_geoms]
        self.device_geom = sisl_replica(device_geom)
        self.mu_i = mu_i
        self.kT_i = kT_i
    
    def Make_Contour(self, sampling, N_F, tol = 1e-6, eps = 1e-12, 
                     pole_mode = 'JieHu2011', save_pics = False,
                     Emin = None, Emax = None, tol_aaa = None):
        """ sampling  :  array of shape (N_lead, N_E) and is the line on which the 
                         W_alc is gonna get fitted on. 
            N_F       : is the number of Pade poles of the Fermi-distibution of lead
                        alpha. (int)
            pole_mode : string, JieHu2011 or Croy2009.
            save_pics : bool, save a plot of the fermi function if true.
            tol and eps controls when the points are close enough to each other
            to count as one.
            "line" contains the points that tbtrans will sample and must 
            be all the needed points for our calculation 
            Here, replicas of the same energy for different leads are counted 
            as one. One contour that combines the samplings given for each lead
            is given to TBtrans.
            """
        
        if pole_mode == 'Croy2009':
            zi, ci = Pade_poles_and_coeffs( N_F ) # Routine for expansion from Croy 2009 appendix 
        if pole_mode == 'JieHu2011':
            zi, ci = Hu_poles( N_F )
        # if pole_mode == 'AAA':
        #     zi, ci = AAA_poles(Emin, Emax, tol)
        F_poles = np.zeros((self.num_leads, 2 * N_F), dtype = np.complex128)
        
        
        
        self.zi = zi
        self.coeffs_fermi = ci.copy()
        self.num_poles = N_F
        self.sampling = sampling
        
        for a in range(self.num_leads):
            for i in range(N_F):
                F_poles[a,i]     = self.mu_i[a] + zi[i]*self.kT_i[a] # The poles of the expanded fermifunction on the upper
                F_poles[a,N_F+i] = self.mu_i[a] - zi[i]*self.kT_i[a] # and lower parts of the complex plane
        
        self.F_poles = F_poles
        
        # Save picture, not really used for anything
        if save_pics:
            for a in range(self.num_leads):
                E_grid = np.linspace(self.mu_i[a]-10, self.mu_i[a]+10, 1000)
                plt.plot(E_grid, FD_expanded(E_grid, zi, 1/self.kT_i[a], self.mu_i[a], coeffs = ci).real)
            plt.savefig('Pade_Expanded_Fermi_functions.png')
            plt.close()
            
        if sampling.shape[0] != self.num_leads or F_poles.shape[0] != self.num_leads: 
            print('Give as many sets of poles as there are leads!\n')
            assert  1 == 0
        
        # Here the various points are collected in one countour.
        # The added "eps" is a really small number that makes sorting easier,
        # because tbtrans sorts the output values after the real part,
        # and tosses away the imaginary part.
        
        line    = sampling[0].copy()
        poles   = F_poles[0].copy() + (np.arange(len(F_poles[0])) + 1) * eps
        pole_it = len(F_poles[0]) + 2
        for a in range(1,self.num_leads):
            for p in sampling[a,:]:
                if (np.abs(line - p) > tol).all():
                    line = np.hstack((line, p))
            for p in F_poles[a,:]:
                if (np.abs(line - p) > tol).all():
                    poles = np.hstack((poles, p + eps * pole_it))
                    pole_it += 1
        
        line = np.sort(np.hstack((line, poles)))
        
        # we make some indexes so we are able to pick out the points on the contour
        # for each lead
        
        sampling_idx = np.zeros( sampling.shape , dtype = np.int32)
        F_poles_idx  = np.zeros( F_poles.shape  , dtype = np.int32)
        sampling_idx[:,:] = -1
        F_poles_idx [:,:] = -1
        
        for a in range(self.num_leads):
            for i in range(len(sampling[a, :])):
                idx = np.where((np.abs(line - sampling[a,i]) <= tol))[0][0]
                sampling_idx[a,i] = idx
            
            for i in range(len(F_poles[ a, :])):
                idx = np.where((np.abs(line - F_poles[a,i])  <= tol))[0][0]
                F_poles_idx[a,i]  = idx
        
        # sanity checks
        assert (F_poles_idx  >= 0).all()
        assert (sampling_idx >= 0).all()
        assert np.allclose(line[F_poles_idx], F_poles)
        assert np.allclose(line[sampling_idx], sampling)
        
        self.Contour      = line
        self.F_poles_idx  = F_poles_idx
        self.sampling_idx = sampling_idx
    
    def Electrodes(self, 
                   names = ['EM', 'EP' ], 
                   semi_infs  = ['-a1','+a1'],
                   kp = [[50,1,1],[50,1,1]],
                   basis = 'SZ',
                   pp_folder = '../pp',
                   overwrite = True,
                   mpi = ''
                   ):
        
        """ Constructs the DFT calculator objects for the electrodes 
            stripped down version, more flags could be added
            This routine is circumventable by assigning 
            self.Device manually (see tutorials). If doing so,
            be sure to give the specify save_SE=True in the 
            initialization of the device instance. 
        """
        
        assert len(names) == len(semi_infs)
        assert len(names) == len(kp)
        assert len(names) == self.num_leads
        
        elecs = []
        for e in range(self.num_leads):
            n = names[e]
            geom = self.elec_geoms[e]
            kT   = self.kT_i[e]
            d  = semi_infs[e]
            k = kp[e]
            
            elecs.append( 
                          SiP(geom.cell, geom.xyz, geom.toASE().numbers, 
                              directory_name = n, sl = n, sm = n,
                              electronic_temperature_mev=kT * 1000,    # in meV
                              overwrite = overwrite,
                              basis = basis,
                              pp_path = pp_folder,
                              semi_inf = d,
                              mpi = mpi,
                              kp = k,
                              dm_tol = '1.d-7')
                         )
        
        self.elecs = elecs
    
    def make_device(self, 
               name = 'Device', 
               solution_method = 'transiesta',
               pp_folder = '../pp',
               k = [1,1,1],
               basis = 'SZ',
               k_tbtrans = [1,1,1],
               elec_inds = None,
               overwrite = True,
               Print = False,
               mpi = '',
               dm_tol = '1.d-7'
               #chem_pots = self.mu_i,#[0.0,0.0],
               ):
        if Print:
            pc = True
            pm = False
        else:
            pc = False
            pm = False
        
        """
        siesta/transiesta object for device
        Again a stripped down version
        This routine is circumventable by assigning 
        self.Device manually(see tutorials)
        If doing so, be sure to give the specify save_SE=True in the 
        initialization of the device instance.
        """
        
        self.Device = SiP(self.device_geom.cell, 
                          self.device_geom.xyz, 
                          self.device_geom.toASE().numbers,
                          directory_name = name,
                          solution_method = solution_method,
                          pp_path = pp_folder,
                          overwrite = overwrite,
                          basis = basis,
                          kp = k,
                          max_scf_it=2000,
                          elec_inds = elec_inds,
                          mpi = mpi,
                          custom_tbtrans_contour = self.Contour,
                          Chem_Pot = [0.0]*len(self.elecs),
                          # self.mu_i,
                          # chem_pots,
                          # self.mu_i.copy(),
                          kp_tbtrans = k_tbtrans,
                          elecs = self.elecs,
                          save_SE = True,
                          print_console=pc,
                          print_mulliken=pm)
        
        if elec_inds is None:
            self.Device.find_elec_inds()
    
    def run_electrodes(self,manual_H = None, parallel_k = False):
        """runs the electrode calculations
           You can also do these steps manually and assign the electrode attribute yourself.
           (See the docstring of make device)
        """
        fois_gras_H = manual_H
        for i,e in enumerate(self.elecs):
            e.fdf(0.0)
            if parallel_k:
                e.set_parallel_k()
            
            if fois_gras_H is not None:
                e.fois_gras(fois_gras_H[i])
            else:
                e.run_siesta_electrode_in_dir()
    
    def run_device(self, morestuff = [], where = 'DEFAULT', tbtrans = True, manual_H = None, manual_k = None):
        """runs the device calculation
           You can also do this manually and assign the Device attribute yourself.
        """
        
        Dev = self.Device
        Dev.fdf(eta = 0.0)
        if manual_k is not None:
            w = np.ones(len(manual_k)) * 1/len(manual_k)
            Dev.manual_k_points(manual_k, w, tbtrans = True)
        
        if len(morestuff)>0:
            Dev.write_more_fdf(morestuff,name = where)
        fois_gras_H = manual_H
        if fois_gras_H is not None:
            Dev.fois_gras(fois_gras_H)
        else:
            Dev.run_siesta_in_dir()
            Dev.run_analyze_in_dir()
        Dev.run_tbtrans_in_dir(DOS_GF = True)
    
    def run_device_non_eq(self,Vi, morestuff = [], where = 'DEFAULT', tbtrans = True, mpi = ''):
        """
        Runs the nonequilibrium calculation with voltages Vi
        Vi should be given in absolute ascending order with
        a zero at the starting position
        
        This is not strictly necessary to do unless you want
        to interpolate the device hamiltonian between voltages
        
        Vi is just an iterable with numbers in it
        """
        done_vi      = [0.0]
        done_calc    = [self.Device]
        hamiltonians = []
        
        for vi in Vi:
            Calc  =   SiP(self.Device.lat, 
                          self.Device.pos_real_space, 
                          self.Device.s,
                          directory_name = self.Device.dir + '_V' + str(np.round(vi,4)),
                          solution_method =  'transiesta',
                          pp_path = self.Device.pp_path,
                          overwrite = True,
                          max_scf_it = 4000,
                          basis = self.Device.basis,
                          kp = self.Device.kp,
                          elec_inds = self.Device.elec_inds,
                          mpi = mpi,
                          Chem_Pot = [-vi/2, vi/2],
                          kp_tbtrans = self.Device.kp_tbtrans,
                          elecs = self.Device.elecs,
                          save_SE = True,
                          print_console=False,
                          print_mulliken=True,
                          NEGF_calc=True,
                          )
            
            Avi = np.array(done_vi)
            dst = np.abs(Avi - vi)
            copy_from = np.where(dst == dst.min())[0][0]
            Calc.copy_DM_from(done_calc[copy_from])
            print(copy_from)
            Calc.fdf( eta = 0.0 )
            if len(morestuff)>0:
                Calc.write_more_fdf(morestuff,name = where)
            Calc.run_siesta_in_dir ()
            Calc.run_analyze_in_dir()
            if tbtrans:
                Calc.run_tbtrans_in_dir()
            done_vi   += [vi  ]
            done_calc += [Calc]
            H,S = Calc.to_sisl(what = 'fromDFT')
            hamiltonians += [H]
        
        self.neq_done_vi      = np.array(done_vi)[1:]
        self.neq_done_calcs   = done_calc[1:]
        nV    =  len(self.neq_done_vi)
        Vi    =  self.neq_done_vi.copy()
        idx   =  np.argsort(self.neq_done_vi)
        Vi    =  Vi[idx]
        neq_H =  [hamiltonians[i] for i in idx]
        nk    = self.Ldense.shape[ 0]
        no    = self.Ldense.shape[-1]
        neq_harr = np.zeros((nV,nk, no, no),dtype = np.complex128)
        
        for ik in range(nk):
            kp = self.tbtk[ik]
            for iV in range(nV):
                hkv = neq_H[iV].H.Hk(k = kp,format = 'array')[:, self.pivot][self.pivot,:]
                neq_harr[iV, ik, :, :] = self.Ldense[ik,0]@hkv@(self.Ldense[ik,0])
        
        self.calculated_neq_H_ortho = neq_harr
        self.calculated_bias        = Vi
    
    def get_H_interpolation_function(self):
        """ Helper function for a obtaining a smooth interpolation of 
            a series of nonequilibrium calculations. Cannot go outside the bias
            window obviously when you call the resulting function. 
            Inspect the arguments of the function by using help or by trial. """
        from Interpolation import make_spline
        return make_spline(self.calculated_bias, self.calculated_neq_H_ortho)
    
    def read_data(self,sub_orbital = [], which_k = None, less_memory = True,
                  D_lead_ortho = True):
        """
        Reads out the gammas, self-energies for the leads
        reads the Device Hamiltonian, overlap, Lowdin-orthogonalizers
        
        You can specify which orbitals to retain using the sub_orbital keyword
        of this function. You need to familiarize yourself with the pivotting
        scheme used in TBtrans, and map your chosen atoms through the pivot indices
        if you want to go from the initial specification in the .fdf files to
        the actual Hamiltonian read into this class.
        
        You can also just include a specific list of k-points using the which_k
        keyword. Check TBtrans/sisl for documentation for k-point ordering.
        
        As said before the main quantities read by this function are the 
        overlap matrix, Hamiltonian, Gamma matrices and self-energies. 
        
        """
        
        Dev = self.Device
        self._sub_orbital = sub_orbital
        #t   = sisl.get_sile(Dev.dir + '/siesta.TBT.nc')
        t   = _READ_get_sile(Dev.dir)
        
        if which_k == None:
            which_k = [i for i in range(len(t.k))]
        
        p   = t.pivot()
        btd = t.btd()
        
        if len(sub_orbital) != 0:
            new_pivot = np.array([i for i in p if i in sub_orbital])
            new_btd   = []
            cs        = np.cumsum(btd)
            cs = np.hstack((np.zeros(1), cs)).astype(np.int32)
            for II in range(len(cs)-1):
                count = 0
                for III in p[cs[II]:cs[II+1]]:
                    if III in sub_orbital:
                        count+=1
                new_btd += [count]
            p   = np.array(new_pivot).astype(np.int32)
            btd = np.array(new_btd  ).astype(np.int32)
        
        # Dev_idx_u = sorted(p)
        P = [0]
        for b in btd:
            P+= [P[-1] + b ]
        
        #SE, inds = read_SE_from_tbtrans( Dev.dir + '/siesta.TBT.SE.nc')
        #E_F      = sisl.get_sile(Dev.dir + '/RUN.fdf').read_fermi_level()
        SE, inds  = _READ_get_SE(Dev.dir)
        E_F       = _READ_get_E_F(Dev.dir)
        
        self.E_F  = E_F
        #H = sisl.get_sile(Dev.dir + '/siesta.TSHS').read_hamiltonian()
        #S = sisl.get_sile(Dev.dir + '/siesta.TSHS').read_overlap()
        H = _READ_get_H(Dev.dir)
        S = _READ_get_S(Dev.dir)
        
        self.read_SE = SE
        self.read_coupling_inds = inds
        self.tbtE    = t.E.copy()
        self.tbtT    = t.transmission().copy()
        self._tbtTk_full  = np.array([t.transmission(kavg = i) for i in range(len(t.k))])
        self._tbtTk       = np.array([t.transmission(kavg = i) for i in which_k])
        
        self.tbtk_full    = t.k.copy()
        self.tbtk         = t.k[which_k]
        self.tbtwkpt_full = t.wkpt.copy()
        self.tbtwkpt      = t.wkpt[which_k]
        SE        = [SE[i][which_k] for i in range(len(SE))]
        self_Es   = [ SE   ]
        self_inds = [ inds ]
        Piv       = [  p   ]
        Part      = [  P   ]
        ## Eg        = t.E
        kv        = t.k[which_k]
        
        self.pivot = p
        self._btd  = btd
        
        # We orthogonalize the device basis and the lead bases through
        # a transformation proposed in Yan Ho Kwok et al. (2013)
        if D_lead_ortho:
            Sig0, Sig1 = read_overlap_data(t, self_inds[0], Dev, H, S)
        else:
            Sig0 = [np.zeros((len(t.k), H.no, H.no), dtype=np.complex128) 
                    for i in range(len(SE))]
            Sig1 = [np.zeros((len(t.k), H.no, H.no), dtype=np.complex128) 
                    for i in range(len(SE))]
        #Sig0, Sig1 = [_Sig0], [_Sig1]
        #if any( [elec.is_RSSE() for elec in self.Device.elecs]):
        #    Sig0 = None
        #    Sig1 = None
        
        Sys = gfunc(H, Piv, Part, sisl_S = S)
        Sys.set_SE(self_Es, self_inds)
        Sys.set_eta(0.0)
        Sys.set_ev(self.Contour)
        Sys.set_kv(kv)
        
        if D_lead_ortho==False:
            iG, Gam, Lowdin, Hamiltonian, Overlap, self_energies = Sys.iG(0)
        else:
            iG, Gam, Lowdin, Hamiltonian, Overlap, self_energies = Sys.iG(0, Sig0=[Sig0], Sig1=[Sig1])
        Sig0 = pivot_and_sub(Sig0, Piv[0], self_inds[0], H.no)
        Sig1 = pivot_and_sub(Sig1, Piv[0], self_inds[0], H.no)
        
        # read M_alpha_D
        # Sigma_0 and Sigma_1 corrections from Yan Ho Kwok et al. (2013)
        # print(self_inds[0])
        SLICES     = iG.all_slices.copy()
        self.Sig0  = Sig0
        self.Sig1  = Sig1
        del Sys
        
        self.self_energies        = self_energies
        self._Slices              = SLICES
        self.BTD_overlap          = Overlap
        self.Nonortho_iG          = iG
        self.Nonortho_Gammas      = Gam
        self.Lowdin               = Lowdin
        self.Nonortho_Hamiltonian = Hamiltonian
        self.BS                   = Hamiltonian.is_zero.copy()
        self.Ortho_Hamiltonian    =  Lowdin[0].BDot(Hamiltonian).BDot(Lowdin[0])   # S^-1/2   H   S^-1/2
        self.Ortho_Gammas         = [Lowdin[0].BDot(Gi         ).BDot(Lowdin[0])   # S^-1/2 Gamma S^-1/2
                                     for Gi in self.Nonortho_Gammas]
        
        try:
            sDM =sisl.get_sile(Dev.dir + '/siesta.TSDE').read_density_matrix()
            NO_DM = np.array([sDM.Dk(k = kvec).toarray()[p,:][:,p]/2 for kvec in kv])
        except:
            NO_DM = None
        
        self.NO_DM = NO_DM
        self.n_orb = self.Nonortho_Hamiltonian.shape[-1]
        if less_memory:
            self.less_memory()
    
    def eliminate_small_numbers(self, small = 1e-10):
        """Use Less memory, its good for your PC. 
           
        """
        for g in self.Ortho_Gammas:
            g.throw_away_less_than(small)
        for g in self.Nonortho_Gammas:
            g.throw_away_less_than(small)
    
    def less_memory(self,dtype=np.complex64):
        """Use Less memory, its good for your PC. 
           See the Block_matrices code and the block_td and block_sparse 
           classes to understand this part better. We are simply
           accessing the elements stored. 
        """
        self.eliminate_small_numbers(small=1e-8)
        del self.read_SE
        def reduce_memory(m):
            for i in range(len(m.vals)):
                m.vals[i] = m.vals[i].astype(dtype)
            m.info()
        def btd_reduce_memory(m):
            for i in range(len(m.Al)):
                m.Al[i] = m.Al[i].astype(dtype)
            for i in range(len(m.Cl)):
                m.Cl[i] = m.Cl[i].astype(dtype)
            for i in range(len(m.Bl)):
                m.Bl[i] = m.Bl[i].astype(dtype)
            m.info(m.diagonal_zeros)
        def use_hermitian(m):
            for I in range(len(m.inds)-1,-1,-1):
                i,j = m.inds[I]
                if i<j:
                    del m.inds[I], m.vals[I]
            m.Hermitian='Yes'
            m.info()
            
        for bm in self.self_energies:
            reduce_memory(bm)
        for bm in self.Nonortho_Gammas:
            reduce_memory(bm)
            use_hermitian(bm)
        for bm in self.Ortho_Gammas:
            reduce_memory(bm)
            use_hermitian(bm)
        btd_reduce_memory(self.Nonortho_iG)
        btd_reduce_memory(self.Nonortho_Hamiltonian)
        
        btd_reduce_memory(self.BTD_overlap)
        reduce_memory(self.Lowdin[0])
        reduce_memory(self.Lowdin[1])
        reduce_memory(self.Ortho_Hamiltonian)
        use_hermitian(self.Ortho_Hamiltonian)
    
    def Fit(self,
            fact = 2.0, 
            Fallback_W = 30.0, 
            NumL = 3, 
            fit_mode = 'all', 
            use_analytical_jac = True,  
            min_method = 'SLSQP',
            ebounds = (-5, 5), 
            wbounds = (0.01, 3.0), 
            gbounds = (None, None),
            tol = -1.0, 
            options = {}, 
            fit_real_part = False,
            specific_bounds = None, 
            alpha_PO = 0.0, 
            cons = 'ascending',
            force_PSD = True, 
            force_PSD_tol = 0.0,
            which_e = None,
            which_k = None,
            init_E  = None,
            init_G  = None,
            exc_idx = None,
            ):
        """
            Here the fitting of either the gammas or the self-energies are done (fit_mode = 'all' or 'all_from_SE')
            You can also rerun a fit, no need to delete previous fit.
            But you can delete a previous fit with a "reset_all_fits()" call.
            fact, together with NumL and ebounds, determines how large broadening is
            on the initial Lorentzian
            Fallback_W: is used when only one sampling point is used (redundant now)
            NumL: Number of Lorentzians used to fit
            use_analytical_jac: use analytical jacobian for minimization
            min_method: the method used by the minimize function. (SLSQP, L-BFGS-B, many more... see scipy docs on minimize)
            ebounds: limits on where the Lorentzian centers can be
            wbounds:  limits on how broad or narrow the lorentzian centers can be
            gbounds: limits on the coeffcients
            tol: the cutoff on which matrix-elements are attempted fitted. The absolute value of over all energies just need to have one larger value than this to get fitted
            fit_real_part : if true only the real part of the Gamma/Selfenergy is fitted
            specific bounds: list \n [ {(ik, iL):((B1, B2), (C1, C2)) ,...}, ... ] of specific bounds on the lorentzian iL at point ik.
            alpha_PO  : (positive number) Adds a term that drives the Lorentzians away from each other
            cons      : additional constraints on each lorentzian.
            force_PSD : forces the matrices with the coefficients for the Lorentzians to be positive semidefinite (or have eigenvalues larger than force_PSD_tol).
            force_PSD_tol: lower bound on how large the eigenvalues can be. float, (NumL,) array or (nk, NumL) array giving smallest eigenvalue for each fitting matrix.
            which_e: Which electrode you want to fit. For polishing individual fits.
            which_k: Which k-point you want to fit. For polishing individual fits.
            Note: This function behaves differently depending on if you have given the code a single energy-point, or multiple. If a single point is given, the code simply gives one Lorentzian with width Fallback_W, which takes the value of the sampled point at the apex if you give the code >1 point, the it instead fits the curve linearly interpolated between the points.
            Todo: Still needs to simultaneusly fit the Hilbert transform....
        """
        if init_E is None:
            init_E = [None] * self.num_leads
        if init_G is None:
            init_G = [None] * self.num_leads
        if exc_idx is None:
            exc_idx = [None] * self.num_leads
        
        self.NumL = NumL
        nk = self.Nonortho_Hamiltonian.shape[0]
        if hasattr(self,'fitted_lorentzians'):
            RERUN = True
        else:
            RERUN = False
            self.fitted_lorentzians   = []
            self.fitted_self_energies = []
            self.NO_fitted_lorentzians= []
            self.sampled_gl_matrices  = [None for i in range(self.num_leads)]
            self._fitting_tags        = [None for i in range(self.num_leads)]
        
        if which_e == None:
            ielecs = range(self.num_leads)
        else:
            assert RERUN
            ielecs = which_e
            assert max(which_e)<self.num_leads
        if which_k is not None: assert RERUN
        
        for e in ielecs:
            idx_real_line   = self.sampling_idx[e]
            idx_fermi_poles = self.F_poles_idx[e]
            zi    = self.Contour[idx_real_line].real # on levelwidth function on real line
            if len(zi)!=1:
                emax = zi.max()
                emin = zi.min()
                nsamples = len(zi)
                gamma = fact * (emax - emin ) * np.ones((nk,nsamples)) / nsamples
                ei    = np.linspace(emin, emax, nsamples) #+ np.random.random(nsamples) * gamma[0]/10
                ei    = np.array([ei] * nk)
            else:
                gamma = Fallback_W * np.ones((nk, 1))
                ei    = np.repeat(zi.copy()[np.newaxis, :],nk, axis=0)      #np.linspace(emin, emax, nsamples)
            
            #Lorentzian parameters
            gel  = self.Ortho_Gammas[e].get_e_subset(idx_real_line)
            self.sampled_gl_matrices[e]=gel.copy()
            self._fitting_tags[e] = e
            
            gel = None
            if specific_bounds is None:
                sb = None
            else:
                sb = specific_bounds[e]
            if isinstance(force_PSD_tol, float) or isinstance(force_PSD_tol, int):
                force_PSD_tol_pass  = 0
                force_PSD_tol_pass += force_PSD_tol
            else:
                force_PSD_tol_pass = force_PSD_tol[e]
            
            print('Finding Lambda matrices:')
            if len(zi)!=1:
                # We start from the Nonorthogonal gamma!
                if fit_mode == 'all':
                    gel = self.Nonortho_Gammas[e].get_e_subset(idx_real_line)
                    
                    if RERUN:
                        print('\nRerun\n')
                        fit_gel = self.NO_fitted_lorentzians[e]
                    else:
                        fit_gel = gel.initialize_lorentzian_fit(zi, NumL, fact = fact, 
                                                                init_E = init_E[e], init_G = init_G[e])
                    t1 = time()
                    
                    NO_gel = gel.fit_sampling_and_Lorenzian(zi, fit_gel, 
                                                             use_analytical_jac = use_analytical_jac, 
                                                             min_method = min_method,
                                                             ebounds = ebounds, wbounds = wbounds,gbounds = gbounds,
                                                             tol = tol,options = options, 
                                                             fit_real_part = fit_real_part,
                                                             force_hermitian = True,
                                                             specific_bounds = sb, alpha_PO = alpha_PO,
                                                             cons = cons,
                                                             force_PSD = force_PSD, force_PSD_tol=force_PSD_tol_pass,
                                                             which_k = which_k,
                                                             exc_idx = exc_idx[e]
                                                             )
                    
                    t2 = time()
                    print('Lorentzian fit took ' + str(t2-t1) + ' seconds.')
                    saved_data = (NO_gel.ei.copy(), NO_gel.gamma.copy())
                    
                    gel = self.Lowdin[0].BDot(NO_gel).BDot(self.Lowdin[0])
                    gel.Lorentzian_basis = True
                    gel.ei    = saved_data[0].copy()
                    gel.gamma = saved_data[1].copy()
                
                elif fit_mode == 'stepwise':
                    count = 0; _nl = 3
                    assert np.mod(NumL, _nl) == 0
                    inds = [];   vals   = []
                    Ei_col = []; Wi_col = []
                    while count * _nl <NumL:
                        
                        if count == 0:
                            diff = self.Nonortho_Gammas[e].get_e_subset(idx_real_line)
                        else:
                            diff = diff.Add(fit_diff.evaluate_Lorentzian_basis(zi).scalar_multiply(-1.0))
                        
                        fit_diff = diff.initialize_lorentzian_fit( zi, _nl, fact = fact, at_extrema = True, de_tol = 0.5)
                        
                        fit_diff = diff.fit_sampling_and_Lorenzian(zi, fit_diff,
                                                                   use_analytical_jac = use_analytical_jac, 
                                                                   min_method = min_method,
                                                                   ebounds = ebounds, wbounds = wbounds,gbounds = gbounds, 
                                                                   tol = tol,options = options,
                                                                   fit_real_part = fit_real_part, 
                                                                   specific_bounds = sb, alpha_PO = alpha_PO,
                                                                   cons=cons,
                                                                   force_PSD = force_PSD,
                                                                   force_PSD_tol = force_PSD_tol_pass,
                                                                   which_k = which_k)
                        
                        inds   += [fit_diff.inds]
                        vals   += [fit_diff.vals]
                        Ei_col += [fit_diff.ei]
                        Wi_col += [fit_diff.gamma]
                        
                        count += 1
                    
                    Ei = np.hstack(Ei_col)
                    Wi = np.hstack(Wi_col)
                    num_blocks = len(self.Nonortho_Gammas[e].inds)
                    new_inds   = []
                    new_blocks = []
                    
                    for B in range(num_blocks):
                        blocks = []
                        B_ind = tuple(self.Nonortho_Gammas[e].inds[B])
                        for L in range(count):
                            blocks += [vals[L][inds[L].index(B_ind)]]
                        
                        new_inds+=[B_ind]
                        new_blocks += [np.concatenate(blocks, axis = 1)]
                    gel = block_sparse(new_inds, new_blocks, Block_shape = self.Nonortho_Gammas[e].Block_shape)
                    gel.Lorentzian_basis = True # Flag it as Lorentzian sum
                    gel.ei = Ei.copy()
                    gel.gamma = Wi.copy()
                
                elif fit_mode == 'all_from_SE':
                    se     = self.self_energies[e].get_e_subset(idx_real_line)
                    if RERUN: fit_se = self.fitted_self_energies[e]
                    else:     fit_se = se.initialize_lorentzian_fit(zi, NumL, fact=fact)
                    
                    t1 = time()
                    
                    fit_se.is_self_energy = True
                    fit_se = se.fit_sampling_and_Lorenzian(zi, fit_se, 
                                                           use_analytical_jac = use_analytical_jac, 
                                                           min_method = min_method,
                                                           ebounds = ebounds, wbounds = wbounds,gbounds = gbounds,
                                                           tol = tol,options = options, 
                                                           fit_real_part = fit_real_part,
                                                           specific_bounds = sb, 
                                                           alpha_PO = alpha_PO, force_hermitian = False,
                                                           cons=cons,
                                                           force_PSD = force_PSD,
                                                           force_PSD_tol = force_PSD_tol_pass,
                                                           which_k = which_k)
                    
                    
                    t2 = time()
                    print('Lorentzian fit took ' + str(t2-t1) + ' seconds.')
                    
                    fit_se.Lorentzian_basis    = True # Flag it as Lorentzian sum
                    if RERUN: self.fitted_self_energies[e] =  fit_se
                    else:     self.fitted_self_energies   += [fit_se]
                    
                    saved_data                 = (fit_se.ei.copy(), fit_se.gamma.copy())
                    fit_se_dag                 = fit_se.copy()
                    fit_se_dag.do_dag()
                    NO_gel          = fit_se.Subtract(fit_se_dag)
                    NO_gel          = NO_gel.scalar_multiply(1.0j)
                    
                    gel = self.Lowdin[0].BDot(NO_gel).BDot(self.Lowdin[0])
                    gel.Lorentzian_basis = True
                    gel.ei    = saved_data[0].copy()
                    gel.gamma = saved_data[1].copy()
                    NO_gel.Lorentzian_basis = True
                    NO_gel.ei    = saved_data[0].copy()
                    NO_gel.gamma = saved_data[1].copy()
                
                if RERUN:self.fitted_lorentzians[e] =  gel;  self.NO_fitted_lorentzians[e] = NO_gel
                else:    self.fitted_lorentzians   += [gel]; self.NO_fitted_lorentzians   += [NO_gel]
            
            else:
                newname_gel = self.sampled_gl_matrices[e]
                newname_gel.gamma = gamma
                newname_gel.ei    = ei
                newname_gel.Lorentzian_basis = True
                self.fitted_lorentzians += [newname_gel]
    
        
        
    def pickle(self, filename, compression = 'lzma'):
        """Saves calculator to file. 
           compression=None or string (default is lzma)
           """
        import pickle as pkl
        if compression == None:
            print('Pickling, no compression')
            f = open(filename +'.Timedep', 'wb')
            pkl.dump(self, f)
            f.close()
        if compression == 'lzma':
            import lzma
            print('Pickling, lzma compression')
            with lzma.open(filename+'.xz','wb') as f:
                pkl.dump(self,f)
        if compression == 'gzip':
            import gzip
            print('Pickling, gzip compression')
            with gzip.open(filename+'.gz','wb') as f:
                pkl.dump(self,f)
        if compression == 'bz2':
            import bz2
            print('Pickling, bz2 compression')
            with bz2.open(filename+'.bz2','wb') as f:
                pkl.dump(self,f)
        #if compression == 'brotli':
        #    import brotli
        #    print('Pickling, bz2 compression')
        #    with brotli.open(filename+'.brotli','wb') as f:
        #        pkl.dump(self,f)
    
    def reset_all_fits(self):
        """Resets all fits"""
        delattr(self, 'fitted_lorentzians')
        delattr(self, 'fitted_self_energies')
        delattr(self, 'sampled_gl_matrices')
        delattr(self, 'NO_fitted_lorentzians')
    
    def diagonalise(self, sorting = 'abs', fixphase = None):
        """Here the eigenvalues and eigenvectors of the coefficient-
           matrices are found for use in the expansion of the lessor/greater 
           self energies.
        """
        
        Gl_eig  = []
        Gl_vec  = []
        GpB_eig = []
        GpB_vec = []
        GpC_eig = []
        GpC_vec = []
        Gpeig   = []
        Gpvec   = []
        
        
        self._gl_matrices   = []
        self._gp_matrices   = []
        self.broadenings        = []
        self.Lorentzian_centers = []
        print('Finding eigenvalues and eigenvectors')
        for e in range(self.num_leads):
            gel = self.fitted_lorentzians[e]
            self.broadenings.append(gel.gamma.copy())
            self.Lorentzian_centers.append(gel.ei.copy())
            idx_fermi_poles = self.F_poles_idx[e]
            gep_bs   = gel.evaluate_Lorentzian_basis(self.Contour[idx_fermi_poles])
            gep  = Blocksparse2Numpy(gep_bs,  self._Slices)
            gel  = Blocksparse2Numpy(gel,  self._Slices)
            
            gepB = (gep + gep.transpose(0,1,3,2).conj())/2
            gepC = (gep - gep.transpose(0,1,3,2).conj())/(2j)
            
            for _i,_si in enumerate(self._Slices):
                for _j, _sj in enumerate(_si):
                    if self.fitted_lorentzians[e].Block(_i, _j) is not None or gep_bs.Block(_i, _j) is not None:
                        assert np.allclose(gel[... , _sj[0], _sj[1]] , self.fitted_lorentzians[e].Block(_i, _j))
                        assert np.allclose(gep[... , _sj[0], _sj[1]] , gep_bs.Block(_i, _j))
                
            self._gl_matrices  += [ gel  ]
            self._gp_matrices  += [ gep  ]
            
            el, vl = Sorted_Eig(gel, hermitian = True, sorting=sorting, fixphase=fixphase)
            assert np.allclose(gel,   gel.transpose(0,1,3,2).conj() )
            assert np.allclose(gepB , gepB.transpose(0,1,3,2).conj())
            assert np.allclose(gepC , gepC.transpose(0,1,3,2).conj())
            
            epB, vpB = Sorted_Eig(gepB, hermitian=True, sorting=sorting, fixphase=fixphase)
            epC, vpC = Sorted_Eig(gepC, hermitian=True, sorting=sorting, fixphase=fixphase)
            epC      = 1j*epC
            
            Gl_eig.append(el)
            Gl_vec.append(vl)
            GpB_eig.append(epB)
            GpB_vec.append(vpB)
            GpC_eig.append(epC)
            GpC_vec.append(vpC)
        
        self.Gl_eig  = np.array(Gl_eig) # indices: lead, k, x, state
        self.Gl_vec  = np.array(Gl_vec) # indices: lead, k, x, state, :
        
        self.GpB_eig = np.array(GpB_eig)
        self.GpB_vec = np.array(GpB_vec)
        
        self.GpC_eig = np.array(GpC_eig)
        self.GpC_vec = np.array(GpC_vec)
        
        assert self.Gl_eig.shape[2] == self.NumL
        print('Maximum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.max(),6)))
        print('Minimum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.min(),6)))
        
        print('If the minimum is negative, you should take extra care!')
        print(r'( if minimum negative Check eigenvalues of $\Gamma$)')
    
    def get_propagation_quantities(self, use_exact_fermi=False):
        """
        Here the quantities originally derived in Popescu et al. (2016) and
        refined in 
           """ + CiteString + """
        are calculated
        """
        
        N_F   = self.num_poles 
        N_L   = self.NumL
        self.N_L = N_L
        nk    = self.Gl_eig[0].shape[0]
        no    = self.n_orb
        nlead = self.num_leads
        Ntot  = N_L + 2*N_F
        comp  = np.complex128
        
        GG_P  = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        GG_M  = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        GL_P  = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        GL_M  = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        Xpp   = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        Xpm   = np.zeros((nk, nlead, Ntot, no),dtype = comp)
        
        
        for k in range(nk):
            for a in range(nlead):
                mu = self.mu_i[a]
                kT = self.kT_i[a]
                beta = 1 / kT
                
                for x in range(Ntot):
                    for c in range(no):
                        if x  < N_L:
                            Wi = self.broadenings[a][k,x]
                            eps = np.array([self.Lorentzian_centers[a][k,x] 
                                            + 1j * Wi])
                            
                            GG_P[k,a,x,c] = -1j / 2 * self.Gl_eig[a,k,x,c] * Wi *\
                                            (1 - FD_expanded(eps      , self.zi, beta, mu = mu,  coeffs = self.coeffs_fermi))
                            GG_M[k,a,x,c] = -1j / 2 * self.Gl_eig[a,k,x,c] * Wi *\
                                            (1 - FD_expanded(eps.conj(), self.zi, beta, mu = mu, coeffs = self.coeffs_fermi))
                            GL_P[k,a,x,c] = +1j / 2 * self.Gl_eig[a,k,x,c] * Wi *\
                                             FD_expanded(eps       , self.zi, beta, mu = mu,     coeffs = self.coeffs_fermi)
                            GL_M[k,a,x,c] = +1j / 2 * self.Gl_eig[a,k,x,c] * Wi *\
                                             FD_expanded(eps.conj(), self.zi, beta, mu = mu,     coeffs = self.coeffs_fermi)
                            Xpp[k,a,x,c]  = eps
                            Xpm[k,a,x,c]  = eps.conj()
                
                sp = slice(0  ,   N_F)
                sm = slice(N_F, 2*N_F)
                
                SB = slice(N_L    , N_L +   N_F)
                SC = slice(N_L+N_F, N_L + 2*N_F)
                
                R = self.coeffs_fermi
                GG_P[k,a,SB,:] = +R[:, None] * self.GpB_eig[a,k,sp,:]/beta
                GG_M[k,a,SB,:] = -R[:, None] * self.GpB_eig[a,k,sm,:]/beta
                GL_P[k,a,SB,:] = +R[:, None] * self.GpB_eig[a,k,sp,:]/beta
                GL_M[k,a,SB,:] = -R[:, None] * self.GpB_eig[a,k,sm,:]/beta
                
                GG_P[k,a,SC,:] = +R[:, None] * self.GpC_eig[a,k,sp,:]/beta
                GG_M[k,a,SC,:] = -R[:, None] * self.GpC_eig[a,k,sm,:]/beta
                GL_P[k,a,SC,:] = +R[:, None] * self.GpC_eig[a,k,sp,:]/beta
                GL_M[k,a,SC,:] = -R[:, None] * self.GpC_eig[a,k,sm,:]/beta
                
                Xpp [k,a,SB,:] = self.F_poles[a,sp,None]
                Xpm [k,a,SB,:] = self.F_poles[a,sm,None]
                Xpp [k,a,SC,:] = self.F_poles[a,sp,None]
                Xpm [k,a,SC,:] = self.F_poles[a,sm,None]
                
                
                    # elif x >= N_L:
                        #     I1 = x  - N_L
                        #     I2 = I1 + N_F
                        #     R = self.coeffs_fermi[I1]
                        #     GG_P[k,a,x,c] = +R*self.Gp_eig[a,k,I1,c]/beta
                        #     GG_M[k,a,x,c] = -R*self.Gp_eig[a,k,I2,c]/beta
                        #     GL_P[k,a,x,c] = +R*self.Gp_eig[a,k,I1,c]/beta 
                        #     GL_M[k,a,x,c] = -R*self.Gp_eig[a,k,I2,c]/beta
                            
                        #     Xpp [k,a,x,c] = self.F_poles[a,I1]
                        #     Xpm [k,a,x,c] = self.F_poles[a,I2]
        
        self.GG_P = GG_P
        self.GG_M = GG_M
        self.GL_P = GL_P
        self.GL_M = GL_M
        self.Xpp  = Xpp
        self.Xpm  = Xpm
        self.diff_ggp_glp = GG_P - GL_P
        self.diff_ggm_glm = GG_M - GL_M
        xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
                                 self.GpB_vec[:,:,0:N_F,:,:].transpose(1,0,2,4,3),
                                 self.GpC_vec[:,:,0:N_F,:,:].transpose(1,0,2,4,3) ), 
                                 axis = 2 )
        Ixi = xi.conj()
        self.xi  = xi
        self.Ixi = Ixi
        
        
        assert (Xpp.imag>0).all()
        assert (Xpm.imag<0).all()
    
    def get_dense_matrices(self, zero_tol = 1e-12):
        """ 
           Prepare the propagation quantities in arrays and get a (bad) initial
           density matrix. 
           
           zero_tol defines the lower cutoff on the cutoff of the eigenvalues
           in the self-energy decompositions. 1e-12 is a very strict value and
           even 1e-5 can be entirely reasonable.
        """
        
        self.Hdense  = Blocksparse2Numpy(self.Ortho_Hamiltonian, self._Slices)
        if hasattr(self, "Hamiltonian_renormalisation_correction"):
            self.Hdense+=self.Hamiltonian_renormalisation_correction
        
        self.Ldense  = Blocksparse2Numpy(self.Lowdin[0], self._Slices)
        self.iLdense = Blocksparse2Numpy(self.Lowdin[1], self._Slices)
        no = self.Hdense.shape[ -1 ]
        nk = self.Hdense.shape[  0 ]
        nlead    = self.num_leads
        nc       = self.GG_P.shape[3]
        nx       = self.GG_P.shape[2]
        He,Hv = np.linalg.eigh(self.Hdense)
        mu_dev = sum(self.mu_i)/len(self.mu_i)
        kT_dev = sum(self.kT_i)/len(self.kT_i)
        self.sigma = np.zeros((nk, no, no),dtype = np.complex128)
        self._device_eigs_vecs = (He, Hv)
        
        for i in range(nk):
            e,v = He[i,0], Hv[i,0]
            f   = 1/(1 + np.exp((e - mu_dev)/kT_dev)) 
            self.sigma[i] = v.dot(np.diag(f)).dot(v.T.conj())
    
        self.Psi_vec   =  np.zeros(self.GG_P.shape + (no,), dtype = np.complex128)
        self.omega_idx = -np.ones((nlead, nx, nc, nlead, nx, nc) ,dtype = np.int64)
        self.psi_idx   = -np.ones((nlead, nx, nc),dtype = np.int64)
        
        it   = 0
        it_2 = 0
        DD1 = np.abs(self.GG_M - self.GL_M)
        DD1 = DD1.max(axis = 0)
        DD2 = np.abs(self.GG_P - self.GL_P)
        DD2 = DD2.max(axis = 0)
        
        for a1 in range(nlead):
            for x1 in range(nx):
                for c1 in range(nc):
                    
                    if np.abs(self.GG_P[:,a1,x1,c1]).max()<zero_tol and np.abs(self.GL_P[:,a1,x1,c1]).max()<zero_tol:
                        pass
                    else:
                        self.psi_idx[a1,x1,c1] = it_2
                        it_2 += 1
                    
                    d1 = DD1[a1,x1,c1]#np.abs(self.GG_M[:,a1,x1,c1] - self.GL_M[:,a1,x1,c1]).max()
                    for a2 in range(nlead):
                        for x2 in range(nx):
                            for c2 in range(nc):
                                d2 = DD2[a2,x2,c2]#np.abs(self.GG_P[:,a2,x2,c2] - self.GL_P[:,a2,x2,c2]).max()
                                if x1 >= self.N_L and x2 >= self.N_L and a1 == a2:
                                    pass
                                elif d1<zero_tol and d2<zero_tol:
                                    pass
                                else:
                                    self.omega_idx[a1,x1,c1,a2,x2,c2] = it
                                    it+=1
        self.omegas = np.zeros((nk, it), dtype = np.complex128)
        self.max_orbital_idx = np.where(self.psi_idx>=0)[2].max()
        
        if self.NO_DM is not None:
            for ik in range(nk):
                self.sigma[ik,:,:] = self.iLdense[ik,0,:,:]@self.NO_DM[ik,:,:]@self.iLdense[ik, 0, :, :]
        
        print('\n----------------\n Storing ' + str(it) + ' values of Omega_{axc,a\'x\'c\'}!\n----------------\n')
        print('\n----------------\n ' + str((self.omega_idx == -1).sum() / self.omega_idx.size * 100) + '% of Omegas are set to zero!')
        print('\n----------------\n ' + str((self.psi_idx == -1).sum() / self.psi_idx.size * 100) + '% of psis are set to zero!')
    
    def tofile(self,name='TDT', tol=1e-7,create_arrays=False, 
                fixphase=None, sorting='abs', exact_fermi=False):
        """Wraps a series of steps for obtaining the end file
           in an easy to use function. 
           specifies the lowest absolute value for the eigenvalues.
           fixphase has been used for testing purposes and
           are inconsenquencial.
           The eigenvalues should we sorted according to obs, as this gives
           the most dense blocks of matrices. 
           
        """
        self.diagonalise(sorting=sorting, fixphase=fixphase)
        self.get_propagation_quantities(use_exact_fermi=exact_fermi)
        self.get_dense_matrices_purenp(zero_tol=tol,create_arrays=create_arrays)
        self.write_to_file(name=name)
    
    def Renormalise_H(self, custom_mat):
        """
            custom_mat: np.ndarray of shape (nk, 1, no, no)
            Here you can give a value to renormalize the Orthogonal Hamiltonian
            Acts as a rigid shift of the Hermitian part of $\Sigma_alpha$
        """
        
        assert len(custom_mat.shape) == 4
        assert np.allclose(custom_mat,
                           custom_mat.transpose(0,1,3,2).conj())
        self.Hamiltonian_renormalisation_correction = custom_mat.copy()
    
    def get_dense_matrices_purenp(self, zero_tol = 1e-15, create_arrays = True):
        """ 
           Prepare the propagation quantities in arrays and get a (bad) initial
           density matrix. 
           
           Furthermore, the Hamiltonian gets the renormalisation given to the 
           class through self.Renormalise_H added to it. 
           
           zero_tol defines the lower cutoff on the cutoff of the eigenvalues
           in the self-energy decompositions. 1e-12 is a very strict value and
           even 1e-5 can be entirely reasonable.
           
           
        """
        
        self.Hdense  = Blocksparse2Numpy(self.Ortho_Hamiltonian, self._Slices)
        if hasattr(self, "Hamiltonian_renormalisation_correction"):
            self.Hdense+=self.Hamiltonian_renormalisation_correction
        
        self.Ldense  = Blocksparse2Numpy(self.Lowdin[0], self._Slices)
        self.iLdense = Blocksparse2Numpy(self.Lowdin[1], self._Slices)
        no = self.Hdense.shape[ -1 ]
        nk = self.Hdense.shape[  0 ]
        nlead    = self.num_leads
        nc       = self.GG_P.shape[3]
        nx       = self.GG_P.shape[2]
        He,Hv = np.linalg.eigh(self.Hdense)
        mu_dev = sum(self.mu_i)/len(self.mu_i)
        kT_dev = sum(self.kT_i)/len(self.kT_i)
        self.sigma = np.zeros((nk, no, no),dtype = np.complex128)
        self._device_eigs_vecs = (He, Hv)
        
        for i in range(nk):
            e,v = He[i,0], Hv[i,0]
            f   = 1/(1 + np.exp((e - mu_dev)/kT_dev)) 
            self.sigma[i] = v.dot(np.diag(f)).dot(v.T.conj())
        
        #self.omega_idx = -np.ones((nlead, nx, nc, nlead, nx, nc) ,dtype = np.int64)
        self.psi_idx   = -np.ones((nlead, nx, nc),dtype = np.int64)
        
        it   = 0
        it_2 = 0
        DD1 = np.abs(self.GG_M - self.GL_M)
        DD1 = DD1.max(axis = 0)
        DD2 = np.abs(self.GG_P - self.GL_P)
        DD2 = DD2.max(axis = 0)
        
        for a1 in range(nlead):
            for x1 in range(nx):
                for c1 in range(nc):
                    
                    if np.abs(self.GG_P[:,a1,x1,c1]).max()<zero_tol and np.abs(self.GL_P[:,a1,x1,c1]).max()<zero_tol:
                        pass
                    else:
                        self.psi_idx[a1,x1,c1] = it_2
                        it_2 += 1
        
        self.max_orbital_idx = np.where(self.psi_idx>=0)[2].max()
        omega_shape          = (nk, nlead, nx, self.max_orbital_idx+1, nlead, nx, nc)
        psi_shape            = (nk, nlead, nx,self.max_orbital_idx+1, no)
        
        if create_arrays:
            self.omegas  = np.zeros(omega_shape, dtype = np.complex128)
            self.Psi_vec   =  np.zeros(psi_shape, dtype = np.complex128)
        
        self.omega_shape = omega_shape
        self.psi_shape   = psi_shape
        self._zero_tol   = zero_tol
        
        if self.NO_DM is not None:
            for ik in range(nk):
                self.sigma[ik,:,:] = self.iLdense[ik,0,:,:]@self.NO_DM[ik,:,:]@self.iLdense[ik, 0, :, :]
    
    def Check_input_to_ODE(self, loose_fermi = True):
        """ Checks the result of the eigendecompositions are indeed consistent
            with the input. For testing purposes. 
            If the ODECheck.txt contains large numbers (say >1e-4),
            increase the number of fermi poles included either in the beginning
            or using the modify_occupations commandline tool.
            """
        
        
        from PadeDecomp import FD
        Sl = self._Slices
        L,iL = self.Ldense, self.iLdense
        Ri = self.coeffs_fermi
        zi = self.zi
        nk = len(L)
        no = L.shape[-1]
        
        # Check that all the quantities are given as in Croy 2016
        print('\n Check the ODECheck.txt file lying in the working directory!\n')
        print('\n The values in the file should be small!\n')
        if hasattr(self, "Hamiltonian_renormalisation_correction"):
            print('Diff in Hamiltonians, Hdense vs input: ')
            print(np.abs(iL@(self.Hdense - self.Hamiltonian_renormalisation_correction)@iL- 
                               Blocksparse2Numpy(self.Nonortho_Hamiltonian, Sl)).sum())
        else:
            diff = np.abs(iL@self.Hdense@iL - Blocksparse2Numpy(self.Nonortho_Hamiltonian, Sl)).max()
            if not np.allclose(iL@self.Hdense@iL, Blocksparse2Numpy(self.Nonortho_Hamiltonian, Sl)):
                print('Warning: max diff between two internal Hamiltonians =', diff)
                print('\n Decide if this is acceptable or stop what you are doing and start debugging.')
        
        assert np.allclose(np.linalg.inv(L), iL)
        PrintFile = open('ODECheck.txt', 'w')
        for ik in range(nk):
            for ia in range(self.num_leads):
                kT,mu = self.kT_i[ia], self.mu_i[ia]
                for l in range(self.NumL):
                    rp = self.fitted_lorentzians[ia].ei[ik,l]
                    ip = self.fitted_lorentzians[ia].gamma[ik,l]
                    Lrz_pole = rp + 1j * ip
                    #Poles
                    assert (self.Xpp[ik,ia,l,:] == Lrz_pole         ).all()
                    assert (self.Xpm[ik,ia,l,:] == np.conj(Lrz_pole)).all()
                    #Eigenvalues and eigenvectors
                    m = np.zeros((no,no), dtype = np.complex128)
                    
                    for c in range(no):
                        f_val_p = FD(Lrz_pole,          1/kT, mu)
                        f_val_m = FD(np.conj(Lrz_pole), 1/kT, mu)
                        
                        T1 = np.allclose(self.GG_P[ik, ia, l, c], -1j/2 * self.Gl_eig[ia, ik, l, c] * ip * (1-f_val_p))
                        T2 = np.allclose(self.GG_M[ik, ia, l, c], -1j/2 * self.Gl_eig[ia, ik, l, c] * ip * (1-f_val_m))
                        T3 = np.allclose(self.GL_P[ik, ia, l, c],  1j/2 * self.Gl_eig[ia, ik, l, c] * ip *    f_val_p )
                        T4 = np.allclose(self.GL_M[ik, ia, l, c],  1j/2 * self.Gl_eig[ia, ik, l, c] * ip *    f_val_m )
                        if loose_fermi==False:
                            assert T1
                            assert T2
                            assert T3
                            assert T4
                        else:
                            if not T1: print('GG_P difference: ', np.abs(self.GG_P[ik, ia, l, c] - (-1j/2 * self.Gl_eig[ia, ik, l, c] * ip * (1-f_val_p))), file = PrintFile)
                            if not T2: print('GG_M difference: ', np.abs(self.GG_M[ik, ia, l, c] - (-1j/2 * self.Gl_eig[ia, ik, l, c] * ip * (1-f_val_m))), file = PrintFile)
                            if not T3: print('GL_P difference: ', np.abs(self.GL_P[ik, ia, l, c] -   1j/2 * self.Gl_eig[ia, ik, l, c] * ip *    f_val_p ) , file = PrintFile)
                            if not T4: print('GL_M difference: ', np.abs(self.GL_M[ik, ia, l, c] -   1j/2 * self.Gl_eig[ia, ik, l, c] * ip *    f_val_m ) , file = PrintFile)
                        
                        vec1 =   self.Gl_vec[ia,ik,l,:,c]
                        vec2 =   self.Gl_vec[ia,ik,l,:,c].conj()
                        O    =   np.multiply.outer(vec1, vec2)
                        m   +=   self.Gl_eig[ia,ik,l,c] * O
                    
                    assert np.allclose(m, self._gl_matrices[ia][ik,l,:,:])
                
                for _f in range(self.num_poles):
                    f  = _f + self.NumL
                    Nf = len(zi)
                    f2 = f+Nf
                    assert (self.Xpp[ik,ia,f,:]  == ( mu + self.zi[_f] * kT )   ).all()
                    assert (self.Xpm[ik,ia,f,:]  == ( mu - self.zi[_f] * kT )   ).all()
                    assert (self.Xpp[ik,ia,f2,:] == ( mu + self.zi[_f] * kT )   ).all()
                    assert (self.Xpm[ik,ia,f2,:] == ( mu - self.zi[_f] * kT )   ).all()
                    
                    m1 = np.zeros((no,no), dtype = np.complex128)
                    m2 = np.zeros((no,no), dtype = np.complex128)
                    
                    for c in range(no):
                        assert np.allclose(self.GG_P[ik, ia, f, c] ,  Ri[_f] * kT * self.GpB_eig[ia, ik, _f, c])
                        assert np.allclose(self.GG_M[ik, ia, f, c] , -Ri[_f] * kT * self.GpB_eig[ia, ik, _f+Nf, c])
                        assert np.allclose(self.GL_P[ik, ia, f, c] ,  Ri[_f] * kT * self.GpB_eig[ia, ik, _f, c])
                        assert np.allclose(self.GL_M[ik, ia, f, c] , -Ri[_f] * kT * self.GpB_eig[ia, ik, _f+Nf, c])
                        
                        assert np.allclose(self.GG_P[ik, ia, f2, c] ,  Ri[_f] * kT * self.GpC_eig[ia, ik, _f, c])
                        assert np.allclose(self.GG_M[ik, ia, f2, c] , -Ri[_f] * kT * self.GpC_eig[ia, ik, _f+Nf, c])
                        assert np.allclose(self.GL_P[ik, ia, f2, c] ,  Ri[_f] * kT * self.GpC_eig[ia, ik, _f, c])
                        assert np.allclose(self.GL_M[ik, ia, f2, c] , -Ri[_f] * kT * self.GpC_eig[ia, ik, _f+Nf, c])
                        
                        vec11 = self.GpB_vec[ia,ik,_f,:,c]
                        O1    = np.multiply.outer(vec11, vec11.conj())
                        m1   += self.GpB_eig[ia,ik,_f,c] * O1
                        
                        vec11 = self.GpC_vec[ia,ik,_f,:,c]
                        O1    = np.multiply.outer(vec11, vec11.conj())
                        m1   += self.GpC_eig[ia,ik,_f,c] * O1
                        
                        
                        vec12 = self.GpB_vec[ia,ik,_f+Nf,:,c]
                        O2    = np.multiply.outer(vec12, vec12.conj())
                        m2   += self.GpB_eig[ia,ik,_f+Nf,c] * O2
                        
                        vec12 = self.GpC_vec[ia,ik,_f+Nf,:,c]
                        O2    = np.multiply.outer(vec12, vec12.conj())
                        m2   += self.GpC_eig[ia,ik,_f+Nf,c] * O2
                    
                    assert np.allclose(m1, self._gp_matrices[ia][ik,_f,:,:])
                    assert np.allclose(m2, self._gp_matrices[ia][ik,_f+Nf,:,:])
        print('Eigendecompositon good', file = PrintFile)
        PrintFile.close()
    
    def Gamma_info(self,lead=0, tol = 1e-3,ik=0, f = None, return_vals=False, only_upper = True):
        """ 
            Useful function for fitting. 
            Give the descendingly largest components of the gamma matrix of 
            electrode lead. 
        """
        Gam = self.Nonortho_Gammas[lead]
        maxes = []
        idx   = []
        nb    = Gam.is_zero.shape[0]
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx'):
            sidx = self._old_sampling_idx[lead]
        
        sparsity = 0
        for I in range(nb):
            for J in range(I,nb):
                block = Gam.Block(I,J)
                if  block is None:
                    continue
                ni,nj = block.shape[-2:]
                for i in range(ni):
                    if only_upper and J == I:
                        jb = i
                    else:
                        jb = 0
                    for j in range(jb, nj):
                        _max = np.abs(block[ik,sidx,i,j]).max()
                        if _max>tol:
                            maxes += [_max]
                            idx   += [(I,J,i,j)]
                        sparsity += 1
        
        maxes = np.array(maxes)
        sort  = np.argsort(maxes)[::-1]
        maxes = maxes[sort]
        idx   = np.array(idx)[sort]
        if return_vals:
            return maxes, idx
        
        for i in range(len(maxes)):
            print(' Max, index: ', str(maxes[i]) + str(idx[i]))
        print('sparsity: ', len(maxes)*100/sparsity,'%' )
    
    def PoleAnalysis(self, lead,tol=1e-2,opts = {}, Wmax = 0.1, decimals = 2, ik = 0):
        """ Wrapper for SciPy signal analysis toolbox. Finds the location of 
            the peaks in the gamma-matrix elements which are larger than tol
            using the scipy find_peaks function.
            
            opts is a dictionary which is parsed to the find_peaks function.
            
            Any poles found by this function should have a widths maller 
            than Wmax.
            
            The poles are rounded to the decimal place given by decimal and
            combined. 
            
            Only the ik th k-point is considered. 
            
        """
        maxes, idx = self.Gamma_info(lead=lead, tol = tol, return_vals = True,ik = ik )
        P1,P2, W1, W2 = [],[],[],[]
        Maxes1,Maxes2 = [],[]
        for _i,v in enumerate(idx):
            I,J,i,j = v
            p1,p2,w1,w2,l = self.Inspect_Lorentzian_fit(lead,I,J,i,j,
                                                        scipy_peak_opts=opts, 
                                                        return_peaks =True, ik = ik);
            P1 += [l[p1]]
            P2 += [l[p2]]
            W1 += [w1]
            W2 += [w2]
            Maxes1 += [np.ones(len(w1))*maxes[_i]]
            Maxes2 += [np.ones(len(w2))*maxes[_i]]
            
        P1     = np.hstack(P1); P2     = np.hstack(P2)
        P1     = np.hstack((P1,P2))
        
        W1     = np.hstack(W1); W2     = np.hstack(W2)
        W1     = np.hstack((W1,W2))
        
        maxes  = np.hstack((np.hstack(Maxes1),np.hstack(Maxes2)))
        del P2, W2, Maxes2
        
        idx1   = np.where(W1<Wmax)[0]
        W1,P1  = W1[idx1], P1[idx1]
        maxes1 = maxes[idx1]
        P1     = np.round(P1,decimals)
        
        P1u    = np.unique(P1)
        W1u    = np.zeros(len(P1u))
        Prio   = np.zeros(len(P1u))
        for i in range(len(P1u)):
            idx     = np.where(P1 == P1u[i])[0]
            W       = W1[idx]
            mx      = maxes1[idx]
            W1u[i]  = (mx*W).sum()/(mx.sum()) # W.min()
            Prio[i] = mx.sum()
        if len(P1u)>0:
            idx = np.argsort(Prio)[::-1]
            return P1u[idx], W1u[idx], Prio[idx]/Prio.max()
        else:
            return np.array([]),np.array([]),np.array([])
    
    def PoleGuess(self, lead, NL, Emin, Emax,
                  fact = 1.0, tol=1e-2, opts = {}, 
                  Wmax = 0.1, decimals = 2, 
                  cutoff = 0.05, ik = 0, 
                  pole_dist = 1.0, pole_fact = 1.0, fillmode = 'adaptive'):
        """
            opts:  is parsed to scipy.signal find_peaks
            fact:  is the coarse grid not at the poles
            pole_dist:  distance is the distance from the coarse poles
            pole_fact:  gets multiplied onto the initial guesses in the narrower pole widths.
        """
        P,W,Pr = self.PoleAnalysis(lead, opts=opts, 
                                   tol=tol, Wmax=Wmax, 
                                   decimals=decimals, ik = ik)
        #if P is None:
        #    return None, None
        print(P.shape)
        if P.shape[0] == 0:
            print('No poles identified... giving you a uniform grid...')
            print('---> Problem: Specifically, P had shape '+str(P.shape)+' inside the PoleGuess function')
            Ei    = np.linspace(Emin, Emax, NL)
            Gi    = np.ones(NL)*fact*0.1*(Emax - Emin)/NL**0.5
            return Ei, Gi
        if not (P.min()>Emin and P.max()<Emax):
            print('Make Emin/Emax smaller/larger')
            assert  1 == 0
        P, W   = P[Pr>cutoff], W[Pr>cutoff]
        W *= pole_fact
        NP = len(P)
        no = NL - NP
        
        g0 = np.ones(no)*fact*0.1*(Emax - Emin)/no**0.5
        dense    =  np.linspace(Emin,Emax, 100 * no)
        dx       =  dense[1] - dense[0]
        dij      =  np.abs(dense[:,None] - P[None,:]).min(axis=1)
        dense    =  dense[dij>pole_dist * g0[0]]
        diff     =  np.diff(dense)
        stops    =  np.where(diff>2*dx)[0]
        segments =  [(Emin, dense[stops[0]])]
        for i in range(len(stops)-1):
            segments += [(dense[stops[i]+1], dense[stops[i+1]])]
        segments+= [(dense[stops[-1]+1], Emax)]
        Lengths  = [s[1] - s[0] for s in segments]
        Ltot     = sum(Lengths)
        Num      = [int(l * no/Ltot) for l in Lengths]
        Num[Num.index(max(Num))] -= sum(Num) - no
        Ec       = []
        for i in range(len(Num)):
            e0,e1  = segments[i]
            npol   = Num[i]
            dE     = e1 - e0
            Ec    += [np.linspace(e0 + dE/(2*npol),e1-dE/(2*npol),npol)]
        
        Ec    = np.hstack(Ec)
        if fillmode == 'adaptive':
            Ei    = np.hstack([Ec, P])
            Gi    = np.hstack([g0, W])
        elif fillmode =='linear':
            Ei    = np.hstack([np.linspace(Ec.min(), Ec.max(), len(Ec)), P])
            Gi    = np.hstack([g0, W])
        sort  = np.argsort(Ei)
        return Ei[sort], Gi[sort]
    
    def adapt_sampling(self, dist = 5.0, dx_coarse = 0.25,
                       # kw for PoleAnalysis
                       tol=1e-2, opts = {}, 
                       Wmax = 0.1, decimals = 2, 
                       ik = 0, cutoff = 0.01):
        """
         dist:  Wi*dist is the radius from which a dense sampling is kept.
         Poles: Poles[e] = Ei  + 1j*Pi are the positions and widths of the samplings 
        """
        C = self.Contour
        if hasattr(self, '_old_sampling_idx'):
            self.sampling_idx = self._old_sampling_idx
        
        new_sampling_idx  =  []
        for e in range(self.num_leads):
            sidx = []
            #print(e)
            Ei, Wi,Pr = self.PoleAnalysis(e, tol = tol, opts = opts, Wmax=Wmax,decimals=decimals, ik=ik)
            #print(Ei)
            Ei, Wi   = Ei[Pr>cutoff], Wi[Pr>cutoff]
            for i in self.sampling_idx[e]:
                if (np.abs(C[i] - C[sidx])>dx_coarse).all() or (np.abs(C[i]-Ei)<dist*Wi).any():
                    sidx += [i]
            new_sampling_idx += [sidx]
        self._old_sampling_idx = self.sampling_idx.copy()
        del self.sampling_idx
        self.sampling_idx = new_sampling_idx
    
    def Inspect_Lorentzian_fit(self, lead,I,J,i,j,Emin = -3.0, Emax = 3.0, 
                               ik = 0,size = 2, center_lines = False, 
                               source='sampled_gl',
                               n_samples        = 100,
                               return_result    = False,
                               scipy_peak_opts  = {},
                               scipy_peaks      = True,
                               return_peaks     = False,
                               pinfo_size       = 8.0,
                               width_nn         = 2,
                               use_dense_grid   = False, 
                               logplot          = False,
                               add_flipped_poles= False,
                               cl_lims = (None, None),
                               NO = True,
                               unit_x = '[eV]',
                               unit_y = '[eV]'
                               ):
        
        # m1 = self.sampled_gl_matrices[lead]
        """ This function is meant to compare the fit to the actual sampled Gammas
        coming from TBtrans. 
        
        I,J denotes the overall blocks, the nonzero elements can be found 
        inspecting the .iszero class attribute of the block_sparse and block_td
        classes. In particular, try e.g. to print .Nonortho_Gammas[0].is_zero 
        if you have already used the read_data command. 
        
        i,j are the sub-indices within the particular block chosen by I and J.
        
        
        The rest of the arguments can be familiarized by experimentation.
        
        """
        #m1 = self.Ortho_Gammas[lead].get_e_subset(self.sampling_idx[lead])
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx') and use_dense_grid:
            sidx = self._old_sampling_idx[lead]
        
        #m1 = self.Ortho_Gammas[lead].Block(I,J)[ik][sidx,i,j]
        if NO:
            m1 = self.Nonortho_Gammas[lead].Block(I,J)[ik][sidx,i,j]
        else:
            m1 = self.Ortho_Gammas[lead].Block(I,J)[ik][sidx,i,j]
        E  = np.linspace(Emin,Emax,n_samples)
        plt.show()
        line = self.Contour[sidx].real
        
        if logplot:
            def M(y):
                return np.log10(np.abs(y))
        else:
            def M(y):
                return y
        
        if not return_peaks:
            mm1 = M(m1.real)
            mm2 = M(m1.imag)
            plt.scatter(line, mm1,label = r'Re[Sampled $\Gamma$]',marker = '*',s = size, color='b')
            plt.scatter(line, mm2,label = r'Im[Sampled $\Gamma$]',marker = '*',s = size, color='y')
            diff = min((mm1.min(), mm2.min())) - max((mm1.max(), mm2.max()))
            plt.ylim(min((mm1.min(), mm2.min())) - diff*0.05,
                     max((mm1.max(), mm2.max())) + diff*0.05
                    )
        if hasattr(self, 'NO_fitted_lorentzians'):
            if NO:
                m2 = self.NO_fitted_lorentzians[lead]
            else:
                m2 = self.fitted_lorentzians[lead]
            #y  = m2.evaluate_Lorentzian_basis(E).Block(I,J)[ik,:,i,j]
            _M = m2.Block(I,J)[ik,:,i,j][None,:,None,None]
            y = evaluate_Lorentz_basis_matrix(_M, E, m2.ei, m2.gamma, tol = 1e-15)[0,:,0,0]
        else:
            y = np.zeros(len(E))*np.nan
        
        if return_result:
            resdic = {'E1':line, 'Gsamp':m1, 'E2':E, 'Gfit': y}
            return resdic
        
        if scipy_peaks:
            opts = {'distance':8,
                    'prominence':(1.0, None),
                    'height':0.1,
                    }
            
            opts.update(scipy_peak_opts)
            x = np.abs(m1.real)            
            peaks1, prop = find_peaks(x,**opts)
            if add_flipped_poles:
                _peaks1,_ = find_peaks(-x,**opts)
                peaks1 = np.hstack((peaks1, _peaks1))
            widths1 = np.zeros(len(peaks1))
            it = 0
            for p in peaks1:
                x     = line[p]
                r     = m1[p].real
                xr,xl = line[p+1], line[p-1]
                yr,yl = m1[p+1].real/r,m1[p-1].real/r
                width = min(np.abs(xr-x)*np.sqrt(yr)/(np.sqrt(1-np.abs(yr))) ,
                            np.abs(xl-x)*np.sqrt(yl)/(np.sqrt(1-np.abs(yl)))  )
                
                if not return_peaks:
                    plt.annotate(str('E = '+str(np.round(line[p],3)))+': '+str(np.round(width,3)), (line[p]-0.8, M(r+0.2)), size=pinfo_size)
                    xlin = np.linspace(x-20*width, x+20*width,200)
                    plt.plot(xlin,M(r*width**2/((xlin-x)**2 + width**2)), linewidth=1.0,alpha=0.55)
                
                widths1[it] = width
                it+=1
            
            x = np.abs(m1.imag)
            peaks2, prop = find_peaks(x, **opts)
            if add_flipped_poles:
                _peaks2,_ = find_peaks(-x,**opts)
                peaks2 = np.hstack((peaks2, _peaks1))
            
            widths2 = np.zeros(len(peaks2))
            it = 0
            for p in peaks2:
                x     = line[p]
                r     = m1.real[p]
                xr,xl = line[p+1], line[p-1]
                yr,yl = m1.real[p+1]/r,m1.real[p-1]/r
                width = min(np.abs(xr-x)*np.sqrt(yr)/(np.sqrt(1-np.abs(yr)))   ,
                            np.abs(xl-x)*np.sqrt(yl)/(np.sqrt(1-np.abs(yl)))    )
                if not return_peaks:
                    plt.annotate(str('E = '+str(np.round(line[p],3)))+': '+str(np.round(width,3)), (line[p]-0.8, M(r+0.2)), size=pinfo_size)
                    xlin = np.linspace(x-5*width, x+5*width,25)
                    plt.plot(xlin,M(r*width**2/((xlin-x)**2 + width**2)), linewidth=1.0,alpha=0.55)
                
                widths2[it] = width
                it+=1
            
            if return_peaks:
                return peaks1, peaks2, widths1, widths2, line
            
            plt.plot(line[peaks1], M(m1.real[peaks1]), "x", color='blue')
            plt.plot(line[peaks2], M(m1.imag[peaks2]), "x", color='r')
        
        plt.plot(E, M(y.real),label = 'Re[Lorentz fit]',linestyle = 'dashed')
        plt.plot(E, M(y.imag),label = 'Im[Lorentz fit]',linestyle = 'dashed')
        
        if center_lines:
            ei   = m2.ei.copy()
            g    = m2.gamma.copy()
            
            if cl_lims[1] is None:
                ymax =  max((y.real.max(), y.imag.max()))
            else:
                ymax = cl_lims[1]
            
            if cl_lims[0] is None:
                ymin =  max((y.real.min(), y.imag.min()))
            else:
                ymin = cl_lims[0]
            
            for i in range(len(ei[ik])):
                plt.vlines(ei[ik, i], ymin, ymax, alpha = 0.25)
                plt.annotate(str(i), (ei[ik,i], ymax + (ymax- ymin)/100), size = 4)
        plt.legend()
        plt.xlabel(r'$E$ '+unit_x)
        if logplot: plt.ylabel(r' $\log_{10}\Gamma(E)$')
        else:       plt.ylabel(r'$\Gamma(E)$ '+unit_y)
    
    def SE_from_lorentzian_fit(self,E, NO = True):
        """ 
           Compute the self-energy using the fit by a Hilbert transformation 
           E is the E-grid on whihc to calculate the self-energy. 
           NO = True gives the SE in the nonorthogonal basis, while False
           gives it in the lowdin orthogonalized basis. 
        """
        
        out = []
        if NO:
            Gams = self.NO_fitted_lorentzians
        else:
            Gams = self.fitted_lorentzians
        for G in Gams:
            im = G.evaluate_Lorentzian_basis(E).scalar_multiply(-1j*1/2)
            re = G.hilbert_transform_lorentzian(E).scalar_multiply(1/2)
            out+=[re.Add(im)]
        self._temp_calculated_SE = out
        return out
    
    def FitNO2O(self):
        """ Used e.g. when curvefit_all has been used, to write the new fitted gammas
        in the non-orthogonal basis to the gammas in the orthognalized basis"""
        for i in range(self.num_leads):
            NO_gel = self.NO_fitted_lorentzians[i]
            L      = self.Lowdin[0]
            gel = L.BDot(NO_gel).BDot(L)
            gel.Lorentzian_basis = True
            gel.ei, gel.gamma = NO_gel.ei.copy(), NO_gel.gamma.copy()
            self.fitted_lorentzians[i] = gel
    
    def setLval(self, lead, I, J, i, j,val, ik = 0, iL = None, hermitian = True):
        """ Internal function used in the fitting algorithm. 
            Sets the value of a particular component of the nonorthogonal 
            Lorentzian fit.
        """
        if iL is None:
            iL = np.arange(self.NumL)
        self.NO_fitted_lorentzians[lead].Block(I,J)[ik,iL,i,j] = val
        if hermitian:
            self.NO_fitted_lorentzians[lead].Block(J,I)[ik, iL, j,i] = np.conj(val)
    
    def printLval(self, lead, I, J, i, j, iL, ik = 0):
        """
            Prints the value of a particular component of the nonorthogonal 
            lorentzian fit.
        """
        print('Lval: ', )
        return self.NO_fitted_lorentzians[lead].Block(I,J)[ik,iL,i,j]
    
    
    def run_curvefit(self, lead, I,J,i,j, ik = 0, use_dense_grid = True, fix_L_idx = []):
        """
            Runs scipy curvefit on the component of the nonorthognal Gamma 
            denoted by the tuple(lead,I,J,i,j). Can sometimes give better 
            results than the matrix inversion based method used in the self.Fit
            function. Uses whatever value is already stored in NO_fitted_lorenzians
            as an initial guess.
        
        """
        
        bij = self.NO_fitted_lorentzians[lead].Block(I,J)
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx') and use_dense_grid:
            sidx = self._old_sampling_idx[lead]
        if bij is not None:
            g0   = self.NO_fitted_lorentzians[lead].Block(I,J)[ik,:,i,j].copy()
        else:
            g0 = np.zeros(len(self.NO_fitted_lorentzians[lead].ei[0,:]))
            return None, None
        m1     = self.Nonortho_Gammas[lead].Block(I,J)[ik][sidx,i,j]
        line   = self.Contour[sidx].real
        ei, wi = self.NO_fitted_lorentzians[lead].ei[ik].copy(),self.fitted_lorentzians[lead].gamma[ik].copy()
        idx0   = np.array(fix_L_idx, dtype=int)
        idx1   = np.setdiff1d(np.arange(len(ei)), idx0)
        
        G1 = g0[idx0] # fixed vals
        G0 = g0[idx1] # variable vals
        N  = len(g0)
        def func(x, *p):
            _p = np.zeros((3, N))
            _p[0] = wi
            _p[1] = ei
            _p[2,idx1] = np.array(p).real
            _p[2,idx0] = G1.real
            return L_sum(x, _p).real
        poptr, pcov = curve_fit(func, line, m1.real, p0=G0.real,**{'maxfev':50*10**5})
        def func(x, *p):
            _p = np.zeros((3, N))
            _p[0] = wi
            _p[1] = ei
            _p[2,idx1] = np.array(p).real
            _p[2,idx0] = G1.imag
            return L_sum(x, _p).real
        popti, pcov = curve_fit(func, line, m1.imag, p0=G0.imag,**{'maxfev':50*10**5})
        POPT        = np.zeros(N,dtype=complex)
        POPT[idx0]  = G1
        POPT[idx1]  = poptr + 1j*popti
        return POPT, g0
    
    def curvefit_all(self, tol, ik = 0,fix_L_idx = []):
        """
            Runs the curvefit function on all components where the exact gamma
            is larger than tol. has to be run for each kpoint specified by ik
        
        """
        for e in range(self.num_leads):
            mx, idx = self.Gamma_info(e, tol = tol, ik = ik, return_vals = True)
            for count in tqdm(range(len(idx))):
                I,J,i,j = idx[count]
                rn, ro  = self.run_curvefit(e,I,J,i,j, ik = ik , fix_L_idx=fix_L_idx)
                if rn is not None:
                    self.setLval(e, I, J, i, j, rn)
    
    def Inspect_Transmission(self,i, j,lead = 0, kpnt = None,
                             return_result = False, invinp = None):
        print('\n Subbed transmission vs TBtrans transmission. \n')
        '''
        This function is for checking the transmission and comparing your subbed 
        transmission. Also, the Gammas in the orthogonal basis is used. Which are 
        the ones directly plotted for comparison when you call "Inspect_Lorentzian_fit".
        '''
        g1      = self.Ortho_Gammas[i]
        g2      = self.Ortho_Gammas[j]
        L, iL   = self.Lowdin
        if invinp is None:
            G_NO    = self.Nonortho_iG.Invert()
        else:
            G_NO    = self.Nonortho_iG.Invert(invinp)
        
        G       = iL.BDot(G_NO).BDot(iL)
        M1 = g1.BDot(G); G.do_dag()
        M2 = g2.BDot(G); G.do_dag()
        _T01 = M1.TrProd(M2)
        if return_result:
            return _T01        
        
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx'):
            sidx = self._old_sampling_idx[lead]
        
        
        plt.show()
        if kpnt == None:
            T01 = (_T01*self.tbtwkpt[:,np.newaxis]).sum(axis = 0)
            plt.plot(self.tbtE[sidx]  , self.tbtT[sidx], label = 'TBtrans')
            plt.plot(self.tbtE[sidx]  , T01[sidx], label = 'This code')
            plt.ylim(-0.01,None)
            plt.title('Transmissions')
            plt.legend()
            plt.show()
        else:
            T01 = _T01[kpnt]
            plt.plot(self.tbtE[sidx]  , self._tbtTk[kpnt,sidx], label = 'TBtrans')
            plt.plot(self.tbtE[sidx]  , T01[sidx], label = 'This code')
            plt.ylim(-0.01,None)
            plt.title('Transmissions')
            plt.legend()
            plt.show()
    
    def Inspect_Poles(self,lead, ik = 0,size = 5, marker_L = '*', marker_F = '.', fermi_poles=True):
        """
           Prints a picture of the poles used in the fit. 
           It simply uses the ei and gamma arrays which is attributes in the 
           self.fitted_lorentzians[i] block-sparse matrix classes.
           
           There are some plotting parameters in the function arguments.
           
        """
        zl = self.fitted_lorentzians[lead].ei[ik] +1j * self.fitted_lorentzians[lead].gamma[ik]
        zf = self.F_poles[lead]
        fig = plt.figure(figsize=(8,1))
        ax = fig.add_subplot(111)
        ax.scatter(zl.real, zl.imag, marker = marker_L, s = size, label = 'Lorentzian poles',c='b')
        if fermi_poles:
            plt.scatter(zf.real, zf.imag, marker = marker_F, s = size, label = 'Fermi poles', c='r')
        plt.title('Poles associated with lead ' + str(lead)+'.')
        plt.xlabel(r'Re[$\chi$]')
        plt.ylabel(r'Im[$\chi$]')
        dz = zl.real.max()- zl.real.min()
        plt.xlim(zl.real.min()-dz/20, zl.real.max()+dz/20)
        outer  = np.abs(np.subtract.outer(zl, zf))
        outer2 = np.abs(np.subtract.outer(zl,zl))
        idx = np.arange(len(zl))
        outer2[idx,idx] = 10**6
        plt.legend()
        print('Minimum distance from Fermi-poles to Lorentzian poles: ' + str(outer.min()) + '/n')
        print('Minimum distance between Lorentzian poles: ' + str(outer2.min()) + '/n')
        print('Minimum value of imaginary part of Lorentzian poles: ' + str(zl.imag.min()))
        plt.savefig('Poles_lead_'+ str(lead)+'_k_'+str(ik)+'.svg')
        
    
    def add_broadening(self,lead, val, ik = 0,idx = None):
        if idx is None:
            idx = np.arange(self.NumL)
        self.fitted_lorentzians[lead].gamma[ik,idx]+=val
    
    def add_move_center(self,lead, val, ik = 0, idx = None):
        if idx is None:
            idx = np.arange(self.NumL)
        self.fitted_lorentzians[lead].ei[ik,idx]+=val
    
    def Get_DOS(self,ik = -1):
        print('ALSO CONTAINS DOS SAMPLED IN FERMI POLES!!!!!!!!!!!!!!!!!!')
        DOS    = self.Nonortho_iG.Invert(BW = 1).Tr()#.imag/np.pi
        
        if ik == -1:
            DOS = -DOS.sum(axis = (0)).imag/np.pi
        else:
            DOS = -DOS[ik,:,:].sum(axis = 1).imag/np.pi
        return DOS
    
    def get_orbital_values(self, dx):
        """
            We can read in the basis of the siesta calculation in the 
            self.Device calculator object, given it has been run and 
            writes the ion.nc files (see siesta compilation dependencies.)
            
        
        """
        B,g = self.Device.read_basis_and_geom()
        VALS = []
        for A in range(len(B)):
            vals = []
            for orb in B[A]:
                x = np.arange(0,orb.R + 2 * dx, dx)
                x[0] = 0.001
                x = np.hstack((np.flip(x[1:]),x))
                
                X,Y,Z = np.meshgrid(x,x,x)
                R     = np.stack((X,Y,Z), axis = -1)
                f     = orb.psi(R)
                vals += [f]
            VALS+=[vals]
        return B,g,VALS
    
    # def Inspect_transmission_from_SE_fit(self,E = np.linspace(-5,5,100), eta = 0.0, i=0, j =1, lead = 0):
    #     S = self.BTD_overlap
    #     H = self.Nonortho_Hamiltonian
    #     if len(self.fitted_self_energies)==0:
    #         print('no fitted self energies!')
    #         return
    #     E = E[np.newaxis, :, np.newaxis, np.newaxis]#+1j * eta
    #     iG = (S.scalar_multiply(E+1j*eta)).add_to_btd(H.scalar_multiply(-1.0))
    #     Gammas = []
    #     for se in self.fitted_self_energies:
    #         se_eval = se.evaluate_Lorentzian_basis(E[0,:,0,0]).scalar_multiply(-1.0)
    #         iG = iG.add_to_btd( se_eval )
    #         se_eval_dag = se_eval.copy()
    #         se_eval_dag.do_dag()
    #         Gammas.append((se_eval.Subtract(se_eval_dag)).scalar_multiply(1j))
    #     G = iG.Invert()
    #     g1      = Gammas[i]
    #     g2      = Gammas[j]
    #     nk = len(g1.vals[0])
    #     slices  = self.Nonortho_iG.all_slices
    #     M1 = g1.BDot(G); G.do_dag()
    #     M2 = g2.BDot(G); G.do_dag()
    #     T01 = ((M1.TrProd(M2))*self.tbtwkpt[:,np.newaxis]).sum(axis = 0)
    #     plt.plot(self.tbtE[self.sampling_idx[lead]] , self.tbtT[self.sampling_idx[lead]], label = 'TBtrans')
    #     plt.plot(E[0,:,0,0] , T01, label = 'Self energies from Lorentzian fit')
    
    def bs2np(self, A):
        """
            Convert a block_sparse matrix class which follows the block pattern
            of the calculation in the self.Device calculation to a dense matrix.
            The block_sparse matrix we are trying to convert must match the
            self._Slices which specifies the block structure.
        """
        return Blocksparse2Numpy(A, self._Slices)
    
    def Inspect_fitted_transmission(self, Eg, eta=1e-3, i=0,j=1, 
                                    return_result=False, kpnt= None, lead=0):
        """
        Not really tested, might be wrong. It seems to be there from when
        the code was being concieved. 
        """
        S   = self.BTD_overlap
        H   = self.Nonortho_Hamiltonian
        if (Eg.imag==0).all():
            Eg = Eg + 1j*eta
        iG  = S.scalar_multiply(-Eg[None,:,None,None]).add_to_btd(H)
        SEs = self.SE_from_lorentzian_fit(Eg, NO=True)
        for se in SEs:
            iG = iG.add_to_btd(se)
        iG = iG.scalar_multiply(-1.0)
        gi = self.NO_fitted_lorentzians[i].evaluate_Lorentzian_basis(Eg)
        gj = self.NO_fitted_lorentzians[j].evaluate_Lorentzian_basis(Eg)
        idxi = gi.is_zero.sum(axis=1)
        idxj = gj.is_zero.sum(axis=0)
        idxi = np.where(idxi>0)[0]
        idxj = np.where(idxj>0)[0]
        mask = np.zeros(gi.is_zero.shape,dtype=int)
        mask[idxi[:,None],idxj[None,:]] = 1
        for i in range(mask.shape[0]):
            mask[i,i] = 1
        #return mask
        
        G  = iG.Invert(mask)
        M1 = gi.BDot(G); G.do_dag()
        M2 = gj.BDot(G); G.do_dag()
        _Tij = M1.TrProd(M2)
        Tij = (_Tij * self.tbtwkpt[:,None]).sum(axis = 0)
        
        if return_result:
            return Eg, Tij 
        
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx'):
            sidx = self._old_sampling_idx[lead]
        
        if kpnt == None:
            plt.plot(self.tbtE[sidx] , self.tbtT[sidx], label = 'TBtrans')
            plt.plot(Eg ,Tij, label = 'Self energies from Lorentzian fit')
        else:
            plt.plot(self.tbtE[sidx] , self._tbtTk[kpnt, sidx], label = 'TBtrans')
            plt.plot(Eg, _Tij[kpnt])
    
    def Inspect_transmission_from_hilbert_transform(self, 
                                                    E = np.linspace(-5,5,100), 
                                                    eta = 0.0, 
                                                    i=0, j=1,
                                                    lead  = 0,
                                                    NO    = True,
                                                    kpnt  = None, figure =None,
                                                    return_result = False):
        
        """
            Allows for comparison of the transmission when the gammas
            from the fit is being used, together with the correction parsed 
            to self.Renormalise_H.
            numpy linalg inv is used for the matrix inversion, so scales 
            pretty badly.
            
            Parameters:
                E : np.linspace like array of floats in which to calculate 
                    the transmission function. Dont add any imaginary part to this.
                eta : float, value of eta for the broadening used when calculating the Greens function for the system.
                i / j: electrode label, from / to
                NO : use the nonorthogonal basis (True, False)
                kpnt: int, specify which k-point you want to calculate the transmission for
                return_result: True/False, return the calculated transmission in an array.
        """
        
        if len(self.fitted_lorentzians)==0:
            print('\n No fitted lorentzians!\n')
            return
        if NO:
            S = self.bs2np(self.BTD_overlap)
            H = self.bs2np(self.Nonortho_Hamiltonian)
            if hasattr(self, "Hamiltonian_renormalisation_correction"):
                H += (self.bs2np(self.Lowdin[1]))@self.Hamiltonian_renormalisation_correction@(self.bs2np(self.Lowdin[1]))
            g1  = self.bs2np(self.NO_fitted_lorentzians[i].evaluate_Lorentzian_basis(E))
            g2  = self.bs2np(self.NO_fitted_lorentzians[j].evaluate_Lorentzian_basis(E))
            SEs = [self.bs2np(a) for a in self.SE_from_lorentzian_fit(E, NO = True)]
        else:
            H   = self.Hdense
            S   = np.eye(H.shape[-1])[np.newaxis,np.newaxis,:,:]
            g1  = self.bs2np(self.fitted_lorentzians[i].evaluate_Lorentzian_basis(E))
            g2  = self.bs2np(self.fitted_lorentzians[j].evaluate_Lorentzian_basis(E))
            SEs = [self.bs2np(a) for a in self.SE_from_lorentzian_fit(E, NO = False)]
        Eg  = E[np.newaxis, :, np.newaxis, np.newaxis]
        G   = S * (Eg  + 1j * eta)  - H - sum(SEs)
        G   = np.linalg.inv(G)
        _Tij = np.trace(g1@G@g2@(G.transpose(0,1,3,2).conj()), axis1 = 2, axis2 = 3)
        Tij = (_Tij * self.tbtwkpt[:,None]).sum(axis = 0)
        
        if return_result:
            return Eg, Tij 
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx'):
            sidx = self._old_sampling_idx[lead]
        
        if kpnt == None:
            plt.plot(self.tbtE[sidx] , self.tbtT[sidx], label = 'TBtrans')
            plt.plot(E ,Tij, label = 'Self energies from Lorentzian fit')
            
        else:
            plt.plot(self.tbtE[sidx] , self._tbtTk[kpnt, sidx], label = 'TBtrans')
            plt.plot(E, _Tij[kpnt], label = 'Self energies from Lorentzian fit')
        plt.legend()
        plt.ylim(0,None)
        if figure is not None:
            plt.savefig(figure)
        
        
        
    def Inspect_dos_from_hilbert_transform(self, 
                                           E = np.linspace(-5,5,100), 
                                           eta = 0.0, 
                                           i=0, j=1,
                                           lead = 0,
                                           kpnt = None,
                                           return_result = False):
        
        if len(self.fitted_lorentzians)==0:
            print('\n No fitted lorentzians!\n')
            return
        S = self.bs2np(self.BTD_overlap)
        H = self.bs2np(self.Nonortho_Hamiltonian)
        if hasattr(self, "Hamiltonian_renormalisation_correction"):
            H += (self.bs2np(self.Lowdin[1]))@self.Hamiltonian_renormalisation_correction@(self.bs2np(self.Lowdin[1]))
        I = np.eye(H.shape[-1])[np.newaxis,np.newaxis,:,:]
        g1  = self.bs2np(self.NO_fitted_lorentzians[i].evaluate_Lorentzian_basis(E))
        g2  = self.bs2np(self.NO_fitted_lorentzians[j].evaluate_Lorentzian_basis(E))
        SEs = [self.bs2np(a) for a in self.SE_from_lorentzian_fit(E)]
        Eg  = E[np.newaxis, :, np.newaxis, np.newaxis]
        G   = S * (Eg  + 1j * eta)  - H - sum(SEs)
        G   = np.linalg.inv(G)
        dos = -np.trace(G,axis1=2,axis2=3).imag/np.pi
        
        if return_result:
            return Eg,dos
        
        if kpnt == None:
            dos = (dos * self.tbtwkpt[:,None]).sum(axis = 0)
            plt.plot(E , dos, label = r'From Lorentzian fit')
        else:
            dos = dos[kpnt]
            plt.plot(E, dos, label = r'From Lorentzian fit')
    
    def Inspect_SE_from_hilbert_transform(self, lead, I, J, i, j, Emin = -3.0, Emax = 3.0, 
                                          ik = 0, size = 2,
                                          return_result = False,
                                          n_samples = 200,
                                          NO = False):
        if not hasattr(self, 'Hamiltonian_renormalisation_correction'):
            self.Hamiltonian_renormalisation_correction=np.zeros((self.Ortho_Hamiltonian.vals[0].shape[0],1,self.n_orb, self.n_orb), dtype=complex)
        
        if NO == False:
            m1   = self.Lowdin[0].BDot(self.self_energies[lead]).BDot(self.Lowdin[0])
            HC   = self.Hamiltonian_renormalisation_correction
        else:
            iL = self.bs2np(self.Lowdin[1])
            m1 = self.self_energies[lead]
            HC = iL@self.Hamiltonian_renormalisation_correction@iL
        
        E    = np.linspace(Emin,Emax,n_samples)
        plt.show()
        sidx = self.sampling_idx[lead]
        if hasattr(self, '_old_sampling_idx'):
            sidx = self._old_sampling_idx[lead]
        
        plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].real,label = r'Re[Sampled $\Sigma$]',marker = '*',s = size, color='darkblue')
        plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].imag,label = r'Im[Sampled $\Sigma$]',marker = '*',s = size, color='pink')
        RES = self.SE_from_lorentzian_fit(E, NO = NO)[lead]
        if hasattr(self, "Hamiltonian_renormalisation_correction"):
            string=' (Shifted!)'
            si,sj = self._Slices[I][J]
            block = RES.Block(I,J) + HC[ik,0,si,sj]#self.Hamiltonian_renormalisation_correction[ik,0,si,sj]
        else:
            string = ''
            block = RES.Block(I,J)
        
        if return_result:
            return E,block[ik,:,i,j]
        
        plt.plot(E,   block[ik,:,i,j].real,label = r'Hilbert Transform'+string,linestyle = 'dashed', color='blue')
        plt.plot(E,   block[ik,:,i,j].imag,label = r'Lorentz fit of $-\Gamma/2$'+string,linestyle = 'dashed', color='red')
        plt.legend()
    
    def write_to_file(self, name='TDT', compressed=True):
        if compressed:
            from Writer import write_to_file_compressed as write
        else:
            from Writer import write_to_file as write
        write(self, name)
    
    def figures(self,subE = None, manual_D = None, 
                axes=[0,1], custom_bp = None, spin=0, 
                bs_lim = (-5, 5), Eg=np.linspace(-5,5,500),
                eta=1e-2, n_gammas = 10, folder='publish',
                scipy_peaks = True, center_lines = False, ik = 0,
                logplot=False, custom_idx = None, Eg_eig = None, loglim = (-5,3),
                n_split = 10):
        """
            Function prints a lot of svg figures to the folder specfied.
            see keyword arguemnt names and experiment with them to get
            a feeling for them. 
        
        """
        
        self.Device.figures(subE = self.sampling_idx[0], manual_D = manual_D, axes=axes, custom_bp = custom_bp, spin=spin)
        plt.close()
        C = Eg.real + 1j*eta
        print('Plotting Gammas!')
        print('....')
        try:
            os.mkdir(folder)
        except:
            pass
        
        #self.Inspect_Lorentzian_fit(0,30,30,10,10,center_lines=True, n_samples=1000); plt.xlim(-5,5)
        for i in range(self.num_leads):
            self.Inspect_Poles(i, ik, fermi_poles=False)
            plt.close()
            E1,E2 = Eg.min(), Eg.max()
            if custom_idx is not None:
                idx = custom_idx[i]
            else:
                _, idx   = self.Gamma_info(i, ik = ik, return_vals = True)
            for v in idx[0:n_gammas]:
                i1,i2,i3,i4 = v
                self.Inspect_Lorentzian_fit(i,i1,i2,i3,i4, 
                                            Emin = Eg.min(), Emax=Eg.max(), 
                                            ik = ik, logplot=logplot, n_samples=len(Eg))
                plt.xlim(E1,E2)
                plt.savefig(folder + '/Gamma_lead_'+str(i)+'_k_'+str(ik)+'_'+str(v)+'.svg')
                plt.close()
                self.Inspect_SE_from_hilbert_transform(i,i1,i2,i3,i4, 
                                                         Emin = Eg.min(), Emax=Eg.max(), 
                                                         ik = ik, n_samples=len(Eg))
                plt.xlim(E1,E2)
                plt.savefig(folder + '/SE_lead_'+str(i)+'_k_'+str(ik)+'_'+str(v)+'.svg')
                plt.close()
        for i in range(self.num_leads):
            for j in range(i+1, self.num_leads):
                self.Inspect_transmission_from_hilbert_transform(E=Eg, eta=eta,i =i , j = j)
                plt.xlabel(r'$E$ [eV]')
                plt.ylabel(r'$T(E)$')
                plt.savefig(folder+'/Transmission_'+str(i)+'_'+str(j)+'.svg')
                plt.close()
                self.Inspect_transmission_from_hilbert_transform(E=Eg, eta=eta, i=i, j=j, kpnt=ik)
                plt.xlabel(r'$E$ [eV]')
                plt.ylabel(r'$T(E)$')
                plt.savefig(folder+'/Transmission_ik_'+str(ik)+'_'+str(i)+'_'+str(j)+'.svg')
                plt.close()
        
        if Eg_eig is not None:
            for i in range(self.num_leads):
                ee_s     = []
                Eg_eig_p = np.array_split(Eg_eig, n_split)
                for ES in Eg_eig_p:
                    ee  = self.bs2np(self.NO_fitted_lorentzians[i].evaluate_Lorentzian_basis(ES.real).eig(self._Slices, hermitian=True)[0])
                    eed = np.array([ee[:,:,i,i] for i in range(self.n_orb)])
                    ee_s += [eed]
                eed = np.concatenate(ee_s,axis=2)
                plt.plot(Eg_eig.real,np.log10(np.min(eed, axis=(1)).T))
                plt.xlabel(r'$E$ [eV]')
                plt.ylabel(r'$\log_{10}\lambda[\mathbf{\Gamma}_{gr}]$')
                plt.ylim(loglim)
                plt.title('Minimum eigenvalue computed = '+str( eed.min()))
                plt.savefig(folder+'/SpectrumLead_'+str(i)+'.svg')
                plt.close()
    
    def cite(self):
        from Zandpack.docstrings import __version__, CiteString, WebPage
        print('------------\n')
        print('Zandpack version: ' + __version__)
        print(CiteString)
        print('Code can be obtained from ' + WebPage)
    
    def make_f_general(self, parallel = False, fastmath = False, nogil = False):
        # The second index on H was for being able to multiply with energy-resolved quantities,
        # which is not needed anymore
        H      = self.Hdense[:,0,:,:].copy()
        Xpp    = self.Xpp.copy()
        Xpm    = self.Xpm.copy()
        GG_P   = self.GG_P.copy()
        GG_M   = self.GG_M.copy()
        GL_P   = self.GL_P.copy()
        GL_M   = self.GL_M.copy()
        nl     = self.num_leads
        nf     = self.num_poles
        no     = H.shape[2]
        nk     = H.shape[0]
        Ntot   = self.NumL + nf
        omega_idx  = self.omega_idx.copy()
        psi_idx    = self.psi_idx.copy()
        
        # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
        xi     = np.concatenate((
                                  self.Gl_vec.transpose(1,0,2,4,3)
                                  , 
                                  self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
                                 ), 
                                axis = 2
                                )
        
        xi = np.ascontiguousarray(xi)
        
        Ixi    = np.concatenate(
                                (
                                  self.Gl_vec.transpose(1,0,2,4,3).conj()
                                  ,
                                  self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
                                ),
                               axis = 2
                               )
        
        Ixi = np.ascontiguousarray(Ixi)
        
        diff_GGP_GLP = GG_P - GL_P
        diff_GGM_GLM = GG_M - GL_M
        self.diff_ggp_glp = diff_GGP_GLP
        self.diff_ggm_glm = diff_GGM_GLM
        
        self.xi  = xi
        self.Ixi = Ixi
        
        for a in range(nl):
            for k in range(nk):
                for xl in range(self.Gl_vec.shape[2]):
                    m = np.zeros((no,no), dtype = np.complex128)
                    for c in range(no):
                        vec1 =   self.Gl_vec[a,k,xl,:,c]
                        vec2 =   self.Gl_vec[a,k,xl,:,c].conj()
                        O    =  np.multiply.outer(vec1, vec2)
                        m   +=   self.Gl_eig[a,k,xl,c] * O
                    assert np.allclose(m, self._gl_matrices[a][k,xl,:,:])
                for xf in range(2 * nf):
                    m = np.zeros((no,no), dtype = np.complex128)
                    for c in range(no):
                        vec1 = self.Gp_vec[a,k,xf,:,c]
                        vec2 = self.Inv_Gp_vec[a,k,xf,c,:]
                        O    = np.multiply.outer(vec1, vec2)
                        m   += self.Gp_eig[a,k,xf,c] * O
                    assert np.allclose(m, self._gp_matrices[a][k,xf,:,:])
        
        @njit(parallel = parallel, fastmath = fastmath, nogil = nogil)
        def f(t, 
              old_sig, old_psi, old_omega,
              dH, delta_variant, dH_given = True):
            #Create the arrays needed
            dt = np.complex128
            D_psi    = np.zeros(old_psi.shape   , dtype = dt)
            D_omega =  np.zeros(old_omega.shape , dtype = dt)
            nk = old_sig.shape[0]
            
            # psi_dagger is needed:
            psi_tilde  = old_psi.conj()
            # Store bias at time t
            delta_t = np.zeros(nl)
            for a in range(nl): delta_t[a] = delta_variant(t, a)
            # Get Hamiltonian
            if dH_given: Ht =  H + dH(t, old_sig)
            else:        Ht =      dH(t, old_sig)
            
            ##### Density matrix EOM:
            D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
            for a in range(nl):
                pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
                D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            D_sig *= 1/hbar
            
            ##### Psi & Omega EOM:
            hbar_sq = hbar**2
            for k in range(nk):
                # Order the xi-vectors in columns (.T)
                xi_k = xi[k].reshape((nl * Ntot * no, no)).T
                old_omega_k = old_omega[k]
                old_sig_k   = old_sig[k,:,:].copy()
                for x in prange(Ntot):
                    for a in range(nl):
                        for c in range(no):
                            # We also use axc_idx etc. later
                            axc_idx   = omega_idx[a,x,c,:,:,:].copy()
                            psi_axc_1 = old_psi[k,a,x,c,:]
                            xi_axc_1  = xi[k,a,x,c,:]
                            xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
                            #We non-zero psi's are has an index that is greater than or equal to zero.
                            
                            ###
                            if psi_idx[a,x,c]>=0:
                                idx       = axc_idx.reshape((nl*Ntot*no))
                                # Use only non-zero Omegas
                                bOOl      =(idx >= 0 )
                                xi_sub    = xi_k[:,bOOl]
                                idx       = idx[bOOl]
                                omega_sub = old_omega_k[idx]
                                # Do sum in the last term of the EOM
                                omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                                # Make the part involving Hamiltonian
                                Hchi      = Ht[k,:,:].copy()
                                for diag in range(no):
                                    Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                                #Calculate
                                D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
                                                          GL_P[k,a,x,c] * xi_axc_1                            +\
                                                          diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
                                                          omega_sum/hbar_sq
                                                        )
                            
                            ###
                            # Differences between the Lambdas:
                            diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                            for a2 in range(nl):
                                for x2 in range(Ntot):
                                    for c2 in range(no):
                                        idx = axc_idx[a2,x2,c2]
                                        # Only non-zero terms calculated:
                                        if idx < 0:
                                            pass
                                        else:
                                            psi_axc_2    = psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
                                            xi_axc_2     = Ixi[k,a2,x2,c2,:]       # 
                                            diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
                                            xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            #Calculate
                                            D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
                                                                      diff_ggm_glm * xi_axc_2.dot(psi_axc_1)                +\
                                                                      diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
                                                              )
            
            return D_sig, D_psi, D_omega
        
        
        return f
    
    def make_f_experimental(self, parallel = False, fastmath = False, nogil = False):
        # The second index on H was for being able to multiply with energy-resolved quantities,
        # which is not needed anymore
        H      = self.Hdense[:,0,:,:].copy()
        Xpp    = self.Xpp.copy()
        Xpm    = self.Xpm.copy()
        GG_P   = self.GG_P.copy()
        GG_M   = self.GG_M.copy()
        GL_P   = self.GL_P.copy()
        GL_M   = self.GL_M.copy()
        nl     = self.num_leads
        nf     = self.num_poles
        no     = H.shape[2]
        nk     = H.shape[0]
        Ntot   = self.NumL + nf
        omega_idx  = self.omega_idx.copy()
        psi_idx    = self.psi_idx.copy()
        
        # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
        xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
                                 self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
                                ), axis = 2)
        
        xi = np.ascontiguousarray(xi)
        
        Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
                                  self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
                                ),axis = 2 )
        Ixi = np.ascontiguousarray(Ixi)
        
        diff_GGP_GLP = GG_P - GL_P
        diff_GGM_GLM = GG_M - GL_M
        self.diff_ggp_glp = diff_GGP_GLP
        self.diff_ggm_glm = diff_GGM_GLM
        
        self.xi  = xi
        self.Ixi = Ixi
        
        @njit(parallel = parallel, fastmath = fastmath, nogil = nogil)
        def f(t, 
              old_sig, old_psi, old_omega,
              dH, delta_variant, dH_given = True):
            #Create the arrays needed
            dt = np.complex128
            D_psi    = np.zeros(old_psi.shape   , dtype = dt)
            D_omega =  np.zeros(old_omega.shape , dtype = dt)
            nk = old_sig.shape[0]
            
            # psi_dagger is needed:
            psi_tilde  = old_psi.conj()
            # Store bias at time t
            delta_t = np.zeros(nl)
            for a in range(nl): delta_t[a] = delta_variant(t, a)
            # Get Hamiltonian
            if dH_given: Ht =  H + dH(t, old_sig)
            else:        Ht =      dH(t, old_sig)
            
            ##### Density matrix EOM:
            D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
            for a in range(nl):
                pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
                D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            D_sig *= 1/hbar
            
            ##### Psi & Omega EOM:
            hbar_sq  = hbar**2
            _1ohbar  = 1/hbar
            _1johbar = 1j /hbar
            for x in prange(Ntot):
                for k in range(nk):
                    for a in range(nl):
                        for c in range(no):
                            # We also use axc_idx etc. later
                            axc_idx   = omega_idx[a,x,c,:,:,:]
                            psi_axc_1 = old_psi[k,a,x,c,:]
                            xi_axc_1  = xi[k,a,x,c,:]
                            xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
                            #We non-zero psi's are has an index that is greater than or equal to zero.
                            if psi_idx[a,x,c]>=0:
                                #Calculate
                                D_psi[k,a,x,c,:]=  ( (Ht[k].dot(psi_axc_1 )  - (Xpp[k,a,x,c] + delta_t[a] ) * psi_axc_1) * _1ohbar
                                                          + GL_P[k,a,x,c] * xi_axc_1
                                                          + diff_GGP_GLP[k,a,x,c]* old_sig[k].dot(xi_axc_1)
                                                   )
                            
                            ###
                            # Differences between the Lambdas:
                            omega_sum = np.zeros(no, dtype = np.complex128)
                            diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                            for a2 in range(nl):
                                for x2 in range(Ntot):
                                    for c2 in range(no):
                                        idx = axc_idx[a2,x2,c2]
                                        # Only non-zero terms calculated:
                                        if idx >= 0:
                                            psi_axc_2    = psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
                                            xi_axc_2     = Ixi[k,a2,x2,c2,:]       # 
                                            diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
                                            xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            #Calculate
                                            D_omega[k, idx] = ( -(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] *_1johbar 
                                                               +    diff_ggm_glm * xi_axc_2.dot(psi_axc_1)
                                                               +    diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
                                                               )
                                            
                                            if psi_idx[a,x,c]>=0:
                                                omega_sum += old_omega[k, idx] * xi[k,a2,x2,c2]
                            
                            if psi_idx[a,x,c]>=0:
                                D_psi[k,a,x,c,:] += (omega_sum/hbar_sq)
                            
            
            return D_sig, -1j * D_psi, D_omega
        
        
        return f
    
    def make_f_purenp(self):
        # The second index on H was for being able to multiply with energy-resolved quantities,
        # which is not needed anymore
        H      = self.Hdense[:,0,:,:]
        Xpp    = self.Xpp
        Xpm    = self.Xpm
        GG_P   = self.GG_P; GL_P   = self.GL_P
        GG_M   = self.GG_M; GL_M   = self.GL_M
        nl     = self.num_leads
        nf     = self.num_poles
        no     = H.shape[2]
        Ntot   = self.NumL + nf
        noT = self.max_orbital_idx + 1
        
        # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
        xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
                                 self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
                                ), axis = 2)
        xi = np.ascontiguousarray(xi)
        
        Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
                                  self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
                                ),axis = 2 )
        Ixi = np.ascontiguousarray(Ixi)
        
        diff_GGP_GLP = GG_P - GL_P
        diff_GGM_GLM = GG_M - GL_M
        self.diff_ggp_glp = diff_GGP_GLP
        self.diff_ggm_glm = diff_GGM_GLM
        
        self.xi  = xi
        self.Ixi = Ixi
        from time import time
        
        def f(t, 
              old_sig, old_psi, old_omega,
              dH, delta_variant, dH_given = True):
            #Create the arrays needed
            dt = np.complex128
            nk = old_sig.shape[0]
            
            # psi_dagger is needed:
            psi_tilde  = old_psi.conj()
            # Store bias at time t
            delta_t = np.zeros(nl)
            for a in range(nl): delta_t[a] = delta_variant(t, a)
            # Get Hamiltonian
            if dH_given: Ht =  H + dH(t, old_sig)
            else:        Ht =      dH(t, old_sig)
            
            ##### Density matrix EOM:
            D_sig =  - 1j * (Ht@old_sig - old_sig@Ht)
            for a in range(nl):
                pi_a   = PI(old_psi[:, a], Ixi[:, a]) 
                D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
            D_sig *= 1/hbar
            ##### Psi & Omega EOM:
            nax = np.newaxis
            Xpp_delta  =  Xpp + delta_t[nax,:, nax, nax]
            Xpm_delta  =  Xpm + delta_t[nax,:, nax, nax]
            
            D_psi  = old_psi @ np.expand_dims(Ht.transpose((0, 2, 1)), (1, 2)) / hbar
            D_psi -= np.expand_dims(Xpp_delta[:,:,:,:noT] / hbar, 4) * old_psi [:,:,:,:noT]
            D_psi += np.expand_dims(GL_P[:,:,:,:noT], 4) * xi[:,:,:,:noT]
            D_psi += np.expand_dims(diff_GGP_GLP[:,:,:,:noT], 4) * (
                                    xi[:,:,:,:noT] @ np.expand_dims(old_sig.transpose((0, 2, 1)), (1, 2))
                                    )
            
            om_shape = (nk, nl*Ntot*noT, nl*Ntot*no)
            xi_shape = (nk, nl*Ntot*no , -1        )
            
            D_psi   += (old_omega.reshape(om_shape) @ xi.reshape(xi_shape)).reshape(D_psi.shape) / (hbar ** 2)
            
            D_omega  = (old_psi[:, :, :, :noT, :].reshape(nk, nl*Ntot*noT, no) @
                          ((diff_GGM_GLM.reshape(nk, nl*Ntot*no, 1)
                          * Ixi.reshape(nk, nl*Ntot*no, no)).transpose((0, 2, 1)))).reshape(old_omega.shape)
            
            D_omega[:,:,:,0:noT, :, :, 0:noT] += ((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1)
                                                            * xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)) @
                                                                  psi_tilde.reshape(nk, nl*Ntot*noT, no)
                                                            .transpose((0, 2, 1))
                                                  ).reshape(nk, nl, Ntot, noT, nl, Ntot, noT)
            
            for k in range(nk):
                D_omega[k]  += (np.subtract.outer(Xpp_delta[k,:,:,:noT], Xpm_delta[k])*old_omega[k])*(1j/hbar)
            
            #D_omega += Xpp_delta[:,:,:,:noT, nax, nax, nax] * (1j / hbar) - Xpm[:, nax, nax, nax, :, :, :] * (1j / hbar)
            
            return D_sig, -1j * D_psi, D_omega
        
        return f
    
    def make_f_gpu(self,dtype = int):
        if k0nfig.GPU == False:
            print('/n GPU is not enabled in config file! /n')
            assert 1 == 0
        
        H      = cp.array(self.Hdense[:,0,:,:]).astype(dtype)
        Xpp    = cp.array(self.Xpp).astype(dtype)
        Xpm    = cp.array(self.Xpm).astype(dtype)
        GG_P   = cp.array(self.GG_P).astype(dtype)
        GG_M   = cp.array(self.GG_M).astype(dtype)
        GL_P   = cp.array(self.GL_P).astype(dtype)
        GL_M   = cp.array(self.GL_M).astype(dtype)
        nl     = self.num_leads
        nf     = self.num_poles
        no     = H.shape[2]
        nk     = H.shape[0]
        Ntot   = self.NumL + nf
        noT = self.max_orbital_idx + 1
        
        # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
        xi     = cp.array(np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
                                          self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
                                          ), axis = 2).astype(dtype) 
                         )
        Ixi    = cp.array(np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
                                          self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
                                          ),axis = 2 ).astype(dtype)
                         )
        xi     = cp.ascontiguousarray(xi).astype(dtype)
        Ixi    = cp.ascontiguousarray(Ixi).astype(dtype)
        
        diff_GGP_GLP = (GG_P - GL_P).astype(dtype)
        diff_GGM_GLM = (GG_M - GL_M).astype(dtype)
        self.diff_ggp_glp = diff_GGP_GLP
        self.diff_ggm_glm = diff_GGM_GLM
        
        self.xi  = xi
        self.Ixi = Ixi
        MM = cp.matmul
        
        def f(t, 
              old_sig, old_psi, old_omega,
              dH, delta_variant, dH_given = True):
            
            delta_t = cp.zeros(nl,dtype = dtype)
            for a in range(nl): delta_t[a] = delta_variant(t, a)
            # Get Hamiltonian
            if dH_given: Ht =  H + dH(t, old_sig)
            else:        Ht =      dH(t, old_sig)
            
            ##### Density matrix EOM:
            D_sig =  - 1j * (cp.matmul(Ht,old_sig) - cp.matmul(old_sig, Ht))
            for a in range(nl):
                pi_a   = PI_gpu(old_psi[:, a], Ixi[:, a]) 
                D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
            D_sig *= 1/hbar
            nax = cp.newaxis
            Xpp_delta  =  Xpp + delta_t[nax,:, nax, nax]
            Xpm_delta  =  Xpm + delta_t[nax,:, nax, nax]
            
            D_psi  = MM(old_psi , cp.expand_dims(Ht.transpose((0, 2, 1)), (1, 2))) / hbar
            D_psi -= cp.expand_dims(Xpp_delta[:,:,:,:noT] / hbar, 4) * old_psi [:,:,:,:noT]
            D_psi += cp.expand_dims(GL_P[:,:,:,:noT], 4) * xi[:,:,:,:noT]
            D_psi += cp.expand_dims( diff_GGP_GLP[:,:,:,:noT], 4) * (
                                 MM( xi[:,:,:,:noT] , cp.expand_dims(old_sig.transpose((0, 2, 1)), (1, 2))  )
                                                                    )
            
            om_shape = (nk, nl*Ntot*noT, nl*Ntot*no); xi_shape = (nk, nl*Ntot*no , -1 )
            D_psi   += MM(old_omega.reshape(om_shape) , xi.reshape(xi_shape)).reshape(D_psi.shape) / (hbar ** 2)
            
            D_omega  = MM(old_psi[:, :, :, :noT, :].reshape(nk, nl*Ntot*noT, no) ,
                          ((diff_GGM_GLM.reshape(nk, nl*Ntot*no, 1)
                          * Ixi.reshape(nk, nl*Ntot*no, no)).transpose((0, 2, 1)))).reshape(old_omega.shape)
            
            D_omega[:,:,:,0:noT, :, :, 0:noT] += MM((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1)
                                                      *        xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)) ,
                                                               old_psi.conj().reshape(nk, nl*Ntot*noT, no)
                                                      .transpose((0, 2, 1))
                                                      ).reshape(nk, nl, Ntot, noT, nl, Ntot, noT)
            
            for k in range(nk):
                D_omega[k]  += ((Xpp_delta[k,:,:,:noT][:,:,:,nax,nax,nax]    
                               - Xpm_delta[k][nax,nax,nax,:,:,:         ])*old_omega[k])*(1j/hbar)
            
            return D_sig, -1j * D_psi, D_omega
    def make_f(self, parallel = False, fastmath = False, nogil = False):
        """ !!!Reference Implementation!!!
            Relatively short and concise numba implementation of the 
            EOMs. Read the code and the paper 
            
            """ +CiteString+""" 
            
            In order to convince yourself it is right.
            
            This function returns a numba JIT-compiled function.
            this function takes the arguments as
                def f(t, 
                      old_sig, old_psi, old_omega,
                      dH, delta_variant, dH_given = True):
            where t is float, old_sig is a (nk,no,no) numpy array,
            old_psi is a (nk, na, nl, noT, no) array, 
            old_omega is a (nk, na, nl, noT,nk, na, nl, noT) array,
            dH is a numba JIT-compiled function taking a float and a (nk,no,no)
                array as its arguments
            delta_variant is a function that takes a float and an int (t,a)
            
        """
        
        
        # The second index on H was for being able to multiply with energy-resolved quantities,
        # which is not needed anymore
        H      = self.Hdense[:,0,:,:].copy()
        Xpp    = self.Xpp.copy();  Xpm    = self.Xpm.copy()
        GG_P   = self.GG_P.copy(); GG_M   = self.GG_M.copy()
        GL_P   = self.GL_P.copy(); GL_M   = self.GL_M.copy()
        nl     = self.num_leads; nf     = self.num_poles
        no     = H.shape[2];     nk     = H.shape[0]
        Ntot   = self.NumL + 2*nf
        omega_idx  = self.omega_idx.copy()
        psi_idx    = self.psi_idx.copy()
        # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
        xi = self.xi.copy(); Ixi = self.Ixi.copy()
        diff_GGP_GLP = GG_P - GL_P
        diff_GGM_GLM = GG_M - GL_M
        self.diff_ggp_glp = diff_GGP_GLP
        self.diff_ggm_glm = diff_GGM_GLM
        
        @njit(parallel = True, fastmath = True)
        def f(t, 
              old_sig, old_psi, old_omega,
              dH, delta_variant, dH_given = True):
            #Create the arrays needed
            dt = np.complex128
            D_psi    = np.zeros(old_psi.shape   , dtype = dt)
            D_omega =  np.zeros(old_omega.shape , dtype = dt)
            nk = old_sig.shape[0]
            # psi_dagger is needed:
            psi_tilde  = old_psi.conj()
            # Store bias at time t
            delta_t = np.zeros(nl)
            for a in range(nl): delta_t[a] = delta_variant(t, a)
            # Get Hamiltonian
            if dH_given: Ht =  H + dH(t, old_sig)
            else:        Ht =      dH(t, old_sig)
            ##### Density matrix EOM:
            D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
            for a in range(nl):
                pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
                D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            D_sig *= 1/hbar
            ##### Psi & Omega EOM:
            hbar_sq = hbar**2
            for k in range(nk):
                # Order the xi-vectors in columns (.T)
                xi_k = xi[k].reshape((nl * Ntot * no, no)).T
                old_omega_k = old_omega[k]
                old_sig_k   = old_sig[k,:,:].copy()
                for x in prange(Ntot):
                    for a in range(nl):
                        for c in range(no):
                            # We also use axc_idx etc. later
                            axc_idx   = omega_idx[a,x,c,:,:,:].copy()
                            psi_axc_1 = old_psi[k,a,x,c,:]
                            xi_axc_1  = xi[k,a,x,c,:]
                            xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
                            #We non-zero psi's are has an index that is greater than or equal to zero.
                            if psi_idx[a,x,c]>=0:
                                idx       = axc_idx.reshape((nl*Ntot*no))
                                # Use only non-zero Omegas
                                bOOl      =(idx >= 0 )
                                xi_sub    = xi_k[:,bOOl]
                                idx       = idx[bOOl]
                                omega_sub = old_omega_k[idx]
                                # Do sum in the last term of the EOM
                                omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                                # Make the part involving Hamiltonian
                                Hchi      = Ht[k,:,:].copy()
                                for diag in range(no):
                                    Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                                #Calculate
                                D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
                                                          GL_P[k,a,x,c] * xi_axc_1                            +\
                                                          diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
                                                          omega_sum/hbar_sq )
                            # Differences between the Lambdas:
                            diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                            for a2 in range(nl):
                                for x2 in range(Ntot):
                                    for c2 in range(no):
                                        idx = axc_idx[a2,x2,c2]
                                        # Only non-zero terms calculated:
                                        if idx < 0:
                                            pass
                                        else:
                                            psi_axc_2    = psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
                                            xi_axc_2     = Ixi[k,a2,x2,c2,:]       # 
                                            diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
                                            xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
                                                                      diff_ggm_glm * xi_axc_2.dot(psi_axc_1)                +\
                                                                      diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
                                                              )
            return D_sig, D_psi, D_omega
        return f
                
    
class sisl_replica:
    # Class meant to make it easier to pickle and unpickle the TD_transport class
    def __init__(self, s):
        self.xyz  = s.xyz
        self.cell = s.cell
        self.Z    = s.atoms.Z
    def toASE(self):
        from ase import Atoms
        return  Atoms(positions = self.xyz, 
                      cell      = self.cell, 
                      numbers   = self.Z)
    def to_sisl(self):
        return sisl.Geometry.fromASE(self.toASE())

# "Overloaded" sile classes, either outputs regular sisl classes
# or the "fake-sisl" classes emulating the regular sisl-behavior.
# depends on if regular TBT.nc files are found or if the fakeTBT files are present
#
# See siesta_python.fake for the return classes
#

def _READ_get_sile(D):
    """ Function depends on the siesta_python.fake module to mimic 
        the sisl sile class that normally interfaces this code to the
        TBtrans code. 
        
    """
    basename     = 'siesta.TBT.nc'
    fakebasename = 'siesta.fakeTBT.npz'
    files = os.listdir(D)
    if fakebasename in files:
        from siesta_python.fake import fakeTBT
        return fakeTBT(D+'/'+fakebasename)
    elif basename in files:
        return sisl.get_sile(D+'/'+basename)
    else:
        print('failed reading siesta-tbt output'); assert 1 == 0

def _READ_get_SE(D):
    """ Function depends on the siesta_python.fake module to mimic 
        the sisl sile class that normally interfaces this code to the
        TBtrans code. 
        The sisl sile is used when a normal TBtrans calculation has been made,
        but if a custom calculation is given, a fakeTBT.npz file is read instead
        
        
    """
    basename     = 'siesta.TBT.SE.nc'
    fakebasename = 'siesta.fakeTBT.SE.npz'
    files = os.listdir(D)
    if fakebasename in files:
        from siesta_python.fake import read_fakeSE_from_tbtrans
        return read_fakeSE_from_tbtrans(D+'/'+fakebasename)
    elif basename in files:
        return read_SE_from_tbtrans(D+'/'+basename)
    
    else:
        print('failed reading siesta-tbt-se output'); assert 1 == 0

def _READ_get_E_F(D):
    """ Function depends on the siesta_python.fake module to mimic 
        the sisl sile class that normally interfaces this code to the
        TBtrans code. 
        The sisl sile is used when a normal TBtrans calculation has been made,
        but if a custom calculation is given, a fakeTBT.npz file is read instead
        
    """
    basename     = 'RUN.fdf'
    fakebasename = 'siesta.fakeTBT.npz'
    files = os.listdir(D)
    if fakebasename in files:
        from siesta_python.fake import fakeTBT
        return fakeTBT(D+'/'+fakebasename).read_fermi_level()
    elif basename in files:
        return sisl.get_sile(D + '/'+basename).read_fermi_level()
    
    else:
        print('failed reading siesta-tbt-se output'); assert 1 == 0

def _READ_get_H(D):
    """ Function depends on the siesta_python.fake module to mimic 
        the sisl sile class that normally interfaces this code to the
        TBtrans code. 
        The sisl sile is used when a normal TBtrans calculation has been made,
        but if a custom calculation is given, a fakeTBT.npz file is read instead
        
    """
    basename     = 'siesta.TSHS'
    fakebasename = 'siesta.fakeTSHS.npz'
    files = os.listdir(D)
    if fakebasename in files:
        from siesta_python.fake import fakeHS
        npz = np.load(D+'/'+fakebasename)
        return fakeHS(npz['H'], npz['k'])
    elif basename in files:
        return sisl.get_sile(D+'/'+basename).read_hamiltonian()
    
    else:
        print('failed reading hamiltonian output'); assert 1 == 0

def _READ_get_S(D):
    """ Function depends on the siesta_python.fake module to mimic 
        the sisl sile class that normally interfaces this code to the
        TBtrans code. 
        The sisl sile is used when a normal TBtrans calculation has been made,
        but if a custom calculation is given, a fakeTBT.npz file is read instead
        
    """
    basename     = 'siesta.TSHS'
    fakebasename = 'siesta.fakeTSHS.npz'
    files = os.listdir(D)
    if fakebasename in files:
        from siesta_python.fake import fakeHS
        npz = np.load(D+'/'+fakebasename)
        return fakeHS(npz['S'], npz['k'])
    elif basename in files:
        return sisl.get_sile(D+'/'+basename).read_hamiltonian()
    
    else:
        print('failed reading overlap output'); assert 1 == 0

##############################################################
############### COMPILED FUNCTIONS ###########################
##############################################################


if k0nfig.NUMBA:
    @njit(parallel = k0nfig.NUMBA_PARALLEL)
    def PI_opti(psi_a, ixi, psi_a_idx):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc = psi_a.shape[2]
        res = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            for c in prange(nc):
                for x in range(nx):
                    if psi_a_idx[x,c]>=0:
                        jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], res[k,:,:] )
        return res/hbar
    
    @njit(cache = k0nfig.CACHE)
    def dot_3d(A,B):
        n = A.shape[0]
        res = np.zeros((n, A.shape[1], B.shape[2]), dtype = np.complex128)
        for i in range(n):
            res[i] = A[i].dot(B[i])
        return res
    
    @njit(cache = k0nfig.CACHE)
    def jit_outer(a,b,res):
        na = len(a)
        nb = len(b)
        for i in range(na):
            res[i,:] += a[i] * b
        return res
    
    @njit(cache = k0nfig.CACHE, parallel = k0nfig.PARALLEL_PI)
    def PI_nb(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc = psi_a.shape[3]
        res = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            reduc_k = np.zeros((nx,nc,nc),dtype = res.dtype)
            for x in prange(nx):
                for c in range(nc_top):
                   jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
            res[k] = reduc_k.sum(axis=0)
        return res/hbar
    
    
    @njit(cache = k0nfig.CACHE, fastmath=k0nfig.FASTMATH)
    def _Q_jit_outer(a,b,res):
        na   = len(a)
        nb   = len(b)
        ac   = np.conj(a)
        bc   = np.conj(b)
        
        for i in range(na):
            ai  = a[i]
            bci = bc[i]
            for j in range(i,nb):
                res[i,j] += ai * b[j] + ac[j]*bci
    
    @njit(cache = k0nfig.CACHE, fastmath=k0nfig.FASTMATH)
    def _Q_jit_outer_v2(a,b,res):
        na   = len(a)
        nb   = len(b)
        ac   = np.conj(a)
        bc   = np.conj(b)
        idx  = np.where(np.abs(b)>1e-13)[0]
        bmin = idx.min()
        bmax = idx.max()+1
        
        for i in range(0, bmin):
            ai  = a[i]
            bci = bc[i]
            for j in range(bmin,bmax):
                res[i,j] += ai * b[j] + ac[j]*bci
        
        for i in range(bmin, bmax):
            ai  = a[i]
            bci = bc[i]
            for j in range(i,nb):
                res[i,j] += ai * b[j] + ac[j]*bci
    
    @njit(cache = k0nfig.CACHE, fastmath = k0nfig.FASTMATH)
    def _Q_make_hermitian(Q):
        na,nb = Q.shape
        for i in range(na):
            for j in range(0,i):
                Q[i,j] = np.conj(Q[j,i])
    
    @njit(cache = k0nfig.CACHE, parallel = k0nfig.PARALLEL_PI, fastmath = k0nfig.FASTMATH)
    def Q_nb(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk     = psi_a.shape[0]
        nx     = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc     = psi_a.shape[3]
        res    = np.zeros((nk, nc, nc), dtype = np.complex128)
        for k in range(nk):
            reduc_k = np.zeros((nx,nc,nc),dtype = res.dtype)
            for x in prange(nx):
                for c in range(nc_top):
                   ###_Q_jit_outer(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
                   _Q_jit_outer_v2(psi_a[k,x,c,:], ixi[k,x,c,:], reduc_k[x])
            res[k] = reduc_k.sum(axis=0)
            _Q_make_hermitian(res[k])
        return res/hbar
    
    def Q_np(psi_a, ixi,nonzero_idx = None):
        """
                
            Returns (Pi + Pi^dagger )/hbar
            Fastest version written so far and is pretty simple.
            
            Both psi_a and ixi is (nk,nx,noT,no) arrays
            
        """
        assert len(psi_a.shape) == 4
        nk,nx,nc_top,nc     = psi_a.shape
        if nonzero_idx is None:
            res    = np.matmul(psi_a.reshape(nk,nx*nc_top,nc).transpose(0,2,1),
                               ixi[:,:,:nc_top,:].reshape(nk,nx*nc_top,nc))
        else:
            res    = np.zeros((nk,nc,nc), dtype=np.complex128)
            subres = np.matmul(psi_a.reshape(nk,nx*nc_top,nc).transpose(0,2,1),
                               ixi[:,:,:nc_top,nonzero_idx
                                   ].reshape(nk,nx*nc_top,len(nonzero_idx)), 
                              )
            res[:,:,nonzero_idx] = subres[:,:,:]
        res   += res.transpose(0,2,1).conj()
        return res/hbar
#######  Here ends the used functions (there are also a couple above)
#######  The stuff below is leftovers of the development.
  



    
###### UNSUPPORTED / DEPRICATED ATM
if k0nfig.GPU:
    def PI_gpu(psi_a, ixi):
        assert len(psi_a.shape) == 4
        nk = psi_a.shape[0]
        nx = psi_a.shape[1]
        nc_top = psi_a.shape[2]
        nc = psi_a.shape[3]
        nax = cp.newaxis
        res = cp.matmul(psi_a[... , nax] , ixi[:,:,0:nc_top, nax, :]).sum(axis = (1,2))
        cp.cuda.Stream.null.synchronize()
        return res/hbar
    
    def Jk_gpu(PI_a):
        return (2*electron_charge/hbar) * cp.trace(PI_a, axis1 = 1, axis2 = 2).real

# Python  +  NumPy functions
def PI_np(psi_a,ixi):
    assert len(psi_a.shape) == 4
    nc_top = psi_a.shape[2]
    nax = np.newaxis
    res = (psi_a[... , nax] @ ixi[:,:,0:nc_top, nax, :]).sum(axis = (1,2))
    return res/hbar

if k0nfig.PI_VERSION == 'NUMBA':
    PI = PI_nb
elif k0nfig.PI_VERSION =='NUMPY':
    PI = PI_np
elif k0nfig.GPU and k0nfig.PI_VERSION=='GPU':
    PI = PI_gpu

def J(PI_a):
    return (2*electron_charge/hbar) * np.trace(PI_a, axis1 = 1, axis2 = 2).sum(axis = 0).real

def Jk(PI_a):
    return (2*electron_charge/hbar) * np.trace(PI_a, axis1 = 1, axis2 = 2).real


def AdaptiveRK4(f, sig0, psi0, omega0, eps, t0, t1,
                dH, delta_func, Ixi,
                h_guess = None, dH_given = True,
                print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
                write_func = None, print_step = 50, plot = False, use_GPU = False,
                atol = None, rtol = None,
                elec_names = ['left', 'right']):
    
    
    # Adaptive timestep RK4
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
                  [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
                  [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
                  [439/216  , -8         , 3680/513  , -845/4104, np.nan],
                  [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
                  ]
                  )
    
    C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
    CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
    CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])
    
    if k0nfig.GPU and use_GPU:
        xp  = cp
        Cur = Jk_gpu
        PI_x = PI_gpu
        save_array = cp.asnumpy
    else:
        xp         = np
        Cur        = Jk
        PI_x       = PI
        save_array = np.asarray
    if atol is None:
        def TERR(k1,k2,k3,k4,k5,k6):
            res = 0.0
            for i in range(3):
                res_I = xp.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
                                       CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] ))**2  )
                res+=res_I
            return xp.sqrt(res)
    else:
        def TERR(k1,k2,k3,k4,k5,k6):
            e_sig = xp.abs(CT[0]*k1[0] + CT[1]*k2[0] + CT[2]*k3[0] +
                           CT[3]*k4[0] + CT[4]*k5[0] + CT[5]*k6[0] )
            
            e_psi = xp.abs(CT[0]*k1[1] + CT[1]*k2[1] + CT[2]*k3[1] +
                           CT[3]*k4[1] + CT[4]*k5[1] + CT[5]*k6[1] )
            
            e_omg = xp.abs(CT[0]*k1[2] + CT[1]*k2[2] + CT[2]*k3[2] +
                           CT[3]*k4[2] + CT[4]*k5[2] + CT[5]*k6[2] )
            
            f1 = ((e_sig - atol[0])/rtol[0] - xp.abs(state_sig)).max()
            f2 = ((e_psi - atol[1])/rtol[1] - xp.abs(state_psi)).max()
            f3 = ((e_omg - atol[2])/rtol[2] - xp.abs(state_omg)).max()
            error = 1**(max((f1,f2,f3)))
            return error
    
    def step_fourth(y_pre, k1,k2,k3,k4,k5,k6):
        res  =(
               y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0],
               y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1],
               y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]
              )
        return res    
    
    def hnew(h, eps, TE):
        return 0.9 * h * (eps / TE) ** (1 / 5)
    
    def scalar_mult(Arr,number):
        Arr*=number

    current_left   = []
    current_right  = []
    density_matrix = []
    times          = []
    
    
    if sig0 is None or psi0 is None or omega0 is None:
        state_sig = xp.load(name+'_last_sig.npy')
        state_psi = xp.load(name+'_last_psi.npy')
        state_omg = xp.load(name+'_last_omega.npy')
        if t0 is None:
            t0        = float(xp.load(name+'_last_time.npy'))
    else:
        state_sig =  sig0.copy()
        state_psi =  psi0.copy()
        state_omg =  omega0.copy()
    
    if h_guess is None:
        h = (t1-t0)/1000
    else:
        h  = 0.0
        h += h_guess
    
    step =  0
    T0   =  0
    T0  += t0
    
    with open(name+'.txt','w') as file:
        file.write('\n\n\n\n\nStart (Wait for compilation)\n')
    time_start= time()
    data = dict()
    for e in elec_names:
        data.update({'current_'+e:[]})
    
    #SIGMAS = np.zeros((6,)+state_sig.shape)
    
    
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            #print(t0)
            dt = A * h
            
            k1 =     f(t0+dt[0], state_sig, state_psi, state_omg, 
                       dH, delta_func, dH_given = dH_given )
            #k1 = tuple([h*v for v in k1])
            [scalar_mult(v,h) for v in k1]
            
            k2 =     f(t0+dt[1], state_sig + B[1,0]*k1[0], 
                                 state_psi + B[1,0]*k1[1], 
                                 state_omg + B[1,0]*k1[2],
                       dH, delta_func, dH_given = dH_given )
            #k2 = tuple([h*v for v in k2])
            [scalar_mult(v,h) for v in k2]
            
            k3 =     f(t0+dt[2], state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
                                 state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
                                 state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k3 = tuple([h*v for v in k3])
            [scalar_mult(v,h) for v in k3]
            
            
            k4 =     f(t0+dt[3], state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
                                 state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
                                 state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k4 = tuple([h*v for v in k4])
            [scalar_mult(v,h) for v in k4]
            
            k5 =     f(t0+dt[4], state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
                                 state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
                                 state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            #k5 = tuple([h*v for v in k5])
            [scalar_mult(v,h) for v in k5]
            
            k6 =     f(t0+dt[5], state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
                                 state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
                                 state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            # k6 = tuple([h*v for v in k6])
            [scalar_mult(v,h) for v in k6]
            
            if fixed_mode:
                break
            else:
                TE  =  TERR(k1, k2, k3, k4, k5, k6)
                h   =  hnew(h, eps, TE)
        
        density_matrix += [save_array(state_sig[:,:,:])]
        for _ie, e in enumerate(elec_names):
            data['current_'+e]+=[save_array(Cur(PI_x(state_psi[:, _ie],Ixi[:, _ie]) ) ) ]
        
        state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6)
        times  +=  [t0]
        t0     +=   h
        
        if np.mod(step, print_step) == 0:
            with open(name + '.txt', 'a') as file:
                file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
                file.write('current timestep: '+str(h) +'fs\n')
                file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
                if write_func is not None:
                    write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
                if plot==True:
                    plt.show()
                    for _ie, e in enumerate(elec_names):
                        plt.plot(xp.array(times), xp.array(data['current_'+e]), label = str(_ie))
                    plt.xlabel('Time [fs]', size = 20)
                    plt.savefig('Current(t)',dpi =300)
                    plt.show()
                    plt.pause(0.05)
            
            xp.save(name + '_last_sig', state_sig)
            xp.save(name + '_last_psi', state_psi)
            xp.save(name + '_last_omega', state_omg)
            xp.save(name + '_last_time', xp.array(times[-1]))
            xp.save('_times', xp.array(times))
            current_keys = [k for k in data.keys() if 'current' in k]
            for ck in current_keys:
                xp.save('_'+ck, save_array(data[ck]))
            
            xp.save('_#electrons_device',xp.trace(save_array(density_matrix),axis1 = 2, axis2=3 ))
            
        step+=1
    
    data.update({'density matrix': save_array(density_matrix)})
    
    runtime = time()-time_start
    
    with open(name + '.txt', 'a') as file:
        file.write('100%\n')
        file.write('Runtime: ' + str(runtime) + 'seconds')
    
    return np.array(times), data

def AdaptiveDOP(f, sig0, psi0, omega0, eps, t0, t1,
                dH, delta_func, Ixi,
                h_guess = None, dH_given = True,
                print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
                write_func = None, print_step = 50, plot = False, use_GPU = False,
                atol = None, rtol = None,
                elec_names = ['left', 'right']):
    
    
    # Adaptive timestep RK4
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
    from DOP54 import A as AA
    from DOP54 import b4,b5
    CT = b5-b4
    A = AA[:,0]
    B = AA[:,1:]
    CH = b5
    
    
    if k0nfig.GPU and use_GPU:
        xp  = cp
        Cur = Jk_gpu
        PI_x = PI_gpu
        save_array = cp.asnumpy
    else:
        xp         = np
        Cur        = Jk
        PI_x       = PI
        save_array = np.asarray
    if atol is None:
        def TERR(k1,k2,k3,k4,k5,k6,k7):
            res = 0.0
            for i in range(3):
                res_I = xp.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
                                       CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] + CT[6] * k7[i]))**2  )
                res+=res_I
            return xp.sqrt(res)
    else:
        def TERR(k1,k2,k3,k4,k5,k6):
            e_sig = xp.abs(CT[0]*k1[0] + CT[1]*k2[0] + CT[2]*k3[0] +
                           CT[3]*k4[0] + CT[4]*k5[0] + CT[5]*k6[0] )
            
            e_psi = xp.abs(CT[0]*k1[1] + CT[1]*k2[1] + CT[2]*k3[1] +
                           CT[3]*k4[1] + CT[4]*k5[1] + CT[5]*k6[1] )
            
            e_omg = xp.abs(CT[0]*k1[2] + CT[1]*k2[2] + CT[2]*k3[2] +
                           CT[3]*k4[2] + CT[4]*k5[2] + CT[5]*k6[2] )
            
            f1 = ((e_sig - atol[0])/rtol[0] - xp.abs(state_sig)).max()
            f2 = ((e_psi - atol[1])/rtol[1] - xp.abs(state_psi)).max()
            f3 = ((e_omg - atol[2])/rtol[2] - xp.abs(state_omg)).max()
            error = 1**(max((f1,f2,f3)))
            return error
    
    def step_fourth(y_pre, k1,k2,k3,k4,k5,k6, k7):
        res  =(
               y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0]+CH[6]*k7[0],
               y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1]+CH[6]*k7[1],
               y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]+CH[6]*k7[2]
              )
        return res
    
    def hnew(h, eps, TE):
        return 0.9 * h * (eps / TE) ** (1 / 5)
    
    current_left   = []
    current_right  = []
    density_matrix = []
    times          = []
    
    
    if sig0 is None or psi0 is None or omega0 is None:
        state_sig = xp.load(name+'_last_sig.npy')
        state_psi = xp.load(name+'_last_psi.npy')
        state_omg = xp.load(name+'_last_omega.npy')
        if t0 is None:
            t0        = xp.load(name+'_last_time.npy')[0]
    else:
        state_sig =  sig0.copy()
        state_psi =  psi0.copy()
        state_omg =  omega0.copy()
    
    if h_guess is None:
        h = (t1-t0)/1000
    else:
        h  = 0.0
        h += h_guess
    
    step =  0
    T0   =  0
    T0  += t0
    
    with open(name+'.txt','w') as file:
        file.write('\n\n\n\n\nStart (Wait for compilation)\n')
    time_start= time()
    data = dict()
    for e in elec_names:
        data.update({'current_'+e:[]})
    
    while t0 <= t1:
        TE = 10 * eps
        while TE > eps:
            #print(t0)
            dt = A * h
            
            k1 =     f(t0+dt[0], state_sig, state_psi, state_omg, 
                       dH, delta_func, dH_given = dH_given )
            k1 = tuple([h*v for v in k1])
            
            k2 =     f(t0+dt[1], state_sig + B[1,0]*k1[0], 
                                 state_psi + B[1,0]*k1[1], 
                                 state_omg + B[1,0]*k1[2],
                       dH, delta_func, dH_given = dH_given )
            k2 = tuple([h*v for v in k2])
            
            k3 =     f(t0+dt[2], state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
                                 state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
                                 state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k3 = tuple([h*v for v in k3])
            
            k4 =     f(t0+dt[3], state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
                                 state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
                                 state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k4 = tuple([h*v for v in k4])
            
            k5 =     f(t0+dt[4], state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
                                 state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
                                 state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            k5 = tuple([h*v for v in k5])
            
            k6 =     f(t0+dt[5], state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
                                 state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
                                 state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            k6 = tuple([h*v for v in k6])
            
            k7 =     f(t0+dt[6], state_sig + B[6,5]*k6[0] + B[6,4]*k5[0] + B[6,3]*k4[0] + B[6,2]*k3[0] + B[6,1]*k2[0] + B[6,0]*k1[0],
                                 state_psi + B[6,5]*k6[1] + B[6,4]*k5[1] + B[6,3]*k4[1] + B[6,2]*k3[1] + B[6,1]*k2[1] + B[6,0]*k1[1],
                                 state_omg + B[6,5]*k6[2] + B[6,4]*k5[2] + B[6,3]*k4[2] + B[6,2]*k3[2] + B[6,1]*k2[2] + B[6,0]*k1[2],
                       dH, delta_func, dH_given = dH_given)
            
            k7 = tuple([h*v for v in k6])
            
            
            
            if fixed_mode:
                break
            else:
                TE  =  TERR(k1, k2, k3, k4, k5, k6, k7)
                h   =  hnew(h, eps, TE)
        
        density_matrix += [save_array(state_sig[:,:,:])]
        for _ie, e in enumerate(elec_names):
            data['current_'+e]+=[save_array(Cur(PI_x(state_psi[:, _ie],Ixi[:, _ie]) ) ) ]
        
        state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6,k7)
        times  +=  [t0]
        t0     +=   h
        
        if np.mod(step, print_step) == 0:
            with open(name + '.txt', 'a') as file:
                file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
                file.write('current timestep: '+str(h) +'fs\n')
                file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
                if write_func is not None:
                    write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
                if plot==True:
                    plt.show()
                    for _ie, e in enumerate(elec_names):
                        plt.plot(xp.array(times), xp.array(data['current_'+e]), label = str(_ie))
                    plt.xlabel('Time [fs]', size = 20)
                    plt.savefig('Current(t)',dpi =300)
                    plt.show()
                    plt.pause(0.05)
            
            xp.save(name + '_last_sig', state_sig)
            xp.save(name + '_last_psi', state_psi)
            xp.save(name + '_last_omega', state_omg)
            xp.save(name + '_last_time', xp.array(times[-1]))
            xp.save('_times', xp.array(times))
            xp.save('_JL',    save_array(current_left))
            xp.save('_JR',    save_array(current_right))
            xp.save('_#electrons_device',xp.trace(save_array(density_matrix),axis1 = 2, axis2=3 ))
        step+=1
    
    data.update({'density matrix': save_array(density_matrix)})
    
    runtime = time()-time_start
    
    with open(name + '.txt', 'a') as file:
        file.write('100%\n')
        file.write('Runtime: ' + str(runtime) + 'seconds')
    
    return np.array(times), data


def three2one(sig, psi, omg):
    nk    = sig.shape[0]
    no    = sig.shape[1]
    noT   = psi.shape[3]
    nlead = psi.shape[1]
    Nm    = psi.shape[2]
    # it lives in device 
    N_sig = no**2 
    #lead, mode, eigen, orbital
    N_psi = nlead * Nm * noT * no
    nz    = omg.shape[-1]#nlead * Nm * noT * nlead * Nm * no
    
    
    return np.hstack((sig[:,:,:].    reshape(nk * N_sig),\
                      psi[:,:,:,:,:].reshape(nk * N_psi),\
                      omg[:,:].      reshape(nk * nz)))



def one2three(y, nk, no, noT, nlead, Nm, nz):
    N_sig = nk * no**2
    N_psi = nk * nlead * Nm * noT * no
    N_omg = nk * nz
    
    return y[0:N_sig].reshape((nk, no, no)), \
           y[N_sig : (N_sig + N_psi)].reshape((nk, nlead, Nm, noT, no)), \
           y[N_sig+N_psi:N_sig+N_psi+N_omg].reshape((nk, nz))

def propagate(g, y0, t_span, atol, rtol, method = 'RK45'):
    t_eval = np.array([t_span[1] ])
    sol = ode(g, t_span, y0, rtol = rtol, atol = atol, t_eval = t_eval)
    return sol.y

def scipy_ode(f, sig0, psi0, omg0, t0, t_eval, dH, delta_variant, Ixi, 
              dH_given = True, method = 'RK45', dt_guess = None, 
              atol = 1e-6, rtol = 1e-4):
    nk    = sig0.shape[0]
    no    = sig0.shape[1]
    noT   = psi0.shape[3]
    nlead = psi0.shape[1]
    Nm    = psi0.shape[2]
    nz    = omg0.shape[-1]#nlead*Nm*noT*nlead*Nm*no
    assert (t0<t_eval).all()
    n_eval = len(t_eval)
    
    def F(t,y):
        y1,y2,y3 = one2three(y, nk, no, noT, nlead, Nm, nz)
        D_sig, D_psi, D_omg = f(t, y1, y2, y3, dH, delta_variant, dH_given = dH_given)
        return three2one(D_sig,D_psi,D_omg)
    
    state_sig =  sig0.copy()
    state_psi =  psi0.copy()
    state_omg =  omg0.copy()
    
    t_prev  = 0
    t_prev += t0
    
    cl =  []
    cr =  []
    rho=  []    
    
    for i in range(n_eval):
        ts = (t_prev , t_eval[i])
        print(ts)
        
        dt =  t_eval[i] - t_prev
        state_sig, state_psi, state_omg = one2three(
                                                      propagate(F, three2one(state_sig, state_psi, state_omg),
                                                                ts, atol, rtol, method = method),
                                                     
                                                      nk, no, noT, nlead, Nm, nz
                                                    )
        
        rho += [state_sig[0]]
        cl  += [Jk(
                    PI(
                        state_psi[:, 0], 
                        Ixi[:, 0]
                        )
                    )
                ]
        t_prev += dt
    
    return t_eval, np.array(rho), np.array(cl), #state_sig, state_psi, state_omg

####################################
##Code Graveyard ## Rest in peace  # 
####################################
#       #        ##       #        #  
#       #        ##       #        #
#    #######     ##    #######     #
#       #        ##       #        #
#       #        ##       #        #
#       #        ##       #        #
#       #        ##       #        #
####################################

# @njit
# def other_psi(psi, xi, Ixi):
#     nk,na,nx,nc = psi.shape[0:4]
#     new_psi     = np.zeros(psi.shape, dtype = np.complex128)
#     xi_conj = xi.conj()
#     norms = (xi * xi_conj).sum(axis = 4)
#     for k in range(nk):
#         for a in range(na):
#             for x in range(nx):
#                 for c in range(nc):
#                     v = psi[k,a,x,c,:].dot(xi_conj[k,a,x,c,:]) / norms[k,a,x,c]
#                     new_psi[k,a,x,c,:] = np.conj(v) * Ixi[k,a,x,c,:]
#     return new_psi




# def AdaptiveRK4_Htnojit(f, sig0, psi0, omega0, eps, t0, t1,
#                         Ht_func, delta_func, Ixi,
#                         h_guess = None,
#                         print_to_file= True, fixed_mode = False, name = 'Runge-Kutta',
#                         write_func = None, print_step = 10, plot = False):
        
#     from time import time
#     # Adaptive timestep RK4
#     # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method
#     A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
#     B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
#                   [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
#                   [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
#                   [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
#                   [439/216  , -8         , 3680/513  , -845/4104, np.nan],
#                   [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
#                   ]
#                   )
    
#     C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
#     CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
#     CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])
    
#     def TERR(k1,k2,k3,k4,k5,k6):
#         res = 0.0
        
#         for i in range(3):
            
#             res_I = np.sum(np.abs((CT[0]*k1[i] + CT[1]*k2[i] + CT[2]*k3[i] +
#                                   CT[3]*k4[i] + CT[4]*k5[i] + CT[5]*k6[i] ))**2  )
#             res+=res_I
#             #print(res_I)
#         return np.sqrt(res)
    
#     def step_fourth(y_pre, k1,k2,k3,k4,k5,k6):
#         res  =(
#                y_pre[0] + CH[0]*k1[0] + CH[1]*k2[0] + CH[2]*k3[0] + CH[3]*k4[0] + CH[4]*k5[0] + CH[5]*k6[0],
#                y_pre[1] + CH[0]*k1[1] + CH[1]*k2[1] + CH[2]*k3[1] + CH[3]*k4[1] + CH[4]*k5[1] + CH[5]*k6[1],
#                y_pre[2] + CH[0]*k1[2] + CH[1]*k2[2] + CH[2]*k3[2] + CH[3]*k4[2] + CH[4]*k5[2] + CH[5]*k6[2]
#               )
#         return res    
    
#     def hnew(h, eps, TE):
#         return 0.9 * h * (eps / TE) ** (1 / 5)
    
#     current_left  = []
#     current_right = []
#     density_matrix = []
#     times = []
#     if h_guess is None:
#         h = (t1-t0)/100
#     else:
#         h  = 0
#         h += h_guess
    
#     state_sig =  sig0.copy()
#     state_psi =  psi0.copy()
#     state_omg =  omega0.copy()
    
#     step = 0
#     T0 = 0
#     T0+= t0
#     with open(name+'.txt','w') as file:
#         file.write('\n\n\n\n\nStart (Wait for compilation)\n')
#     time_start= time()
#     while t0 <= t1:
#         TE = 10 * eps
        
#         Ht = Ht_func(t0, state_sig, state_psi, state_omg)
        
#         while TE > eps:
#             dt = A * h
            
#             k1 =     f(t0+dt[0], Ht, state_sig, state_psi, state_omg, 
#                        delta_func)
#             k1 = tuple([h*v for v in k1])
            
#             k2 =     f(t0+dt[1], Ht,
#                                  state_sig + B[1,0]*k1[0], 
#                                  state_psi + B[1,0]*k1[1], 
#                                  state_omg + B[1,0]*k1[2],
#                        delta_func)
#             k2 = tuple([h*v for v in k2])
            
#             k3 =     f(t0+dt[2], Ht,
#                                  state_sig + B[2,1]*k2[0] + B[2,0]*k1[0],
#                                  state_psi + B[2,1]*k2[1] + B[2,0]*k1[1],
#                                  state_omg + B[2,1]*k2[2] + B[2,0]*k1[2],
#                        delta_func)
#             k3 = tuple([h*v for v in k3])
            
#             k4 =     f(t0+dt[3], Ht,
#                                  state_sig + B[3,2]*k3[0] + B[3,1]*k2[0] + B[3,0]*k1[0],
#                                  state_psi + B[3,2]*k3[1] + B[3,1]*k2[1] + B[3,0]*k1[1],
#                                  state_omg + B[3,2]*k3[2] + B[3,1]*k2[2] + B[3,0]*k1[2],
#                        delta_func)
#             k4 = tuple([h*v for v in k4])
            
#             k5 =     f(t0+dt[4], Ht,
#                                  state_sig + B[4,3]*k4[0] + B[4,2]*k3[0] + B[4,1]*k2[0] + B[4,0]*k1[0],
#                                  state_psi + B[4,3]*k4[1] + B[4,2]*k3[1] + B[4,1]*k2[1] + B[4,0]*k1[1],
#                                  state_omg + B[4,3]*k4[2] + B[4,2]*k3[2] + B[4,1]*k2[2] + B[4,0]*k1[2],
#                        delta_func)
#             k5 = tuple([h*v for v in k5])
            
            
#             k6 =     f(t0+dt[5], Ht,
#                                  state_sig + B[5,4]*k5[0] + B[5,3]*k4[0] + B[5,2]*k3[0] + B[5,1]*k2[0] + B[5,0]*k1[0],
#                                  state_psi + B[5,4]*k5[1] + B[5,3]*k4[1] + B[5,2]*k3[1] + B[5,1]*k2[1] + B[5,0]*k1[1],
#                                  state_omg + B[5,4]*k5[2] + B[5,3]*k4[2] + B[5,2]*k3[2] + B[5,1]*k2[2] + B[5,0]*k1[2],
#                        delta_func)
#             k6 = tuple([h*v for v in k6])
#             if fixed_mode:
#                 break
#             else:
#                 TE = TERR(k1, k2, k3, k4, k5, k6)
#                 h  = hnew(h, eps, TE)
        
#         density_matrix += [state_sig[0,:,:]]
        
#         current_left  += [J(
#                     PI(
#                         state_psi[:, 0], 
#                         Ixi[:, 0]
#                         )
#                     )
#                   ]
#         current_right  += [J(
#                                 PI(
#                                     state_psi[:, 1], 
#                                     Ixi[:, 1]
#                                     )
#                             )
#                           ]
        
#         #print(np.abs(state_sig).sum() , np.abs(state_psi).sum(), np.abs(state_omg).sum())
#         state_sig, state_psi, state_omg = step_fourth((state_sig, state_psi, state_omg), k1,k2,k3,k4,k5,k6)
#         times += [t0]
#         t0 += h
#         if np.mod(step, print_step) == 0:
#             with open(name + '.txt', 'a') as file:
#                 file.write(str((((t0 - T0)/(t1-T0)))*100 ) + ' %\n')
#                 file.write('current timestep: '+str(h) +'fs\n')
#                 file.write('delta t: ' + str(time() - time_start) + 'seconds\n')
#                 if write_func is not None:
#                     write_func(file, t0-h, state_sig.copy(), state_psi.copy(), state_omg.copy())
#                 if plot==True:
#                     plt.plot(np.array(times), np.array(current_left))
            
#             #print('\n '+ str(h))
#         step+=1
    
#     data = {'current left':  np.array(current_left),
#             'current right': np.array(current_right),
#             'density matrix': np.array(density_matrix),
#             'last sigma': state_sig,
#             'last psi':   state_psi,
#             'last omega': state_omg}
#     runtime = time()-time_start
    
#     with open(name + '.txt', 'a') as file:
#         file.write('100%\n')
#         file.write('Runtime: ' + str(runtime) + 'seconds')
    
#     return np.array(times), data





# def F(t,x):
#     return np.array(f(t, x[0], x[1],x[2], 0,0))

    
# def f(t,x,y,z, a,b, dH_given = True):
#     if t<5:
#         return (1*x + -2*y) * np.sin(t*5), (2*z+x), (-y-z-x)
#     else:
#         return (1*x + -2*y), (2*z+x), (-y-z-x)

# # def dH(t):
# #     return 0


# def delta(a,t):
#     return 0

# xi = np.random.random(1)

# a,b,c = np.array([1.0]),np.array([2.0]),np.array([3.0])

# t,d = AdaptiveRK4(f, a, b, c, 1e-10, -10, 0,
#                   dH, delta, xi,
#                   h_guess = None, dH_given = True,
#                   print_to_file= True, fixed_mode = False)



    
    
    
    
    # def make_f_safe(self):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy() 
    #     GG_M   = self.GG_M.copy() 
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.sampling_idx.shape[1] + nf
    #     omega_idx  = self.omega_idx.copy()
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,3,4)
    #                              , 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)), 
    #                              axis = 2)
        
    #     #xi_inv = 
        
    #     xi_conj = xi.conj()
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
        
        
        
    #     self.diff_GGM_GLM = diff_GGM_GLM.copy()
    #     self.diff_GGP_GLP = diff_GGP_GLP.copy()
    #     self.xi = xi
        
    #     @njit
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
            
    #         dt = np.complex128
            
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
            
    #         # For equations (4), (20) & # (21) Croy & Popescu 2016
    #         nk = old_sig.shape[0]
            
    #         # For use in psi eom and omega eom
    #         psi_conj = old_psi.conj()
            
    #         # Lead energy shift at time t´
    #         delta_t = np.zeros(nl)
            
    #         for a in range(nl):
    #             delta_t[a] = delta_variant(t, a)
            
    #         # Hamiltonian at time t
    #         #if dH_given:
    #         Ht = H + dH(t)
    #         #else:
    #         #    Ht = dH(t)
    #         #print(delta_t)
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         #print(D_sig)
    #         for a in range(nl):
    #             pi_a   = PI(old_psi[:, a], xi_conj[:, a]) #### Transpose removed, MAYBE NOT?????
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
            
    #         #####
            
    #         ##### Psi vector EOM:
    #         hbar_sq = hbar**2
    #         for k in range(nk):
    #             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
    #             old_omega_k = old_omega[k]
                
    #             for a in range(nl):
    #                 for x in range(Ntot):
    #                     for c in range(no):
    #                         Hchi = Ht[k,:,:].copy()
    #                         idx  = omega_idx[a,x,c,:,:,:].copy().reshape((nl*Ntot*no))
    #                         bOOl = ( idx >= 0 )
    #                         xi_sub    = xi_k[:,bOOl]
    #                         idx       = idx[bOOl]
    #                         omega_sub = old_omega_k[idx]
    #                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                            
    #                         for diag in range(no):
    #                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                            
    #                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(old_psi[k,a,x,c,:] ) / hbar                     +\
    #                                                   GL_P[k,a,x,c]*xi[k,a,x,c,:]                              +\
    #                                                   diff_GGP_GLP[k,a,x,c]* old_sig[k,:,:].dot(xi[k,a,x,c,:])  +\
    #                                                   omega_sum/hbar_sq 
    #                                                   )
                            
            
    #         for k in range(nk):
    #             for a1 in range(nl):
    #                 for x1 in range(Ntot):
    #                     for c1 in range(no):
    #                         psi_axc_1 = old_psi[k,a1,x1,c1,:]
    #                         xi_axc_1  =      xi[k,a1,x1,c1,:]
    #                         xp_axc_1  =     Xpp[k,a1,x1,c1] + delta_t[a1]
                            
    #                         diff_ggp_glp = diff_GGP_GLP[k,a1,x1,c1]
                            
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = omega_idx[a1,x1,c1,a2,x2,c2]
    #                                     if idx < 0:
    #                                         pass
    #                                     else:
    #                                         psi_axc_2    = psi_conj[k,a2,x2,c2,:]
    #                                         xi_axc_2     =  xi_conj[k,a2,x2,c2,:]
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            
    #                                         D_omega[k, idx] = ( -1j*(xm_axc_2  - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
    #                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
    #                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
    #                                                           )
    #         return D_sig, D_psi, D_omega
        
    #     return f
    
    
    # def make_f_exp(self, parallel = True, fastmath = True):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy() 
    #     GG_M   = self.GG_M.copy() 
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.sampling_idx.shape[1] + nf
    #     omega_idx  = self.omega_idx.copy()
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,3,4)
    #                              , 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)), 
    #                              axis = 2)
        
    #     xi_conj = xi.conj()
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
        
    #     #self.diff_GGM_GLM = diff_GGM_GLM.copy()
    #     #self.diff_GGP_GLP = diff_GGP_GLP.copy()
    #     #self.xi = xi
        
    #     @njit(parallel = parallel, fastmath = fastmath)
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
            
    #         dt = np.complex128
            
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
            
    #         # For equations (4), (20) & # (21) Croy & Popescu 2016
    #         nk = old_sig.shape[0]
            
    #         # For use in psi eom and omega eom
    #         psi_conj = old_psi.conj()
            
            
            
    #         # Lead energy shift at time t´
    #         delta_t = np.zeros(nl)
            
    #         for a in range(nl):
    #             delta_t[a] = delta_variant(t, a)
            
    #         # Hamiltonian at time t
    #         #if dH_given:
    #         Ht = H + dH(t)
    #         #else:
    #         #    Ht = dH(t)
    #         #print(delta_t)
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         #print(D_sig)
    #         for a in range(nl):
    #             pi_a   = PI(old_psi[:, a], xi_conj[:, a]) #### Transpose removed, MAYBE NOT?????
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
            
    #         #####
            
    #         ##### Psi vector EOM:
    #         hbar_sq = hbar**2
    #         for k in range(nk):
    #             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
    #             old_omega_k = old_omega[k]
                
    #             for a in range(nl):
    #                 for x in prange(Ntot):
    #                     for c in range(no):
    #                         Hchi = Ht[k,:,:].copy()
    #                         idx  = omega_idx[a,x,c,:,:,:].copy().reshape((nl*Ntot*no))
    #                         bOOl = ( idx >= 0 )
    #                         xi_sub    = xi_k[:,bOOl]
    #                         idx       = idx[bOOl]
    #                         omega_sub = old_omega_k[idx]
    #                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
                            
    #                         for diag in range(no):
    #                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                            
    #                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(old_psi[k,a,x,c,:] ) / hbar                     +\
    #                                                   GL_P[k,a,x,c]*xi[k,a,x,c,:]                              +\
    #                                                   diff_GGP_GLP[k,a,x,c]* old_sig[k,:,:].dot(xi[k,a,x,c,:])  +\
    #                                                   omega_sum/hbar_sq 
    #                                                   )
                            
                            
    #                         psi_axc_1 = old_psi[k,a,x,c,:]
    #                         xi_axc_1  =      xi[k,a,x,c,:]
    #                         xp_axc_1  =     Xpp[k,a,x,c] + delta_t[a]
                            
    #                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                            
                            
                            
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = omega_idx[a,x,c,a2,x2,c2]
    #                                     if idx < 0:
    #                                         pass
    #                                     else:
    #                                         psi_axc_2    = psi_conj[k,a2,x2,c2,:]
    #                                         xi_axc_2     =  xi_conj[k,a2,x2,c2,:]
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                            
    #                                         D_omega[k, idx] = ( -1j*(xm_axc_2  - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
    #                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
    #                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
    #                                                           )
    #         return D_sig, D_psi, D_omega
        
    #     return f
##############################################################################
            # self.broadenings.append(gel.gamma.copy())
            # self.Lorentzian_centers.append(gel.ei.copy())
            
            # gep2  = self.Ortho_Gammas[e].get_e_subset(idx_fermi_poles)
            # gep   = gel.evaluate_Lorentzian_basis(self.Contour[idx_fermi_poles])
            # gep  = Blocksparse2Numpy(gep,  SLICES)
            # gel  = Blocksparse2Numpy(gel,  SLICES)
            # gep2 = Blocksparse2Numpy(gep2, SLICES)
            
            # self._gp2_matrices += [ gep2 ]
            # EIG = np.linalg.eig
            # el, vl = Sorted_Eig(gel)#
            # assert np.allclose(gel, gel.transpose(0,1,3,2) )
            # ep, vp = Sorted_Eig(gep)#
            # assert np.allclose(gep, gep.transpose(0,1,3,2) )
            # Gl_eig.append(el)
            # Gl_vec.append(vl)
            # Gp_eig.append(ep)
            # Gp_vec.append(vp)
            # Gl.append(gel)
            # Gp.append(gep)
        
        # self.Gl_eig = np.array(Gl_eig) # indices: lead, k, x, state
        # self.Gl_vec = np.array(Gl_vec) # indices: lead, k, x, state, :
        # self.Gp_eig = np.array(Gp_eig)
        # self.Gp_vec = np.array(Gp_vec)
        # assert self.Gl_eig.shape[2] == NumL
        # self._Inv_Gl_vec = np.linalg.inv(self.Gl_vec)
        # self.Inv_Gp_vec  = np.linalg.inv(self.Gp_vec)
        # print('Maximum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.max(),6)))
        # print('Minimum of eigenvalues of Lorentzian Gammas: ' + str(np.round(self.Gl_eig.min(),6)))

        


# def make_f_general_V3(self, parallel = False, fastmath = False):
#     # The second index on H was for being able to multiply with energy-resolved quantities,
#     # which is not needed anymore
#     H      = self.Hdense[:,0,:,:].copy()
#     Xpp    = self.Xpp.copy()
#     Xpm    = self.Xpm.copy()
#     GG_P   = self.GG_P.copy()
#     GG_M   = self.GG_M.copy()
#     GL_P   = self.GL_P.copy()
#     GL_M   = self.GL_M.copy()
#     nl     = self.num_leads
#     nf     = self.num_poles
#     no     = H.shape[2]
#     nk     = H.shape[0]
#     Ntot   = self.NumL + nf
#     omega_idx  = self.omega_idx.copy()
#     psi_idx    = self.psi_idx.copy()
    
#     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
#     xi     = np.concatenate((
#                               self.Gl_vec.transpose(1,0,2,4,3)
#                               , 
#                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
#                              ), 
#                             axis = 2
#                             )
    
#     xi = np.ascontiguousarray(xi)
    
#     Ixi    = np.concatenate(
#                             (
#                               self.Gl_vec.transpose(1,0,2,4,3).conj()
#                               ,
#                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
#                             ),
#                            axis = 2
#                            )
    
#     Ixi = np.ascontiguousarray(Ixi)
    
#     diff_GGP_GLP = GG_P - GL_P
#     diff_GGM_GLM = GG_M - GL_M
#     self.diff_ggp_glp = diff_GGP_GLP
#     self.diff_ggm_glm = diff_GGM_GLM
    
#     self.xi  = xi
#     self.Ixi = Ixi
    
#     for a in range(nl):
#         for k in range(nk):
#             for xl in range(self.Gl_vec.shape[2]):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 =   self.Gl_vec[a,k,xl,:,c]
#                     vec2 =   self.Gl_vec[a,k,xl,:,c].conj()
#                     O    =  np.multiply.outer(vec1, vec2)
#                     m   +=   self.Gl_eig[a,k,xl,c] * O
#                 assert np.allclose(m, self._gl_matrices[a][k,xl,:,:])
#             for xf in range(2 * nf):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 = self.Gp_vec[a,k,xf,:,c]
#                     vec2 = self.Inv_Gp_vec[a,k,xf,c,:]
#                     O    = np.multiply.outer(vec1, vec2)
#                     m+=self.Gp_eig[a,k,xf,c] * O
#                 assert np.allclose(m, self._gp_matrices[a][k,xf,:,:])
    
#     @njit(parallel = parallel, fastmath = fastmath)
#     def f(t, 
#           old_sig, old_psi, old_omega,
#           dH, delta_variant, dH_given = True):
#         #Create the arrays needed
#         dt = np.complex128
#         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
#         D_omega =  np.zeros(old_omega.shape , dtype = dt)
#         nk = old_sig.shape[0]
        
#         # psi_dagger is needed:
#         psi_tilde  = old_psi.conj()
#         # Store bias at time t
#         delta_t = np.zeros(nl)
#         for a in range(nl): delta_t[a] = delta_variant(t, a)
#         # Get Hamiltonian
#         if dH_given: Ht =  H + dH(t, old_sig)
#         else:        Ht =      dH(t, old_sig)
        
#         ##### Density matrix EOM:
#         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
#         for a in range(nl):
#             pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
#             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
#         D_sig *= 1/hbar
        
#         ##### Psi & Omega EOM:
#         hbar_sq = hbar**2
#         for k in range(nk):
#             # Order the xi-vectors in columns (.T)
#             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
#             old_omega_k = old_omega[k]
#             old_sig_k   = old_sig[k,:,:].copy()
#             for x in prange(Ntot):
#                 x_omega_idx = omega_idx[:,x,:,:,:,:]
#                 psi_kx = old_psi[k,:,x,:,:]
#                 xi_kx  = xi[k,:,x]
#                 Xpp_kx = Xpp[k,:,x]
                
#                 for a in range(nl):
#                     xa_omega_idx = x_omega_idx[a]
#                     psi_kax      = psi_kx[a]
#                     xi_kax       = xi_kx [a]
#                     Xpp_kax      = Xpp_kx[a]
                    
#                     for c in range(no):
#                         # We also use axc_idx etc. later
#                         #axc_idx   = omega_idx[a,x,c,:,:,:].copy()
#                         axc_idx    = xa_omega_idx[c,:,:,:].copy()
                        
#                         #psi_axc_1 = old_psi[k,a,x,c,:]
#                         psi_axc_1  = psi_kax[c,:]
                        
#                         #xi_axc_1  = xi[k,a,x,c,:]
#                         xi_axc_1   = xi_kax[c,:]
                        
#                         #xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
#                         xp_axc_1   = Xpp_kax[c] + delta_t[a]
                        
#                         #We non-zero psi's are has an index that is greater than or equal to zero.
                        
#                         ###
#                         if psi_idx[a,x,c]>=0:
#                             idx       = axc_idx.reshape((nl*Ntot*no))
#                             # Use only non-zero Omegas
#                             bOOl      =(idx >= 0 )
#                             xi_sub    = xi_k[:,bOOl]
#                             idx       = idx[bOOl]
#                             omega_sub = old_omega_k[idx]
#                             # Do sum in the last term of the EOM
#                             omega_sum = (omega_sub * xi_sub).sum(axis = 1)
#                             # Make the part involving Hamiltonian
#                             Hchi      = Ht[k,:,:].copy()
#                             for diag in range(no):
#                                 Hchi[diag,diag] +=  - xp_axc_1
#                             #Calculate
#                             D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
#                                                       GL_P[k,a,x,c] * xi_axc_1                            +\
#                                                       diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
#                                                       omega_sum/hbar_sq
#                                                     )
#                         ###
                        
#                         # Differences between the Lambdas:
#                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
#                         for a2 in range(nl):
#                             psi_a_2         = psi_tilde[k,a2]
#                             xi_a_2          = Ixi[k,a2]
#                             diff_ggm_glm_a_2= diff_GGM_GLM[k,a2]
#                             xm_a_2          = Xpm[k,a2] + delta_t[a2]
                            
#                             for x2 in range(Ntot):
#                                 psi_ax_2         = psi_a_2[x2]
#                                 xi_ax_2          = xi_a_2[x2]
#                                 diff_ggm_glm_ax_2= diff_ggm_glm_a_2[x2]
#                                 xm_ax_2          = xm_a_2[x2]
                                
#                                 for c2 in range(no):
#                                     idx = axc_idx[a2,x2,c2]
#                                     # Only non-zero terms calculated:
#                                     if idx < 0:
#                                         pass
#                                     else:
#                                         psi_axc_2    = psi_ax_2[c2]          #psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
#                                         xi_axc_2     = xi_ax_2[c2]           #Ixi[k,a2,x2,c2,:]       # 
#                                         diff_ggm_glm = diff_ggm_glm_ax_2[c2] #diff_GGM_GLM[k,a2,x2,c2]
#                                         xm_axc_2     = xm_ax_2[c2]            #Xpm[k,a2,x2,c2] + delta_t[a2]
#                                         #Calculate
#                                         D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
#                                                                   diff_ggm_glm * xi_axc_2.dot(psi_axc_1)                +\
#                                                                   diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
#                                                           )
        
#         return D_sig, D_psi, D_omega
    
    
#     return f
#### This method is expendable, always make the optimization 
#### in "make_f_general" and then just add in down here
# def make_f_general_v2(self, parallel = False, fastmath = False):
#     # The second index on H was for being able to multiply with energy-resolved quantities,
#     # which is not needed anymore
#     H      = self.Hdense[:,0,:,:].copy()
#     Xpp    = self.Xpp.copy()
#     Xpm    = self.Xpm.copy()
#     GG_P   = self.GG_P.copy()
#     GG_M   = self.GG_M.copy()
#     GL_P   = self.GL_P.copy()
#     GL_M   = self.GL_M.copy()
#     nl     = self.num_leads
#     nf     = self.num_poles
#     no     = H.shape[2]
#     nk     = H.shape[0]
#     Ntot   = self.NumL + nf
#     omega_idx  = self.omega_idx.copy()
    
#     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
#     xi     = np.concatenate((
#                               self.Gl_vec.transpose(1,0,2,4,3)
#                               , 
#                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
#                              ), 
#                             axis = 2
#                             )
    
#     xi = np.ascontiguousarray(xi)
    
#     Ixi    = np.concatenate(
#                             (
#                               self.Gl_vec.transpose(1,0,2,4,3).conj()
#                               ,
#                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
#                             ),
#                            axis = 2
#                            )
    
#     Ixi = np.ascontiguousarray(Ixi)
    
#     diff_GGP_GLP = GG_P - GL_P
#     diff_GGM_GLM = GG_M - GL_M
#     self.diff_ggp_glp = diff_GGP_GLP
#     self.diff_ggm_glm = diff_GGM_GLM
    
#     self.xi  = xi
#     self.Ixi = Ixi
    
#     for a in range(nl):
#         for k in range(nk):
#             for xl in range(self.Gl_vec.shape[2]):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 =   self.Gl_vec[a,k,xl,:,c]
#                     vec2 =   self.Gl_vec[a,k,xl,:,c].conj()
#                     O    =  np.multiply.outer(vec1, vec2)
#                     m   +=   self.Gl_eig[a,k,xl,c] * O
#                 assert np.allclose(m, self._gl_matrices[a][k,xl,:,:])
#             for xf in range(2 * nf):
#                 m = np.zeros((no,no), dtype = np.complex128)
#                 for c in range(no):
#                     vec1 = self.Gp_vec[a,k,xf,:,c]
#                     vec2 = self.Inv_Gp_vec[a,k,xf,c,:]
#                     O    = np.multiply.outer(vec1, vec2)
#                     m+=self.Gp_eig[a,k,xf,c] * O
#                 assert np.allclose(m, self._gp_matrices[a][k,xf,:,:])
#     ####    VERSION 2    #####
#     @njit(parallel = parallel, fastmath = fastmath)
#     def f(t, Ht,
#           old_sig, old_psi, old_omega,
#           delta_variant):
        
#         dt = np.complex128
#         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
#         D_omega =  np.zeros(old_omega.shape , dtype = dt)
#         nk = old_sig.shape[0]
#         psi_tilde  = old_psi.conj()
        
#         delta_t = np.zeros(nl)
#         for a in range(nl): delta_t[a] = delta_variant(t, a)
#         #if dH_given: Ht =  H + dH(t, old_sig)
#         #else:        Ht =      dH(t, old_sig)
        
#         ##### Density matrix EOM:
#         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
#         for a in range(nl):
#             pi_a   = PI(old_psi[:, a], Ixi[:, a]) 
#             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
#         D_sig *= 1/hbar
        
#         ##### Psi & Omega EOM:
#         hbar_sq = hbar**2
#         for k in range(nk):
#             xi_k = xi[k].reshape((nl * Ntot * no, no)).T
#             old_omega_k = old_omega[k]
#             old_sig_k   = old_sig[k,:,:].copy()
#             for x in prange(Ntot):
#                 for a in range(nl):
#                     for c in range(no):
#                         # We also use axc_idx later
#                         axc_idx = omega_idx[a,x,c,:,:,:].copy()
#                         idx     = axc_idx.reshape((nl*Ntot*no))
                        
#                         bOOl      =(idx >= 0 )
#                         xi_sub    = xi_k[:,bOOl]
#                         idx       = idx[bOOl]
#                         omega_sub = old_omega_k[idx]
#                         omega_sum = (omega_sub * xi_sub).sum(axis = 1)
#                         psi_axc_1 = old_psi[k,a,x,c,:]
#                         xi_axc_1  = xi[k,a,x,c,:]
#                         xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
#                         Hchi      = Ht[k,:,:].copy()
                        
#                         for diag in range(no):
#                             Hchi[diag,diag] +=  - ( Xpp[k,a,x,c]  + delta_t[a] )
                        
#                         D_psi[k,a,x,c,:]= -1j * ( Hchi.dot(psi_axc_1 ) / hbar                         +\
#                                                   GL_P[k,a,x,c] * xi_axc_1                            +\
#                                                   diff_GGP_GLP[k,a,x,c]* old_sig_k.dot(xi_axc_1)      +\
#                                                   omega_sum/hbar_sq
#                                                 )
                        
#                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
                        
#                         for a2 in range(nl):
#                             for x2 in range(Ntot):
#                                 for c2 in range(no):
#                                     idx = axc_idx[a2,x2,c2]
#                                     if idx < 0:
#                                         pass
#                                     else:
#                                         psi_axc_2    = psi_tilde[k,a2,x2,c2,:]
#                                         xi_axc_2     = Ixi[k,a2,x2,c2,:]
#                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
#                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
                                        
#                                         D_omega[k, idx] = ( -1j*(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] / hbar      +\
#                                                                  diff_ggm_glm * xi_axc_2.dot(psi_axc_1)  +\
#                                                                  diff_ggp_glp * psi_axc_2.dot(xi_axc_1)
#                                                           )
        
#         return D_sig, D_psi, D_omega
    
#     return f


    # def make_f_most_experimental(self, parallel = False, fastmath = False, nogil = False):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:].copy()
    #     Xpp    = self.Xpp.copy()
    #     Xpm    = self.Xpm.copy()
    #     GG_P   = self.GG_P.copy()
    #     GG_M   = self.GG_M.copy()
    #     GL_P   = self.GL_P.copy()
    #     GL_M   = self.GL_M.copy()
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     nk     = H.shape[0]
    #     Ntot   = self.NumL + nf
    #     omega_idx  = self.omega_idx.copy()
    #     psi_idx    = self.psi_idx.copy()
    #     no_top = 0; no_top += self.max_orbital_idx + 1
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
    #                              self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
    #                             ), axis = 2)
        
    #     xi = np.ascontiguousarray(xi)
        
    #     Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
    #                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
    #                             ),axis = 2 )
    #     Ixi = np.ascontiguousarray(Ixi)
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
    #     self.diff_ggp_glp = diff_GGP_GLP
    #     self.diff_ggm_glm = diff_GGM_GLM
        
    #     self.xi  = xi
    #     self.Ixi = Ixi
        
    #     @njit(parallel = parallel, fastmath = fastmath, nogil = nogil)
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
    #         #Create the arrays needed
    #         dt = np.complex128
    #         D_psi    = np.zeros(old_psi.shape   , dtype = dt)
    #         D_omega =  np.zeros(old_omega.shape , dtype = dt)
    #         nk = old_sig.shape[0]
            
    #         # psi_dagger is needed:
    #         psi_tilde  = old_psi.conj()
    #         # Store bias at time t
    #         delta_t = np.zeros(nl)
    #         for a in range(nl): delta_t[a] = delta_variant(t, a)
    #         # Get Hamiltonian
    #         if dH_given: Ht =  H + dH(t, old_sig)
    #         else:        Ht =      dH(t, old_sig)
            
    #         ##### Density matrix EOM:
    #         D_sig = - 1j * (dot_3d(Ht , old_sig) - dot_3d(old_sig , Ht))
    #         for a in range(nl):
    #             pi_a   = PI_opti(old_psi[:, a], Ixi[:, a], psi_idx[a,:,:])  #PI(old_psi[:, a], Ixi[:, a]) 
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
    #         D_sig *= 1/hbar
            
    #         ##### Psi & Omega EOM:
    #         hbar_sq  = hbar**2
    #         _1ohbar  = 1/hbar
    #         _1johbar = 1j /hbar
    #         for x in prange(Ntot):
    #             for k in range(nk):
    #                 for a in range(nl):
    #                     for c in range(no_top):
    #                         # We also use axc_idx etc. later
    #                         axc_idx   = omega_idx[a,x,c,:,:,:]
    #                         psi_axc_1 = old_psi[k,a,x,c,:]
    #                         xi_axc_1  = xi[k,a,x,c,:]
    #                         xp_axc_1  = Xpp[k,a,x,c] + delta_t[a]
    #                         # We non-zero psi's are has an index that is greater than or equal to zero.
    #                         # if psi_idx[a,x,c]>=0:
    #                         # Calculate
    #                         D_psi[k,a,x,c,:]=  ( (Ht[k].dot(psi_axc_1 )  - (Xpp[k,a,x,c] + delta_t[a] ) * psi_axc_1) * _1ohbar
    #                                                       + GL_P[k,a,x,c] * xi_axc_1
    #                                                       + diff_GGP_GLP[k,a,x,c]* old_sig[k].dot(xi_axc_1)
    #                                                )
                            
    #                         ###
    #                         #  Differences between the Lambdas:
    #                         omega_sum = np.zeros(no, dtype = np.complex128)
    #                         diff_ggp_glp = diff_GGP_GLP[k,a,x,c]
    #                         for a2 in range(nl):
    #                             for x2 in range(Ntot):
    #                                 for c2 in range(no):
    #                                     idx = axc_idx[a2,x2,c2]
    #                                     # Only non-zero terms calculated:
    #                                     if idx >= 0:
    #                                         psi_axc_2    = psi_tilde[k,a2,x2,c2,:] # psi_tilde was the conjugate of psi, see top of function
    #                                         xi_axc_2     = Ixi[k,a2,x2,c2,:]       # 
    #                                         diff_ggm_glm = diff_GGM_GLM[k,a2,x2,c2]
    #                                         xm_axc_2     = Xpm[k,a2,x2,c2] + delta_t[a2]
    #                                         # Calculate
    #                                         D_omega[k, idx] = ( -(xm_axc_2 - xp_axc_1  ) * old_omega[k, idx] *_1johbar 
    #                                                           +    diff_ggm_glm * np.sum(xi_axc_2  * psi_axc_1)
    #                                                           +    diff_ggp_glp * np.sum(psi_axc_2 * xi_axc_1 )
    #                                                            )
                                            
    #                                         #if psi_idx[a,x,c]>=0:
    #                                         omega_sum += old_omega[k, idx] * xi[k,a2,x2,c2]
                            
    #                         #if psi_idx[a,x,c]>=0:
    #                         D_psi[k,a,x,c,:] += (omega_sum/hbar_sq)
                            
            
    #         return D_sig, -1j * D_psi, D_omega
        
        
    #     return f
    
# def Inspect_SE_lorentzian_fit(self, lead,I,J,i,j,Emin = -3.0, Emax = 3.0, ik = 0,size = 2):
#     if len(self.fitted_self_energies)==0:
#         print('no fitted self energies!')
#         return
    
#     m1 = self.self_energies[lead]
#     m2 = self.fitted_self_energies[lead]
#     E  = np.linspace(Emin,Emax,1000)
#     plt.show()
#     sidx = self.sampling_idx[lead]
#     plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].real,label = r'Re[Sampled $\Sigma$]',marker = '*',s = size)
#     plt.scatter(self.Contour[sidx].real, m1.Block(I,J)[ik,sidx,i,j].imag,label = r'Im[Sampled $\Sigma$]',marker = '*',s = size)
    
#     plt.plot(E, m2.evaluate_Lorentzian_basis(E).Block(I,J)[ik,:,i,j].real,label = 'Re[Lorentz fit]',linestyle = 'dashed')
#     plt.plot(E, m2.evaluate_Lorentzian_basis(E).Block(I,J)[ik,:,i,j].imag,label = 'Im[Lorentz fit]',linestyle = 'dashed')
#     plt.legend()

    
    
    # def make_f_purenp_opti(self):
    #     # The second index on H was for being able to multiply with energy-resolved quantities,
    #     # which is not needed anymore
    #     H      = self.Hdense[:,0,:,:]
    #     Xpp    = self.Xpp
    #     Xpm    = self.Xpm
    #     GG_P   = self.GG_P
    #     GG_M   = self.GG_M
    #     GL_P   = self.GL_P
    #     GL_M   = self.GL_M
    #     nl     = self.num_leads
    #     nf     = self.num_poles
    #     no     = H.shape[2]
    #     Ntot   = self.NumL + nf
    #     Nlr    = self.NumL
        
    #     #omega_idx  = self.omega_idx
    #     psi_idx       = self.psi_idx
    #     noT = 0; noT += self.max_orbital_idx + 1
        
    #     # index ordering: kidx, lead_idx, mode_idx, matrix_idx....
    #     xi     = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3), 
    #                               self.Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,4,3)
    #                             ), axis = 2)
        
    #     xi = np.ascontiguousarray(xi)
        
    #     Ixi    = np.concatenate((self.Gl_vec.transpose(1,0,2,4,3).conj(),
    #                               self.Inv_Gp_vec[:,:,0:nf,:,:].transpose(1,0,2,3,4)
    #                             ),axis = 2 )
    #     Ixi = np.ascontiguousarray(Ixi)
        
    #     diff_GGP_GLP = GG_P - GL_P
    #     diff_GGM_GLM = GG_M - GL_M
    #     self.diff_ggp_glp = diff_GGP_GLP
    #     self.diff_ggm_glm = diff_GGM_GLM
        
    #     self.xi  = xi
    #     self.Ixi = Ixi
        
    #     def f(t, 
    #           old_sig, old_psi, old_omega,
    #           dH, delta_variant, dH_given = True):
    #         #Create the arrays needed
    #         dt = np.complex128
    #         nk = old_sig.shape[0]
            
    #         # psi_dagger is needed:
    #         psi_tilde  = old_psi.conj()
    #         # Store bias at time t
    #         delta_t = np.zeros(nl)
    #         for a in range(nl): delta_t[a] = delta_variant(t, a)
    #         # Get Hamiltonian
    #         if dH_given: Ht =  H + dH(t, old_sig)
    #         else:        Ht =      dH(t, old_sig)
            
    #         ##### Density matrix EOM:
    #         D_sig =  - 1j * (Ht@old_sig - old_sig@Ht)
    #         for a in range(nl):
    #             pi_a   = PI_np(old_psi[:, a], Ixi[:, a]) 
    #             #print(pi_a.shape)
    #             D_sig += pi_a + pi_a.conj().transpose(0,2,1)
            
    #         D_sig *= 1/hbar
    #         ##### Psi & Omega EOM:
    #         nax = np.newaxis
    #         Xpp_delta  =  Xpp + delta_t[nax,:, nax, nax]
    #         Xpm_delta  =  Xpm + delta_t[nax,:, nax, nax]
            
    #         D_psi  = old_psi @ np.expand_dims(Ht.transpose((0, 2, 1)), (1, 2)) / hbar
    #         D_psi -= np.expand_dims(Xpp_delta[:,:,:,:noT] / hbar, 4) * old_psi [:,:,:,:noT]
    #         D_psi += np.expand_dims(GL_P[:,:,:,:noT], 4) * xi[:,:,:,:noT]
    #         D_psi += np.expand_dims(diff_GGP_GLP[:,:,:,:noT], 4) * (
    #                                 xi[:,:,:,:noT] @ np.expand_dims(old_sig.transpose((0, 2, 1)), (1, 2))
    #                                 )
            
    #         om_shape = (nk, nl*Ntot*noT, nl*Ntot*no)
    #         xi_shape = (nk, nl*Ntot*no , -1        )
            
    #         D_psi   += (old_omega.reshape(om_shape) @ xi.reshape(xi_shape)).reshape(D_psi.shape) / (hbar ** 2)
            
            
            
    #         s_ll = (nk, nl, Nlr, noT, nl, Nlr, no)
    #         s_lf = (nk, nl, Nlr, noT, nl, nf , no)
    #         s_fl = (nk, nl, nf , noT, nl, Nlr, no)
            
    #         D_omega  = np.zeros(old_omega.shape,dtype = dt)
            
    #         D_omega[:,:,0:Nlr,:,:,0:Nlr, :]  +=  (old_psi[:, :, 0:Nlr, :noT, :].reshape(nk, nl*Nlr*noT, no) @
    #                                             ((diff_GGM_GLM[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, 1)
    #                                             * Ixi[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, no)).transpose((0, 2, 1)))).reshape(s_ll)
    #         D_omega[:,:,0:Nlr,:,:,Nlr:Ntot, :]  +=  (old_psi[:,:,0:Nlr,:noT,:].reshape(nk, nl*Nlr*noT, no) @
    #                                               ((diff_GGM_GLM[:,:,Nlr:Ntot,:].reshape(nk, nl*nf*no, 1)
    #                                               * Ixi[:,:,Nlr:Ntot,:].reshape(nk, nl*nf*no, no)).transpose((0, 2, 1)))).reshape(s_lf)
    #         D_omega[:,:,Nlr:Ntot,:,:,0:Nlr, :]  +=  (old_psi[:,:,Nlr:Ntot,:noT,:].reshape(nk, nl*nf*noT, no) @
    #                                               ((diff_GGM_GLM[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, 1)
    #                                               * Ixi[:,:,0:Nlr,:].reshape(nk, nl*Nlr*no, no)).transpose((0, 2, 1)))).reshape(s_fl)
            
    #         D_omega[:,:,:,0:noT, :, :, 0:noT] += ((diff_GGP_GLP[:,:,:,:noT].reshape(nk, nl*Ntot*noT, 1)
    #                                               * xi[:,:,:,:noT].reshape(nk, nl*Ntot*noT, no)) @
    #                                                 old_psi.conj().reshape(nk, nl*Ntot*noT, no)
    #                                                 .transpose((0, 2, 1))
    #                                                 ).reshape(nk, nl, Ntot, noT, nl, Ntot, noT)
            
    #         for k in range(nk):
    #             D_omega[k]  += (np.subtract.outer(Xpp_delta[k,:,:,:noT], Xpm_delta[k])*old_omega[k])*(1j/hbar)
            
    #         return D_sig, -1j * D_psi, D_omega
        
        
    #     return f





# def run_curvefit_SE(self,lead,I,J,i,j,ik=0, use_dense_grid=True, fix_L_idx = []):
#     sidx = self.sampling_idx[lead]
#     bij  = self.NO_fitted_lorentzians[lead].Block(I,J)
    
#     if hasattr(self, '_old_sampling_idx') and use_dense_grid:
#         sidx = self._old_sampling_idx[lead]
#     m1   = self.self_energies[lead].Block(I,J)[ik,sidx,i,j]
    
#     if bij is not None:
#         g0   = bij[ik,:,i,j].copy()
#     else:
#         g0 = np.zeros(len(self.NO_fitted_lorentzians[lead].ei[0,:]))
#         return None, None
    
#     line   = self.Contour[sidx].real
#     ei, wi = self.NO_fitted_lorentzians[lead].ei[ik].copy(),self.fitted_lorentzians[lead].gamma[ik].copy()
    
#     idx0 = np.array(fix_L_idx, dtype=int)
#     idx1 = np.array([i for i in range(len(ei)) if i not in idx0])
#     G1   = g0[idx0] # fixed vals
#     G0   = g0[idx1] # variable vals
#     N    = len(g0)
#     def func(x, *p):
#         _p = np.zeros((3, N))
#         _p[0] = wi
#         _p[1] = ei
#         _p[2,idx1] = np.array(p).real
#         _p[2,idx0] = G1.real
#         return (KK_L_sum(x, _p)/2-L_sum(x, _p)/2).real
    
#     poptr, pcov = curve_fit(func, line, m1.real, p0=G0.real)
#     def func(x, *p):
#         _p = np.zeros((3, N))
#         _p[0] = wi
#         _p[1] = ei
#         _p[2,idx1] = np.array(p).real
#         _p[2,idx0] = G1.imag
#         return (KK_L_sum(x, _p)/2-L_sum(x, _p)/2).real
#     popti, pcov = curve_fit(func, line, m1.imag, p0=G0.imag)
#     popt = poptr + 1j * popti
#     POPT = np.zeros(N, dtype=complex)
#     POPT[idx0] = G1
#     POPT[idx1] = popt
#     return POPT, g0
    
    
    
