#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:53:07 2026

@author: aleks
"""
import inspect

def stepwise_const_bias(ControlInstance, 
                        n_steps, step_height, step_width = 40.0, softness = 3.0, 
                        modocc_kw = {},
                        scf_kw    = {},
                        psi0_kw   = {},
                        print_all_kws = False,
                        linearization_scheme = "lin_odm",
                        nozand=True, 
                        mpi="mpirun ",
                        arcname = "stepped_bias_calc",
                        skip_steady_state = False
                        ):
    """
    Automated stepwise time-dependent calculation.
    Args:
        ControlInstance: Initialized Control instance (from Zandpack.wrapper Control)
        n_steps : How many steps there is in the bias function
        step_height: How high the steps are in the bias function
        step_width: How long time in between each step 
        softness: How abrupt the step from one plateau to the next is.
        modocc_kw : dict, containing keywords that are passed to the modify_occupations tool (Some default values defined inside the function)
        scf_kw : dict, containing keywords that are passed to the SCF tool (Some default values defined inside the function)
        psi0_kw: dict, containing keywords that are passed to the psinought tool
    """
    C = ControlInstance
    # Default values
    # These kws are parsed to the modify_occupation tool
    _modocc_kw = {"eigtol": 1e-3, 
                  "N_F":25, 
                  "kT_i":[0.025, 0.025]}
    _modocc_kw.update(modocc_kw)
    # These kws are parsed to the SCF tool
    _scf_kw = {"DM_randomness":0.0, 
               "write_dm_every":10, 
               "weight":0.15,
               "drho_tol":1e-6, 
               "write_progress":True
               }
    _scf_kw.update(scf_kw)
    
    # These kws are parsed to the psinought tool
    _psi0_kw = {}
    _psi0_kw.update(psi0_kw)
    
    more_imports = ["from Zandpack.Pulses import stairs",
                    "step_height="+str(step_height),
                    "step_width="+str(step_width),
                    "softness="+str(softness),
                    "n_steps="+str(n_steps), 
                    ]
    # Note that the code this function is not run here.
    # inspect.getsource is used to write it to the Bias.py
    # file. This is why the "more_imports" is needed, 
    # to write the various inputs to the Bias.py also.
    def bias(t, a):
        V = stairs(t, step_height, step_width, n_steps, softness=softness)
        if a == 0:
            return  V
        else:
            return -V
    if skip_steady_state == False:
        C.write_bias(bias=bias, hook=C.hook, more_imports=more_imports)
        C.write_initial()
        C.modify_occupation(**_modocc_kw)
        C.run_scf(**_scf_kw)
        C.check()
        C.hook.scheme=linearization_scheme
        C.hook_linearize()
        C.write_bias(bias=bias, hook=C.hook, more_imports=more_imports)
        C.write_initial()
        _scf_kw_2 = _scf_kw.copy()
        _scf_kw_2.update({"DM_start_file":None})
        C.run_scf(**_scf_kw_2)
        C.run_psinought(**_psi0_kw)
    if nozand:
        C.input.orthogonal=False
        C.write_bias(bias=bias, hook=C.hook, more_imports=more_imports)
        C.write_initial()
        C.run_nozand(mpi)
    else:
        C.input.orthogonal=True
        C.write_bias(bias=bias, hook=C.hook, more_imports=more_imports)
        C.write_initial()
        C.run_nozand(mpi)
    if arcname is not None:
        C.archive_calculation(arcname)

def const_bias_and_sine(ControlInstance,
                        V, Ampl, w, tstart= 0.0,s = 2.5,
                        nozand = True, mpi = 'mpirun '):
    """
    Function for running a series of calculations with a bias function as
    V(t) = C + A*sin(wt). Please note you have to give this function a 
    Control object where you have the full equilibrium (V=0) steady state already (And much preferably
    the linearization also). The sine will be introduced at tstart
    Args:
        ConstrolInstance: Control object from the Zandpack.wrapper module
        V: scalar or 1D array of voltages.
        Ampl: scalar or 1D array of amplitudes for the sine part.
        w: scalar or 1D array of frequencies for the sine part
        tstart: starting time where the periodic perturbation is introduced
        s: how smoothly the periodic perturbation is introduced
    Returns
        A series of folders with the various calculations in.
    """
    C = ControlInstance
    
    import numpy as np
    if isinstance(V, float):
        V = np.array([V])
    V = V[np.argsort(np.abs(V))]
    if isinstance(Ampl, float):
        Ampl = np.array([Ampl])
    if isinstance(w, float):
        w = np.array([w])
    SSDM = [C.sigma]
    SSV  = [0.0]
    
    V    = np.round(V, 4)
    Ampl = np.round(Ampl,4)
    w   = np.round(w, 4)
    for vi in V:
        D = np.abs(np.array(SSV) - vi)
        idx = np.where(D == D.min())[0][0]
        DMSTART = SSDM[idx]
        np.save(C.working_dir + "/UseThisDM.npy", DMSTART)
        first_step = True
        for ai in Ampl:
            for wi in w:
                def bias(t,a):
                    env = 1-1/(np.exp((t-tstart)/s) + 1.0)
                    V = vi + env * ai * np.sin(wi * t)
                    if a == 0: return  V
                    else:      return -V
                more_imports = ["vi="+str(vi), "ai="+str(ai), "wi="+str(wi),
                                "tstart="+str(tstart), "s="+str(s), ]
                if first_step:
                    C.input.orthogonal=True
                    C.write_bias(bias=inspect.getsource(bias), hook=C.hook, 
                                 more_imports=more_imports, dm_diff_tol = 1.0)
                    C.write_initial()
                    C.run_scf(DM_randomness=0.0, write_dm_every=10, weight=0.1,
                              DM_start_file="UseThisDM.npy",
                              drho_tol = 1e-7, Nonequilibrium = True,
                              # Contour="../mycontour_2.npy", 
                              write_progress=True, adaptive_mixer = True
                              )
                    if C.scf_status==False:
                        print("Seems like SCF has trouble converging")
                    SSDM += [C.sigma.copy()]
                    SSV  += [vi + 0.0]
                    C.run_psinought()
                    if False in C.psinought_status:
                        print("Seems like psinought has problems converging")
                first_step = False
                if nozand:
                    C.input.orthogonal=False
                    C.write_bias(bias=bias, hook=C.hook, 
                                 more_imports=more_imports, dm_diff_tol = 1.0)
                    C.write_initial()
                    C.run_nozand(mpi)
                else:
                    C.input.orthogonal=True
                    C.write_bias(bias=bias, hook=C.hook, 
                                 more_imports=more_imports, dm_diff_tol = 1.0)
                    C.write_initial()
                    C.run_nozand(mpi)
                C.archive_calculation(C.input.name 
                                      + "_save_V_"+str(vi)
                                      + "_A_"+str(ai)
                                      + "_w_"+str(wi))