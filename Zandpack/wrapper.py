#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:47:40 2026

@author: aleks
"""
# from Zandpack.td_constants import hbar
import numpy as np
import inspect
import os, sisl
import time
from  Zandpack.plot import J, DM
from copy import deepcopy
from pickle import load
import datetime
from Zandpack.PadeDecomp import Hu_poles, FD_expanded
from textwrap import dedent

glob_test = False
# Wrapper classes for more easily control a zandpack calculation
# directly from python.

class Input:
    def __init__(self, 
                 name,
                 t0=0.0, 
                 t1=50.0, 
                 eps=1e-7, 
                 usesave=True,
                 LoadFromFull=True,
                 checkpoints = None,
                 save_checkpoints=False,
                 print_timings=False,
                 stepsize=0.1,
                 saveevery=50,
                 n_dm_compress=10, 
                 save_PI = False,
                 verbose=True,
                 orthogonal=True,
                 compress_mat=True,
                 dm_triu_only=True,
                 dm_occ_only=False,
                 ):
        self.name = name
        self.t0   = t0 
        self.t1   = t1
        self.eps  = eps
        self.usesave = usesave
        self.loadfromfull=LoadFromFull
        self.checkpoints = checkpoints
        if checkpoints is None:
            self.checkpoints = np.linspace(t0,t1, 5)[1:]
        self.save_checkpoints = save_checkpoints
        self.print_timings = print_timings
        self.stepsize=stepsize
        self.n_dm_compress=n_dm_compress
        self.save_PI = save_PI
        self.verbose = verbose
        self.orthogonal = True
        self.saveevery = saveevery
        self.compress_mat=compress_mat
        self.dm_triu_only=dm_triu_only
        self.dm_occ_only=dm_occ_only
    def write_initial(self, prefix):
        text = "from Zandpack.td_constants import hbar\n"
        text+= "from Zandpack.Loader import load_dictionary\n"
        text+= "import numpy as np\n"
        text+="name=\""+str(self.name)+"\"\n"
        text+="t0="+str(self.t0)+"\n"
        text+="t1="+str(self.t1)+"\n"
        text+="eps="+str(self.eps)+"\n"
        text+="usesave="+str(self.usesave)+"\n"
        text+="LoadFromFull="+str(self.loadfromfull)+"\n"
        text+="checkpoints="+str([float(v) for v in self.checkpoints])+"\n"
        text+="save_checkpoints="+str(self.save_checkpoints)+"\n"
        text+="print_timings="+str(self.print_timings)+"\n"
        text+="stepsize="+str(self.stepsize)+"\n"
        text+="n_dm_compress="+str(self.n_dm_compress)+"\n"
        text+="save_PI="+str(self.save_PI)+"\n"
        text+="saveevery="+str(self.saveevery)+"\n"
        text+="Adir = name + \'/Arrays/\'\n"
        text+="Arrs = load_dictionary(Adir)\n"
        text+="compress_mat="+str(self.compress_mat)+"\n"
        text+="dm_triu_only="+str(self.dm_triu_only)+"\n"
        text+="dm_occ_only="+str(self.dm_occ_only)+"\n"
        with open(prefix + "/Initial.py", "w") as f:
            f.write(text)
        if self.verbose:
            print("Wrote Initial.py file")
    def write_bias(self, prefix, more_imports = None, mpi4py=True,
                   os_envvar=[], bias=None, dH=None, 
                   hook = None, use_lin=False, dm_diff_tol = 1e-4):
        text = "import numpy as np\n"
        text+= "import sisl, os\n"
        text+= "from Zandpack.Help import TDHelper\n"
        if mpi4py:
            text+="from mpi4py import MPI; rank = MPI.COMM_WORLD.Get_rank(); size = MPI.COMM_WORLD.Get_size()\n"
        else:
            text+="rank=0 \n"
        if more_imports is not None:
            if isinstance(more_imports, str):
                text += more_imports + "\n"
            if isinstance(more_imports, list):
                for l in more_imports:
                    text += l + "\n"
        for ev in os_envvar:
            text += "_"+ev+"=os.environ[\"" +ev+"\"]\n"
        text+= "name=\""+str(self.name)+"\"\n"
        text+= "orth= "+str(self.orthogonal)+"\n"
        text+= "Hlp = TDHelper(name); nlead=Hlp.num_leads\n"
        text+= "L   = Hlp.Lowdin; iL = Hlp.invLowdin\n"
        text+= "# Standard functions for various transformations \n# between orthogonal and nonorthogonal basis\n"
        text+= "def sigO2NO(DMlike): return L  @ DMlike @ L\n"
        text+= "def sigNO2O(DMlike): return iL @ DMlike @ iL\n"
        text+= "def HamO2NO(Hlike):  return iL @ Hlike  @ iL\n"
        text+= "def HamNO2O(Hlike):  return L  @ Hlike  @ L\n"
        text+= "def sig2mul_O(dm):\n"
        text+= "    nodm = sigO2NO(dm)\n"
        text+= "    return Hlp.S @ nodm + nodm @ Hlp.S\n"
        text+= "def sig2mul_NO(dm):\n"
        text+= "    return Hlp.S @ dm + dm @ Hlp.S\n"
        text+= "def dissipator(t,sig): return 0.0\n"
        text+= "if Hlp.orb_pos is not None and rank == 0:\n    P1 = np.average(Hlp.pos_elecorbs[0],axis=0)\n"
        text+= "    P2 = np.average(Hlp.pos_elecorbs[1],axis=0)\n"
        text+= "    drc= P2 - P1; drc *= 1/np.linalg.norm(drc)\n"
        text+= "    Po = (Hlp.orb_pos - P1).dot(drc) \n"
        text+= "    X0 = Po.min(); X1 = Po.max()\n"
        text+= "    def ramp_field(r,t):\n"
        text+= "        # Modify this at will. \n        # this is a crude 2E version \n"
        text+= "        X = r.dot(drc); F = bias(t,0) + (bias(t,1) - bias(t,0)) *(X - X0)/(X1 - X0)\n"
        text+= "        return F \n"
        text+= "else: \n    def ramp_field(r,t): return 0.0\n"
        #if dH is None:
        text += "def Ramp(t): \n"
        text += "    return Hlp.approxfield2mat(t, ramp_field, orthogonal=orth)\n"
        #else:
        #   text+= dedent(inspect.getsource(dH))
        text += '\n'
        if isinstance(bias, str):
            text += dedent(bias)
        else:
            text += dedent(inspect.getsource(bias))
        text+='\n'
        text+="def dH(t, sig):\n"
        text+="    out = np.zeros(sig.shape,dtype=complex)\n"
        text+="    out+= Ramp(t)\n"
        text+="    DL  = [bias(t, a) for a in range(nlead) ]\n"
        text+="    out+=  Hlp.lead_dev_dyncorr(DeltaList = DL, orthogonal=orth)\n"
        text+="    out+= kohn_sham(sig, orthogonal=orth)\n"
        text+="    return out\n"
        if hook is None:
            text+="def kohn_sham(sig):\n"
            text+="    return 0.0"
        else:
            text+="from "+hook.scriptname.replace(".py","") +" import H_from_DFT\n"
            text+="# below, if linearization is active, we check if the DM used for linearization and the \n"
            text+="# and orthogonal DM from the Arrays directory are the same (in the orth.basis or nonorth. depending on the case)\n"
            if hook.scheme == 'lin_mul':
                text+="from Zandpack.wrapper import Mull_Lin_NO as Linear\n"
                text+="lin_dh = Linear(\"MullikenLin_NO.npz\", rank)\n"
                text+="if rank == 0:\n"
                text+="    assert lin_dh.FileNotFound == False\n"
                text+="    assert np.abs(lin_dh.dm0-sigO2NO(Hlp.DM0)).max() < "+str(dm_diff_tol)+"\n"
                text+="    if np.abs(lin_dh.dm0-sigO2NO(Hlp.DM0)).max() > 1e-7:\n"
                text+="        print(\"It seems the DM changed since you linearized the Hamiltonian?\")\n"
                text+="ldH = lin_dh.linearized_H\n"
                text+="def kohn_sham(sig, orthogonal=True):\n"
                text+="    if orthogonal:\n"
                text+="        return HamNO2O( lin_dh.H0 + ldH(sigO2NO(sig))) - (Hlp.H0 - Hlp.Hcorr)\n"
                text+="    else:\n"
                text+="        return lin_dh.H0 + ldH(sig) - HamO2NO(Hlp.H0 - Hlp.Hcorr)\n"
            elif hook.scheme == 'lin_odm':
                text+="from Zandpack.wrapper import DM_Lin_O as Linear\n"
                text+="lin_dh = Linear(\"DM_Lin_O.npz\", rank)\n"
                text+="if rank == 0:\n"
                text+="    assert lin_dh.FileNotFound == False\n"
                text+="    assert np.abs(lin_dh.dm0-Hlp.DM0).max() < "+str(dm_diff_tol)+"\n"
                text+="    if np.abs(lin_dh.dm0-Hlp.DM0).max() > 1e-7:\n"
                text+="        print(\"It seems the DM changed since you linearized the Hamiltonian?\")\n"
                text+="ldH = lin_dh.linearized_H\n"
                text+="def kohn_sham(sig, orthogonal=True):\n"
                text+="    if orthogonal:\n"
                text+="        return HamNO2O( lin_dh.H0 + ldH(sig)) - (Hlp.H0 - Hlp.Hcorr)\n"
                text+="    else:\n"
                text+="        return lin_dh.H0 + ldH(sigNO2O(sig)) - HamO2NO(Hlp.H0 - Hlp.Hcorr)\n"
            elif hook.scheme == 'lin_nodm':
                text+="from Zandpack.wrapper import DM_Lin_NO as Linear\n"
                text+="lin_dh = Linear(\"DM_Lin_NO.npz\", rank)\n"
                text+="if rank == 0:\n"
                text+="    assert lin_dh.FileNotFound == False\n"
                text+="    assert np.abs(lin_dh.dm0-sigO2NO(Hlp.DM0)).max() < "+str(dm_diff_tol)+"\n"
                text+="    if np.abs(lin_dh.dm0-sigO2NO(Hlp.DM0)).max() > 1e-7:\n"
                text+="        print(\"It seems the DM changed since you linearized the Hamiltonian?\")\n"
                text+="ldH = lin_dh.linearized_H\n"
                text+="def kohn_sham(sig, orthogonal=True):\n"
                text+="    if orthogonal:\n"
                text+="        return HamNO2O( lin_dh.H0 + ldH(sigO2NO(sig))) - (Hlp.H0 - Hlp.Hcorr)\n"
                text+="    else:\n"
                text+="        return lin_dh.H0 + ldH(sig) - HamO2NO(Hlp.H0 - Hlp.Hcorr)\n"
            elif hook.scheme == 'full':
                text+="def kohn_sham(sig, orthogonal=True):\n"
                text+="    if orthogonal:\n"
                text+="        return HamNO2O(H_from_DFT(sigO2NO(sig))) - (Hlp.H0 - Hlp.Hcorr)\n"
                text+="    else:\n"
                text+="        return H_from_DFT(sig) - HamO2NO(Hlp.H0 - Hlp.Hcorr)\n"
                
        with open(prefix + "/Bias.py", "w") as f:
            f.write(text)
        if self.verbose:
            print("Wrote Bias.py file")
    def pickle(self, filename):
        """Saves input class to file"""
        import pickle as pkl
        f = open(filename +'.input', 'wb')
        pkl.dump(self, f)
        f.close()
    def copy(self):
        out = deepcopy(self)
        return out

def fmt_str_cmd(s):
    if isinstance(s, np.ndarray):
        s2 = str([float(i) for i in list(s)])
    else:
        s2 = str(s)
    for v in ["[", "]", "(", ")", " "]:
        s2 = s2.replace(v, "")
    return s2

class Control:
    def __init__(self, input_class, source_files=None, logfile = 'cmds_1.txt', 
                 livelog="cmds.txt", prepend_env_vars = ["OMP_NUM_THREADS=1", "NUMBA_NUM_THREADS=1"]):
        self.input = input_class
        # source files is the folder written by TD_Transport 
        # when using "tofile"
        self.srcf    = source_files
        self.working_dir = None
        self.textlog = []
        self._rawlog = []
        self.basedir = os.getcwd()
        self.txtlogfile = logfile
        self.livelog = livelog
        self.prepend_env_vars = prepend_env_vars
        self._first_logwrite = True
    @property
    def scf_status(self):
        self.into_wd()
        try:
            msg = open("SCF_MESSAGE.txt",'r').read()
            if "success" in msg.lower():
                out =  True
            else:
                out =  False
        except:
            print('SCF hasnt run yet it seems')
            out =  False
        self.out_wd()
        return out
    
    @property
    def psinought_status(self):
        self.into_wd()
        try:
            msg = open("psinought_dpsi_MESSAGE.txt",'r').read()
            if "success" in msg.lower():
                dpsi  =  True
            else:
                dpsi  =  False
        except:
            print('It seems psinought has not run / finished')
            dpsi=False
        try:
            msg = open("psinought_dsig_MESSAGE.txt",'r').read()
            if "success" in msg.lower():
                dsig  =  True
            else:
                dsig  =  False
        except:
            dsig=False
        self.out_wd()
        return {'dpsi_conv': dpsi, 'dsig_conv':dsig}
    
    def set_direc(self, folder):
        self.working_dir = folder
        self.wdir_abspath = os.getcwd() + folder
    def into_wd(self):
        #self.orig_dir = os.getcwd()
        os.chdir(self.working_dir)
        self.textlog += ['cd into '+str(self.working_dir)+'\n']
        self._rawlog += ['cd '+self.working_dir+"\n"]
        
    def out_wd(self):
        os.chdir(self.basedir)
        self.textlog += ['cd into '+str(self.basedir)+'\n']
        self._rawlog += ['cd '+self.basedir]
    def create_wd(self):
        if self.working_dir is None:
            print("WARNING! You need to set working directory...")
            return
        if os.path.isdir(self.working_dir) == False:
            self.systemcall('mkdir ' + self.working_dir)
    def init(self, overwrite=True):
        self.create_wd()
        if os.path.isdir(self.working_dir+"/"+self.input.name+'_SRC'):
            if overwrite:
                cmd = "rm -rf "+self.working_dir+"/"+self.input.name+'_SRC'
                self.systemcall(cmd)
                cmd = "cp -R "+self.srcf + " "+self.working_dir+"/"+self.input.name+'_SRC'
                self.systemcall(cmd)
                print("Removed: " +self.working_dir+"/"+self.input.name+'_SRC')
                print("and put in copy from "+self.srcf)
            else:
                pass
        else:
            cmd = "cp -R "+self.srcf + " "+self.working_dir+"/"+self.input.name+'_SRC'
            self.systemcall(cmd)
    def write_bias(self, prefix=None, **kwargs):
        if prefix is None:
            _prefix = self.working_dir
        else:
            _prefix = prefix
        self.input.write_bias(_prefix, **kwargs)
    def write_initial(self, prefix=None, **kwargs):
        if prefix is None:
            _prefix = self.working_dir
        else:
            _prefix = prefix
        self.input.write_initial(_prefix, **kwargs)
    def set_hook(self, hook, write = True, **kwargs):
        self.hook = hook
        if write:
            self.write_hook(**kwargs)
    def write_hook(self, **kwargs):
        self.hook.write_hook(self.input.name, self.working_dir, self.basedir,  
                             **kwargs)
    #def noneq_run(self, V):
    #    
    def init_from_other(self, file_or_dir, newname):
        """
        This function is useful if you have a working directory and you want to reuse
        the input files for another calculation.
        """
        self.create_wd()
        self.systemcall("cp -rs $PWD/"+file_or_dir
                        + " $PWD/"+self.working_dir+"/"+newname)
        
    def copy_state(self, other_controller):
        B = other_controller
        self.create_wd()
        self.init_from_other(B.working_dir +'/'+B.input.name, self.input.name)
        self.create_subfolder(self.input.name+"_save")
        self.init_from_other(B.working_dir+'/'+B.input.name+"_save/last_psi.npy", 
                             self.input.name+"_save/last_psi.npy")
        self.init_from_other(B.working_dir+'/'+B.input.name+"_save/last_omg.npy", 
                             self.input.name+"_save/last_omg.npy")
    def create_subfolder(self, name):
        self.into_wd()
        os.mkdir(name)
        self.textlog+=['made subfolder '+name+'\n']
        self._rawlog+=['mkdir '+name+'\n']
        self.out_wd()
    def systemcall(self, cmd):
        now = time.ctime()
        t1 = time.time()
        if glob_test==False:
            exst = os.system(cmd)
            exst = str(exst)
        else:
            exst = '?'
        t2 = time.time()
        self.rawlog(now+"\n")
        self.rawlog(cmd + '\n')
        self.rawlog("exit status: " +exst+ '\n')
        self.rawlog(str(datetime.timedelta(seconds=t2-t1))+'\n\n')
    def rawlog(self, s):
        if type(self.livelog) is str:
            if os.path.isfile(self.livelog) == False:
                mode = "w"
            else:
                mode = "a"
            with open(self.livelog,mode) as f:
                f.write(s)
        if self._first_logwrite:
            self.livelog = os.getcwd() + "/"+self.livelog
            self._first_logwrite = False
        self._rawlog += [s]
    @property
    def sigma(self):
        dmpath = self.basedir+"/"+self.working_dir+'/'+self.input.name+'/Arrays/DM_Ortho.npy'
        try:
            return np.load(dmpath)
        except:
            print("failed to load from "+dmpath)
    
    @property
    def psi0(self):
        psipath = self.basedir+"/"+self.working_dir+'/'+self.input.name+'_save/last_psi.npy'
        try:
            return np.load(psipath)
        except:
            print("failed to load from "+psipath)
    @property
    def scf_H(self):
        Hpath = self.basedir+"/"+self.working_dir+'/'+self.input.name+'/Arrays/SCF_Hlast_Ortho.npy'
        try:
            return np.load(Hpath)
        except:
            print("Failed to load H from "+Hpath+". Did you run SCF first?")
    def check(self):
        HO = self.scf_H
        if HO is None:
            print("No scf H in the directory.")
        else:
            assert np.allclose(HO, np.conj(HO.transpose(0,2,1)))
            e = np.linalg.eigvalsh(HO)
            N_F = np.load(self.basedir+"/"+self.working_dir+'/'+self.input.name+'/Arrays/num_poles_fermi.npy') // 2 
            kT = np.load(self.basedir+"/"+self.working_dir+'/'+self.input.name+'/Arrays/kT_i.npy').min()
            x,c = Hu_poles(N_F)
            F1 = FD_expanded(e.ravel(), x, 1 / kT, coeffs = c)
            def fd(E):
                return 1 / (1 + np.exp(E / kT))
            F2 = fd(e.ravel())
            maxdiff = np.abs(F1 - F2).max()
            if np.abs(F1 - F2).max()<1e-10:
                print("Fermi expansion very good")
                print("maxdiff: " + str(maxdiff))
            elif np.abs(F1 - F2).max()<1e-8:
                print("Fermi expansion good")
                print("maxdiff: " + str(maxdiff))
            elif np.abs(F1 - F2).max()<1e-5:
                print("Fermi expansion mediocre")
                print("maxdiff: " + str(maxdiff))
            elif np.abs(F1 - F2).max()>=1e-5:
                print("Fermi expansion bad")
                print("maxdiff: " + str(maxdiff))
            
    def modify_occupation(self, N_F=None,     eigtol=None, 
                                mu_i = None,  kT_i=None,
                                newlead=None, scale_gamma=None):
        """
        
        If None is given, the default value of the tool is used
        These can be seen using taking modify_occupations --help
        Parameters
        ----------
        N_F : TYPE, optional
            Number of poles in the Fermi function. .
        eigtol : TYPE, optional
             eigenvalues with absval smaller than this is neglected.
        mu_i : list / 1d array, len(mu_i) = #elecs
            DESCRIPTION. 
        kT_i : list / 1d array, len(mu_i) = #elecs
            DESCRIPTION. 
        newlead : a script contaning functions for adding another lead
            DESCRIPTION. 
        scale_gamma : list of scalars, that will scale the given lead with the factor.
            DESCRIPTION. 
        """
        this_frame = inspect.currentframe()
        arg_values = inspect.getargvalues(this_frame)
        kwargs = {}
        kwargs["file"]    = self.input.name+'_SRC'
        kwargs["outfile"] = self.input.name
        for k in arg_values.args:
            if k=='self':
                continue
            kwargs[k] = arg_values.locals[k]
        self.run_cmd_standard("modify_occupations", " > modocc.out", **kwargs)
    def make_ts_contour(self, E1   = None, N_C= None,  N_F = None, 
                               fact = None, kT = None,  name= None,
                               pp_path = None):
        """
        
        Runs the make_ts_contour tool
        Parameters
        ----------
        E1 : TYPE, optional
            lower point of contour
        N_C : TYPE, optional
            number of points for the wide arch 
        N_F : TYPE, optional
            number of points for the finer part close to z=0 (?? ).
        fact : TYPE, optional
             fact * kT  is the cutoff for the fermi part 
        kT : TYPE, optional
            thermal energy
        name : TYPE, optional
            output contour name
        pp_path : TYPE, optional
            Transiesta is called through a SiP class, it needs a .psf/.psml
            file directory. 
        -------
        """
        this_frame = inspect.currentframe()
        arg_values = inspect.getargvalues(this_frame)
        kwargs = {}
        for k in arg_values.args:
            if k=='self':
                continue
            kwargs[k] = arg_values.locals[k]
        self.run_cmd_standard("make_ts_contour", " > tscont.out", **kwargs)
    def run_scf(self, Contour = None,  kT     = None, kweights = None, 
                      drho_tol= None,  history= None, weight   = None, 
                      DM_start_file=None, nprocs= None, backend = None, 
                      DM_out_file  =None, UfUd  = None, real_line_integral=None, 
                      real_line_min=None, real_line_max=None, real_line_N=None,
                      real_line_exact_fermi=None, Nonequilibrium = None, 
                      DM_randomness  = None, random_on_diag_only = None, 
                      memory_conserve= None, fact_kT= None, tolerance  = None,
                      quadvec_workers= None, Bias   = None, save_last_H= None, 
                      write_dm_every=None, write_progress=None, adaptive_mixer = False):
        this_frame = inspect.currentframe()
        arg_values = inspect.getargvalues(this_frame)
        kwargs = {}
        kwargs["file"]=self.input.name
        for k in arg_values.args:
            if k=='self':
                continue
            kwargs[k] = arg_values.locals[k]
        print('Running SCF')
        exc = " ".join(self.prepend_env_vars + ["SCF"])
        self.run_cmd_standard(exc," > scf.out", **kwargs)
    def run_psinought(self,maxiter=None, checkderivative=None,dl=None,
                           steptol=None, add_random=None, random_weight=None,
                           L_info =None, Axb_solver=None, start_psi = None,
                           memory_save = None, Xpp_optim = None,  printfile=None,
                           extreme_memory_save = None, Woodbury_inv =None,
                           prec_to_disk=None, outer_einsum=None,
                           use_preconditioner = None ):
        """
        Please note the argument "Woodbury_inv" is a slightly different name
        compared to the actual psinought tool. This is because one cannot write the 
        argument with the dash in the arg name.... Woodbury_inv is changed
        to Woodbury-inv during the passing of arguments in the actual terminal call. 
        """
        this_frame = inspect.currentframe()
        arg_values = inspect.getargvalues(this_frame)
        kwargs = {}
        kwargs["file"]=self.input.name
        for k in arg_values.args:
            if k=='self':
                continue
            if k=="Woodbury_inv":
                k2 ="Woodbury-inv"
                kwargs[k2] = arg_values.locals[k]
            else:
                kwargs[k] = arg_values.locals[k]
        exc = " ".join(self.prepend_env_vars + ["psinought"])
        print('Running psinought')
        self.run_cmd_standard(exc," > psi0.out", **kwargs)
    def run_cmd_standard(self, CMD, out, **kwargs):
        cmd = CMD + " Dir=$PWD "
        for kw in kwargs.keys():
            val = kwargs[kw]
            if val is not None:
                v    = fmt_str_cmd(val)
                cmd += " "+kw+"="+v
        self.into_wd()
        self.systemcall(cmd + out)
        self.textlog += ['Ran '+cmd + "\n"]
        self.out_wd()
    def run_zand(self, mpi="mpirun "):
        self.textlog += ['Executing zand....\n']
        exc = " ".join(self.prepend_env_vars + [mpi, "zand"])
        print('Running zand')
        self.run_cmd_standard(exc, " > zand.out",)
    def run_nozand(self, mpi="mpirun "):
        self.textlog += ['Executing nozand....\n']
        exc = " ".join(self.prepend_env_vars + [mpi, "nozand"])
        print('Running nozand')
        self.run_cmd_standard(mpi+"nozand ", " > nozand.out")
    def write_log(self,ftxt):
        with open(ftxt, "w") as f:
            for l in self.textlog:
                f.write(l)
    def write_rawlog(self):
        ftxt = self.txtlogfile
        with open(ftxt, "w") as f:
            for l in self._rawlog:
                f.write(l)
    def read_current(self):
        return J([self.working_dir+"/"+self.input.name+'_save'])
    def read_dm(self):
        return DM([self.working_dir+"/"+self.input.name+'_save'])
    def pickle(self, filename):
        """Saves controller to file"""
        import pickle as pkl
        f = open(filename +'.control', 'wb')
        pkl.dump(self, f)
        f.close()
    def hook_linearize(self):
        print('Running hook linearization')
        if self.scf_status == False:
            print("There is no postive message from the SCF tool that the DM has been converged. \
                  Are you sure you are linearizing around equilibrium?")
        self.into_wd()
        scr = self.hook.scriptname.replace(".py","")
        if self.hook.scheme == 'lin_odm':
            fnc = 'linearize_odm'
        elif self.hook.scheme == 'lin_mul':
            fnc = 'linearize_mulliken'
        elif self.hook.scheme == 'lin_nodm':
            fnc = 'linearize_nodm'
        elif self.hook.scheme == 'full':
            print('Dont use hook_linearize when the hook scheme is \"full\"!')
            fnc = "empty"
        cmd = "python -c \"from "+scr+" import "+fnc+" as F; F() \" > hook_linearize.out"
        self.systemcall(cmd)
        self.out_wd()
    def archive_calculation(self, arc_name, keep_psi_omg_in_arc = False, clean_original = True,):
        self.into_wd()
        archive_calculation(self.input.name+"_save", arc_name, 
                            keep_psi_omg_in_arc = keep_psi_omg_in_arc,
                            clean_original = clean_original)
        self.out_wd()
class transiesta_hook:
    def __init__(self, indev, scheme, nsc=(1,1,1)):
        if type(indev) is str:
            dev = load(open(indev, 'rb'))
        else:
            dev = indev
        self.fermi_level = dev.read_fermi_level_from_out()
        self.devg  = sisl.get_sile(dev.dir+'/siesta.XV').read_geometry()
        self.devg.set_nsc(nsc)
        dev2 = dev.clone(dev.dir+'_dm2h')
        self.devg.write(dev2.dir+'/geom.xyz')
        self.dev = dev2
        self.orig_dev = dev
        self.set_onlyhsetup()
        self.scheme    = scheme
        self.use_full  = True
        self.scriptname = None
        assert scheme in ['lin_odm', 'lin_nodm', 'lin_mul','full'], "The given scheme is not implemented."
    def set_onlyhsetup(self):
        self.dev.write_more_fdf(["HSetupOnly true \n", 
                                 "SaveHS true\n",
                                 "User.Basis.NetCDF true\n",
                                 "DirectPhi true\n",
                                 ], 
                                name="DEFAULT")
    def write_hook(self,ArrayDir, wdir, bdir,  suffix ='', 
                   dq=0.005, Rmax = 12.5, use_orig_S=False, siesta_mpi = ""):
        scriptname = 'transiesta'
        if len(suffix)>0:
            scriptname += '_'+suffix
        scriptname+= '.py'
        if self.dev.solution_method=='transiesta':
            dmtype='TSDE'
        else:
            dmtype='DM'
        gnsc = list(self.devg.nsc)
        gnsc = [int(v) for v in gnsc]
        self.scriptname = scriptname
        code = (f"""\
import sisl, os
import numpy as np
from tqdm import tqdm
init_dm_from = \"{self.orig_dev.dir+"/"+self.orig_dev.sl}\"; sgs = sisl.get_sile
DevDir = \"{self.dev.dir}\"; Devsl = \"{self.dev.sl}\"
dmfile = DevDir+"/"+Devsl+'.'+"{dmtype}"
wdir, bdir = \"{wdir}\", \"{bdir}\"
gdev = sgs(bdir+'/'+DevDir+'/geom.xyz').read_geometry()
gdev.set_nsc({gnsc}); E_F = {self.fermi_level}
gorb = sisl.get_sile(bdir+'/'+DevDir+"/RUN.fdf").read_geometry()
Rorb = gorb.xyz[gorb.o2a(np.arange(gorb.no))]
piv  = np.load(\"{ArrayDir}\"+\"/Arrays/pivot.npy\")
_L   = np.load(\"{ArrayDir}\"+\"/Arrays/S^(-0.5).npy\")
_L   = np.linalg.inv(_L); S = _L @ _L
L    = np.load(\"{ArrayDir}\"+\"/Arrays/S^(-0.5).npy\")
iL   = np.linalg.inv(L)
# Standard functions for various transformations \n# between orthogonal and nonorthogonal basis
def sigO2NO(DMlike): return L  @ DMlike @ L
def sigNO2O(DMlike): return iL @ DMlike @ iL
def HamO2NO(Hlike):  return iL @ Hlike  @ iL
def HamNO2O(Hlike):  return L  @ Hlike  @ L
nlead= int(np.load(\"{ArrayDir}\"+\"/Arrays/num_leads.npy\"))
for ilead in range(nlead):
    if {use_orig_S}:
        S += np.load(\"{ArrayDir}\"+\"/Arrays/Sig1_NO_\"+str(ilead)+\".npy\")
Rorb = Rorb[piv].copy()
Dij  = np.linalg.norm(Rorb[:,None,:] - Rorb[None,:,:],axis=2)
I, J = [], []
for i in piv:
    for j in piv: I+=[i]; J+=[j]
ts,I,J= False, np.array(I), np.array(J)
dmf = sgs(bdir+'/'+init_dm_from+'.'+"{dmtype}").read_density_matrix()
if "{dmtype}"=="TSDE":
    edmf = sgs(bdir+'/'+init_dm_from+'.'+"{dmtype}").read_energy_density_matrix()
    ts   = True
W = sisl.io.siesta.tsdeSileSiesta(bdir+"/"+dmfile)
def DFT():
    os.chdir(bdir+'/'+DevDir)
    W.write_density_matrices(dmf, edmf, E_F)
    os.system("{siesta_mpi} siesta RUN.fdf > RUN.out")
    if os.path.isfile("siesta.HSX"):
        Hload = sgs('siesta.HSX').read_hamiltonian(geometry = gdev)
        Hload.set_nsc((1,1,1))
        os.system("rm siesta.HSX")
    elif os.path.isfile("siesta.0.HSX"):
        Hload = sgs('siesta.0.HSX').read_hamiltonian(geometry = gdev)
        Hload.set_nsc((1,1,1))
        os.system("rm siesta.0.HSX")
    else:
        assert 1 == 0, "Couldnt find any Hamiltonian. Check the siesta directory??"
    os.chdir(bdir+'/'+wdir)
    Hload.shift(-E_F)
    return Hload
def dm2DM(nosig):
    ravelled = nosig[0].ravel()
    for i in range(len(I)):
        dmf[I[i],J[i],0] = 2*ravelled[i].real
        if ts: edmf[I[i],J[i],0] = 0.0
def H_from_DFT(nosig):
    dm2DM(nosig)
    return DFT().Hk()[:,piv][piv,:].toarray()
from scipy.linalg import solve_sylvester, fractional_matrix_power
S_S = solve_sylvester; FMP = fractional_matrix_power
dq  = {dq}
Rmax = {Rmax}

def linearize_mulliken():
    dm0  = np.load(\"{ArrayDir}\"+\"/Arrays/DM_Ortho.npy\")
    dm0  = sigO2NO(dm0)
    try:
        other_dm = np.load('MullikenLin_NO.npz')["DM0"]
        other_dq = np.load('MullikenLin_NO.npz')["dq"]
        if np.allclose(dm0, other_dm) and np.allclose(dq, other_dq):
            print("Found MullikenLin_NO.npz with matching DM!")
            return 
        else:
            pass
    except:
        pass
    def sig2mul(dm): return S @ dm + dm @ S
    def mul2sig(Q):
        if len(Q.shape) == 2: out =  S_S(S,S, Q)
        else:                 out =  np.array([S_S(S[i],S[i],Q[i] ) for i in range(len(Q))])
        return out
    Q0, H0 = sig2mul(dm0), H_from_DFT(dm0)
    no   = dm0.shape[-1]
    dHdQ = np.zeros((no, no, no),dtype=np.complex64)
    for i in range(no):
        Qv = Q0.copy()
        if Qv[0,i,i]+dq > 1.0:
            sgn=-1.
        else:
            sgn= 1.
        Qv[0,i,i] += dq * sgn
        res = (H_from_DFT(mul2sig(Qv)) - H0)/(dq * sgn)
        idx = np.where(Dij[i]<Rmax)[0]
        for I in idx:
            for J in idx:
                dHdQ[i,I,J] = res[I,J]
    np.savez_compressed("MullikenLin_NO.npz", 
                        dHdQ=dHdQ, DM0=dm0, 
                        H0=H0,     Q0=Q0, 
                        dq=dq, S=S, Rmax = Rmax, dm_in_ortho_basis=False)
def linearize_odm():
    dm0  = np.load(\"{ArrayDir}\"+\"/Arrays/DM_Ortho.npy\")
    try:
        other_dm = np.load('DM_Lin_O.npz')["DM0"]
        other_dq = np.load('DM_Lin_O.npz')["dq"]
        if np.abs(dm0 - other_dm).max()<(dq/10) and np.allclose(dq, other_dq):
            print("Found DM_Lin_O.npz with matching DM!")
            return 
        else:
            pass
    except:
        pass
    H0   = H_from_DFT(sigO2NO(dm0))
    no   = dm0.shape[-1]
    dHdQ = np.zeros((no, no, no),dtype=np.complex64)
    for i in tqdm(range(no)):
        dm = dm0.copy()
        if dm[0,i,i] + dq > 0.5:
            sgn = -1.0
        else:
            sgn =  1.0
        dm[0,i,i] += dq * sgn
        res = (H_from_DFT(sigO2NO(dm)) - H0)/(dq * sgn)
        idx = np.where(Dij[i]<Rmax)[0]
        for I in idx:
            for J in idx:
                dHdQ[i,I,J] = res[I,J]
    np.savez_compressed("DM_Lin_O.npz", 
                        dHdQ=dHdQ, DM0=dm0, 
                        H0=H0, Rmax=Rmax, 
                        dq=dq, S=S, dm_in_ortho_basis=True)

def linearize_nodm():
    dm0  = np.load(\"{ArrayDir}\"+\"/Arrays/DM_Ortho.npy\")
    dm0  = sigO2NO(dm0)
    try:
        other_dm = np.load('DM_Lin_NO.npz')["DM0"]
        other_dq = np.load('DM_Lin_NO.npz')["dq"]
        if np.abs(dm0 - other_dm).max()<(dq/10) and np.allclose(dq, other_dq):
            print("Found DM_Lin_NO.npz with matching DM!")
            return 
        else:
            pass
    except:
        pass
    H0   = H_from_DFT(dm0)
    no   = dm0.shape[-1]
    dHdQ = np.zeros((no, no, no),dtype=np.complex64)
    for i in tqdm(range(no)):
        dm = dm0.copy()
        if dm[0,i,i] + dq > 0.5:
            sgn = -1.0
        else:
            sgn =  1.0
        dm[0,i,i] += dq * sgn
        res = (H_from_DFT(dm) - H0)/(dq * sgn)
        idx = np.where(Dij[i]<Rmax)[0]
        for I in idx:
            for J in idx:
                dHdQ[i,I,J] = res[I,J]
    np.savez_compressed("DM_Lin_NO.npz", 
                        dHdQ=dHdQ, DM0=dm0, 
                        H0=H0, Rmax=Rmax, 
                        dq=dq, S=S, dm_in_ortho_basis=False)

    
    
def empty():
    return None
""")
        with open(wdir+'/'+scriptname, "w") as f:
            f.write(code)

class Mull_Lin_NO:
    def __init__(self, file, rank):
        if rank != 0:
            return
        try:
            f = np.load(file)
            self._f = f
            self.dHdQ = f["dHdQ"].transpose(1,2,0).copy()
            self.dm0  = f["DM0"]
            self.H0   = f["H0"]
            self.Q0   = f["Q0"]
            self.q0   = np.diag(self.Q0[0])
            self.dq   = f["dq"]
            self.S    = f["S"]
            self.FileNotFound = False
            assert f["dm_in_ortho_basis"] == False
        except:
            self.FileNotFound = True
        
    def linearized_H(self, sigNO):
        Q = self.S @ sigNO + sigNO @ self.S
        q = np.diag(Q[0])
        dq = q - self.q0
        dH = self.dHdQ @ dq
        return dH

class DM_Lin_O:
    def __init__(self, file, rank):
        if rank != 0:
            return
        try:
            f = np.load(file)
            self._f = f
            self.dHdQ = f["dHdQ"].transpose(1,2,0).copy()
            self.dm0  = f["DM0"]
            self.H0   = f["H0"]
            self.dq   = f["dq"]
            self.S    = f["S"]
            self.q    = np.diag(self.dm0[0])
            self.FileNotFound = False
            assert f["dm_in_ortho_basis"] == True
        except:
            self.FileNotFound = True
    def linearized_H(self, sigO):
        dq = np.diag(sigO[0]) - self.q
        dH = self.dHdQ @ dq
        return dH
class DM_Lin_NO:
    def __init__(self, file, rank):
        if rank != 0:
            return
        try:
            f = np.load(file)
            self._f = f
            self.dHdQ = f["dHdQ"].transpose(1,2,0).copy()
            self.dm0  = f["DM0"]
            self.H0   = f["H0"]
            self.dq   = f["dq"]
            self.S    = f["S"]
            self.q    = np.diag(self.dm0[0])
            self.FileNotFound = False
            assert f["dm_in_ortho_basis"] == False
        except:
            self.FileNotFound = True
    def linearized_H(self, sigNO):
        dq = np.diag(sigNO[0]) - self.q
        dH = self.dHdQ @ dq
        return dH

def archive_calculation(name, arc_name, keep_psi_omg_in_arc = False, clean_original = True,):
    os.system("cp -R "+name + " " + arc_name)
    if keep_psi_omg_in_arc == False:
        os.system("rm "+arc_name + "/last_psi.npy")
        os.system("rm "+arc_name + "/last_omg.npy")
    if clean_original:
        os.system("rm "+name+"/DM*.npy")
        os.system("rm "+name+"/current*.npy")
        os.system("rm "+name+"/times*.npy")
    

        
        



