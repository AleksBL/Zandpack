#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 16:47:40 2026

@author: aleks
"""
from Zandpack.td_constants import hbar
import numpy as np
# Wrapper classes for more easily control a zandpack calculation
# directly from python.

class Initial:
    def __init__(self, 
                 name,
                 t0, 
                 t1, 
                 eps, 
                 usesave=True,
                 LoadFromFull=True,
                 checkpoints = None,
                 save_checkpoints=False,
                 print_timings=False,
                 stepsize=0.1,
                 n_dm_compress=10, 
                 save_PI = False,
                 verbose=True,
                 orthogonal=True,
                 ):
        self.name = name
        self.t0   = t0 
        self.t1   = t1
        self.eps  = eps
        self.usesave = usesave
        self.loadfromfull=LoadFromFull
        self.checkpoints = checkpoints
        if checkpoints is None:
            self.checkpoints = np.linspace(t0,t1, 10)
        self.save_checkpoints = save_checkpoints
        self.print_timings = print_timings
        self.stepsize=stepsize
        self.n_dm_compress=n_dm_compress
        self.save_PI = save_PI
        self.verbose = verbose
        self.orthogonal=True
    def write_initial(self, prefix):
        text = "from Zandpack.td_constants import hbar\n"
        text+= "from Zandpack.Loader import load_dictionary\n"
        text+= "import numpy as np\n"
        
        text+="name=\""+str(self.name)+"\"\n"
        text+="t0="+str(self.t0)+"\n"
        text+="t1="+str(self.t1)+"\n"
        text+="usesave="+str(self.usesave)+"\n"
        text+="checkpoints="+str([float(v) for v in self.checkpoints])+"\n"
        text+="save_checkpoints="+str(self.save_checkpoints)+"\n"
        text+="print_timings="+str(self.print_timings)+"\n"
        text+="stepsize="+str(self.stepsize)+"\n"
        text+="n_dm_compress="+str(self.n_dm_compress)+"\n"
        text+="save_PI="+str(self.save_PI)+"\n"
        text+="Adir = name + \'/Arrays/\'\n"
        text+="Arrs = load_dictionary(Adir)\n"
        with open(prefix + "/Initial.py", "w") as f:
            f.write(text)
        if self.verbose:
            print("Wrote Initial.py file")
    def write_bias(self, prefix, more_imports = None, mpi4py=False,os_envvar=[]):
        text = "import numpy as np\n"
        text+= "import sisl, os\n"
        text+= "from Zandpack.Help import TDHelper\n"
        if mpi4py:
            text += "from mpi4py import MPI; rank = MPI.COMM_WORLD.Get_rank(); size = rank = MPI.COMM_WORLD.Get_size()\n"
        if more_imports is not None:
            if isinstance(more_imports, str):
                text += more_imports
            if isinstance(more_imports, list):
                for l in more_imports:
                    text += l + "\n"
        for ev in os_envvar:
            text += "_"+ev+"=os.environ[\"" +ev+"\"]\n"
        text+= "name=\""+str(self.name)+"\"\n"
        text+= "Hlp = TDHelper(name); nlead=Hlp.num_leads\n"
        text+= "def sigO2NO(DMlike): return Hlp.Lowdin @ DMlike @ Hlp.Lowdin\n"
        text+= "def sigNO2O(DMlike): return Hlp.invLowdin @ DMlike @ Hlp.invLowdin\n"
        text+= "def HamO2NO(Hlike):  return Hlp.invLowdin @ Hlike @ Hlp.invLowdin\n"
        text+= "def HamNO2O(Hlike):  return Hlp.Lowdin@Hlike @ Hlp.Lowdin\n"
        text+= "def LeadDevOrthCorr_O(t):\n"
        text+= "    return  Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in range(nlead) ]) \n" 
        text+= "def LeadDevOrthCorr_NO(t):\n"
        text+= "    return  Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in range(nlead) ], orthogonal=False) \n"
        text+= "def LeadDevOrthCorr_O(t):\n"
        text+= "    return  Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in range(nlead) ]) \n" 
        text+= "def LeadDevOrthCorr_NO(t):\n"
        text+= "    return  Hlp.lead_dev_dyncorr(DeltaList = [bias(t, a) for a in range(nlead) ], orthogonal=False)\n"
        text+= "def sig2mul_O(dm):\n"
        text+= "    nodm = sigO2NO(dm)\n"
        text+= "    return Hlp.S @ nodm + nodm @ Hlp.S\n"
        text+= "def sig2mul_NO(dm):\n"
        text+= "    return Hlp.S @ dm + dm @ Hlp.S\n"
        text+= "def dissipator(t,sig): return 0.0\n"
        with open(prefix + "/Bias.py", "w") as f:
            f.write(text)
        if self.verbose:
            print("Wrote Bias.py file")
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        