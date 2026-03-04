from pickle import load
import lzma
import numpy as np
from Zandpack.FittingTools import find_correction
R = load(lzma.open('ZGNR.xz','rb'))
Eg = np.linspace(-6.5,6.5,750)
NL = 81
opts= {'height':1.0, 'distance':5}
Emin, Emax = -8.0, 8.
pdist, fact, pfact, cf = 0.1,0.9,0.8,0.1
fm      = 'linear'
E1,G1=R.PoleGuess(0,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=1.0,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)
E2,G2=R.PoleGuess(1,NL,Emin,Emax,fact=fact,cutoff=cf,
                  tol=1.0,decimals=3,pole_dist=pdist,
                  pole_fact=pfact,opts=opts)
init_E = [E1[None,:], E2[None,:]]
init_G = [G1[None,:], G2[None,:]]
alpha_PO = 0.0
min_tol  = -100000*np.ones(NL)[None,:]
mt1, mt2 = min_tol.copy(),min_tol.copy()
tol_ele   = 1e-3
def run_mini(its,method, which_e = None):
    R.Fit(fact = 1.0, NumL = NL,
          force_PSD_tol = [mt1,mt2],
          min_method = method,
          ebounds    = (-15.,15.),
          wbounds    = (0.001, 5),
          tol     = tol_ele,                 
          options={'maxiter':its,'gtol':1e-5},
          fit_real_part = False,
          alpha_PO=alpha_PO,init_E=init_E,
          init_G  =init_G, which_e =which_e,
          )
run_mini(0,'COBYLA')    # First refinement
R.curvefit_all(0.0001)  # Second refinement
R.NO_fitted_lorentzians[0].iterative_PSD(
    maxit=100, n=40,nW=15,
    lbtol = -0.001,fact=.5,add_last = False)
R.NO_fitted_lorentzians[1].iterative_PSD(
    maxit=100, n=40,nW=15,
    lbtol = -0.001,fact=.5,
    add_last = False)
R.FitNO2O()             # Third refinement
C = find_correction(R, Emin = -1, Emax = 1)
R.Renormalise_H(C)      # Fourth refinement
# Terms in eq. %\textcolor{green}{\eqref{eq:SE_EigDecomp}}% in which %\textcolor{green}{$|\Delta_{\alpha x c}^{</>}|<$}%tol
# are excluded in the timepropagation. 
R.tofile('TDZGNR_src', tol=1e-4)
