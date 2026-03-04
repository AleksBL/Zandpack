from pickle import load
import lzma; import numpy as np
from Zandpack.plot import plt
from Block_matrices.Croy import gamma_from_centres_matrix as LHS
from Block_matrices.Croy import gamma_from_centers_RHS    as RHS
from Block_matrices.Croy import L_sum
Nl1, Nl2,i,j = 20,80,0,1
# Read in the saved file
R  = load(lzma.open('ZGNR.xz','rb'))
idx=R.sampling_idx[0]
# Define poles
ei1=np.linspace(-5,5,Nl1);gi1=np.ones(Nl1)*0.35
ei2=np.linspace(-5,5,Nl2);gi2=np.ones(Nl2)*0.12
v=R.Nonortho_Gammas[0].Block(0,0)# Pick a block
#Plotted in Fig. %\textcolor{green}{\ref{fig:Fit1}}%
x,y= R.Contour[idx].real, v[0,idx,i,j].real
x2 = np.linspace(-7,7,10000)
# Set up and solve eq. %\textcolor{green}{\eqref{eq:GammaEquation}}%
M1 = LHS(ei1,gi1); v1  = RHS(ei1,gi1,y,x)
M2 = LHS(ei2,gi2); v2  = RHS(ei2,gi2,y,x)
Gam1 = np.linalg.inv(M1).dot(v1)
Gam2 = np.linalg.inv(M2).dot(v2)
C1  = np.vstack([gi1,ei1,Gam1])
C2  = np.vstack([gi2,ei2,Gam2])
Fit1= L_sum(x2, C1)#Plotted in Fig. %\textcolor{green}{\ref{fig:Fit1}}%
Fit2= L_sum(x2, C2)#Plotted in Fig. %\textcolor{green}{\ref{fig:Fit1}}%
