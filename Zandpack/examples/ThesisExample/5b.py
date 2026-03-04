from pickle import load
from Zandpack.TimedependentTransport import TD_Transport as TDT
import numpy as np
# Load File
Dev = load(open('Dev.SiP','rb'))
gd   = Dev.to_sisl()
ge1,ge2  = Dev.elecs[0].to_sisl(),Dev.elecs[1].to_sisl()
# Initialize TD_Transport object
R    = TDT([ge1,ge2], gd,kT_i=[.025, .025], mu_i = [.0, .0])
# Define energy sampling grid with broadening
line = np.linspace(-8,10,201)+1e-2j+1e-3
line = np.vstack((line,line))
#Choose initial pole exp., modifiable later
#Default is to use the one from ref. %\textcolor{green}{\cite{hu2011pade}}%
R.Make_Contour(line,5)#5 terms in eq. %\textcolor{green}{\eqref{eq:FermiExp}}%
Dev.custom_tbtrans_contour = R.Contour
Dev.fdf(eta = 1.0e-2)
Dev.run_tbtrans_in_dir()
# Set Device parameter in TD_Transport class
R.Device = Dev
cC,cH,it,sub_idx = 4,1,0,[]
# Extract %\textcolor{green}{$p_z$}% part
for i in range(len(Dev.s)):
    if Dev.s[i]==6:
        sub_idx += [it+2]
        it += cC
    elif Dev.s[i]==1:
        it += cH
# Read data from transport calculation
R.read_data(sub_orbital=sub_idx)
R.pickle('ZGNR')
