import numpy as np
import matplotlib.pyplot as plt
from TimedependentTransport.mpi_tools import combine_currents as CC

t,J = CC(['propfile_save'], n=3)
t = t[:-1]
#idx = t>11.5
#t   = t[idx]
D1  = np.load('RK4.npz')
D2  = np.load('ScipyODE.npz')

plt.plot(D1['t'], D1['Jl'])
plt.plot(t[:-10], J[0]#[idx]
[:-10])

plt.scatter(D2['t'], D2['jl'][:,0],s = 5, marker = 'x', color = 'k')
plt.scatter(D2['t'], D2['jl'][:,1],s = 5, marker = 'x', color = 'k')
plt.scatter(D2['t'], D2['jl'][:,2],s = 5, marker = 'x', color = 'k')


plt.show()

