import matplotlib.pyplot as plt
from TimedependentTransport.mpi_tools import combine_currents
import sys
name = sys.argv[1]

t,J = combine_currents(['TDT_save'])
plt.plot(t, J[0])
plt.savefig(name+'_Jl.svg')
plt.close()
plt.plot(t, J[1])
plt.savefig(name+'_Jr.svg')

