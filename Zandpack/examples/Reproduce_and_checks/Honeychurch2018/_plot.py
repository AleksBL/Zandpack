import matplotlib.pyplot as plt
from Zandpack.mpi_tools import combine_currents as CC
t, J = CC(['TDT_Honeychurch2018_save'])
plt.plot(t[:-1],J[0])
plt.show()

