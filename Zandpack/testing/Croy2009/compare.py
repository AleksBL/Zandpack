import numpy as np
from Zandpack.plot import plt, J, DM

R1 = np.load("Croy2009_Results.npz")

JL = R1["Jl"]
JR = R1["Jr"]
T  = R1["t"]

t2,j2 = J(["TDT_Croy2009_save_nozand"])
plt.plot(T, JL)
plt.plot(T, JR)

plt.plot(t2, j2[0], linestyle="dashed")
plt.plot(t2, j2[1], linestyle="dotted")

plt.show()

t2,j2 = J(["TDT_Croy2009_save_zand"])
plt.plot(T, JL)
plt.plot(T, JR)

plt.plot(t2, j2[0], linestyle="dashed")
plt.plot(t2, j2[1], linestyle="dotted")
