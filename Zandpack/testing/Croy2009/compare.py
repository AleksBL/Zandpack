import numpy as np
from Zandpack.plot import plt, J, DM

R1 = np.load("Croy2009_Results.npz")

JL = R1["rk4_jl"]
JR = R1["rk4_jr"]
T  = R1["rk4_t"]

sp_jl = R1["scipy_jl"]
sp_t = R1["scipy_t"]


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

## Some sort of small scatterplot of scipy points
plt.scatter(sp_t, sp_jl)
plt.savefig("Croy2009_compare.png",dpi=300)
