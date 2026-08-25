import numpy as np
from Zandpack.plot import plt, J, DM

R1 = np.load("croy2016_results.npz")

JL = R1["rk4_jl"]
JR = R1["rk4_jr"]
T  = R1["rk4_t"]
# JL2 = R1["scipy_jl"]
# T2  = R1["scipy_t"]


t2,j2 = J(["TDT_Croy2016_save_nozand"])
plt.plot(T, JL, label="rk4")
plt.plot(T, JR, label="rk4")
#plt.plot(T2, JL2, label = 'scipy', linestyle="dashdot")
plt.plot(t2, j2[0], linestyle="dashed", label="nozand")
plt.plot(t2, j2[1], linestyle="dotted", label="nozand")
plt.legend()

plt.show()

t2,j2 = J(["TDT_Croy2016_save_zand"])
plt.plot(T, JL, label="RK4")
plt.plot(T, JR, label="RK4")
# plt.plot(T2, JL2, label = 'scipy', linestyle="dashdot")

plt.plot(t2, j2[0], linestyle="dashed", label="zand")
plt.plot(t2, j2[1], linestyle="dotted", label="zand")

plt.show()
