import numpy as np
from Zandpack.plot import J

t1,j1 = J(['calc_0/TDT2T_G0_save'])
t2,j2 = J(['calc_1/TDT2T_G1_save'])
t3,j3 = J(['calc_2/TDT2T_G2_save'])
np.savez_compressed('data_calc012.npz', t1=t1,t2=t2,t3=t3,
                                        j1=j1,j2=j2,j3=j3)


