import numpy as np
from Bias import dH
from Initial import  steps_for_bondcurrent
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]



name = 'TDT_save'
H0 = np.load('TDT/Arrays/H_Ortho.npy')

files = os.listdir(name)
f_times = [f for f in files if 'times' in f]
f_dm    = [f for f in files if 'DM'    in f]

f_times.sort(key=natural_keys)
f_dm   .sort(key=natural_keys)

print(f_times)
print(f_dm)

BC = []
T  = []
DM = []

for i,f in enumerate(f_dm):
    print(i)
    dms = np.load(name+'/'+f)[::steps_for_bondcurrent]
    ts  = np.load(name+'/'+f_times[i])[::steps_for_bondcurrent]
    
    bc = np.stack([2*(H0 + dH(ts[k], dms[0]))*np.imag(dms[k]).transpose(0,2,1) 
                      for k in range(len(dms))],axis=0)
    
    BC += [bc]
    T  += [ts]
    DM += [dms]
   
BC = np.vstack(BC)
DM = np.vstack(DM)
T  = np.hstack(T)


np.save(name+'_bondcurrents_over_time', BC)
np.save(name+'times_for_bc',T)
np.save(name+'_dm_over_time', DM)

