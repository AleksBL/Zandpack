import numpy as np
D = 0.27

def pairing_field(ik, i,j,s1,s2):
    if i==j and s1!=s2:
        return D
    else:
        return 0.0
        

