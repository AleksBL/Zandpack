#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:42:18 2023

@author: aleksander
"""
import os
import numpy as np
from time import time

def flexload(filepath, force_array = True,return_arc = False):
    # First, try the actual file name
    if return_arc and '.npz' in filepath:
        return np.load(filepath)
    
    try:
        f = np.load(filepath)
        if isinstance(f, np.ndarray):
            return f
        else:
            return f[f.files[0]]
    except:
        k = 1
    # Secondly, assume the file is actually .npz instead of .npy
    try:
        return np.load(filepath.replace('.npy','.npz'))['arr_0']
    except:
        k = 2
    print('flexload error code: ', k)
    print("input file path: " + filepath)
    assert 1 == 0

class load_dictionary:
    def __init__(self, dir):
        self.dir = dir
        self.timer = time
        self.times = [self.timer()]
    def __getitem__(self, x):
        self.times += [self.timer()]
        return flexload(self.dir+x+'.npy')
    def __str__(self):
        return "load_dictionary class instance, using "+str(self.dir)
