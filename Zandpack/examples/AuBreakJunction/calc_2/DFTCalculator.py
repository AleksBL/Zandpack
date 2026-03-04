#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:49:26 2022

@author: aleksander
"""

import sisl
from time import sleep
import os
from DFTObj import Dev

sleep_time_1 = 1e-2
sleep_time_2 = 1e-2

siesta_command = 'mpirun -np 3 siesta RUN.fdf > RUN.out'

def check_for_new_dm():
    files = os.listdir()
    if 'NEWDM.txt' in files:
        sleep(sleep_time_1)
        os.remove('NEWDM.txt')
        return True
    return False

def signal_new_H():
    with open('NEWH.txt','w') as f:
        f.write('some text')
        f.close()
    
    print('New H!')

def check_for_stop():
    files = os.listdir()
    if 'STOP.txt' in files:
        return True
    return False
cond = True
print('\n')
print('----------------------')
print('DFT Calculator started')
print('----------------------')
print('\n')

f = open('siestascript.out','w')
f.write('Start\n')
f.close()

while cond:
    if check_for_new_dm():
        os.chdir(Dev.dir)
        os.system(siesta_command)
        os.chdir('..')
        signal_new_H()
        with open('siestascript.out', 'a') as f:
            f.write('Finished H Setup \n ')
    if check_for_stop():
        cond = False
    sleep(sleep_time_2)
with open('siestascript.out','a') as f:
    f.write('siesta calculator was killed')

