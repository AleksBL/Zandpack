#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 12:34:01 2022

@author: aleksander
"""

from time import time, sleep
import numpy as np

def pad_string(string, Len):
    n = len(string)
    return string + (" " * (Len - n))

class TimeBot:
    def __init__(self, name, print_timings, name_str_len = 60, num_space = 13):
        self.dict = {}
        self.name = name
        self.skip = {}
        self.print_timings = print_timings
        self.name_str_len = name_str_len
        self.num_space    = num_space
    
    def names(self):
        return self.dict.keys()
    
    def _time(self, name, t, skip_next = False):
        
        if name not in self.skip.keys():
            skip_is_false = True
        elif self.skip[name] == False:
            skip_is_false = True
        else:
            skip_is_false = False
        
        if skip_is_false:
            if name in self.names():
                self.dict[name]+=[t]
            else:
                self.dict.update({name:[t]})
                self.skip.update({name:False})
        else:
            self.skip[name] = False
        
        if skip_next:
            self.skip[name] = True
    
    def time(self,name, skip_next = False):
        if self.print_timings == False:
            return
        
        t = time()
        if isinstance(skip_next, bool):
            for n in name:
                self._time(n, t, skip_next = skip_next)
        elif isinstance(skip_next, list):
            for i,n in enumerate(name):
                self._time(n, t, skip_next = skip_next[i])
    def reset(self):
        self.dict = {}
        self.skip = {}
    
    def generate_timings(self):
        mystring =""+self.name + "\n"
        if self.print_timings == False:
            return mystring +'Not Timed\n'
        for n in self.names():
            v         = np.array(self.dict[n])
            timings   = v[1:][::2] - v[::2]#np.diff(v)
            arr_string= np.array2string(timings, precision = 4)[1:-1]
            numbers   = arr_string.split()
            arr_string= "".join([pad_string(v, self.num_space) for v in numbers])
            nt        = len(timings)
            initial_string = n + " timed " + str(nt) + " times: "
            initial_string = pad_string(initial_string,self.name_str_len)
            mystring += initial_string + arr_string +' sec\n'
        mystring+="\n"
        return mystring



def timethis():
    T = TimeBot('noname', True)
    T.time(['Outer'])
    for i in range(5):
        T.time(['O1'])
        sleep(0.5)
        T.time(['O1',
                "O2"])
        sleep(0.1)
        T.time(['O2',
                ])
    T.time(['Outer'])
    print(T.generate_timings())

        
    