#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:16:48 2022

@author: aleksander
"""
import numpy as np

A = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
B = np.array([[np.nan   ,      np.nan,    np.nan ,    np.nan, np.nan],
              [1/4      ,      np.nan,    np.nan ,    np.nan, np.nan],
              [3/32     ,      9/32  ,    np.nan ,    np.nan, np.nan],
              [1932/2197, -7200/2197 , 7296/2197 ,    np.nan, np.nan],
              [439/216  , -8         , 3680/513  , -845/4104, np.nan],
              [-8/27    , 2          , -3544/2565, 1859/4104, -11/40]
              ]
              )

C  = np.array([25/216, 0, 1408 / 2565 , 2197 / 4104, -1/5, np.nan ])
CH = np.array([16/135, 0, 6656 /12825 , 28561/56430, -9/50 , 2/55 ])
CT = np.array([1/360 , 0, -128/4275   , -2197/75240,  1/50 , 2/55 ])

