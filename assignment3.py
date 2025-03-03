#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP example provided by Dr. Benjamin Hourahine in
# PH510. Modifications made by kfb22143 - Licensed under the MIT License.
# See LICENSE file for details.
"""
Created on Mon Mar  3 15:07:49 2025

@author: nadia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import SeedSequence, default_rng
#from mpi4py import MPI

N = 1000
ss = SeedSequence(12345)
dimensions = 6

# Spawn off 10 child SeedSequences to pass to child processes.
child_seeds = ss.spawn(dimensions)
streams = [default_rng(s) for s in child_seeds]
print(type(streams))
print(streams[2].random())
print(streams[3].random())

x = streams[0].uniform(-1, 1, size=N)
y = streams[1].uniform(-1, 1, size=N)
z = streams[2].uniform(-1, 1, size=N)

def twodregion(x,y):
    r2 = x**2 + y**2
    return x**2 + y**2 < 1

def threedregion(x,y,z):
    r2 = x**2 + y**2 + z**2
    return x**2 + y**2 + z**2 < 1

def fourdregion(w,x,y,z):
    r2 = w**2 + x**2 + y**2 + z**2
    return w**2 + x**2 + y**2 + z**2 < 1

def fivedregion(v,w,x,y,z):
    r2 = v**2 + w**2 + x**2 + y**2 + z**2
    return v**2 + w**2 + x**2 + y**2 + z**2 < 1

def sixdregion(u,v,w,x,y,z):
    r2 = u**2 + v**2 + w**2 + x**2 + y**2 + z**2
    return u**2 + v**2 + w**2 + x**2 + y**2 + z**2 < 1

print(twodregion(0,1))
print(threedregion(0,1,1))
print(fourdregion(0,1,1,1))
print(fivedregion(0,1,1,1,1))
print(sixdregion(0,1,1,1,1,1))

inside = twodregion(x, y)
plt.scatter(x[inside], y[inside], color='blue', label='Inside Circle')
plt.scatter(x[~inside], y[~inside], color='red', label='Outside Circle')
plt.legend()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Random Points in 2D")
plt.grid()


#if points(x=0):
 #   return points(x=0)
#if points(x=1):
#    return points(x=0)
#return plt.plot(x,y)
    


