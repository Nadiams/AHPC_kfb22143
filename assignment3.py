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
#from mpi4py import MPI

N = 1000

x = np.random.uniform(-1, 1.5, size=N)
y = np.random.uniform(-1, 1.5, size=N)

from numpy.random import SeedSequence, default_rng

ss = SeedSequence(12345)

# Spawn off 10 child SeedSequences to pass to child processes.
child_seeds = ss.spawn(10)
streams = [default_rng(s) for s in child_seeds]
print(type(streams))
print(streams[2].random())
print(streams[3].random())

def twodregion(x,y):
    r2 = x**2 + y**2
    return 1 if r2<1 else 0

print(twodregion(0,1))


plt.scatter(x, y, color='blue', label='Random Points')
plt.legend()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Points")
plt.grid()


#if points(x=0):
 #   return points(x=0)
#if points(x=1):
#    return points(x=0)
#return plt.plot(x,y)
    


