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


plt.scatter(x, y, color='blue', label='Random Points')
plt.legend()
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Points")

plt.show()


#if points(x=0):
 #   return points(x=0)
#if points(x=1):
#    return points(x=0)
#return plt.plot(x,y)
    


