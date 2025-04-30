#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP, parallel variance and the Partial differential
#equations Handout 7 examples provided by Dr. Benjamin Hourahine in PH510.
#Modifications made by kfb22143 - Licensed under the MIT License.
#See LICENSE file for details.
"""
Created on Tue Mar 18 12:32:13 2025

@author: nadia
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng


def overrelaxation_method(self):
        """
            Method to solve Poissons equation.
            Implement a relaxation (or over-relaxation) method to solve Poisson’s equation for a
square N × N grid, with a grid spacing of h and specified charges at the grid sites (f ).
This will be used as an independent check for your Monte Carlo results.
        """

N = 4 # Sets the size of the grid.
phi = np.zeros([N, N]) # creates an array of zeros in a NxN (4x4) grid 
for i in range(0,N): # creates a grid of these zeros
    phi[0,i] = 1 # sets the first line, [0,i] all = 1 from which we can calculate numerical values.
    phi[N-1, i] = 1
    phi[i, 0] = 1
    phi[i, N-1] = 1
print('Initial phi with boundary conditions:')
print(phi)
for itters in range(1000): # Repeats the solver 1000 times.
    for i in range(1, N-1):
        for j in range(1, N-1): # enables the relaxer to navigate the grid and protects it from encountering neighbours outside the grid.
           # print(i,j)
            phi[i,j] = 1/4 * ( phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]) # Used phi[i,j] to specifically alter each part of the grid.
print('phi now=',phi)



            #phi[i,j] = omega * ( f[i,j] + 1/4* (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]) ) + (1 - omega) * initial_phi[i,j]  
        #num_samples = [self.rng]
    #    while omega += 1, 2:
        #for n in num_samples:
          #  if omega += 1, 2:
               # inner = phi[i+1:j] + phi[i-1:j] + phi[i:j+1] + phi[i:j-1]
               # bracket =  f[i:j] + 1/4 * inner
               # poisson = omega * bracket + (1 - omega) * initial_phi[i:j]       
        #return poisson
        # h is the grid spacing. Over-relaxation is faster but not necessary for the assignment.
#N = 4
#phi = np.array([N, N])
#for i in range(1, N-1):
    #for j in range(1, N-1):
       # print(i,j)