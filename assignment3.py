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

class MonteCarlo:
    def __init__(self, num_samples=1000000, dimensions=6, seed=12345):
        """
        To initialize the Monte Carlo class.
        """
        self.num_samples = num_samples
        self.dimensions = dimensions
        self.seed = seed
        self.rng = default_rng(SeedSequence(seed))

    def sample_points(self):
        """
        To generate random points.
        """
        return self.rng.uniform(-1, 1, size=(self.dimensions, self.num_samples))

    def in_hypersphere(self, points):
        """
        Check if points are inside the unit hypersphere.
        """
        return np.sum(points**2, axis=0) < 1

    def estimate_volume(self):
        """
        Estimate the volume of a hypersphere using Monte Carlo.
        """
        points = self.sample_points()
        inside_count = np.sum(self.in_hypersphere(points))
        volume_fraction = inside_count / self.num_samples
        cube_volume = 2**self.dimensions
        return volume_fraction * cube_volume

if __name__ == "__main__":
   # num_samples = 1000000
    #dimensions = [2, 3, 4, 5]
    mc_2d = MonteCarlo(num_samples=1000000, dimensions=2)
    print(f"Estimated volume for 2D (circle): {mc_2d.estimate_volume()}")

    mc_3d = MonteCarlo(num_samples=1000000, dimensions=3)
    print(f"Estimated volume for 3D (sphere): {mc_3d.estimate_volume()}")
    
    mc_4d = MonteCarlo(num_samples=1000000, dimensions=4)
    print(f"Estimated volume for 4D (sphere): {mc_3d.estimate_volume()}")
    
    mc_5d = MonteCarlo(num_samples=1000000, dimensions=5)
    print(f"Estimated volume for 5D (sphere): {mc_3d.estimate_volume()}")
    
    mc_6d = MonteCarlo(num_samples=1000000, dimensions=6)
    print(f"Estimated volume for 6D (sphere): {mc_3d.estimate_volume()}")
    

print(__name__)

#if points(x=0):
 #   return points(x=0)
#if points(x=1):
#    return points(x=0)
#return plt.plot(x,y)
    


