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
    num_samples = 1000000
    dimensions = [2, 3, 4, 5, 6]
    for d in dimensions:
        mc_simulator = MonteCarlo(num_samples, d)
        volume_estimate = mc_simulator.estimate_volume()
        print(f"Estimated volume for {d}D hypersphere: {volume_estimate:.6f}")