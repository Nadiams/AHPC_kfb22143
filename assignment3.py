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
    def __init__(self, num_samples=10000, dimensions=5, seed=12345):
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

    def in_hyperspace(self, points):
        """
        Check if points are inside the unit hyperspace.
        """
        return np.sum(points**2, axis=0) < 1

    def estimate_volume(self):
        """
        Estimate the volume of a hyperspace using Monte Carlo.
        """
        points = self.sample_points()
        inside_count = np.sum(self.in_hyperspace(points))
        volume_fraction = inside_count / self.num_samples
        cube_volume = 2**self.dimensions
        return volume_fraction * cube_volume

    def twodimensionscatter(self):
        """
        Visualise sampled points in 2D.
        """
        points = self.sample_points()
        inside = self.in_hyperspace(points)

        plt.figure(figsize=(6, 6))
        plt.scatter(points[0][inside], points[1][inside], color='blue', label='Inside Circle', s=1)
        plt.scatter(points[0][~inside], points[1][~inside], color='red', label='Outside Circle', s=1)
        plt.legend()
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Monte Carlo Sampling of a 2D Circle")
        plt.grid()
        
    def threedimensionscatter(self):
        """
            Visualise sampled points in 3D.
        """
        points = self.sample_points()
        inside = self.in_hyperspace(points)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(points[0][inside], points[1][inside], points[2][inside], 'bo', label='Inside Sphere', markersize=1)
        ax.plot(points[0][~inside], points[1][~inside], points[2][~inside], 'ro', label='Outside Sphere', markersize=1)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_title("Monte Carlo Sampling of a 3D Sphere")
        ax.legend()

if __name__ == "__main__":
    num_samples = 100000
    dimensions_list = [2, 3, 4, 5]
    mc_results = []

    for d in dimensions_list:
        mc_simulator = MonteCarlo(num_samples=num_samples, dimensions=d)
        volume_estimate = mc_simulator.estimate_volume()
        mc_results.append(volume_estimate)
        print(f"Estimated volume for {d}D hyperspace: {volume_estimate:.6f}")
        if d == 2:
            mc_simulator.twodimensionscatter()
        elif d == 3:
            mc_simulator.threedimensionscatter()

class GaussianIntegrator:
    def __init__(self, num_samples=10000000, dimensions=1, sigma=1.0, x0=0.0):
        self.num_samples = num_samples
        self.dimensions = dimensions
        self.sigma = sigma
        self.x0 = x0
        
        """
        self.num_samples = num_samples
        self.dimensions = dimensions
        self.seed = seed
        self.rng = default_rng(SeedSequence(seed))

    def gaussian(self, x):
        """
        Gaussian function f(x) = 1 / (sigma * sqrt(2 * pi)) * exp(-(x - x0)^2 / (2 * sigma^2))
        """
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - self.x0) ** 2) / (2 * self.sigma ** 2))

