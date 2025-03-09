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
import math
from mpmath import mp, mpf

class MonteCarloIntegrator:
    def __init__(
            self, function, lower_bounds, upper_bounds, num_samples=100000
    ):
        """
           To initialise the Monte Carlo class.
            Args:
                function: The function to integrate.
                lower_bounds: List of lower bounds for each dimension.
                upper_bounds: List of upper bounds for each dimension.
                num_samples: Number of random samples to take.
        """
        self.function = function
        self.lower_bounds = np.array(lower_bounds, dtype=float)
        self.upper_bounds = np.array(upper_bounds, dtype=float)
        self.num_samples = num_samples
        self.dimensions = len(lower_bounds)
        self.rng = default_rng(SeedSequence())

    def integrate(self):
        """
            Performs the Monte Carlo integration to estimate the integral.
            Returns:
                The value computed by the integral.
        """
        samples = self.rng.uniform(self.lower_bounds, self.upper_bounds, 
                           (self.num_samples, self.dimensions))
        function_values = np.apply_along_axis(self.function, 1, samples)

        volume = np.prod(self.upper_bounds - self.lower_bounds)
        integral_estimate = volume * np.mean(function_values)

        return integral_estimate

    def transform_variable(self, t):
        """
            Computes the integral of f(x) over (-∞, ∞) using the
            transformation x = t / (1 - t^2).
            Returns:
                The transformed variable.
                The Jacobian determinant.
        """
        x = t / (1 - t**2)
        jacobian = (1 + t**2) / (1 - t**2)**2
        return x, jacobian

class ContainedRegion(MonteCarloIntegrator):
    def __init__(self, num_samples=10000, dimensions=5, seed=12345):
        """
            This class inherits from previous class to compute the volume
            (region) of a hyperspace using Monte Carlo.
        """
        self.num_samples = num_samples
        self.dimensions = dimensions
        self.seed = seed
        self.rng = default_rng(SeedSequence(seed))

        lower_bounds = [-1] * dimensions
        upper_bounds = [1] * dimensions

        def inside_hyperspace(point):
            return 1 if np.sum(point**2) <= 1 else 0

        super().__init__(
            inside_hyperspace, lower_bounds, upper_bounds, num_samples
        )

    def sample_points(self):
        """
            To generate random points within the unit cube.
        """
        return self.rng.uniform(
            -1, 1, size=(self.num_samples, self.dimensions)
        )

    def twodimensionscatter(self):
        """
            Visualise sampled points in 2D.
        """
        points = self.sample_points()
        inside = np.sum(points**2, axis=0) < 1

        plt.figure(figsize=(6, 6))
        plt.scatter(
            points[0][inside], points[1][inside], color='blue', 
            label='Inside Circle', s=1
        )
        plt.scatter(
            points[0][~inside], points[1][~inside], color='red', 
            label='Outside Circle', s=1
        )
        plt.legend(loc='upper right')
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Monte Carlo Sampling of a 2D Circle")
        plt.grid()

    def threedimensionscatter(self):
        """
            Visualise sampled points in 3D.
        """
        points = self.sample_points()
        inside = np.sum(points**2, axis=0) < 1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            points[0][inside], points[1][inside], points[2][inside], 
            color='blue', s=1, label='Inside Sphere'
        )
        ax.scatter(
            points[0][~inside], points[1][~inside], points[2][~inside], 
            color='red', s=1, label='Outside Sphere'
        )
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_title("Monte Carlo Sampling of a 3D Sphere")
        ax.legend()

class GaussianIntegrator(MonteCarloIntegrator):
    def __init__(self, num_samples=100000, dimensions=1, sigma=1.0, x0=0.0):
        """
            Monte Carlo integration of a Gaussian function.
        """
        self.sigma = sigma
        self.x0 = x0

        lower_bounds = [-5 * sigma] * dimensions
        upper_bounds = [5 * sigma] * dimensions

        super().__init__(
            self.gaussian, lower_bounds, upper_bounds, num_samples
        )

    def gaussian(self, x):
        """
            Gaussian function f(x) = 1 / (sigma * sqrt(2 * pi))
            * exp(-(x - x0)^2 / (2 * sigma^2))
        """
        return (1 / (self.sigma * np.sqrt(2 * np.pi))) * np.exp(-(
            (x - self.x0) ** 2) / (2 * self.sigma ** 2))

def parallelmontecarlo(num_samples, dimensions):
    """
        Class to use MPI Parallelism.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    mc_simulator = ContainedRegion(num_samples=num_samples, dimensions=dimensions)
    lower_bounds = np.array([-1] * dimensions, dtype=float)
    upper_bounds = np.array([1] * dimensions, dtype=float)
    volume = np.prod(upper_bounds - lower_bounds)
    local_samples = num_samples // size
    rng = np.random.default_rng(seed=rank)
    count_inside = 0
    for i in range(local_samples):
        point = rng.uniform(-1, 1, dimensions)
        if np.sum(point**2) <= 1:
            count_inside += 1
    region_volume = (2**dimensions) * (count_inside / local_samples)
    total_volumes = comm.gather(region_volume, root=0)
    if rank == 0:
        for i in range(0, dimensions):
            mean_volume = np.mean(total_volumes)
            variance = np.var(total_volumes)
            print(f"The {dimensions}D Hyperspace Volume: {mean_volume:.6f} ± {np.sqrt(variance):.6f}")

if __name__ == "__main__":
    parallelmontecarlo(num_samples=1000000, dimensions=5)
    num_samples = 100000
    dimensions_list = [2, 3, 4, 5]

    for d in dimensions_list:
        mc_simulator = ContainedRegion(num_samples=num_samples, dimensions=d)
        volume_estimate = mc_simulator.integrate()
        print(
            f"The volume for {d}D hyperspace: {volume_estimate:.6f}"
        )

        if d == 2:
            mc_simulator.twodimensionscatter()
        elif d == 3:
            mc_simulator.threedimensionscatter()

    gaussian_integrator = GaussianIntegrator(
        num_samples=1000000, dimensions=1, sigma=1.0, x0=0.0
    )
    integral_value = gaussian_integrator.integrate()
    print(
        f"The integral of Gaussian: {integral_value:.6f}"
    )