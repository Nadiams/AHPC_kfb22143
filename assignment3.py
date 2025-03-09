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
from numpy.random import SeedSequence, default_rng
from mpi4py import MPI

class MonteCarloIntegrator:
    """
        To initialise the Monte Carlo class.
    """
    def __init__(self, function, lower_bounds, upper_bounds, num_samples=100000):
        """
            Initialises parameters.
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

    def parallel_monte_carlo(self):
        """
            Function to use MPI Parallelism.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_samples = self.num_samples // size
        samples = default_rng().uniform(self.lower_bounds, self.upper_bounds,
				(local_samples, self.dimensions)
		)
        count_inside = np.sum(np.sum(samples**2, axis=1) <= 1)
        region_volume = (2 ** self.dimensions) * (count_inside / local_samples)
        total_volumes = comm.gather(region_volume, root=0)

        if rank == 0:
            mean_volume = np.mean(total_volumes)
            variance = np.var(total_volumes)
            print(
                f"The {self.dimensions}D Hyperspace Volume: {mean_volume:.6f}"
                f" ± {np.sqrt(variance):.6f}"
            )

    def integrate(self):
        """
            Performs the Monte Carlo integration to estimate the integral
            in parallel across multiple processors.
            Returns:
                The value computed by the integral.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        local_samples = self.num_samples // size
        samples = self.rng.uniform(self.lower_bounds, self.upper_bounds,
                                  (local_samples, self.dimensions)
        )
        volume = np.prod(self.upper_bounds - self.lower_bounds)
        function_values = np.mean(
                            [self.function(sample) for sample in samples]
                        )
        integral_value = volume * function_values
        return integral_value

    def transform_variable(self, t):
        """
        Computes the integral of f(x) over (-∞, ∞) using the transformation
        x = t / (1 - t^2).
        Returns:
                The transformed variable.
                The Jacobian determinant.
        """
        x = t / (1 - t**2)
        jacobian = (1 + t**2) / (1 - t**2)**2
        return x, jacobian

class ContainedRegion(MonteCarloIntegrator):
    """
        This class inherits from previous class to compute the volume (region)
        of a hyperspace using Monte Carlo.
    """
    def __init__(self, num_samples=10000, dimensions=5, seed=12345):
        """
            Initialises parameters.
            Args:
                num_samples, dimensions, seed.
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
    """
        Monte Carlo integration of a Gaussian function.
    """
    def __init__(self, num_samples=100000, dimensions=1, sigma=1.0, x0=0.0):
        """
            Initialises parameters for Gaussian function.
            Args:
                num_samples, dimensions, sigma, x0.
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
                (x - self.x0) ** 2) / (2 * self.sigma ** 2)
        )

if __name__ == "__main__":
    MAIN_NUM_SAMPLES = 1000000
    dimensions_list = [2, 3, 4, 5]

    for d in dimensions_list:
        mc_simulator = ContainedRegion(num_samples=MAIN_NUM_SAMPLES,
                                       dimensions=d
                                       )
        mc_simulator.parallel_monte_carlo()
        volume_estimate = mc_simulator.integrate()
        print(f"The volume for {d}D hyperspace: {volume_estimate:.6f}")
        if d == 2:
            mc_simulator.twodimensionscatter()
        elif d == 3:
            mc_simulator.threedimensionscatter()

    gaussian_integrator = GaussianIntegrator(num_samples=MAIN_NUM_SAMPLES,
                                             dimensions=1, sigma=1.0, x0=0.0
    )
    integral_value_gaussian = gaussian_integrator.integrate()
    print(f"The integral of Gaussian: {integral_value_gaussian:.6f}")