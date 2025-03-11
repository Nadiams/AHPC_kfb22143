#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP example provided by Dr. Benjamin Hourahine in
# PH510. Modifications made by kfb22143 - Licensed under the MIT License.
# See LICENSE file for details.
"""
Created on Mon Mar  3 15:07:49 2025

@author: nadia
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
from mpi4py import MPI

class Error:
    """
        Class to calculate the error (mean + variance) in parallel.
    """
    def __init__(self, N, mean, variance):
        """
        Initialises objects into Error class.
        Args:
            num_samples, mean, and variance.
        """
        self.N = N
        self.mean = mean
        self.variance = variance

    def __add__(self, other):
        """
        Adds two objects together.
        """
        temporary = copy.deepcopy(self)
        temporary.N += other.N
        temporary.mean = ( self.N * self.mean + other.N * other.mean
                          ) / temporary.N
        temporary.variance = self.parallel_variance(self.N, self.mean,
                            self.variance, other.N, other.mean, other.variance
        )
        return temporary

    def parallel_variance(self, n_a, mean_a, var_a, n_b, mean_b, var_b):
        """
        Computes the combined variance for two groups of samples.
        """
        total_samples = n_a + n_b
        delta_mean = mean_b - mean_a
        final_m2 = (var_a * (n_a - 1) + var_b * (n_b - 1) +
                       delta_mean**2 * n_a * n_b / total_samples)
        final_variance = final_m2 / (total_samples - 1)
        return final_variance

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
        self.params = {
            'function': function,
            'num_samples': num_samples,
            'dimensions': len(lower_bounds),
            'bounds': {
                'lower': np.array(lower_bounds, dtype=float),
                'upper': np.array(upper_bounds, dtype=float)
            }
        }
        self.rng = default_rng(SeedSequence())
        self.mpi_info = {
            'comm': MPI.COMM_WORLD,
            'rank': MPI.COMM_WORLD.Get_rank(),
            'size': MPI.COMM_WORLD.Get_size()
        }

    def parallel_monte_carlo(self):
        """
		Function to use MPI Parallelism.
		"""
        region_samples = self.params['num_samples'] // self.mpi_info['size']
        samples = self.rng.uniform(
            self.params['bounds']['lower'],
            self.params['bounds']['upper'],
            (region_samples, self.params['dimensions'])
        )
        count_inside = np.sum(np.sum(samples**2, axis=1) <= 1)
        region_volume = (2 ** self.params['dimensions']
                         ) * (count_inside / region_samples)
        total_volumes = self.mpi_info['comm'].gather(region_volume, root=0)

        if self.mpi_info['rank'] == 0:
            mean_volume = np.mean(total_volumes)
            variance = np.var(total_volumes)
            print(
                f"The {self.params['dimensions']}D Hyperspace Volume:"
                f"  {mean_volume:.4f} ± {np.sqrt(variance):.4f}"
            )
            return mean_volume, variance

    def integrate(self):
        """
    		Performs the Monte Carlo integration to estimate the integral in
            parallel across multiple processors.
            Returns:
                The value computed by the integral.
		"""
        region_samples = self.params['num_samples'] // self.mpi_info['size']
        samples = self.rng.uniform(
            self.params['bounds']['lower'],
            self.params['bounds']['upper'],
            (region_samples, self.params['dimensions'])
        )
        volume = np.prod(self.params['bounds']['upper'] -
                         self.params['bounds']['lower'])
        function_values = np.mean([self.params['function'](sample)
                                   for sample in samples])
        region_integral_value = volume * function_values
        return region_integral_value


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
    def __init__(self, num_samples, dimensions=1, sigma=1.0, x0=0.0):
        """
            Initialises parameters for Gaussian function.
            Args:
                num_samples, dimensions, sigma, x0.
        """
        self.sigma = sigma
        self.x0 = x0
        self.dimensions = dimensions
        self.num_samples = num_samples
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
        normalization_factor = 1 / (self.sigma * np.sqrt(2 * np.pi
                                                         ))**self.dimensions
        exponent = -np.sum((x - self.x0)**2) / (2 * self.sigma**2)
        gaussian_output = normalization_factor * np.exp(exponent)
        return gaussian_output

    def transform_variable(self):
        """
        	Computes the integral of f(x) over (-∞, ∞) using the transformation
            x = t / (1 - t^2).
            Returns:
                The transformed variable.
                The Jacobian determinant.
    	"""
        t = self.rng.uniform(-1, 1, self.num_samples)
        x = t / (1 - t**2)
        jacobian = (1 + t**2) / (1 - t**2)**2
        gaussian_value = self.gaussian(x)
        adjusted_value = gaussian_value * jacobian
        integral = np.mean(adjusted_value)
        return integral

    def plot_gaussian_1d(self):
        """
            Plot of 1D gaussian.
        """
        x_values = np.linspace(-1, 1, 500)
        y_values = self.calculate_gaussian_1d(x_values)
        plt.plot(x_values, y_values, label="Gaussian (1D)", color="blue")
        plt.legend(loc='upper right')
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Gaussian Distribution in 1D")
        plt.grid()

if __name__ == "__main__":
    MAIN_NUM_SAMPLES = 1000000
    dimensions_list = [2, 3, 4, 5]

    for d in dimensions_list:
        mc_simulator = ContainedRegion(num_samples=MAIN_NUM_SAMPLES, dimensions=d)
        mean_volume, variance = mc_simulator.parallel_monte_carlo()
        volume_estimate = mc_simulator.integrate()

        if mc_simulator.mpi_info['rank'] == 0:
            print(f"The volume for {d}D hyperspace: {volume_estimate:.4f}")
        
        if d == 2:
            mc_simulator.twodimensionscatter()
        elif d == 3:
            mc_simulator.threedimensionscatter()

    for dim in [1, 6]:
        gaussian_integrator = GaussianIntegrator(num_samples=MAIN_NUM_SAMPLES,
                                    dimensions=dim, sigma=1.0, x0=0.0)
        integral_value = gaussian_integrator.integrate()
        print(f"The integral of Gaussian ({dim}D): {integral_value:.4f}")

    gaussian_integrator.plot_gaussian_1d()