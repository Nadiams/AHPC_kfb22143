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
    def __init__(self, n_samples, mean, var):
        """
        Initialises objects into Error class.
        Args:
            num_samples, mean, and variance.
        """
        self.n_samples= n_samples
        self.mean = mean
        self.variance = var

    def __add__(self, other):
        """
        Adds two objects together.
        """
        temporary = copy.deepcopy(self)
        temporary.n_samples+= other.n_samples
        temporary.mean = ( self.n_samples* self.mean + other.n_samples* other.mean
                          ) / temporary.N
        temporary.variance = self.parallel_variance(
                    (self.n_samples, self.mean, self.variance),
                    (other.n_samples, other.mean, other.variance)
        )
        return temporary

    def parallel_variance(self, samples_a, samples_b):
        """
        Computes the combined variance for two groups of samples.
        """
        n_a, mean_a, var_a = samples_a
        n_b, mean_b, var_b = samples_b
        total_samples = n_a + n_b
        delta_mean = mean_b - mean_a
        final_m2 = (var_a * (n_a - 1) + var_b * (n_b - 1) + (
            delta_mean**2 * n_a * n_b / total_samples)
        )
        return final_m2 / (total_samples - 1)

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

        local_error = Error(region_samples, np.mean(samples), np.var(samples))
        stats = self.mpi_info['comm'].gather(local_error, root=0)

        if self.mpi_info['rank'] == 0:
            total_samples = sum(error.n_samples for error in stats)
            weighted_mean = sum(error.n_samples * error.mean for error in stats)
            weighted_variance = sum((error.n_samples - 1) * error.variance for
                                    error in stats
            )

            global_mean = weighted_mean / total_samples
            global_variance = (weighted_variance + sum(
                error.n_samples * (error.mean - global_mean)**2 for error in stats
                )) / (total_samples - 1)

            print(f"The {self.params['dimensions']}D Hyperspace Volume:"
                  f"{global_mean:.4f} ± {np.sqrt(global_variance):.4f}")
            return global_mean, global_variance

        return None, None

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

        return float(region_integral_value)


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
        inside = np.sum(points**2, axis=1) <= 1

        plt.figure(figsize=(6, 6))
        plt.scatter(
        points[inside, 0], points[inside, 1], color='blue',
        label='Inside Circle', s=1
        )
        plt.scatter(
        points[~inside, 0], points[~inside, 1], color='red',
        label='Outside Circle', s=1
        )
        plt.legend(loc='upper right')
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Monte Carlo Sampling of a 2D Circle")
        plt.grid()
        plt.savefig("scatter_2d.png")

    def threedimensionscatter(self):
        """
            Visualise sampled points in 3D.
        """
        points = self.sample_points()
        inside = np.sum(points**2, axis=1) <= 1

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            points[inside, 0], points[inside, 1], points[inside, 2],
            color='blue', s=1, label='Inside Sphere'
        )
        ax.scatter(
            points[~inside, 0], points[~inside, 1], points[~inside, 2],
            color='red', s=1, label='Outside Sphere'
        )

        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")
        ax.set_title("Monte Carlo Sampling of a 3D Sphere")
        ax.legend()
        plt.savefig("scatter_3d.png")

    def hyperspace_region_demo(self):
        """
            Hyperspace as a percentage of inner area to show the region.
        """
        points = self.sample_points()
        inner = np.sum(points**2, axis=1) <= 1
        inner_percentage = np.sum(inner) / self.num_samples
        print(f"Percentage inside hyperspace: {inner_percentage:.4f}")

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
        self.variance = 0
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
        normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi))
                                )**self.dimensions
        exponent = -np.sum((x - self.x0)**2) / (2 * self.sigma**2)
        gaussian_output = normalisation_factor * np.exp(exponent)
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
        if self.mpi_info['rank'] == 0:
            x_values = np.linspace(-1, 1, 500)
            y_values = self.gaussian(x_values)
            #z=np.linspace(-1, 1, 500)
            y_error = np.sqrt(y_values)
            computed_integral = self.integrate()
            print(f"x: {x_values}")
            print(f"y: {y_values}")
            print(f"yerr: {y_error}")
            print(f"Integral: {computed_integral}")
            #y_error = error_value
            plt.errorbar(
                x_values,
                y_values,
                xerr=None,
                yerr=y_error,
                label="Gaussian (1D)",
                fmt='o',
                color="blue"
            )
            plt.plot(x_values, y_values, label="Gaussian (1D)", color="blue")
            plt.legend(loc='upper right')
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.title("Gaussian Distribution in 1D")
            plt.savefig("gaussian_1d.png")
            plt.grid()

    def plot_gaussian_6d(self):
        """
            Plot of 6D gaussian.
        """
        if self.mpi_info['rank'] == 0:
            sixd_gaussian = np.random.normal(self.x0, self.sigma, size=(500, 6))
            plt.scatter(sixd_gaussian[:, 0], sixd_gaussian[:, 1], color="blue",
                        label="6D Gaussian Projection")
            plt.legend(loc='upper right')
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.title("Gaussian Distribution in 6D")
            plt.grid()
            plt.savefig("gaussian_6d.png")

    def plot_transformed_gaussian(self):
        """
            Plot the Gaussian over all space using the transformation.
        """
        t_values = np.linspace(-0.99, 0.99, 500)
        x_values = t_values / (1 - t_values**2)
        gaussian_values = self.gaussian(x_values)

        plt.figure(figsize=(8, 6))
        plt.plot(x_values, gaussian_values, label="Transformed Gaussian", color="blue")
        plt.xlabel("Transformed Variable x")
        plt.ylabel("Gaussian Function Value")
        plt.title("Transformed Gaussian Distribution")
        plt.grid()
        plt.legend()
        plt.savefig("gaussian_transformed.png")

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
        else:
            mc_simulator.hyperspace_region_demo()

    for dim in [1, 6]:
        gaussian_integrator = GaussianIntegrator(num_samples=MAIN_NUM_SAMPLES,
                                    dimensions=dim, sigma=1.0, x0=0.0)
        integral_value = gaussian_integrator.integrate()
        if gaussian_integrator.mpi_info['rank'] == 0:
            print(f"The integral of Gaussian ({dim}D): {integral_value:.4f}")
    gaussian_integrator = GaussianIntegrator(num_samples=MAIN_NUM_SAMPLES,
                                             dimensions=1, sigma=1.0, x0=0.0
    )
    integral_value = gaussian_integrator.transform_variable()
    if gaussian_integrator.mpi_info['rank'] == 0:
        print(f"The integral of the transformed Gaussian: {integral_value:.4f}")
    gaussian_integrator.plot_transformed_gaussian()
    gaussian_integrator.plot_gaussian_1d()
    gaussian_integrator.plot_gaussian_6d()

    MPI.Finalize()