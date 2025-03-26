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
                          ) / temporary.n_samples
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

class MonteCarloIntegrator(Error):
    """
	To initialise the Monte Carlo class.
	"""
    def __init__(self, function, lower_bounds, upper_bounds, num_samples=1000000):
        """
		Initialises parameters.
		Args:
			function: The function to integrate.
			lower_bounds: List of lower bounds for each dimension.
			upper_bounds: List of upper bounds for each dimension.

			num_samples: Number of random samples to take.
		"""
        lower_bounds = np.array(lower_bounds, dtype=float)
        upper_bounds = np.array(upper_bounds, dtype=float)

        self.params = {
            'function': function,
            'num_samples': num_samples,
            'dimensions': len(lower_bounds),
            'bounds': {'lower': lower_bounds, 'upper': upper_bounds},
            'volume': np.prod(upper_bounds - lower_bounds)
        }
        self.mpi_info = {
            'comm': MPI.COMM_WORLD,
            'rank': MPI.COMM_WORLD.Get_rank(),
            'size': MPI.COMM_WORLD.Get_size()
        }

        self.rng = default_rng(SeedSequence(self.mpi_info['rank']))

        super().__init__(num_samples, mean=0, var=0)

    def parallel_monte_carlo(self):
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

        function_values = np.array([self.params['function'](x) for x in samples])
        local_mean = np.mean(function_values)
        local_variance = np.var(function_values, ddof=1)

        local_integral = self.params['volume'] * local_mean

        global_integral = self.mpi_info['comm'].reduce(local_integral, op=MPI.SUM, root=0)
        global_variance = self.mpi_info['comm'].reduce(local_variance, op=MPI.SUM, root=0)

        if self.mpi_info['rank'] == 0:
            self.mean = global_integral
            self.variance = global_variance / self.mpi_info['size']
            return global_integral, global_variance
        return None, None

class ContainedRegion(MonteCarloIntegrator):
    """
        This class inherits from previous class to compute the volume (region)
        of a hyperspace using Monte Carlo.
    """
    def __init__(self, num_samples=100000, dimensions=5, seed=12345):
        """
            Initialises parameters.
            Args:
                num_samples, dimensions, seed.
        """

        super().__init__(
            function=self.inside_hyperspace,
            lower_bounds=[-1]*dimensions,
            upper_bounds=[1]*dimensions,
            num_samples=num_samples
        )

        self.num_samples = num_samples
        self.dimensions = dimensions
        self.seed = seed
        #self.mpi_info = {
         #   'comm': MPI.COMM_WORLD,
          #  'rank': MPI.COMM_WORLD.Get_rank(),
           # 'size': MPI.COMM_WORLD.Get_size()
        #}
        self.rng = default_rng(SeedSequence(self.mpi_info['rank']))

        #lower_bounds = [-1] * dimensions
        #upper_bounds = [1] * dimensions

    def inside_hyperspace(self, point):
        """
            Points inside hyperspace.
        """
        return 1 if np.sum(point**2) <= 1 else 0

    def sample_points(self):
        """
            To generate random points within the unit cube.
        """
        return self.rng.uniform(
            -1, 1, size=(self.num_samples, self.dimensions)
        )

    def calculate_hyperspace_volume(self):
        """
            Estimates the volume of the hyperspace in d, dimensions using the
            Monte Carlo integrator.
            Returns:
                Estimated volume.
        """
        local_volume, local_variance = self.parallel_monte_carlo()
        if self.mpi_info['rank'] == 0:
            standard_error = np.sqrt(local_variance / self.num_samples)
            return local_volume * (2 ** self.dimensions), standard_error
        return 0.0, 0.0

    def hyperspace_region_demo(self):
        """
            Hyperspace as a percentage of inner area to show the region.
            Returns:
                inner_percentage,
                f-string.
        """
        points = self.sample_points()
        inner = np.sum(points**2, axis=1) <= 1
        inner_percentage = np.sum(inner) / self.num_samples
        if self.mpi_info['rank'] == 0:
            print(f"Percentage inside hyperspace: {inner_percentage:.4f}")

    def  plot_points_in_hyperspace(self):
        """
            Function to compute the points inside and outside the hyperspace.
            Returns:
                points_inside,
                points_outside.
        """
        points_inside = []
        points_outside = []

        for _ in range(self.num_samples):
            point = np.random.uniform(-1, 1, self.dimensions)
            if np.linalg.norm(point) <= 1:
                points_inside.append(point)
            else:
                points_outside.append(point)

        points_inside = np.array(points_inside)
        points_outside = np.array(points_outside)

        return points_inside, points_outside

    def twodimensionscatter(self):
        """
            Visualise sampled points in 2D.
        """
        if self.mpi_info['rank'] == 0:
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

class GaussianIntegrator(MonteCarloIntegrator):
    """
        Monte Carlo integration of a Gaussian function.
    """
    def __init__(self, num_samples, dimensions=1, sigma=1.0, x0=0.0, method='no_sub'):
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
        self.method = method

        if method == 'no_sub':
            lower_bounds = [-5 * sigma] * dimensions
            upper_bounds = [5 * sigma] * dimensions
        elif method == 'sub':
            lower_bounds = [-1] * dimensions
            upper_bounds = [1] * dimensions

        super().__init__(
            self.gaussian, lower_bounds, upper_bounds, num_samples
        )

    def gaussian(self, x):
        """
            Gaussian function f(x) = 1 / (sigma * sqrt(2 * pi))
            * exp(-(x - x0)^2 / (2 * sigma^2))
        """
        if self.method == 'no_sub':
            normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi))
                                )**self.dimensions
            exponent = -np.sum((x - self.x0) ** 2, axis=-1) / (2 * self.sigma ** 2)
            gaussian_output = normalisation_factor * np.exp(exponent)
            return gaussian_output


    def plot_gaussian_1d(self):
        """
            Plot of 1D gaussian.
        """
        if self.mpi_info['rank'] == 0:
            plt.figure()
            x_values = np.linspace(-5, 5, 500)
            y_values = np.array([self.gaussian(x) for x in x_values])
            #z=np.linspace(-1, 1, 500)
            #y_error = np.sqrt(y_values)
            y_std = np.std(y_values) / np.sqrt(len(y_values))
            computed_integral, _ = self.parallel_monte_carlo()
            #print(f"x: {x_values}")
            #print(f"y: {y_values}")
            #print(f"yerr: {y_std}")
            print(f"Integral: {computed_integral}")
            if len(x_values) != len(y_values):
                print("Mismatch in lengths: x has", len(x_values), "and y has", len(y_values))
            #y_error = error_value
            plt.errorbar(
                x_values,
                y_values,
                xerr=None,
                yerr=y_std,
                label="Gaussian (1D)",
                fmt='o',
                color="blue"
            )
            plt.plot(x_values, y_values, label="Gaussian (1D)", color="blue")
            plt.legend(loc='upper right')
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.xlim(-5.0, 5.0)
            plt.title("Gaussian Distribution in 1D")
            plt.savefig("gaussian_1d.png")
            plt.grid()

    def plot_gaussian_6d(self):
        """
            Plot of 6D gaussian.
        """
        if self.mpi_info['rank'] == 0:
            plt.figure()
            sixd_gaussian = np.random.normal(self.x0, self.sigma, size=(500, 6))
            plt.scatter(sixd_gaussian[:, 0], sixd_gaussian[:, 1], color="blue",
                        label="6D Gaussian Projection")
            plt.legend(loc='upper right')
            plt.xlabel("x-axis")
            plt.ylabel("y-axis")
            plt.xlim(-6.0, 6.0)
            plt.title("Gaussian Distribution in 6D")
            plt.grid()
            plt.savefig("gaussian_6d.png")

    def plot_transformed_gaussian(self):
        """
            Plot the Gaussian over all space using the transformation.
        """
        if self.rank == 0:
            x_values, transformed_values, integral = self.transform_variable()

            plt.figure(figsize=(8, 6))
            plt.scatter(x_values, transformed_values,
                    label="Transformed Gaussian", color="blue", s=5
            )
            plt.axhline(y=integral, color='red', linestyle='dashed',
                    label=f"Mean Integral: {integral:.4f}"
            )
            plt.xlabel("Transformed Variable x")
            plt.ylabel("Gaussian Function Value")
            plt.xlim(-6.0, 6.0)
            plt.title("Transformed Gaussian Distribution")
            plt.grid()
            plt.legend()
            plt.savefig("gaussian_transformed.png")

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        MAIN_NUM_SAMPLES = 1000000
        dimensions_list = [2, 3, 4, 5]

        for d in dimensions_list:
            mc_simulator = ContainedRegion(num_samples=MAIN_NUM_SAMPLES, dimensions=d)
            volume_estimate, _ = mc_simulator.parallel_monte_carlo()

            if mc_simulator.mpi_info['rank'] == 0:
                print(f"The volume for {d}D hyperspace: {volume_estimate:.4f}")

                if d == 2:
                    mc_simulator.twodimensionscatter()
                elif d == 3:
                    mc_simulator.threedimensionscatter()
                else:
                    mc_simulator.hyperspace_region_demo()

        for dim in [1, 6]:
            gaussian_integrator = GaussianIntegrator(
                num_samples=MAIN_NUM_SAMPLES, dimensions=dim, sigma=1.0, x0=0.0
            )
            integral_value, _ = gaussian_integrator.parallel_monte_carlo()

            if gaussian_integrator.mpi_info['rank'] == 0:
                print(f"The integral of Gaussian ({dim}D): {integral_value:.4f}")
            x_transformed, adjusted_value, transformed_integral = (
                gaussian_integrator.transform_variable()
            )

            if gaussian_integrator.mpi_info['rank'] == 0:
                print(f"The integral of the transformed Gaussian:"
                      f"{transformed_integral:.4f}"
                )
                gaussian_integrator.plot_transformed_gaussian()
                gaussian_integrator.plot_gaussian_1d()
                gaussian_integrator.plot_gaussian_6d()

    MPI.Finalize()