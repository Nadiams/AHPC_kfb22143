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
    
    def compute_error(self):
        return np.sqrt(self.variance / self.n_samples) if self.n_samples > 0 else 0

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
        function_values = np.array(
            [fx for fx in (self.params['function'](x) for x in samples) if fx is not None],
            dtype=np.float64
            )
        function_values = function_values[np.isfinite(function_values)]
        if function_values.size > 0:
            local_mean = np.mean(function_values)
            local_variance = np.var(function_values, ddof=1)
        else:
            local_mean, local_variance = 0.0, 0.0
        local_integral = self.params['volume'] * local_mean
        global_integral = self.mpi_info['comm'].allreduce(local_integral, op=MPI.SUM) / self.mpi_info['size']
        global_variance = self.mpi_info['comm'].allreduce(local_variance, op=MPI.SUM) / self.mpi_info['size']
        if self.mpi_info['rank'] == 0:
            self.mean = global_integral
            self.variance = global_variance
            standard_error = self.compute_error()
            print("\n================ Monte Carlo Integration =================")
            print(f"Final 6D Integral: {global_integral:.6f}")
            print(f"Estimated Variance: {global_variance:.6f}")
            if global_integral == 0.0:
                print("Error: Integral computed as 0.0! Check function evaluation.")
            print("========================================================\n")
            return global_integral, global_variance, standard_error
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

        elif self.method == 'sub':
            #transformed_x = x / (1 - x**2)
            #dx_dt = (1 + x**2) / (1 - x**2)**2
            #normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi
            #                                            )))**self.dimensions
            #exponent = -np.sum((transformed_x - self.x0) ** 2, axis=-1
             #                  ) / (2 * self.sigma ** 2)
            #gaussian_value = normalisation_factor * np.exp(exponent)
            #return gaussian_value * np.prod(dx_dt, axis=-1)
            return self.sub_function(x)

    def sub_function(self, t):
        """
            docustring
        """
        x = t / (1 - t**2)
        normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi
                                                    )))**self.dimensions
        exponent = -((x - self.x0) ** 2) / (2 * self.sigma ** 2)
        gaussian_value = normalisation_factor * np.exp(exponent)
        dx_dt = (1 + t**2) / (1 - t**2)**2
        output = gaussian_value * dx_dt
        return output

    def plot_sub(self):
        """
        Plot of transformed gaussian.
        """
        if self.mpi_info['rank'] == 0:
            plt.figure(figsize=(8, 6))
            x_values = np.linspace(-0.5, 0.5, 10)
            y_values = self.sub_function(x_values)
            plt.plot(x_values, y_values)
            plt.grid()
            plt.savefig("sub_plot1.png")

    def plot_gaussian_1d(self):
        """Plot 1D Gaussian function."""
        if MPI.COMM_WORLD.Get_rank() == 0:
            plt.figure(figsize=(8, 6))
            x_values = np.linspace(-5, 5, 500)
            y_values = self.gaussian(x_values[:, np.newaxis])
            plt.plot(x_values, y_values, label="Gaussian Function", color="blue", linewidth=2)
            #plt.errorbar(
            #    x_values,
            #    y_values,
            #    xerr=None,
            #    yerr=y_std,
            #    label="Gaussian (1D)",
            #    fmt='o',
            #    color="blue"
            #)
            plt.xlabel("x")
            plt.ylabel("Gaussian f(x)")
            plt.title("1D Gaussian Function")
            plt.legend()
            plt.grid()
            plt.savefig("gaussian_1d.png")

    def plot_gaussian_6d(self):
        """Plot a 6D Gaussian projection (scatter of first two dimensions)."""
        if MPI.COMM_WORLD.Get_rank() == 0:
            plt.figure(figsize=(8, 6))
            x_values = np.linspace(-5 * self.sigma, 5 * self.sigma, 100)
            y_values = np.linspace(-5 * self.sigma, 5 * self.sigma, 100)
            X, Y = np.meshgrid(x_values, y_values)
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x_sample = np.array([X[i, j], Y[i, j], 0, 0, 0, 0])
                    Z[i, j] = self.gaussian(x_sample)
            plt.contourf(X, Y, Z, levels=50, cmap="viridis")
            plt.colorbar(label="Gaussian Value")
            plt.xlabel("x-axis (Dim 1)")
            plt.ylabel("y-axis (Dim 2)")
            plt.title("Projected 6D Gaussian (First 2 Dimensions)")
            plt.grid()
            plt.savefig("gaussian_6d_contour.png")

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        MAIN_NUM_SAMPLES = 1000000
        dimensions_list = [2, 3, 4, 5]

        for d in dimensions_list:
            mc_simulator = ContainedRegion(num_samples=MAIN_NUM_SAMPLES, dimensions=d)
            volume_estimate, _ = mc_simulator.parallel_monte_carlo()
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
            print(f"The integral of Gaussian ({dim}D): {integral_value:.4f}")

            if dim == 1:
                gaussian_integrator.plot_gaussian_1d()
                gaussian_integrator.plot_sub()

            if dim == 6:
                gaussian_integrator.plot_gaussian_6d()

    integrator = GaussianIntegrator(MAIN_NUM_SAMPLES, dimensions=6, sigma=1.0, x0=0.0, method='no_sub')
    integral, variance = integrator.parallel_monte_carlo()
    print(f"6D Integral: {integral:.4f}, Variance: {variance:.4f}")

    integrator = GaussianIntegrator(MAIN_NUM_SAMPLES, dimensions=1, sigma=1.0, x0=0.0, method='sub')
    integral, variance = integrator.parallel_monte_carlo()
    print(f"1D Integral over all space: {integral:.4f}, Variance: {variance:.4f}")
    integratorsub = GaussianIntegrator(MAIN_NUM_SAMPLES, dimensions=1, sigma=1, x0=1, method='sub')
    integratorsub.plot_sub()
    MPI.Finalize()