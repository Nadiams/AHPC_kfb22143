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
        Returns:
            temporary
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
        Args:
            the mean, variance and sample size.
        Returns:
            variance
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
        """
            Function to find the Standard Error
            Returns:
                Standard Error, 0.
        """
        if self.n_samples > 0:
            return np.sqrt(self.variance / self.n_samples)
        return 0

class MonteCarloIntegrator(Error):
    """
	Monte Carlo class which uses the multi-dimensional Monte Carlo quadrature 
    formula to compute other integrals quickly. Inherits from the Error class.
	"""
    def __init__(self, function, lower_bounds, upper_bounds, num_samples=1000000):
        """
		Initialises parameters.
		Args:
			function: The function to integrate.
			lower_bounds,
			upper_bounds,
            num_samples.
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
            parallel, also finds the variance on this by inheriting from the
            Error class.
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
            [fx
             for fx in (self.params['function'](x) for x in samples)
             if fx is not None],
            dtype=np.float64
        )
        function_values = function_values[np.isfinite(function_values)]
        if function_values.size > 0:
            local_mean = np.mean(function_values)
            local_variance = np.var(function_values, ddof=1)
        else:
            local_mean, local_variance = 0.0, 0.0
        local_integral = self.params['volume'] * local_mean
        total_samples = self.mpi_info['comm'].allreduce(
            region_samples, op=MPI.SUM
        )
        global_integral = self.mpi_info['comm'].allreduce(
            local_integral, op=MPI.SUM
        ) / self.mpi_info['size']

        global_variance = self.mpi_info['comm'].allreduce(
            local_variance, op=MPI.SUM
        ) / self.mpi_info['size']
        error_object = Error(total_samples, global_integral, global_variance)
        standard_error = error_object.compute_error()
        if self.mpi_info['rank'] == 0:
            print("\n================ Monte Carlo Integration ===============")
            print(f"Final {self.params['dimensions']}D Integral: "
                  f"{global_integral:.6f}")
            print(f"Estimated Variance: {global_variance:.6f}")
            print(f"Standard Error: {standard_error:.6f}")
            if global_integral == 0:
                print("Error: Integral computed as 0!")
            print("========================================================\n")

        return (
            global_integral,
            global_variance,
            standard_error
        ) if self.mpi_info['rank'] == 0 else (None, None, None)

    def plot_monte_carlo_convergence(self):
        """
            Plots Monte Carlo convergence.
            Args:
                the output of the monte carlo integral,
                num_samples.
            Returns:
                Plot of the monte carlo integral value and the num_samples.
        """
        if self.mpi_info['rank'] != 0:
            return
        plt.figure(figsize=(8, 6))
        sample_sizes = np.logspace(2, 6, num=20, dtype=int)
        estimates = []

        for num_samples in sample_sizes:
            self.params['num_samples'] = num_samples
            global_integral, _, _ = self.parallel_monte_carlo()
            estimates.append(global_integral)

        plt.plot(sample_sizes, estimates, marker="o", linestyle="-",
                 label="Monte Carlo Estimate", color="blue")
        plt.axhline(y=1.0, color='red', linestyle="dashed", label="Expected Value")
        plt.xscale("log")
        plt.xlabel("Number of Samples")
        plt.ylabel("Integral Estimate")
        plt.title("Monte Carlo Convergence")
        plt.legend()
        plt.grid()
        plt.savefig("monte_carlo_convergence.png")

class ContainedRegion(MonteCarloIntegrator):
    """
        This class inherits from previous class to compute the volume (region)
        of a hyperspace using the Monte Carlo integral.
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
        self.rng = default_rng(SeedSequence(self.mpi_info['rank']))

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
        local_integral, local_variance, _ = self.parallel_monte_carlo()
        all_integrals = self.mpi_info['comm'].gather(local_integral, root=0)
        all_variances = self.mpi_info['comm'].gather(local_variance, root=0)
        if self.mpi_info['rank'] == 0:
            combined_error = Error(self.num_samples, 0, 0)
            for worker_integral, worker_variance in zip(all_integrals, all_variances):
                worker_samples = self.num_samples // self.mpi_info['size']
                worker_error = Error(worker_samples, worker_integral, worker_variance)
                combined_error += worker_error
            final_volume = combined_error.mean * (2 ** self.dimensions)
            standard_error = combined_error.compute_error()

            print("\n==== Monte Carlo Estimation of Hypersphere Volume =====")
            print(f"Estimated Volume: {final_volume:.6f}")
            print(f"Estimated Variance: {combined_error.variance:.6f}")
            print(f"Standard Error: {standard_error:.6f}")
            print("========================================================\n")

            return final_volume, standard_error

        return None, None

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
            point = self.rng.uniform(-1, 1, self.dimensions)
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
        ax.set_box_aspect([1, 1, 1])
        ax.legend()
        plt.savefig("scatter_3d.png")

class GaussianIntegrator(MonteCarloIntegrator):
    """
        Class which inherits from the MonteCarloIntegrator class to compute
        a 1D, 6D gaussians and a gaussian over all space which uses 
        substitution.
    """
    def __init__(self, num_samples, dimensions=1, sigma=1.0,
                 x0=0.0, seed=12345, method='no_sub'):
        """
            Initialises parameters for Gaussian function.
            Args:
                num_samples, dimensions, sigma, x0, seed, method.
        """
        self.sigma = sigma
        self.x0 = x0
        self.dimensions = dimensions
        self.num_samples = num_samples
        self.method = method

        self.mpi_info = {
            'comm': MPI.COMM_WORLD,
            'rank': MPI.COMM_WORLD.Get_rank(),
            'size': MPI.COMM_WORLD.Get_size()
        }
        self.rng = default_rng(SeedSequence(self.mpi_info['rank']))
        lower_bounds = [-5 * sigma] * dimensions
        upper_bounds = [5 * sigma] * dimensions
        if method == 'sub':
            lower_bounds = [-1] * dimensions
            upper_bounds = [1] * dimensions

        super().__init__(
            self.gaussian, lower_bounds, upper_bounds, num_samples
        )

    def sub_function(self, t):
        """
            Substitution which allows the gaussian integral to be solved over
            all space.
            Args:
                t.
            Returns:
                new gaussian function.
        """
        epsilon = 1e-10
        t = np.clip(t, -1 + epsilon, 1 - epsilon)
        x = t / (1 - t**2)
        normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi
                                                    )))**self.dimensions
        exponent = -((x - self.x0) ** 2
                           ) / (2 * self.sigma ** 2)
        gaussian_value = normalisation_factor * np.exp(exponent)
        dx_dt = (1 + t**2) / (1 - t**2)**2
        output = gaussian_value * dx_dt
        return output

    def gaussian(self, x):
        """
            This is the Gaussian function.
            Args:
                self, 
                x.
            Returns:
                Gaussian integral in 1D, 6D or uses the substitution.
        """
        if self.method == 'no_sub':
            normalisation_factor = (1 / (self.sigma * np.sqrt(2 * np.pi))
                                )**self.dimensions
            exponent = -np.sum((x - self.x0) ** 2, axis=-1) / (2 * self.sigma ** 2)
            gaussian_output = normalisation_factor * np.exp(exponent)
            return gaussian_output

        elif self.method == 'sub':
            return self.sub_function(x)

    def compute_gaussian_integral(self):
        """
        Computes the Monte Carlo integral of the Gaussian function.
        Args:
            self, the gaussian function output.
        Returns:
            integral,
            variance,
            standard error.
        """
        integral, variance, standard_error = self.parallel_monte_carlo()

        if self.mpi_info['rank'] == 0:
            print("\n===== Monte Carlo Estimation of Gaussian Integral ======")
            print(f"Estimated Integral: {integral:.6f}")
            print(f"Estimated Variance: {variance:.6f}")
            print(f"Standard Error: {standard_error:.6f}")
            print("========================================================\n")
        return integral, variance, standard_error

    def plot_gaussian_1d(self):
        """
        Plot of the 1D Gaussian integral function.
        """
        if MPI.COMM_WORLD.Get_rank() == 0:
            plt.figure(figsize=(8, 6))
            x_values = self.rng.uniform(-5, 5, 500)
            y_values = self.gaussian(x_values[:, np.newaxis])
            plt.scatter(
                x_values,
                y_values,
                label="Gaussian Function",
                color="black",
                linewidth=2
            )
            #y_errors = np.sqrt(y_values)
            plt.xlabel("x")
            plt.ylabel("Gaussian f(x)")
            plt.title("1D Gaussian Function")
            plt.legend()
            plt.grid()
            plt.savefig("gaussian_1d_final.png")

    def plot_sub(self):
        """
        Plot of the gaussian integral over all space using the substitution.
        Returns:
            sub_plot_final.png
        """
        if self.mpi_info['rank'] == 0:
            plt.figure(figsize=(8, 6))
            x_values = self.rng.uniform(-0.5, 0.5, 500)
            y_values = self.sub_function(x_values)
            plt.scatter(x_values, y_values, labrl="Gaussian Function")
            plt.title("Gaussian Function Over All Space Using Substitution")
            #y_errors = np.sqrt(y_values)
            plt.grid()
            plt.savefig("sub_plot_final.png")

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        MAIN_NUM_SAMPLES = 1000000
        dimensions_list = [2, 3, 4, 5]

        for d in dimensions_list:
            mc_simulator = ContainedRegion(num_samples=MAIN_NUM_SAMPLES, dimensions=d)
            volume_estimate, _, _ = mc_simulator.parallel_monte_carlo()
            print(f"The volume for {d}D hyperspace: {volume_estimate:.4f}")
            if d == 2:
                mc_simulator.twodimensionscatter()
            elif d == 3:
                mc_simulator.threedimensionscatter()

        for dim in [1, 6]:
            gaussian_integrator = GaussianIntegrator(
                num_samples=MAIN_NUM_SAMPLES, dimensions=dim, sigma=1.0, x0=0.0
            )
            integral_value, variance, _ = gaussian_integrator.parallel_monte_carlo()
            print(f"The integral of Gaussian ({dim}D): {integral_value:.4f}")

            if dim == 1:
                gaussian_integrator.plot_gaussian_1d()
                gaussian_integrator.plot_monte_carlo_convergence()
                if rank == 0:
                    gaussian_integrator.plot_sub()

    integrator = GaussianIntegrator(
        num_samples=10000,
        dimensions=6,
        sigma=1.0,
        x0=0.0,
        method='no_sub'
    )
    integral, variance, _ = integrator.parallel_monte_carlo()
    print(f"Final Estimation for 6D Gaussian: "
          f"{integral:.4f}, Variance: {variance:.4f}")

    integrator = GaussianIntegrator(
        num_samples=100000,
        dimensions=1,
        sigma=1.0,
        x0=0.0,
        method='sub'
    )
    integral, variance, _ = integrator.parallel_monte_carlo()
    print(f"Final Estimation for 1D Gaussian (Substitution): "
          f"{integral:.4f}, Variance: {variance:.4f}")

    integratorsub = GaussianIntegrator(
        num_samples=100000,
        dimensions=1,
        sigma=1,
        x0=1,
        method='sub'
    )

    if rank == 0:
        integratorsub.plot_sub()
        integratorsub.plot_monte_carlo_convergence()

    MPI.Finalize()