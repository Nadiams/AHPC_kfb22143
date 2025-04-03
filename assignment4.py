#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP, parallel variance and the Partial differential
#equations Handout 7 examples provided by Dr. Benjamin Hourahine in PH510.
#Modifications made by kfb22143 - Licensed under the MIT License.
#See LICENSE file for details.
"""
Created on Tue Mar 18 12:32:13 2025

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
			function: The function to integrate over.
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
            initial_mean = np.mean(function_values)
            initial_variance = np.var(function_values, ddof=1)
        else:
            initial_mean, initial_variance = 0, 0
        initial_integral = self.params['volume'] * initial_mean
        total_samples = self.mpi_info['comm'].allreduce(
            region_samples, op=MPI.SUM
        )
        combined_integral = self.mpi_info['comm'].allreduce(
            initial_integral, op=MPI.SUM
        ) / self.mpi_info['size']

        combined_variance = self.mpi_info['comm'].allreduce(
            initial_variance, op=MPI.SUM
        ) / self.mpi_info['size']
        error_object = Error(total_samples,  combined_integral,  combined_variance)
        standard_error = error_object.compute_error()
        if self.mpi_info['rank'] == 0:
            print("\n================ Monte Carlo Integration ===============")
            print(f"Final {self.params['dimensions']}D Integral: "
                  f"{ combined_integral:.6f}")
            print(f"Estimated Variance: { combined_variance:.6f}")
            print(f"Standard Error: {standard_error:.6f}")
            if  combined_integral == 0:
                print("Error: Integral computed as 0!")
            print("========================================================\n")

            return  combined_integral,  combined_variance, standard_error
        return None, None, None

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
            combined_integral, _, _ = self.parallel_monte_carlo()
            estimates.append( combined_integral)

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