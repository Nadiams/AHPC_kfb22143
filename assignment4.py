#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP example provided by Dr. Benjamin Hourahine in
# PH510. Modifications made by kfb22143 - Licensed under the MIT License.
# See LICENSE file for details.
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



    MPI.Finalize()