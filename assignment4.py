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
        function_values = []
        for _ in range(region_samples):
            _, hit = self.perform_single_walk()
            if hit is not None:
                function_values.append(hit)
        function_values = np.array(function_values, dtype=np.float64)

        if function_values.size > 0:
            initial_mean = np.mean(function_values)
            initial_variance = np.var(function_values, ddof=1)
        else:
            initial_mean, initial_variance = 0, 0
        initial_integral = initial_mean

        total_samples = self.mpi_info['comm'].allreduce(
            len(function_values), op=MPI.SUM
        )
        combined_integral = self.mpi_info['comm'].allreduce(
            initial_integral * len(function_values), op=MPI.SUM
        )
        combined_integral = combined_integral / total_samples if total_samples > 0 else 0

        combined_variance = self.mpi_info['comm'].allreduce(
            initial_variance * (len(function_values) - 1), op=MPI.SUM
        )
        combined_variance = combined_variance / (total_samples - 1) if total_samples > 1 else 0

        self.n_samples = total_samples
        self.mean = combined_integral
        self.variance = combined_variance
        standard_error = self.compute_error()

        if self.mpi_info['rank'] == 0:
            print("\n================ Monte Carlo Random Walk ===============")
            print(f"Final Estimate: {combined_integral:.6f}")
            print(f"Estimated Variance: {combined_variance:.6f}")
            print(f"Standard Error: {standard_error:.6f}")
            if combined_integral == 0:
                print("Monte Carlo Integral = 0")
            print("========================================================\n")

            return combined_integral, combined_variance, standard_error
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

class RandomWalkGreenSolver(MonteCarloIntegrator):
    """
        Class to create walkers who follow a random path with specific
        start points, boundaries and charges acting on them.
    """
    def __init__(self, N, L, walkers, max_steps, seed=12345):
        """
        Initialised variables to inherit them from the
        Monte Carlo (Error) Class.
        """
        lower_bounds = [0, 0]
        upper_bounds = [L, L]
        super().__init__(
            function=None,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            num_samples=walkers
        )

        self.N = N
        self.L = L
        self.h = L / (N - 1)
        self.max_steps = max_steps

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.rng = default_rng(SeedSequence(seed + self.rank))

    def greens_walker(self, start):
        """
            Function of the walker in Green's function to use random
            sampling to follow a path starting from a specific point.
            Args:
	        self,
	        start
	    Returns:
                    the mean visit count of green's function,
		    the error (standard deviation) of this mean.
        """
        initial_walkers = self.params['num_samples'] // self.mpi_info['size']
        visits = np.zeros(initial_walkers, int)
        i0, j0 = start
    
        for walker_index in range(initial_walkers):
            i, j = i0, j0
            count = 0
            steps = 0
            while steps < self.max_steps:
                if self.rng.random() < 0.5:
                    i += self.rng.choice([-1, 1])
                else:
                    j += self.rng.choice([-1, 1])
                steps += 1
                if (i, j) == (i0, j0):
                    count += 1
                if i in (0, self.N - 1) or j in (0, self.N - 1):
                    break
            visits[walker_index] = count
        allv = None
        if self.rank == 0:
            allv = np.empty(self.params['num_samples'], int)
        self.comm.Gather(visits, allv, root=0)
    
        if self.rank == 0:
            mean_vis = allv.mean()
            var_vis  = allv.var(ddof=1)
            greens_mean   = self.h**2 * mean_vis
            self.mean     = greens_mean
            self.variance = self.h**4 * var_vis
            return greens_mean, self.compute_error()

class Charges_Boundary_Grids(RandomWalkGreenSolver):
    """
        Class to define boundary conditions, charge distributions
	    and start points.
	"""
    def __init__(self, N=5, L=10):
        """
            Initialised variables to inherit them from the
	        random walker class (Monte Carlo(Error)).
        """
        self.N = N
        self.L = L
        self.h = L / (N - 1)
        self.boundaries = self.boundary_conditions()
        self.charges = self.charge_distribution()

    def boundary_conditions(self):
        """
            Defined boundary conditions in functions to create a 
            loop for each point. A while loop may have been better.
        """

        def boundary_a(i, j):
            """
                All edges uniformly at +1 V.
            """
            return 1

        def boundary_b(i, j):
            """
                Top and bottom edges at +1 V, left and right at -1 V
            """
            if i == 0 or i == self.N - 1:
                return 1
            elif j == 0 or j == self.N - 1:
                return -1
            return 0

        def boundary_c(i, j):
            """
                Top and left edges at +2 V, bottom edge at 0 V and
                right side edge at -4 V.
            """
            if i == 0:
                return 2
            elif i == self.N - 1:
                return 0
            elif j == 0:
                return 2
            elif j == self.N - 1:
                return -4
            return 0

        return {
            "Case A": boundary_a,
            "Case B": boundary_b,
            "Case C": boundary_c
        }

    def charge_distribution(self):
        """
            Set the three different charge distributions.
        """
        N = self.N
        L = self.L
        uniform_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                uniform_charge[i, j] = 10.0 / (L * L)

        gradient_charge = np.zeros((N, N))
        for i in range(N):
            density = 1.0 - (i / (N - 1))
            for j in range(N):
                gradient_charge[i, j] = density

        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        exp_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dx = x[j] - L / 2
                dy = y[i] - L / 2
                r = np.sqrt(dx**2 + dy**2)
                exp_charge[i, j] = np.sum(np.exp(-2000 * r))

        charge_distributions = {
            "Uniform (10C)": uniform_charge,
            "Gradient (top to bottom)": gradient_charge,
            "Exponential (centered)": exp_charge
        }

        return charge_distributions

    def coords_to_index(self, x):
        """
            To obtain better results, increased the grid size.
            Had to convert the coordinates of the specific points
            into indexes of the grid/matrix.
        """
        return int(round(x / self.h))

    def potential_from_green(self, greens_matrix, points):
        """
            Function to find the Green's function potential output.
            Args:
                 self,
                 greens_matrix,
                 points.
                 Returns:
                 results of green's function.
        """
        N, h = self.N, self.h
        results = {}
        for boundaryconditions_label, boundaryconditions_func in self.boundaries.items():
            phi_boundary_grid = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    phi_boundary_grid[i, j] = boundaryconditions_func(i, j)
            for charge_label, charge_density in self.charges.items():
                key = f"{charge_label} + {boundaryconditions_label}"
                potentials = {}
                for x, y in points:
                    i0 = self.coords_to_index(y)
                    j0 = self.coords_to_index(x)
                    phi_from_charges = 0.0
                    for i in range(1, N - 1):
                        for j in range(1, N - 1):
                            phi_from_charges += (
                                greens_matrix[i0, j0, i, j] * charge_density[i, j] * h**2
                                )
                    boundary_phi = 0
                    for j in range(N):
                        boundary_phi += greens_matrix[i0, j0, 0, j] * phi_boundary_grid[0, j]
                        boundary_phi += greens_matrix[i0, j0, N - 1, j] * phi_boundary_grid[N - 1, j]
                    for i in range(1, N - 1):
                        boundary_phi += greens_matrix[i0, j0, i, 0] * phi_boundary_grid[i, 0]
                        boundary_phi += greens_matrix[i0, j0, i, N - 1] * phi_boundary_grid[i, N - 1]
                    potentials[(x, y)] = phi_from_charges + boundary_phi
                results[key] = potentials
        return results

def plot_green_at_points(green, L=10):
    """
        Plot of green's function with specific points.
    
        Args:
        2D array of Green's function outputs,
        specific grid points (5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1
    """
    N = green.shape[0]
    h = L / (N - 1)
    grid_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]

    plt.figure(figsize=(10, 8))
    plt.imshow(green, origin='lower', cmap='viridis', extent=[0, N-1, 0, N-1])
    plt.colorbar(label="Green's function")
    plt.title("Green's Function at Grid Points")
    plt.savefig("Greens_colourmap")

    for x_position, y_position in grid_points:
        j_index = int(round(x_position / h))
        i_index = int(round(y_position / h))

        plt.plot(j_index, i_index, 'ro')
        label = f"({x_position:.1f},{y_position:.1f}) cm"
        plt.text(j_index, i_index, label, color='white')
    plt.xlabel("x (grid index)")
    plt.ylabel("y (grid index)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("plotgreenatpoints.png")
    plt.show()

if __name__ == "__main__":
    N = 5
    L = 10.0
    walkers = 100000
    max_steps = 100000
    h = L / (N - 1)
    sample_points = [(5.0, 5.0), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]
    def coord_to_index(xy):
        x, y = xy
        return int(round(y / h)), int(round(x / h))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        solver = RandomWalkGreenSolver(N, L, walkers, max_steps)
        greens_matrix = np.zeros((N, N, N, N))
        for i in range(1, N-1):
            for j in range(1, N-1):
                greens_output, _ = solver.greens_walker((i, j))
                greens_matrix[:, :, i, j] = greens_output
        sim = Charges_Boundary_Grids(N=N, L=L)
        green_potentials = sim.potential_from_green(greens_matrix, sample_points)
        print("\n=== Potential (V) using Green’s function ===")
        for case_label, potentials_at_points in green_potentials.items():
            print(f"\nCase: {case_label}")
            for point, potential_value in potentials_at_points.items():
                print(f"  Point ({point[0]:.2f}, {point[1]:.2f}),  {potential_value:.4f} V")
    if rank == 0:
        print("\n=== Monte Carlo mean of Green's Function w/ Error ===")
        for point in sample_points:
            grid_index = coord_to_index(point)
            greens_output, greens_error = solver.greens_walker(grid_index)
            print(f"  Point ({point[0]:.2f}, {point[1]:.2f}), "
                  f"G = {greens_output:.4f} ± {greens_error:.4f}")
