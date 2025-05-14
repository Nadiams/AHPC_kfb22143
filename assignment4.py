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

class overrelaxation(MonteCarloIntegrator):
    """
            Method to solve Poissons equation.
            Implement a relaxation (or over-relaxation) method to solve Poisson’s equation for a
    square N × N grid, with a grid spacing of h and specified charges at the grid sites (f ).
    This will be used as an independent check for your Monte Carlo results.
    """
    def __init__(self, N=4, h=1, seed=12345,
                 num_samples=1000, dimensions=1):
        self.N = N
        self.h = h
        self.seed = seed
        self.dimensions = dimensions
        self.num_samples = num_samples
        self.rng = default_rng(SeedSequence(seed))
        self.phi = np.zeros((N, N))
        super().__init__(
            function=self.inside_hyperspace,
            lower_bounds=[-1]*dimensions,
            upper_bounds=[1]*dimensions,
            num_samples=num_samples
        )
    def laplace(self):
        for i in range(self.N):
            self.phi[0, i] = 1
            self.phi[self.N-1, i] = 1
            self.phi[i, 0] = 1
            self.phi[i, self.N-1] = 1
        print("Initial φ with boundary conditions:")
        print(self.phi)
    def inside_hyperspace(self, x):
        return 0.0

    def overrelaxation_method(self, f, max_iter=100, tol=1e-5):
        """
                Method to solve Poissons equation.
                Implement a relaxation (or over-relaxation) method to solve Poisson’s equation for a
        square N × N grid, with a grid spacing of h and specified charges at the grid sites (f ).
        This will be used as an independent check for your Monte Carlo results.
        """
        self.phi = self.phi.copy()
        omega = 2 / (1 + np.sin(np.pi / self.N))

        for iters in range(max_iter):
            old_phi = self.phi.copy()
            for i in range(1, self.N - 1):
                for j in range(1, self.N - 1):
                    poisson = 1/4 * (
                        self.phi[i+1, j] + self.phi[i-1, j] +
                        self.phi[i, j+1] + self.phi[i, j-1] +
                        (self.h ** 2) * f(i, j)
                    )
                    self.phi[i, j] = (1 - omega) * self.phi[i, j] + omega * poisson
            if np.max(np.abs(self.phi - old_phi)) < tol:
                print(f"Converged after {iters+1} iterations.")
                break
        print("phi after over-relaxation:")
        print(self.phi)
        return self.phi


    def plot_potential(self):
        """
        Plots the 2D grid showing potential values for phi.
        """
        plt.imshow(self.phi, origin='lower', cmap='viridis')
        plt.colorbar(label='Potential φ')
        plt.title('Solution of Poisson’s Equation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

# Task 2

# Random Walk Solver for Poisson's Equation (Green's Function)

class randwalker(MonteCarloIntegrator):
    """
    Implement a random-walk method to solve Poisson’s equation
    for a square N × N grid, using random walkers to obtain the Green’s function.
    """
    def __init__(self, N=4, walkers=10000, tol=1e-5, max_steps=20000, h=1.0, seed=12345,
                 num_samples=1000, dimensions=1, L=10):
        self.N = N
        self.walkers = walkers
        self.max_steps = max_steps
        self.L = L
        self.h = L / (N - 1)
        self.tol = tol
        self.seed = seed
        self.dimensions = dimensions
        self.num_samples = num_samples
        self.rng = default_rng(SeedSequence(seed))
        self.phi = np.zeros((N, N))

        super().__init__(
            function=self.inside_hyperspace,
            lower_bounds=[-1]*dimensions,
            upper_bounds=[1]*dimensions,
            num_samples=num_samples
        )
    def inside_hyperspace(self, x):
        return 0.0
    def laplace(self):
        for i in range(self.N):
            self.phi[0, i] = 1
            self.phi[self.N-1, i] = 1
            self.phi[i, 0] = 1
            self.phi[i, self.N-1] = 1

        print("Initial φ with boundary conditions:")
        print(self.phi)

    def random_walk_solver(self):
        self.laplace()
        N = self.N
        visit_count = np.zeros((N, N))
        for _ in range(self.walkers):
            i, j = self.rng.integers(1, N-1), self.rng.integers(1, N-1)
            for _ in range(self.max_steps):
                if self.rng.integers(0, 2):
                    i += self.rng.choice([-1, 1])
                else:
                    j += self.rng.choice([-1, 1])
                if i in [0, N-1] or j in [0, N-1]:
                    break
                visit_count[i, j] += 1

        green = (self.h ** 2) * visit_count / self.walkers
        green_solve = np.sum(green * self.phi)

        print("Visit count per grid point:\n", visit_count)
        print("\nEstimated Green’s function:\n", green)
        print(f"\nGreen’s function solution (sum φ·G): {green_solve:.6f}")

        return (green, green_solve)

    def solve_with_charge(self, tol=1e-5, max_iter=2000,
                                   boundary_func=None, f=None):
        self.laplace()
        N = self.N
        for i in range(self.N):
            self.phi[0, i] = boundary_func(0, i)
            self.phi[self.N - 1, i] = boundary_func(self.N - 1, i)
            self.phi[i, 0] = boundary_func(i, 0)
            self.phi[i, self.N - 1] = boundary_func(i, self.N - 1)
        print("Initial φ with boundary conditions:")
        print(self.phi)
        for iteration in range(1, max_iter + 1):
            old_phi = self.phi.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    self.phi[i,j] = 1/4 * (
                        self.phi[i+1,j] + self.phi[i-1,j] +
                        self.phi[i,j+1] + self.phi[i,j-1] +
                        (self.h ** 2) * f(i, j)
                        )
            max_delta = np.max(np.abs(self.phi - old_phi))
            if max_delta < self.tol:
                print(f"Converged after {iteration} iterations (Δφₘₐₓ = {max_delta:.2e}).")
                break
            print("Final φ after relaxation:")
        print(self.phi)
        return self.phi

class Charges_Boundary_Grids(randwalker):
    def __init__(self, N=32, L=10):
        self.N = N
        self.L = L
        self.h = L / (N - 1)
        self.boundaries = self.boundary_conditions()
        self.charges = self.charge_distribution()
        self.solver = overrelaxation(N=N, h=self.h)
    
    def boundary_conditions(self):
        def boundary_a(i, j):
            return 1

        def boundary_b(i, j):
            if i == 0 or i == self.N - 1:
                return 1
            elif j == 0 or j == self.N - 1:
                return -1
            return 0

        def boundary_c(i, j):
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
        """
        N = self.N
        L = self.L
        uniform_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                uniform_charge[i, j] = 10.0 / (L * L)
        print("\nCharge distribution: Uniform (10 C total)")
        print(uniform_charge)

        gradient_charge = np.zeros((N, N))
        for i in range(N):
            density = 1.0 - (i / (N - 1))
            for j in range(N):
                gradient_charge[i, j] = density
        print("\nCharge distribution: Gradient (top→bottom 1→0)")
        print(gradient_charge)

        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        exp_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dx = x[j] - L / 2
                dy = y[i] - L / 2
                r = np.sqrt(dx**2 + dy**2)
                exp_charge[i, j] = np.exp(-2000 * r)
        print("\nCharge distribution: Exponential (centered)")
        print(exp_charge)

        charge_distributions = {
            "Uniform (10C)":           uniform_charge,
            "Gradient (top→bottom)":   gradient_charge,
            "Exponential (centered)":  exp_charge
        }

        return charge_distributions

    def points_in_charge_dist(self):
        results = {}
        for bc_label, bc_func in self.boundaries.items():
            for charge_label, charge_array in self.charges.items():
                def f(i, j):
                    return charge_array[i, j]

                print(f"\nRunning: {charge_label} + {bc_label}")
                phi = self.solver.solve_with_charge(charge_func=f, boundary_func=bc_func)
                key = f"{charge_label} + {bc_label}"
                results[key] = phi
                print(f"Finished: {key}\n")
        return results

    def evaluate_points(self, green, grid_size=10):
        N = green.shape[0]
        grid_spacing = grid_size / (N - 1)
        for x, y in [(5,5),(2.5,2.5),(0.1,2.5),(0.1,0.1)]:
            i = int(round(y / grid_spacing))
            j = int(round(x / grid_spacing))
            val = green[i, j]
            print(f"G({x},{y}) into grid[{i},{j}] = {val:.4f}")


    def plot_green(self, green):
        """
        Plots the 2D grid showing Green's function values.
        """
        plt.imshow(green, origin='lower', cmap='viridis')
        plt.colorbar(label='Green\'s function')
        plt.title('Solution of Poisson’s Equation (Green\'s Function)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def plot_green_at_points(self, green, grid_size=10):
        N = green.shape[0]
        grid_spacing = grid_size / (N - 1)
        grid_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]
        plt.imshow(green, origin='lower', cmap='viridis')
        plt.colorbar(label="Green's function")
        plt.title("Green's Function at Grid Points")
        for point in grid_points:
            x_position = point[0]
            y_position = point[1]
            j_index = int(round(x_position / grid_spacing))
            i_index = int(round(y_position / grid_spacing))
            plt.plot(j_index, i_index, 'ro')
            label = f"({x_position:.1f},{y_position:.1f}) cm"
            plt.text(j_index + 0.5, i_index + 0.5, label,
                     color='white', fontsize=5)
        plt.xlabel("x (grid index) cm")
        plt.ylabel("y (grid index) cm")
        plt.show()

if __name__ == "__main__":
    solver = randwalker()
    green, green_solve = solver.random_walk_solver()
    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        print("Charge Relaxation-Green's Simulation\n")
        sim = Charges_Boundary_Grids(N=4, L=10)
        all_fields = sim.points_in_charge_dist()
        for label, phi in all_fields.items():
            print(f"\nEvaluating potential field: {label}")
            sim.evaluate_points(phi)

    MPI.Finalize()
