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
import random
#from mpi4py import MPI
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
    def __init__(self, N=4, walkers=10000, max_steps=20000, h=1.0, seed=12345,
                 num_samples=1000, dimensions=1):
        self.N = N
        self.walkers = walkers
        self.max_steps = max_steps
        self.h = h
        self.seed = seed
        self.dimensions = dimensions
        self.num_samples = num_samples
        self.rng = default_rng(SeedSequence(seed))
        self.phi = np.zeros((N, N))
        for i in range(N):
            self.phi[0, i] = 1
            self.phi[N-1, i] = 1
            self.phi[i, 0] = 1
            self.phi[i, N-1] = 1
        print("Initial φ with boundary conditions:")
        print(self.phi)
        super().__init__(
            function=self.inside_hyperspace,
            lower_bounds=[-1]*dimensions,
            upper_bounds=[1]*dimensions,
            num_samples=num_samples
        )

    def test_func(i, j):
        return i * j
    def overrelaxation_method(f):
        """
                Method to solve Poissons equation.
                Implement a relaxation (or over-relaxation) method to solve Poisson’s equation for a
        square N × N grid, with a grid spacing of h and specified charges at the grid sites (f ).
        This will be used as an independent check for your Monte Carlo results.
        """
        N = 4 # Sets the size of the grid.
        h = 1
        omega = 2 / ( 1 + np.sin(np.pi/N) )
        #f = 0
        phi = np.zeros([N, N]) # creates an array of zeros in a NxN (4x4) grid 
        for i in range(0,N): # creates a grid of these zeros
            phi[0,i] = 1 # sets the first line, [0,i] all = 1
            phi[N-1, i] = 1
            phi[i, 0] = 1
            phi[i, N-1] = 1
        print('Initial phi with boundary conditions:')
        print(phi)
        for itters in range(100): # Repeats the solver 1000 times.
            for i in range(1, N-1):
                for j in range(1, N-1): # enables the relaxer to navigate the grid and protects it from encountering neighbours outside the grid.
                   # print(i,j)
                    poisson = 1/4 * (
                        phi[i+1, j] + phi[i-1, j] +
                        phi[i, j+1] + phi[i, j-1] +
                        (h**2) * f(i, j)
                        ) # Used phi[i,j] to specifically alter each part of the grid.
                    phi[i, j] = (1 - omega) * phi[i, j] + omega * poisson
        print('phi after over-relaxation method:')
        print(phi)
        return phi
    phi=overrelaxation_method(test_func)

# Task 2

# Random Walk Solver for Poisson's Equation (Green's Function)

class randwalker(MonteCarloIntegrator):
    """
    Implement a random-walk method to solve Poisson’s equation
    for a square N × N grid, using random walkers to obtain the Green’s function.
    """
    def __init__(self, N=4, walkers=10000, max_steps=20000, h=1.0, seed=12345,
                 num_samples=1000, dimensions=1):
        self.N = N
        self.walkers = walkers
        self.max_steps = max_steps
        self.h = h
        self.seed = seed
        self.dimensions = dimensions
        self.num_samples = num_samples

        self.rng = default_rng(SeedSequence(seed))
        self.phi = np.zeros((N, N))
        for i in range(N):
            self.phi[0, i] = 1
            self.phi[N-1, i] = 1
            self.phi[i, 0] = 1
            self.phi[i, N-1] = 1
        print("Initial φ with boundary conditions:")
        print(self.phi)
        super().__init__(
            function=self.inside_hyperspace,
            lower_bounds=[-1]*dimensions,
            upper_bounds=[1]*dimensions,
            num_samples=num_samples
        )

    def random_walk_solver(self):
        N = self.N
        walkers = self.walkers
        max_steps = self.max_steps
        h = self.h
        phi = np.zeros((N, N))
        visit_count = np.zeros((N, N))
        green = (h**2) * visit_count / walkers

        for i in range(N):
            phi[0, i] = 1
            phi[N-1, i] = 1
            phi[i, 0] = 1
            phi[i, N-1] = 1

        print('Initial phi with boundary conditions:')
        print(phi)

        for _ in range(walkers):
            i, j = random.randint(1, N-2), random.randint(1, N-2)
            for _ in range(max_steps):
                if random.randint(0, 1):
                    i += random.choice([-1, 1])
                else:
                    j += random.choice([-1, 1])

                if 0 <= i < N and 0 <= j < N:
                    visit_count[i, j] += 1

                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    break

        green = (h**2) * visit_count / walkers
        green_solve = np.sum(green * phi)

        print("Visit count per grid point:\n", visit_count)
        print("\nEstimated Green’s function:\n", green)
        print(f"\nGreen’s function solution (sum φ·G): {green_solve:.6f}")

        return green, green_solve

    def overrelaxation_with_charge(N=4, h=0.01, max_iter=1000, tol=1e-5,
                                   boundary_func=None, f=None):
        phi = np.zeros((N, N))
        for i in range(N):
            phi[0,i] = boundary_func(0, i) # sets the first line, [0,i] all = 1
            phi[N-1, i] = boundary_func(N-1,  i)
            phi[i, 0] = boundary_func(i, 0)
            phi[i, N-1] = boundary_func(i, N-1)
        print("Initial φ with boundary conditions:")
        print(phi)
        for iteration in range(1, max_iter + 1):
            old_phi = phi.copy()
            for i in range(1, N-1):
                for j in range(1, N-1):
                    phi[i,j] = 1/4 * ( phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]) # Used phi[i,j] to specifically alter each part of the grid.
            max_delta = np.max(np.abs(phi - old_phi))
            if max_delta < tol:
                print(f"Converged after {iteration} iterations (Δφₘₐₓ = {max_delta:.2e}).")
            break
            print("Final φ after relaxation:")
        print(phi)
        return phi
    
    N = 4
    L = 0.1 # length of the grid
    h = L / (N - 1) # spacing
    def boundary_a(i, j):
        return 1  # All edges +1 V
    
    def boundary_b(i, j):
        if i == 0 or i == N-1:
            return 1  # Top and bottom +1 V
        elif j == 0 or j == N-1:
            return -1  # Left and right -1 V
        return 0
    
    def boundary_c(i, j):
        if i == 0:
            return 2  # Top +2 V
        elif i == N-1:
            return 0  # Bottom 0 V
        elif j == 0:
            return 2  # Left +2 V
        elif j == N-1:
            return -4  # Right -4 V
        return 0
    
    boundary_conditions = [("a", boundary_a), ("b", boundary_b), ("c", boundary_c)]
    results = {}
    
    for label, func in boundary_conditions:
        results[f"phi_{label}"] = overrelaxation_with_charge(N, boundary_func=func)

    def evaluate_green_points(green, grid_size=10):
        N = green.shape[0]
        grid_spacing = grid_size / (N - 1)
        grid_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]
        for x, y in grid_points:
            i = int(round(y / grid_spacing))
            j = int(round(x / grid_spacing))
            value = green[i, j]
            print(f"Green's function at ({x} cm, {y} cm) grid[{i}, {j}] = {value:.4f}")

    def charge_distributions(N, L):
        """
        """
    
        #  10 C charge, spread uniformly over the whole grid
        uniform_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                uniform_charge[i, j] = 10.0 / (L * L)
        print("\nCharge distribution: Uniform (10 C total)")
        print(uniform_charge)
        
        # A uniform charge gradient from the top the the bottom of the grid, where the charge
        # density at the top of the grid is 1 Cm−2 and 0 at the bottom sites
        gradient_charge = np.zeros((N, N))
        for i in range(N):
            density = 1.0 - (i / (N - 1))
            for j in range(N):
                gradient_charge[i, j] = density
        print("\nCharge distribution: Gradient (top→bottom 1→0)")
        print(gradient_charge)
        
        # An exponentially decaying charge distribution, exp −2000|r|, placed at the centre of
        # the grid.
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        exp_charge = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dx = x[j] - L/2
                dy = y[i] - L/2
                r = np.sqrt(dx**2 + dy**2)
                exp_charge[i, j] = np.exp(-2000 * r)
        print("\nCharge distribution: Exponential (centered)")
        print(exp_charge)
        charge_distributions = {
            "Uniform (10C)":           uniform_charge,
            "Gradient (top→bottom)":   gradient_charge,
            "Exponential (centered)":  exp_charge
        }
        points = {
            "(5,5) cm":    (16, 16),
            "(2.5,2.5) cm":(8,   8),
            "(0.1,2.5) cm":(8,   0),
            "(0.1,0.1) cm":(0,   0)
        }
        boundary_conditions = {
            "Case A": boundary_a,
            "Case B": boundary_b,
            "Case C":        boundary_c
        }
charge_distributions(N, L)

    def plot_potential(phi):
        """
        Plots the 2D grid showing potential values for phi.
        """
        plt.imshow(phi, origin='lower', cmap='viridis')
        plt.colorbar(label='Potential φ')
        plt.title('Solution of Poisson’s Equation')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def plot_green(green):
        """
        Plots the 2D grid showing Green's function values.
        """
        plt.imshow(green, origin='lower', cmap='viridis')
        plt.colorbar(label='Green\'s function')
        plt.title('Solution of Poisson’s Equation (Green\'s Function)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    green = green / np.max(green) 
    plot_potential(phi)
    plot_green(green)

    def plot_green_at_points(green, grid_size=10):
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
    
plot_green_at_points(green)

if __name__ == "__main__":
    solver = RandWalker()
    green, green_solve = solver.random_walk_solver()



