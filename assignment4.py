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

def overrelaxation_method():
    """
            Method to solve Poissons equation.
            Implement a relaxation (or over-relaxation) method to solve Poisson’s equation for a
    square N × N grid, with a grid spacing of h and specified charges at the grid sites (f ).
    This will be used as an independent check for your Monte Carlo results.
    """
    N = 32 # Sets the size of the grid.
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
                phi[i,j] = 1/4 * ( phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1]) # Used phi[i,j] to specifically alter each part of the grid.
    print('phi after over-relaxation method:')
    print(phi)
    return phi

phi=overrelaxation_method()

# Task 2

# Random Walk Solver for Poisson's Equation (Green's Function)
def random_walk_solver():
    """
    Implement a random-walk method to solve Poisson’s equation
    for a square N × N grid, using random walkers to obtain the Green’s function.
    """

    N = 32 # Sets the size of the grid
    phi = np.zeros([N, N])  # Creates an array of zeros in a NxN (4x4) grid
    walkers = 10000
    max_steps = 20000
    h = 1.0  # Grid spacing
    start_point = (1, 1)
    visit_count = np.zeros((N, N))# To track the number of visits to each grid point
    green = (h**2) * visit_count / walkers
    for i in range(N):
        phi[0, i] = 1        # Top boundary
        phi[N-1, i] = 1      # Bottom boundary
        phi[i, 0] = 1        # Left boundary
        phi[i, N-1] = 1      # Right boundary
    print('Initial phi with boundary conditions:')
    print(phi)

    for walker in range(walkers):    # Random walk for each walker
        i = random.randint(1, N-2)  # Random row inside the grid
        j = random.randint(1, N-2)  # Random column inside the grid
        current_position = (i, j)  # Initialise walker position
        for step in range(max_steps):  # Walk with a maximum limit on the number of steps
            i, j = current_position  # Current position of the walker
            if random.randint(0, 1):  # Pick row to move to
                if i > 0:
                    current_position = (i-1, j)
                else:
                    current_position = (i+1, j)
            else:  # Pick column to move to
                if j > 0:
                    current_position = (i, j-1)
                else:
                    current_position = (i, j+1)
    
            visit_count[current_position] += 1
    
            if current_position[0] == 0: # Once an edge is reached, need to stop the function continuing.
                break
            elif current_position[0] == N-1:
                break
            elif current_position[1] == 0:
                break
            elif current_position[1] == N-1:
                break

    green = (h**2) * visit_count / walkers
    
    print('Visit count for each grid point:')
    print(visit_count)
    
    print('Estimated Green\'s function for charge density:')
    print(green)
    return(green)
green = random_walk_solver()

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

def evaluate_green_points(green, grid_size=10):
    N = green.shape[0]
    grid_spacing = grid_size / (N - 1)
    physical_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]
    for point in physical_points:
        x_position = point[0]
        y_position = point[1]
        print(x_position)
        print(y_position)
        j_index = int(round(x_position / grid_spacing))  # colmns, convert integers into grid indices
        i_index = int(round(y_position / grid_spacing))  # rows
        point_number = green[i_index, j_index]
        print(f"Green's function at ({x_position} cm, {y_position} cm) "
              f"grid[{i_index}, {j_index}] = {point_number:.6f}")
evaluate_green_points(green)