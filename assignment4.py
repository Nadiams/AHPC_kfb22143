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
    #start_point = (1, 1) # check
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
    green_solve = np.sum(green * phi)
    
    print('Visit count for each grid point:')
    print(visit_count)
    
    print('Estimated Green\'s function for charge density:')
    print(green)
    print('green solve', green_solve)
    return green, green_solve
green = random_walk_solver()[0]
green_solve = random_walk_solver()[1]


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
    grid_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]
    for point in grid_points:
        x_position = point[0]
        y_position = point[1]
        print(x_position)
        print(y_position)
        j_index = int(round(x_position / grid_spacing))  # columns, convert integers into grid indices
        i_index = int(round(y_position / grid_spacing))  # rows
        point_number = green[i_index, j_index]
        print(f"Green's function at ({x_position} cm, {y_position} cm) "
              f"grid[{i_index}, {j_index}] = {point_number:.4f}")
evaluate_green_points(green)

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

# Task 4

def overrelaxation_with_charge(N=32, h=0.01, max_iter=1000, tol=1e-5,
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

N = 32
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

phi_a = overrelaxation_with_charge(N, boundary_func=boundary_a)
phi_b = overrelaxation_with_charge(N, boundary_func=boundary_b)
phi_c = overrelaxation_with_charge(N, boundary_func=boundary_c)

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
