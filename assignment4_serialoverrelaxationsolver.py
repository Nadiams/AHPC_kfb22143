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
# Task 1
# f(i,j) = 0 everywhere
def f_zero(i, j):
    return 0
def overrelaxation_method(f_func, N=100, h=1.0, omega=None, max_iter=10000, tol=1e-5, top=1, bottom=1, left=1, right=1):
    """
    Over-relaxation solver for Poisson's equation on a square grid.
    Args:
        f_func: function f(i,j) represents the charge distribution
        N: grid size
        h: grid spacing
        omega: optimum relaxation factor
        max_iter: maximum number of iterations
        tol: tolerance of the convergence (limit)
    Returns:
        phi: final potential of the grid
    """
    phi = np.zeros((N, N))
    omega = 2 / (1 + np.sin(np.pi / N))
    phi[0, :] = 1
    phi[-1, :] = 1
    phi[:, 0] = 1
    phi[:, -1] = 1

    for iteration in range(max_iter):
        old_phi = phi.copy()
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                f_ij = f_func(i, j)
                surrounding_phi_average = 1/4 * (
                    phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]
                )
                phi[i, j] = omega * (h**2 * f_ij + surrounding_phi_average) + (1 - omega) * old_phi[i, j]

        max_change = np.max(np.abs(phi - old_phi))
        if max_change < tol:
            print(f"Maximum change in φ = {max_change:.4e}")
            break

    return phi

phi = overrelaxation_method(f_zero)

# Task 5

L = 10.0
N = 100
h = L / (N - 1)

def zero_charge(i, j):
    return 0.0

def uniform_charge(i, j):
    return 10.0

def gradient_charge(i, j):
    return (N - 1 - i) / (N - 1)

def exp_charge(i, j):
    y= i * h
    x = j * h
    center_x, center_y = L / 2, L / 2
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    return np.exp(-2000 * r)

charge_distributions = {
        "Uniform (10C)": uniform_charge,
        "Gradient (top→bottom)": gradient_charge,
        "Exponential (centered)": exp_charge
        }
points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]

def coords_to_index(x, y, h):
    i = int(round(y / h))
    j = int(round(x / h))
    return i, j

boundary_conditions = {
    "1st boundary condition": {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
    "2nd boundary condition": {'top': 1, 'bottom': 1, 'left': -1, 'right': -1},
    "3rd boundary condition": {'top': 2, 'bottom': 0, 'left': 2, 'right': -4}
}

results = []

for boundary_name, boundary_values in boundary_conditions.items():
    for charge_name, charge_function in charge_distributions.items():
        potential_grid = overrelaxation_method(
            f_func=charge_function,
            N=N,
            h=h,
            top=boundary_values['top'],
            bottom=boundary_values['bottom'],
            left=boundary_values['left'],
            right=boundary_values['right']
        )

        for point in points:
            row_index, col_index = coords_to_index(*point, h)
            results.append((
                f"{boundary_name}, {charge_name}",
                point,
                potential_grid[row_index, col_index]
            ))

print("Condition Location (cm) Potential (V)")
for description, point, potential in results:
    print(f"{description} {point} {potential:.4e}")
