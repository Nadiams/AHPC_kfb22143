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

import numpy as np

# Task 1
def zero_charge(_i, _j):
    """
        Zero charge distribution
        Returns:
                0
    """
    return 0

def overrelaxation_method(f_func, n=100, spacing=1, omega=None,
                          max_iter=10000, tol=1e-5,
                          top=1, bottom=1, left=1, right=1):
    """
    Over-relaxation solver for Poisson's equation on a square grid.
    Args:
        f_func: function f(i,j) represents the charge distribution
        n: grid size (N)
        spacing: grid spacing (h)
        omega: optimum relaxation factor
        max_iter: maximum number of iterations
        tol: tolerance of the convergence (limit)
        top, bottom, left, right: boundary values
    Returns:
        phi: final potential of the grid
    """
    phi = np.zeros((n, n))
    omega = 2 / (1 + np.sin(np.pi / n)) if omega is None else omega
    phi[0, :] = top
    phi[-1, :] = bottom
    phi[:, 0] = left
    phi[:, -1] = right

    for _ in range(max_iter):
        old_phi = phi.copy()
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                f_ij = f_func(i, j)
                avg_neighbors = 0.25 * (
                    phi[i + 1, j] + phi[i - 1, j] +
                    phi[i, j + 1] + phi[i, j - 1]
                )
                phi[i, j] = omega * (spacing**2 * f_ij + avg_neighbors) + \
                            (1 - omega) * old_phi[i, j]

        max_change = np.max(np.abs(phi - old_phi))
        if max_change < tol:
            print(f"Maximum change in φ = {max_change:.4e}")
            break

    return phi
phi = overrelaxation_method(zero_charge)

# Task 5 setup
LENGTH = 10
N = 100
GRID_SPACING = LENGTH / (N - 1)

def zero_charge(_i, _j):
    """
        Zero charge distribution
        Returns:
                0
    """
    return 0

def uniform_charge(_i, _j):
    """
        Uniform charge distribution.
    """
    return 10

def gradient_charge(i, _j):
    """
        Linear gradient from  1 cm^−2 top to 0 cm^-2bottom.
    """
    return (N - 1 - i) / (N - 1)

def exp_charge(i, j):
    """
        exp −2000|r| decay from the center.
    """
    y = i * GRID_SPACING
    x = j * GRID_SPACING
    center_x, center_y = LENGTH / 2, LENGTH / 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    return np.sum(np.exp(-2000 * r))

charge_distributions = {
    "Uniform (10C)": uniform_charge,
    "Gradient (top→bottom)": gradient_charge,
    "Exponential (centered)": exp_charge
}

sample_points = [(5, 5), (2.5, 2.5), (0.1, 2.5), (0.1, 0.1)]

def coords_to_index(x_val, y_val, spacing_val):
    """Convert physical coordinates to grid indices."""
    i_idx = int(round(y_val / spacing_val))
    j_idx = int(round(x_val / spacing_val))
    return i_idx, j_idx

boundary_conditions = {
    "1st boundary condition": {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
    "2nd boundary condition": {'top': 1, 'bottom': 1, 'left': -1, 'right': -1},
    "3rd boundary condition": {'top': 2, 'bottom': 0, 'left': 2, 'right': -4}
}

results = []

for boundary_name, bc_values in boundary_conditions.items():
    for charge_name, charge_func in charge_distributions.items():
        potential = overrelaxation_method(
            f_func=charge_func,
            n=N,
            spacing=GRID_SPACING,
            top=bc_values['top'],
            bottom=bc_values['bottom'],
            left=bc_values['left'],
            right=bc_values['right']
        )

        for point in sample_points:
            i_idx, j_idx = coords_to_index(*point, GRID_SPACING)
            results.append((
                f"{boundary_name}, {charge_name}",
                point,
                potential[i_idx, j_idx]
            ))

print("Condition Location (cm) Potential (V)")
for desc, pt, pot in results:
    print(f"{desc} {pt} {pot:.4f}")