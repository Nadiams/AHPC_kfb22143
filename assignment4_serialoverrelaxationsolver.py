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

# f(i,j) = 0 everywhere
def f_zero(i, j):
    return 0
def overrelaxation_method(f_func, N=4, h=1.0, omega=None, max_iter=1000, tol=1e-5):
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
    if omega is None:
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
                avg_neighbors = 1/4 * (
                    phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]
                )
                phi[i, j] = omega * (h**2 * f_ij + avg_neighbors) + (1 - omega) * old_phi[i, j]

        max_change = np.max(np.abs(phi - old_phi))
        if max_change < tol:
            print(f"Converged after {iteration+1} iterations with max Δφ = {max_change:.2e}")
            break

    return phi

phi = overrelaxation_method(f_zero)