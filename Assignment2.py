#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:01:27 2025

@author: nadia
"""

import math
import numpy as np

class Vector:
    """
        Vector class for Cartesian vectors in 3D space.
    """
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k

    def __str__(self):
        return f"({self.i}, {self.j}, {self.k})"

    def __add__(self, other):
        return Vector(self.i + other.i, self.j + other.j, self.k + other.k)

    def __sub__(self, other):
        return Vector(self.i - other.i, self.j - other.j, self.k - other.k)

    def norm(self):
        """Computes magnitude of the vector"""
        return math.sqrt(self.i**2 + self.j**2 + self.k**2)

    def __array__(self, dtype=None):
        if dtype:
            return np.array([self.i, self.j, self.k], dtype=dtype)
        else:
            return np.array([self.i, self.j, self.k])

v1 = Vector(1, 0, 0)
v2 = Vector(0, 1, 0)

print("Vector v1:", v1)
print("Vector v2:", v2)
print("Magnitude of v1:", v1.norm())
print("v1 + v2:", v1 + v2)
print("v1 - v2:", v1 - v2)