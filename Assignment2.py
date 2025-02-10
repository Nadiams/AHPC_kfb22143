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
        return f"({self._i:.2f}, {self._j:.2f}, {self._k:.2f})"

    def __add__(self, other):
        return Vector(self.i + other.i, self.j + other.j, self.k + other.k)

    def __sub__(self, other):
        return Vector(self.i - other.i, self.j - other.j, self.k - other.k)

    def norm(self):
        """Calculates the magnitude of the vector"""
        return math.sqrt(self.i**2 + self.j**2 + self.k**2)
    
class ExtendedVector(Vector):
    """ 
        Using inheritance to take previous methods used in parent class to 
        pass to child class.
    """
    def __init__(self, i, j, k):
        super().__init__(i, j, k)

    
    def dot(self, other):
        """Calculates the dot (scalar) product of two vectors"""
        return self.i * other.i + self.j * other.j + self.k * other.k

    def cross(self, other):
        """Calculates the cross product of two vectors"""
        return Vector(
            self.j * other.k - self.k * other.j,
            self.k * other.i - self.i * other.k,
            self.i * other.j - self.j * other.i
        )
    
    def __array__(self, dtype=None):
        if dtype:
            return np.array([self.i, self.j, self.k], dtype=dtype)
        else:
            return np.array([self.i, self.j, self.k])

v1 = ExtendedVector(1, 0, 0) # i
v2 = ExtendedVector(0, 1, 0) # j
v3 = ExtendedVector(1, 0, 0) # k

print("Vector v1:", v1)
print("Vector v2:", v2)
print("Magnitude of v1:", v1.norm())
print("v1 + v2:", v1 + v2)
print("v1 - v2:", v1 - v2)
print("Dot Product v1 · v2:", v1.dot(v2))
print("Cross Product v1 × v2:", v1.cross(v2))