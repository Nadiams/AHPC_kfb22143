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
        self._i = i
        self._j = j
        self._k = k

    def __str__(self):
        return f"({self._i:.2f}, {self._j:.2f}, {self._k:.2f})"

    def __add__(self, other):
        return Vector(self._i + other._i, self._j + other._j, self._k + other._k)

    def __sub__(self, other):
        return Vector(self._i - other._i, self._j - other._j, self._k - other._k)

    def norm(self):
        """Calculates the magnitude of the vector"""
        return math.sqrt(self._i**2 + self._j**2 + self._k**2)
    
    def dot(self, other):
        return self._i * other._i + self._j * other._j + self._k * other._k

    def cross(self, other):
        return Vector(
            self._j * other._k - self._k * other._j,
            self._k * other._i - self._i * other._k,
            self._i * other._j - self._j * other._i
        )

    def __array__(self, dtype=None):
        if dtype:
            return np.array([self._i, self._j, self._k], dtype=dtype)
        else:
            return np.array([self._i, self._j, self._k])
    
class SphericalPolarVector(Vector):
    """ 
            Using inheritance to take previous methods used in parent class to 
            pass to child class.
    """
    def __init__(self, r, theta, phi):

        i = r * math.sin(theta) * math.cos(phi)  # x-component
        j = r * math.sin(theta) * math.sin(phi)  # y-component
        k = r * math.cos(theta)                  # z-component
        
        super().__init__(i, j, k)

        self._r = r
        self._theta = theta
        self._phi = phi

    def __str__(self):
        return f"(r={self._r:.2f}, θ={math.degrees(self._theta):.2f}°, φ={math.degrees(self._phi):.2f}°)"

    def cartesian(self, vector):
        """Converts Cartesian coordinates vectors to Spherical-Polar coordinates."""
        r = math.sqrt(vector._i**2 + vector._j**2 + vector._k**2)
        theta = math.acos(vector._k / r) if r != 0 else 0
        phi = math.atan2(vector._j, vector._i)

        return SphericalPolarVector(r, theta, phi) 
    
# Converts cartesian coordinates to spherical-polar coordinates
cartesian_vector = Vector(1, 1, 1)
sphericalpolar_vector = SphericalPolarVector(0, 0, 0) 
sphericalpolar_vector = sphericalpolar_vector.cartesian(cartesian_vector) 

print(f"Spherical-Polar Vector: {sphericalpolar_vector}")

print(f"Spherical-Polar Vector as a NumPy array: {np.array(sphericalpolar_vector)}")

v1 = Vector(1, 0, 0) # i
v2 = Vector(0, 1, 0) # j
v3 = Vector(1, 0, 0) # k

print("Vector v1:", v1)
print("Vector v2:", v2)
print("Magnitude of v1:", v1.norm())
print("v1 + v2:", v1 + v2)
print("v1 - v2:", v1 - v2)
print("Dot Product v1 · v2:", v1.dot(v2))
print("Cross Product v1 × v2:", v1.cross(v2))