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
        
            
    def cartesian_to_spherical(self):
        """
            This is where the conversion occurs.
        """
        r = self.norm()
        theta = math.acos(self._k / r) if r != 0 else 0
        phi = math.atan2(self._j, self._i)

        return SphericalPolarVector(r, theta, phi)
    
class SphericalPolarVector(Vector):
    """ 
            Using inheritance to take previous methods used in parent class to 
            pass to child class.
    """
    def __init__(self, r, theta, phi):
        self._r = r
        self._theta = theta
        self._phi = phi
        
        i = r * math.sin(theta) * math.cos(phi)  # x-component
        j = r * math.sin(theta) * math.sin(phi)  # y-component
        k = r * math.cos(theta)                  # z-component
        
        super().__init__(i, j, k)

    def __str__(self):
        return f"(r={self._r:.2f}, θ={math.degrees(self._theta):.2f}°, φ={math.degrees(self._phi):.2f}°)"

v1 = Vector(1, 0, 0) # i=1, j=0, k=0
v2 = Vector(0, 1, 0) # i=0, j=1, k=0
v3 = Vector(0, 0, 1) # i=0, j=0, k=1
sphericalpolar_vector1 = v1.cartesian_to_spherical()
sphericalpolar_vector2 = v2.cartesian_to_spherical()
sphericalpolar_vector3 = v3.cartesian_to_spherical()

cartesian_sub = v1 - v2
sphericalpolar_sub = cartesian_sub.cartesian_to_spherical()

cartesian_add = v1 + v2
sphericalpolar_add = cartesian_add.cartesian_to_spherical()

cartesian_mag = v1.norm()
#sphericalpolar_mag = cartesian_mag.cartesian_to_spherical()

cartesian_cross = v1.cross(v2)
sphericalpolar_cross = cartesian_cross.cartesian_to_spherical()

cartesian_dot = v1.dot(v2)
#sphericalpolar_dot = cartesian_dot.cartesian_to_spherical()


#print("sphericalpolar_dot", sphericalpolar_dot)
print("sphericalpolar_cross", sphericalpolar_cross)
#print("sphericalpolar_mag", sphericalpolar_mag)
print(f"Spherical-Polar Addition (v1 + v2): {sphericalpolar_add}")
print()


print(f"Spherical-Polar Vector (v1): {sphericalpolar_vector1}")
print(f"Spherical-Polar Vector (v2): {sphericalpolar_vector2}")
print(f"Spherical-Polar Subtraction (v1 - v2): {sphericalpolar_sub}")
print(f"Cartesian Subtratction (v1 - v2) {cartesian_sub}")
print("Vector v1:", v1)
print("Vector v2:", v2)
print("Magnitude of v1:", v1.norm())
print("v1 + v2:", v1 + v2)
print("v1 - v2:", v1 - v2)
print("Dot Product v1 · v2:", v1.dot(v2))
print("Cross Product v1 × v2:", v1.cross(v2))
print("sphericalv1 - sphericalv2:", sphericalpolar_vector1 - sphericalpolar_vector2)



