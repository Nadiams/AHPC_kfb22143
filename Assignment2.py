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
        """
            Initializes vector with i, j and k components.
        """
        self._i = i
        self._j = j
        self._k = k

    def __str__(self):
        """
            String representation of Cartesian vector.
        """
        return f"({self._i:.2f}, {self._j:.2f}, {self._k:.2f})"

    def __add__(self, other):
        """
            Args: vector components
            Returns: Cartesian vector
            Adds vectors.
        """
        return Vector(self._i + other._i, self._j + other._j, self._k + other._k)

    def __sub__(self, other):
        """
            Args: vector components
            Returns: Cartesian vector
            Subtracts vectors.
        """
        return Vector(self._i - other._i, self._j - other._j, self._k - other._k)

    def norm(self):
        """
            Calculates the magnitude of the vector.
        """
        return math.sqrt(self._i**2 + self._j**2 + self._k**2)

    def dot(self, other):
        """
            Calculates the dot product of two vectors.
        """
        return self._i * other._i + self._j * other._j + self._k * other._k

    def cross(self, other):
        """
            Calculates the cross product of two vectors.
        """
        return Vector(
            self._j * other._k - self._k * other._j,
            self._k * other._i - self._i * other._k,
            self._i * other._j - self._j * other._i
        )

    def __array__(self, dtype=None):
        """
            Creates an array to contain the vector components, to display.
        """
        if dtype:
            return np.array([self._i, self._j, self._k], dtype=dtype)
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
    Uses inheritance to take previous methods used in parent class to pass to child class.
    """
    def __init__(self, r, theta, phi):
        """
            Initialises the vector with r, theta and phi components to convert
            from cartesian i, j and k components.
            Args: 
                i, j and k
            Returns: 
                r, theta and phi.
        """
        self._r = r
        self._theta = theta
        self._phi = phi

        i = r * math.sin(theta) * math.cos(phi)  # x-component
        j = r * math.sin(theta) * math.sin(phi)  # y-component
        k = r * math.cos(theta)  # z-component

        super().__init__(i, j, k)

    def __str__(self):
        """
        String representation of the vector in spherical-polar form.
        """
        return (
            f"(r={self._r:.2f}, "
            f"θ={math.degrees(self._theta):.2f}°, "
            f"φ={math.degrees(self._phi):.2f}°)"
    )
v1 = Vector(1, 0, 0)  # i=1, j=0, k=0
v2 = Vector(0, 1, 0)  # i=0, j=1, k=0
v3 = Vector(0, 0, 1)  # i=0, j=0, k=1
sphericalpolar_vector1 = v1.cartesian_to_spherical()
sphericalpolar_vector2 = v2.cartesian_to_spherical()
sphericalpolar_vector3 = v3.cartesian_to_spherical()

cartesian_sub = v1 - v2
sphericalpolar_sub = cartesian_sub.cartesian_to_spherical()

cartesian_add = v1 + v2
sphericalpolar_add = cartesian_add.cartesian_to_spherical()

cartesian_mag = v1.norm()
# Spherical-Polar Magnitude is the r-component.

cartesian_cross = v1.cross(v2)
sphericalpolar_cross = cartesian_cross.cartesian_to_spherical()

cartesian_dot = v1.dot(v2)
# sphericalpolar_dot = cartesian_dot.cartesian_to_spherical()
sphericalpolar_dot = sphericalpolar_vector1.dot(sphericalpolar_vector2)
sphericalpolar_dot = round(sphericalpolar_dot, 10)

print("Cartestian Vector Form")
print("Cartesian Vector 1:", v1)
print("Cartesian Vector 2:", v2)
print(f"Cartesian Subtraction: {cartesian_sub}")
print(f"Cartesian Addition: {cartesian_add}")
print(f"Cartesian Magnitude of v1: {v1.norm()}")
print(f"Cartesian Dot Product: {v1.dot(v2)}")
print(f"Cartesian Cross Product: {v1.cross(v2)}")
print()
print("Spherical-Polar Vector Form")
print(f"Spherical-Polar Vector 1: {sphericalpolar_vector1}")
print(f"Spherical-Polar Vector 2: {sphericalpolar_vector2}")
print(f"Spherical-Polar Subtraction: {sphericalpolar_sub}")
print(f"Spherical-Polar Addition: {sphericalpolar_add}")
print(f"Spherical-Polar Dot Product: {sphericalpolar_dot}")
print(f"Spherical-Polar Cross Product: {sphericalpolar_cross}")
print(f"Magnitude of sphericalpolar_vector1: {sphericalpolar_vector1._r}")  # It is the r component.
print(f"Magnitude of sphericalpolar_vector2: {sphericalpolar_vector2._r}")  # It is the r component.

# Task 3

# 4 Triangles with Cartesian Points

vv1 = (1, 0, 0)  # i=1, j=0, k=0
vv2 = (0, 1, 0)  # i=0, j=1, k=0
vv3 = (0, 0, 1)  # i=0, j=0, k=1

t1 = (vv1, vv2, vv3)

u1 = (-1,-1,-1)
u2 = (0,-1,-1)
u3 = (-1,0,-1)


t2 = (u1, u2, u3)

p1 = (1,0,0)
p2 = (0,0,1)
p3 = (0,0,0)

t3 = (p1, p2, p3)

q1 = (0,0,0)
q2 = (1,-1,0)
q3 = (0,0,1)

t4 = (q1, q2, q3)

print("t1", t1)
print("t2", t2)
print("t3", t3)
print("t4", t4)

Area1 = 0.5 * vv1.cross(vv2).norm()

print("Area1")





