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
    
    def trianglearea(self, v2, v3):
        """
            Area of a triangle
            Args:
                v1, v2, v3: Vector objects representing the vertices.
            Returns:
                Area.
        """
        side1 = v2 - self
        side2 = v3 - self
        cross_product = side1.cross(side2)
        return 0.5 * cross_product.norm()
    
    def angleproduct(self, theta):
        """
        Args:
            theta
        Returns:
            cos_angle
        """
        dot_product = self.dot(theta)
        norm_product = self.norm() * theta.norm()
        cos_angle = dot_product / norm_product
        return math.degrees(math.acos(cos_angle))
    
    def triangleangles(self, v2, v3):
        """
            Args:
                self, v2 and v3
    
            Returns:
                Angles of the triangles.
        """
        self._v1 = v1
        self._v2 = v2
        self._v3 = v3

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

#v1 = Vector(1, 0, 0)  # i=1, j=0, k=0
#v2 = Vector(0, 1, 0)  # i=0, j=1, k=0
#v3 = Vector(0, 0, 1)  # i=0, j=0, k=1

# 4 Triangles with Cartesian Points

vv1 = Vector(0, 0, 0)  # i=1, j=0, k=0
vv2 = Vector(0, 1, 0)  # i=0, j=1, k=0
vv3 = Vector(0, 0, 1)  # i=0, j=0, k=1

u1 = Vector(-1,-1,-1)
u2 = Vector(0,-1,-1)
u3 = Vector(-1,0,-1)

p1 = Vector(1, 0, 0)
p2 = Vector(0, 0, 1)
p3 = Vector(0, 0, 0)

q1 = Vector(0,0,0)
q2 = Vector(1,-1,0)
q3 = Vector(0,0,1)

# Task 3(a)

area1 = vv1.trianglearea(vv2, vv3)
print("Cartesian Triangle 1 Area: ", area1)
area2 = u1.trianglearea(u2, u3)
print("Cartesian Triangle 2 Area: ", area2)
area3 = p1.trianglearea(p2, p3)
print("Cartesian Triangle 3 Area: ", area3)
area4 = q1.trianglearea(q2, q3)
print("Cartesian Triangle 4 Area: ", area4)

# Task 3(b)

# Task 1 and 2

sphericalpolar_vector1 = vv1.cartesian_to_spherical()
sphericalpolar_vector2 = vv2.cartesian_to_spherical()
sphericalpolar_vector3 = vv3.cartesian_to_spherical()

cartesian_sub = vv1 - vv2
sphericalpolar_sub = cartesian_sub.cartesian_to_spherical()

cartesian_add = vv1 + vv2
sphericalpolar_add = cartesian_add.cartesian_to_spherical()

cartesian_mag = vv1.norm()
# Spherical-Polar Magnitude is the r-component.

cartesian_cross = vv1.cross(vv2)
sphericalpolar_cross = cartesian_cross.cartesian_to_spherical()

cartesian_dot = vv1.dot(vv2)
# sphericalpolar_dot = cartesian_dot.cartesian_to_spherical()
sphericalpolar_dot = sphericalpolar_vector1.dot(sphericalpolar_vector2)
sphericalpolar_dot = round(sphericalpolar_dot, 10)

print("Cartestian Vector Form")
print("Cartesian Vector 1:", vv1)
print("Cartesian Vector 2:", vv2)
print(f"Cartesian Subtraction: {cartesian_sub}")
print(f"Cartesian Addition: {cartesian_add}")
print(f"Cartesian Magnitude of v1: {vv1.norm()}")
print(f"Cartesian Dot Product: {vv1.dot(vv2)}")
print(f"Cartesian Cross Product: {vv1.cross(vv2)}")
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