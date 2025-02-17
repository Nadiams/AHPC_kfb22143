#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licence
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

    def trianglearea(self, v2, v3):
        """
            Area of a triangle
            Args:
                self = v1, v2, v3, i.e. Vector objects which represent the
                vertices.
            Returns:
                Area.
        """
        side1 = v2 - self
        side2 = v3 - self
        crossproduct = side1.cross(side2)
        return 0.5 * crossproduct.norm()
    
    def angleproduct(self, angle):
        """
        Args:
            self=v1, angle
        Returns:
            cos_angle
        """
        dot_product = self.dot(angle)
        norm_product = self.norm() * angle.norm()
        cos_angle = dot_product / norm_product
        return math.degrees(math.acos(cos_angle))
    
    def triangleangles(self, v2, v3):
        """
            Args:
                self=v1, v2 and v3
    
            Returns:
                Angles of the triangles.
        """
        side1 = v2 - self
        side2 = v3 - self
        side3 = v3 - v2

        angle1 = side1.angleproduct(side2)
        angle2 = side2.angleproduct(side3)
        angle3 = side3.angleproduct(side1)

        return angle1, angle2, angle3

class SphericalPolarVector(Vector):
    """
    Uses inheritance to take previous methods used in parent class to pass to child class.
    """
    def __init__(self, r, theta, phi):
        """
            Initializes the vector with spherical-polar r, theta and phi components
            to convert from Cartesian i, j and k components.
            Args: 
                i, j and k
            Returns: 
                r, theta and phi.
        """
        self._i = r * math.sin(math.radians(theta)) * math.cos(math.radians(phi))  # x-component
        self._j = r * math.sin(math.radians(theta)) * math.sin(math.radians(phi))  # y-component
        self._k = r * math.cos(math.radians(theta))  # z-component

    def __add__(self, other):
        """
            Args: vector components
            Returns: Cartesian vector
            Adds vectors.
        """
        return SphericalPolarVector(self._i + other._i, self._j + other._j, self._k + other._k)
    
    def __str__(self):
        """
            String representation of the vector in spherical-polar form.
        """
        r = np.sqrt(self._i**2 + self._j**2 + self._k**2)
        theta = math.acos(self._k / r)
        theta = math.degrees(theta)
        phi = math.atan2(self._j, self._i)
        phi = math.degrees(phi)
        if phi < 0:
            phi += 360
        return f"(r={r:.2f}, theta={theta:.2f}, phi={phi:.2f})"

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
print("q1", q1)

# 4 Triangles with Spherical-Polar Points
a1 = SphericalPolarVector(0, 0, 0)
a2 = SphericalPolarVector(1, 0, 0)
a3 = SphericalPolarVector(1, 90, 0)

b1 = SphericalPolarVector(1, 0, 0)
b2 = SphericalPolarVector(1, 90, 0)
b3 = SphericalPolarVector(1, 90, 180)

c1 = SphericalPolarVector(0, 0, 0)
c2 = SphericalPolarVector(2, 0, 0)
c3 = SphericalPolarVector(2, 90, 0)

d1 = SphericalPolarVector(1, 90, 0)
d2 = SphericalPolarVector(1, 90, 180)
d3 = SphericalPolarVector(1, 90, 270)
d3x2 = d3 + d3
print("addition check", d3x2)
print("d3", d3)
print(f"(checking:{d1})")

# Task 3(a)



# Task 3(b)
print("Cartestian Vector Form")
angles1 = vv1.triangleangles(vv2, vv3)
print(f"Cartesian Triangle 1: 1st Angle: {angles1[0]:.2f}°, "
      f"2nd Angle: {angles1[1]:.2f}°,"
      f"3rd Angle: {angles1[2]:.2f}°"
      )

angles2 = u1.triangleangles(u2, u3)
print(f"Cartesian Triangle 2: 1st Angle: {angles2[0]:.2f}°, "
      f"2nd Angle: {angles2[1]:.2f}°,"
      f"3rd Angle: {angles2[2]:.2f}°"
      )

angles3 = p1.triangleangles(p2, p3)
print(f"Cartesian Triangle 3: 1st Angle: {angles3[0]:.2f}°, "
      f"2nd Angle: {angles3[1]:.2f}°,"
      f"3rd Angle: {angles3[2]:.2f}°"
      )

angles4 = q1.triangleangles(q2, q3)
print(f"Cartesian Triangle 4: 1st Angle: {angles4[0]:.2f}°, "
      f"2nd Angle: {angles4[1]:.2f}°,"
      f"3rd Angle: {angles4[2]:.2f}°"
      )
print()
# Task 3 (c)
print("Spherical-Polar Vector Form")

cartesian_sub = vv1 - vv2

cartesian_add = vv1 + vv2

cartesian_mag = vv1.norm()
cartesian_cross = vv1.cross(vv2)

cartesian_dot = vv1.dot(vv2)

print()
print("Cartestian Vector Form")
print("Cartesian Vector 1:", vv1)
print("Cartesian Vector 2:", vv2)
print(f"Cartesian Subtraction: {cartesian_sub}")
print(f"Cartesian Addition: {cartesian_add}")
print(f"Cartesian Magnitude of v1: {vv1.norm()}")
print(f"Cartesian Dot Product: {vv1.dot(vv2)}")
print(f"Cartesian Cross Product: {vv1.cross(vv2)}")

print()
#g1 = Vector(1, 0, 0)
#g2 = Vector(1, 0, 1)
#cross_check = (g1.cross(g2))
#mag_cross = cross_check.norm()
#print(f"Cartesian Cross Product: {0.5 * mag_cross}")