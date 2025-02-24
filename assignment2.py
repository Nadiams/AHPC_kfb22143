#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is based on an OOP example provided by Dr. Benjamin Hourahine in
# PH510. Modifications made by kfb22143 - Licensed under the MIT License.
# See LICENSE file for details.
"""
Created on Mon Feb 10 14:01:27 2025

@author: nadia
"""
import copy
import math
import numpy as np

class Vector:
    """
        Vector class for Cartesian vectors in 3D space.
    """
    def __init__(self, i, j, k):
        """
            Initializes vector with i, j and k components.
            Args:
                i: The x-component of the vector.
                j: The y-component of the vector.
                k: The z-component of the vector.
            Returns:
                initialises components.
        """
        self._i = i
        self._j = j
        self._k = k

    def get_i(self):
        """
        Getter for retrieving the data.
        Args:
            None.
        Returns:
            The x-component of the vector (i).
        """
        return self._i

    def get_j(self):
        """
        Getter for retrieving the data.
        Args:
            None.
        Returns:
           The y-component of the vector (j).
        """
        return self._j

    def get_k(self):
        """
        Getter for retrieving the data.
        Args:
            None.
        Returns:
            The z-component of the vector (k).
        """
        return self._k

    def __str__(self):
        """
            Returns a string representation of the vector.
            Returns:
                str: The vector in (i, j, k) format.
        """
        return f"({self._i:.2f}, {self._j:.2f}, {self._k:.2f})"

    def __add__(self, other):
        """
            Adds compoents of a vector together.
            Args: vector components.
            Returns: sum of Cartesian vectors as a new vector.
            Adds vectors.
        """
        temporary = copy.deepcopy(self)
        temporary._i += other._i
        temporary._j += other._j
        temporary._k += other._k
        return temporary

    def __sub__(self, other):
        """
        Subtracts comonents of one vector from the other.
            Args: vector components.
            Returns: New cartesian vector.
        """
        temporary = copy.deepcopy(self)
        temporary._i -= other._i
        temporary._j -= other._j
        temporary._k -= other._k
        return temporary

    def norm(self):
        """
            Calculates the magnitude of the vector.
        """
        return math.sqrt(self.get_i()**2 + self.get_j()**2 + self.get_k()**2)

    def dot(self, other):
        """
            Calculates the dot product of two vectors.
        """
        dotproduct = (
			self.get_i() * other.get_i() + self._j * other.get_j()
			+ self.get_k() * other.get_k()
		)
        return 0.0 if abs(dotproduct) < 1e-10 else dotproduct

    def cross(self, other):
        """
            Calculates the cross product of two vectors.
        """
        return Vector(
            self.get_j() * other.get_k() - self.get_k() * other.get_j(),
            self.get_k() * other.get_i() - self.get_i() * other.get_k(),
            self.get_i() * other.get_j() - self.get_j() * other.get_i()
        )

    def __array__(self, dtype=None):
        """
            Creates an array to contain the vector components, to display
            by converting the vector into a NumPy array.
        """
        if dtype:
            return np.array([self.get_i(), self.get_j(), self.get_k()], dtype=dtype)
        return np.array([self.get_i(), self.get_j(), self.get_k()])

    def trianglearea(self, v2, v3):
        """
            Calculates the area of a triangle from three vertices (vectors).
            Args:
                self (v1), v2, v3
                i.e. Vector objects which represent the vertices.
            Returns:
                Area of the triangle.
        """
        side1 = v2 - self
        side2 = v3 - self
        crossproduct = side1.cross(side2)
        return 0.5 * crossproduct.norm()

    def angleproduct(self, angle):
        """
            Calculates the angle (degrees) between two vectors using the dot product.
            Args:
                self (v1), angle
            Returns:
                The angle in degrees.
        """
        dot_product = self.dot(angle)
        norm_product = self.norm() * angle.norm()
        cos_angle = dot_product / norm_product
        return math.degrees(math.acos(cos_angle))

    def triangleangles(self, v2, v3):
        """
            Calculates the angle (degrees) of the triangles 
            across three vertices (vectors).
            Args:
                v2 (Vector): Second vertex of the triangle.
                v3 (Vector): Third vertex of the triangle.
            Returns:
                The three angles of the triangle in degrees.
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
    A class which represents a vector in spherical polar coordinates.
    """
    def __init__(self, r, theta, phi):
        """
            Initializes the vector with spherical-polar coordinates:
            r, theta and phi components to convert to Cartesian i, j and k 
            components.
            Args:
                r: Radius.
                theta: Polar angle in degrees.
                phi: Azimuthal angle in degrees.
        """
        self._i = r * math.sin(math.radians(theta)) * math.cos(math.radians(phi))  # x-component
        self._j = r * math.sin(math.radians(theta)) * math.sin(math.radians(phi))  # y-component
        self._k = r * math.cos(math.radians(theta))  # z-component
        super().__init__(self._i, self._j, self._k)
    def __str__(self):
        """
            Returns a string representation of the vector in spherical-polar 
            form.
            Returns:
                str: The vector in (r, θ, φ) format.
        """
        r = np.sqrt(self.get_i()**2 + self.get_j()**2 + self.get_k()**2)
        theta = math.acos(self.get_k() / r)
        theta = math.degrees(theta)
        phi = math.atan2(self.get_j(), self.get_i())
        phi = math.degrees(phi)
        if phi < 0:
            phi += 360
        return f"(r={r:.2f}, θ={theta:.2f}°, φ={phi:.2f}°)"

# 4 Triangles with Cartesian Points
vv1 = Vector(0, 0, 0)  # i=0, j=0, k=0
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

print("Task 1: Cartestian Vector Form")
print("Vector 1:", vv1)
print("Vector 2:", vv2)
print(f"Subtraction: {vv1 - vv2}")
print(f"Addition: {vv1 + vv2}")
print(f"Magnitude of Vector 1: {vv1.norm()}")
print(f"Dot Product: {vv1.dot(vv2)}")
print(f"Cross Product: {vv1.cross(vv2)}")
print()

# Task 2

print("Task 2: Spherical-Polar Vector Form")
print(f"Spherical-Polar Vector 1: {b3}")
print(f"Spherical-Polar Vector 2: {d3}")
print(f"Subtraction: {d3 - b3}")
print(f"Addition: {d3 + b3}")
print(f"Magnitude of Spherical-Polar Vector 2: {d3.norm()}")
print(f"Dot Product: {d3.dot(b3)}")
print(f"Cross Product: {b3.cross(d3)}")
print()

# Task 3(a)

print("Task 3(a): Cartesian Vector Form")
print(f"Area of Triangle 1: {vv1.trianglearea(vv2, vv3):.2f}")
print(f"Area of Triangle 2: {u1.trianglearea(u2, u3):.2f}")
print(f"Area of Triangle 3: {p1.trianglearea(p2, p3):.2f}")
print(f"Area of Triangle 4: {q1.trianglearea(q2, q3):.2f}")
print()

# Task 3(b)

print("Task 3(b): Cartestian Vector Form")
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

print("Task 3(c): Spherical-Polar Vector Form")
print()
print(f"Area of Triangle 1: {a1.trianglearea(a2, a3):.2f}")
print(f"Area of Triangle 2: {b1.trianglearea(b2, b3):.2f}")
print(f"Area of Triangle 3: {c1.trianglearea(c2, c3):.2f}")
print(f"Area of Triangle 4: {d1.trianglearea(d2, d3):.2f}")
print()
angles4 = a1.triangleangles(a2, a3)
print(f"Spherical-Polar Triangle 1: 1st Angle: {angles1[0]:.2f}°, "
      f"2nd Angle: {angles1[1]:.2f}°,"
      f"3rd Angle: {angles1[2]:.2f}°"
      )

angles5 = b1.triangleangles(b2, b3)
print(f"Spherical-Polar Triangle 2: 1st Angle: {angles2[0]:.2f}°, "
      f"2nd Angle: {angles2[1]:.2f}°,"
      f"3rd Angle: {angles2[2]:.2f}°"
      )

angles6 = c1.triangleangles(c2, c3)
print(f"Spherical-Polar Triangle 3: 1st Angle: {angles3[0]:.2f}°, "
      f"2nd Angle: {angles3[1]:.2f}°,"
      f"3rd Angle: {angles3[2]:.2f}°"
      )

angles7 = d1.triangleangles(d2, d3)
print(f"Spherical-Polar Triangle 4: 1st Angle: {angles4[0]:.2f}°, "
      f"2nd Angle: {angles4[1]:.2f}°,"
      f"3rd Angle: {angles4[2]:.2f}°"
      )