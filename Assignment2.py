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

  #  def cartesian_to_spherical(self):
   #     """
    #        This is where the conversion occurs.
     #   """
      #  r = self.norm()
       # theta = math.acos(self._k / r) if r != 0 else 0
        #phi = math.atan2(self._j, self._i)
        #return SphericalPolarVector(r, theta, phi)
        
    def cartesian_to_spherical(self):
        """
            Converts Cartesian vector to spherical-polar coordinates.
            Args:
                _i, _j and _k.
            Returns:
                r, theta and phi in spherical-polar coordinates.
        """
        r = self.norm()
        if r == 0:
            return 0, 0, 0
    
        theta = math.acos(self._k / r)
        phi = math.atan2(self._j, self._i)
        return r, math.degrees(theta), math.degrees(phi)

    
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
        self._r = r
        self._theta = math.radians(theta)
        self._phi = math.radians(phi)
        self._i = r * math.sin(self._theta) * math.cos(self._phi)  # x-component
        self._j = r * math.sin(self._theta) * math.sin(self._phi)  # y-component
        self._k = r * math.cos(self._theta)  # z-component

        super().__init__(self._i, self._j, self._k)
        
    def spherical_to_cartesian(self):
        """
            Converts from spherical-polar form to Cartesian form.
        """
        return self._i, self._j, self._k

    def cartesian_to_spherical(self):
        """
            To convert from Cartesian back to spherical-polar using inheritance.
            Args:
                r, theta and phi.
            Returns:
                theta and phi in degrees.
        """
       # r = self.norm()
       #theta = math.acos(np.clip(self._k / r, -1.0, 1.0))
       #phi = math.atan2(self.j, self.i)
        return super().cartesian_to_spherical()

#    def cartesian_to_spherical(self):
 #       """
 #       To convert from Cartesian to Spherical-Polar form.
  #      Args:
   #         r, theta, and phi.
    #    Returns:
     #       theta and phi in degrees.
      #  """
       # r = self.norm()
        #theta = math.acos(np.clip(self._k / r, -1.0, 1.0))
        #phi = math.atan2(self._j, self._i)

#        return r, math.degrees(theta), math.degrees(phi)

    def sphericalangleproduct(self, sph_angle):
        """
            Args:
                sph_angle (Vector)
            Returns:
                cos_angle or 0 if Vector=(0,0,0)
        """
        sph_dot_product = self.dot(sph_angle)
        sph_norm_product = self.norm() * sph_angle.norm()
        if sph_norm_product == 0:
            return 0
        sph_cos_angle = np.clip(sph_dot_product / sph_norm_product, -1.0, 1.0)
        return math.degrees(math.acos(sph_cos_angle))

    def sphericaltriangleangles(self, a2, a3):
        """
            Args:
                self=a1, a2 and a3
            Returns:
                Angles of the triangles in spherical-polar form in degrees.
        """
        sph_side1 = SphericalPolarVector(*((a2 - self).cartesian_to_spherical()))
        sph_side2 = SphericalPolarVector(*((a3 - self).cartesian_to_spherical()))
        sph_side3 = SphericalPolarVector(*((a3 - a2).cartesian_to_spherical()))

        sph_angle1 = sph_side1.sphericalangleproduct(sph_side2)
        sph_angle2 = sph_side2.sphericalangleproduct(sph_side3)
        sph_angle3 = sph_side3.sphericalangleproduct(sph_side1)

        return sph_angle1, sph_angle2, sph_angle3

    def sphericaltrianglearea(self, a2, a3):
        """
        Area of a triangle.
        Args:
            self=a1, a2, a3
        Returns:
            Area=int or Area=0 if vectors are collinear (0,0,0).
        """
        if a2.norm() == 0 or a3.norm() == 0:
            return 0
        v1 = a2 - self
        v2 = a3 - self
        crossproduct = v1.cross(v2)
        numerator = abs(self.dot(crossproduct))
        denom = 1 + self.dot(a2) + a2.dot(a3) + self.dot(a3)
        return 2 * np.arctan2(numerator, denom)

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

# Task 3(a)
print("Spherical-Polar Vector Form")
sph_area1 = a1.sphericaltrianglearea(a2, a3)
print(f"Area of Triangle 1: {sph_area1:.2f}")
sph_area2 = b1.sphericaltrianglearea(b2, b3)
print(f"Area of Triangle 2: {sph_area2:.2f}")
sph_area3 = c1.sphericaltrianglearea(c2, c3)
print(f"Area of Triangle 3: {sph_area3:.2f}")
sph_area4 = d1.sphericaltrianglearea(d2, d3)
print(f"Area of Triangle 4: {sph_area4:.2f}")
print()
print("Cartestian Vector Form")
area1 = vv1.trianglearea(vv2, vv3)
print(f"Area of Triangle 1: {area1:.2f}")
area2 = u1.trianglearea(u2, u3)
print(f"Area of Triangle 2: {area2:.2f}")
area3 = p1.trianglearea(p2, p3)
print(f"Area of Triangle 3: {area3:.2f}")
area4 = q1.trianglearea(q2, q3)
print(f"Area of Triangle 4: {area4:.2f}")
print()
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
sph_angles1 = a1.sphericaltriangleangles(a2, a3)
print(f"Spherical-Polar Triangle 1: 1st Angle: {sph_angles1[0]:.2f}°, "
      f"2nd Angle: {sph_angles1[1]:.2f}°,"
      f"3rd Angle: {sph_angles1[2]:.2f}°"
      )

sph_angles2 = b1.sphericaltriangleangles(b2, b3)
print(f"Spherical-Polar Triangle 2: 1st Angle: {sph_angles2[0]:.2f}°, "
      f"2nd Angle: {sph_angles2[1]:.2f}°,"
      f"3rd Angle: {sph_angles2[2]:.2f}°"
      )

sph_angles3 = c1.triangleangles(c2, c3)
print(f"Spherical-Polar Triangle 3: 1st Angle: {sph_angles3[0]:.2f}°, "
      f"2nd Angle: {sph_angles3[1]:.2f}°,"
      f"3rd Angle: {sph_angles3[2]:.2f}°"
      )

sph_angles4 = d1.triangleangles(d2, d3)
print(f"Spherical-Polar Triangle 4: 1st Angle: {sph_angles4[0]:.2f}°, "
      f"2nd Angle: {sph_angles4[1]:.2f}°,"
      f"3rd Angle: {sph_angles4[2]:.2f}°"
      )

# Task 1 and 2

sphericalpolar_vector1 = SphericalPolarVector(*vv1.cartesian_to_spherical())
sphericalpolar_vector2 = SphericalPolarVector(*vv2.cartesian_to_spherical())
sphericalpolar_vector3 = SphericalPolarVector(*vv3.cartesian_to_spherical())

cartesian_sub = vv1 - vv2
sphericalpolar_sub = cartesian_sub.cartesian_to_spherical()

cartesian_add = vv1 + vv2
sphericalpolar_add = cartesian_add.cartesian_to_spherical()

cartesian_mag = vv1.norm()
cartesian_cross = vv1.cross(vv2)
sphericalpolar_cross = cartesian_cross.cartesian_to_spherical()

cartesian_dot = vv1.dot(vv2)
sphericalpolar_dot = sphericalpolar_vector1.sphericalangleproduct(sphericalpolar_vector2)

sphericalpolar_vector1 = SphericalPolarVector(*vv1.cartesian_to_spherical())
sphericalpolar_vector2 = SphericalPolarVector(*vv2.cartesian_to_spherical())

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
print("Spherical-Polar Vector Form")
print(f"Spherical-Polar Vector 1: {sphericalpolar_vector1}")
print(f"Spherical-Polar Vector 2: {sphericalpolar_vector2}")
print(f"Spherical-Polar Subtraction: {sphericalpolar_sub}")
print(f"Spherical-Polar Addition: {sphericalpolar_add}")
print(f"Spherical-Polar Dot Product: {sphericalpolar_dot}")
print(f"Spherical-Polar Cross Product: {sphericalpolar_cross}")
print(f"Magnitude of sphericalpolar_vector1: {sphericalpolar_vector1._r}")
print(f"Magnitude of sphericalpolar_vector2: {sphericalpolar_vector2._r}")