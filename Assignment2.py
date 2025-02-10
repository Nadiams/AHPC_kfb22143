#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:01:27 2025

@author: nadia
"""

class Vector:
    """
        Vector class for Cartesian vectors in 3D space
    """
    def __init__(self, i, j, k):
        self.i = i
        self.j = j
        self.k = k
    
    def __str__(self):
        """
        Assumes floating point when printing
        """
        return f"Vector: ({self.i:2f}, {self.j:2f}, {self.k:2f})"

    def __add__(self,other):
        """
        Adds the strings together to create vector.
        """
        return Vector(self.i + other.i, self.j + other.j, self.k + other.k)

    def norm(self):
        """computes magnitude of vector"""
        return math.sqrt(self.i**2+self.j**2+self.k**2)
    
i = Vector(1.,0.,0.)
j = Vector(0.,1.,0.)
k = Vector(0.,0.,1.)

print(i)
print(j)
print(k)