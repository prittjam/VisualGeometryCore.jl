"""
Python interface for circles
"""

import numpy as np
from .core import VisualGeometryCore, jl
from .conics import Ellipse


class Circle:
    """Python wrapper for Circle functionality"""
    
    def __init__(self, center, radius):
        """
        Create a Circle
        
        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point (x, y)
        radius : float
            Circle radius
        """
        VisualGeometryCore.ensure_initialized()
        
        self._center = np.asarray(center)
        self._radius = float(radius)
        
        if self._center.shape != (2,):
            raise ValueError("Center must be 2D point")
        if self._radius <= 0:
            raise ValueError("Radius must be positive")
    
    @property
    def center(self):
        """Get circle center as NumPy array"""
        return self._center.copy()
    
    @property
    def radius(self):
        """Get circle radius"""
        return self._radius
    
    def to_ellipse(self):
        """Convert circle to Ellipse representation"""
        semi_axes = np.array([self._radius, self._radius])
        return Ellipse(center=self._center, semi_axes=semi_axes, angle=0.0)
    
    def to_homogeneous_conic(self):
        """Convert to HomogeneousConic representation"""
        return self.to_ellipse().to_homogeneous_conic()
    
    def points(self, n_points=100):
        """
        Generate points on the circle boundary
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
            
        Returns:
        --------
        points : ndarray, shape (n_points, 2)
            Points on circle boundary
        """
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        x = self._center[0] + self._radius * np.cos(theta)
        y = self._center[1] + self._radius * np.sin(theta)
        
        return np.column_stack([x, y])
    
    def contains_point(self, point):
        """
        Check if a point is inside the circle
        
        Parameters:
        -----------
        point : array-like, shape (2,)
            Point to test
            
        Returns:
        --------
        bool
            True if point is inside circle
        """
        point = np.asarray(point)
        distance = np.linalg.norm(point - self._center)
        return distance <= self._radius
    
    def distance_to_point(self, point):
        """
        Calculate distance from point to circle boundary
        
        Parameters:
        -----------
        point : array-like, shape (2,)
            Point to measure distance to
            
        Returns:
        --------
        float
            Distance to circle boundary (negative if inside)
        """
        point = np.asarray(point)
        distance_to_center = np.linalg.norm(point - self._center)
        return distance_to_center - self._radius
    
    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"