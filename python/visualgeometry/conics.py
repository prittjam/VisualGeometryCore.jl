"""
Python interface for conic sections (ellipses, circles, etc.)
"""

import numpy as np
from .core import VisualGeometryCore, jl


class HomogeneousConic:
    """Python wrapper for HomogeneousConic from VisualGeometryCore.jl"""
    
    def __init__(self, matrix=None, julia_conic=None):
        """
        Create a HomogeneousConic
        
        Parameters:
        -----------
        matrix : array-like, shape (3, 3)
            Homogeneous conic matrix
        julia_conic : Julia object
            Existing Julia HomogeneousConic object
        """
        VisualGeometryCore.ensure_initialized()
        
        if julia_conic is not None:
            self._julia_obj = julia_conic
        elif matrix is not None:
            matrix = np.asarray(matrix)
            if matrix.shape != (3, 3):
                raise ValueError("Conic matrix must be 3x3")
            julia_matrix = VisualGeometryCore.numpy_to_julia(matrix)
            self._julia_obj = jl.HomogeneousConic(julia_matrix)
        else:
            raise ValueError("Either matrix or julia_conic must be provided")
    
    @property
    def matrix(self):
        """Get the 3x3 conic matrix as NumPy array"""
        return VisualGeometryCore.julia_to_numpy(self._julia_obj.C)
    
    def __repr__(self):
        return f"HomogeneousConic(\n{self.matrix}\n)"


class Ellipse:
    """Python wrapper for Ellipse from VisualGeometryCore.jl"""
    
    def __init__(self, center=None, semi_axes=None, angle=0.0, julia_ellipse=None):
        """
        Create an Ellipse
        
        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point (x, y)
        semi_axes : array-like, shape (2,)
            Semi-major and semi-minor axes (a, b)
        angle : float
            Rotation angle in radians
        julia_ellipse : Julia object
            Existing Julia Ellipse object
        """
        VisualGeometryCore.ensure_initialized()
        
        if julia_ellipse is not None:
            self._julia_obj = julia_ellipse
        elif center is not None and semi_axes is not None:
            center = np.asarray(center)
            semi_axes = np.asarray(semi_axes)
            
            if center.shape != (2,):
                raise ValueError("Center must be 2D point")
            if semi_axes.shape != (2,):
                raise ValueError("Semi-axes must be 2D")
                
            julia_center = VisualGeometryCore.numpy_to_julia(center)
            julia_axes = VisualGeometryCore.numpy_to_julia(semi_axes)
            
            self._julia_obj = jl.Ellipse(julia_center, julia_axes, angle)
        else:
            raise ValueError("Either (center, semi_axes) or julia_ellipse must be provided")
    
    @property
    def center(self):
        """Get ellipse center as NumPy array"""
        return VisualGeometryCore.julia_to_numpy(self._julia_obj.center)
    
    @property
    def semi_axes(self):
        """Get semi-axes as NumPy array"""
        return VisualGeometryCore.julia_to_numpy(self._julia_obj.semi_axes)
    
    @property
    def angle(self):
        """Get rotation angle in radians"""
        return float(self._julia_obj.angle)
    
    def to_homogeneous_conic(self):
        """Convert to HomogeneousConic representation"""
        julia_conic = jl.HomogeneousConic(self._julia_obj)
        return HomogeneousConic(julia_conic=julia_conic)
    
    def points(self, n_points=100):
        """
        Generate points on the ellipse boundary
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
            
        Returns:
        --------
        points : ndarray, shape (n_points, 2)
            Points on ellipse boundary
        """
        theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        
        # Generate points on unit circle
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Scale by semi-axes
        a, b = self.semi_axes
        x *= a
        y *= b
        
        # Rotate
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        
        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle
        
        # Translate to center
        cx, cy = self.center
        x_rot += cx
        y_rot += cy
        
        return np.column_stack([x_rot, y_rot])
    
    def __repr__(self):
        return f"Ellipse(center={self.center}, semi_axes={self.semi_axes}, angle={self.angle:.3f})"