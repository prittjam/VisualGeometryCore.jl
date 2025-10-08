"""
Python interface for circles
"""

import numpy as np
from .core import VisualGeometryCore, jl
from .conics import Ellipse


class Circle:
    """Python wrapper for Circle functionality"""

    def __init__(self, center=None, radius=None, julia_circle=None):
        """
        Create a Circle

        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point (x, y)
        radius : float
            Circle radius
        julia_circle : Julia GeometryBasics.Circle object
            Existing Julia Circle object
        """
        VisualGeometryCore.ensure_initialized()

        if julia_circle is not None:
            # Extract center and radius from Julia Circle
            self._center = np.array([float(julia_circle.center[1]), float(julia_circle.center[2])])
            self._radius = float(julia_circle.r)
        elif center is not None and radius is not None:
            self._center = np.asarray(center)
            self._radius = float(radius)

            if self._center.shape != (2,):
                raise ValueError("Center must be 2D point")
            if self._radius <= 0:
                raise ValueError("Radius must be positive")
        else:
            raise ValueError("Either (center, radius) or julia_circle must be provided")
    
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
        Generate points on the circle boundary using Julia backend
        
        Uses Julia's GeometryBasics.decompose for high-precision point generation.
        Falls back to pure Python implementation if Julia backend unavailable.
        
        Parameters:
        -----------
        n_points : int, optional
            Number of points to generate (default: 100)
            
        Returns:
        --------
        points : ndarray, shape (n_points, 2)
            Points on circle boundary as [x, y] coordinates
            
        Examples:
        ---------
        >>> circle = Circle([0, 0], 1.0)
        >>> points = circle.points(64)
        >>> points.shape
        (64, 2)
        >>> # Verify points are on circle boundary
        >>> distances = np.linalg.norm(points, axis=1)
        >>> np.allclose(distances, 1.0)
        True
        """
        try:
            # Try to use Julia decompose for better accuracy
            from .decompose import decompose_circle
            return decompose_circle(self._center, self._radius, n_points)
        except Exception:
            # Fallback to pure Python implementation
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

    def to_torch_tensor(self):
        """
        Convert circle to PyTorch tensor [x, y, r] for ML workflows

        Returns:
        --------
        torch.Tensor
            Tensor with shape (3,) containing [center_x, center_y, radius]

        Example:
        --------
        >>> circle = Circle(center=[100, 200], radius=15)
        >>> tensor = circle.to_torch_tensor()
        >>> tensor
        tensor([100., 200., 15.])
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_torch_tensor(). Install with: pip install torch")

        return torch.tensor([self._center[0], self._center[1], self._radius])

    def to_mpl_circle(self, **kwargs):
        """
        Convert to matplotlib Circle patch for quick plotting

        Parameters:
        -----------
        **kwargs
            Additional arguments passed to matplotlib.patches.Circle
            (e.g., facecolor, edgecolor, alpha, fill)

        Returns:
        --------
        matplotlib.patches.Circle
            Circle patch ready for adding to matplotlib axes

        Example:
        --------
        >>> import matplotlib.pyplot as plt
        >>> circle = Circle(center=[100, 200], radius=15)
        >>> fig, ax = plt.subplots()
        >>> patch = circle.to_mpl_circle(fill=False, edgecolor='r')
        >>> ax.add_patch(patch)
        >>> plt.show()
        """
        try:
            from matplotlib.patches import Circle as MPLCircle
        except ImportError:
            raise ImportError("Matplotlib is required for to_mpl_circle(). Install with: pip install matplotlib")

        return MPLCircle(self._center, self._radius, **kwargs)

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"