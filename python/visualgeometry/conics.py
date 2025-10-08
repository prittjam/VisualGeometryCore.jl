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
            # Convert to HomogeneousConic directly from tuple (column-major order)
            # Julia matrices are column-major, so we need to transpose
            matrix_col_major = matrix.T.flatten()
            self._julia_obj = jl.HomogeneousConic(tuple(matrix_col_major))
        else:
            raise ValueError("Either matrix or julia_conic must be provided")
    
    @property
    def matrix(self):
        """Get the 3x3 conic matrix as NumPy array"""
        # HomogeneousConic is a 3x3 matrix wrapper, convert directly
        return VisualGeometryCore.julia_to_numpy(self._julia_obj)
    
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
                
            # Create Julia Ellipse using proper constructor
            self._julia_obj = jl.seval(f"Ellipse(Point2f({center[0]}, {center[1]}), {semi_axes[0]}f0, {semi_axes[1]}f0, {angle}f0)")
        else:
            raise ValueError("Either (center, semi_axes) or julia_ellipse must be provided")
    
    @property
    def center(self):
        """Get ellipse center as NumPy array"""
        return VisualGeometryCore.julia_to_numpy(self._julia_obj.center)
    
    @property
    def semi_axes(self):
        """Get semi-axes as NumPy array [a, b]"""
        return np.array([float(self._julia_obj.a), float(self._julia_obj.b)])
    
    @property
    def rotation(self):
        """Get rotation angle in radians"""
        return float(self._julia_obj.θ)
    
    def to_homogeneous_conic(self):
        """Convert to HomogeneousConic representation"""
        julia_conic = jl.HomogeneousConic(self._julia_obj)
        return HomogeneousConic(julia_conic=julia_conic)
    
    def points(self, n_points=100):
        """
        Generate points on the ellipse boundary using Julia backend
        
        Uses Julia's GeometryBasics.decompose for high-precision point generation
        with proper handling of rotation and scaling. Falls back to pure Python 
        implementation if Julia backend unavailable.
        
        Parameters:
        -----------
        n_points : int, optional
            Number of points to generate (default: 100)
            
        Returns:
        --------
        points : ndarray, shape (n_points, 2)
            Points on ellipse boundary as [x, y] coordinates
            
        Examples:
        ---------
        >>> ellipse = Ellipse([0, 0], [2, 1], np.pi/4)
        >>> points = ellipse.points(64)
        >>> points.shape
        (64, 2)
        >>> # Verify points satisfy ellipse equation
        >>> errors = ellipse.evaluate_points(points)
        >>> np.allclose(errors, 0, atol=1e-10)
        True
        """
        try:
            # Try to use Julia decompose for better accuracy
            from .decompose import decompose_ellipse
            return decompose_ellipse(self.center, self.semi_axes, self.rotation, n_points)
        except Exception:
            # Fallback to pure Python implementation
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            
            # Generate points on unit circle
            x = np.cos(theta)
            y = np.sin(theta)
            
            # Scale by semi-axes
            a, b = self.semi_axes
            x *= a
            y *= b
            
            # Rotate
            cos_angle = np.cos(self.rotation)
            sin_angle = np.sin(self.rotation)
            
            x_rot = x * cos_angle - y * sin_angle
            y_rot = x * sin_angle + y * cos_angle
            
            # Translate to center
            cx, cy = self.center
            x_rot += cx
            y_rot += cy
            
            return np.column_stack([x_rot, y_rot])
    
    def evaluate_points(self, points):
        """
        Evaluate ellipse equation at given points

        Parameters:
        -----------
        points : ndarray, shape (n, 2)
            Points to evaluate

        Returns:
        --------
        values : ndarray, shape (n,)
            Ellipse equation values (should be ~0 for points on ellipse)
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        # Translate to ellipse center
        centered = points - self.center

        # Rotate by -angle to align with axes
        cos_angle = np.cos(-self.rotation)
        sin_angle = np.sin(-self.rotation)

        x_aligned = centered[:, 0] * cos_angle - centered[:, 1] * sin_angle
        y_aligned = centered[:, 0] * sin_angle + centered[:, 1] * cos_angle

        # Evaluate ellipse equation: (x/a)² + (y/b)² - 1
        a, b = self.semi_axes
        return (x_aligned / a) ** 2 + (y_aligned / b) ** 2 - 1

    def to_torch_tensor(self):
        """
        Convert ellipse to PyTorch tensor [x, y, a, b, θ] for ML workflows

        Returns:
        --------
        torch.Tensor
            Tensor with shape (5,) containing [center_x, center_y, semi_major, semi_minor, rotation_radians]

        Example:
        --------
        >>> ellipse = Ellipse(center=[100, 200], semi_axes=[20, 10], angle=np.pi/4)
        >>> tensor = ellipse.to_torch_tensor()
        >>> tensor
        tensor([100.0000, 200.0000, 20.0000, 10.0000, 0.7854])
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_torch_tensor(). Install with: pip install torch")

        return torch.tensor([
            self.center[0],
            self.center[1],
            self.semi_axes[0],
            self.semi_axes[1],
            self.rotation
        ])

    def to_mpl_ellipse(self, **kwargs):
        """
        Convert to matplotlib Ellipse patch for quick plotting

        Parameters:
        -----------
        **kwargs
            Additional arguments passed to matplotlib.patches.Ellipse
            (e.g., facecolor, edgecolor, alpha, fill)

        Returns:
        --------
        matplotlib.patches.Ellipse
            Ellipse patch ready for adding to matplotlib axes

        Example:
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> ellipse = Ellipse(center=[100, 200], semi_axes=[20, 10], angle=np.pi/4)
        >>> fig, ax = plt.subplots()
        >>> patch = ellipse.to_mpl_ellipse(fill=False, edgecolor='b')
        >>> ax.add_patch(patch)
        >>> plt.show()
        """
        try:
            from matplotlib.patches import Ellipse as MPLEllipse
        except ImportError:
            raise ImportError("Matplotlib is required for to_mpl_ellipse(). Install with: pip install matplotlib")

        # matplotlib.patches.Ellipse takes (xy, width, height, angle_degrees)
        # width = 2*a, height = 2*b, angle in degrees
        return MPLEllipse(
            self.center,
            2 * self.semi_axes[0],  # width = 2*a
            2 * self.semi_axes[1],  # height = 2*b
            angle=np.degrees(self.rotation),  # convert radians to degrees
            **kwargs
        )

    def __repr__(self):
        return f"Ellipse(center={self.center}, semi_axes={self.semi_axes}, rotation={self.rotation:.3f})"