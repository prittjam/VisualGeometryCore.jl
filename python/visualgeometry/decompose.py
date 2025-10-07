"""
Python interface for GeometryBasics decompose functionality
"""

import numpy as np
from .core import VisualGeometryCore, jl


def decompose_circle(center, radius, resolution=32):
    """
    Decompose a circle into boundary points using Julia GeometryBasics
    
    Parameters:
    -----------
    center : array-like, shape (2,)
        Circle center
    radius : float
        Circle radius
    resolution : int
        Number of points to generate
        
    Returns:
    --------
    ndarray, shape (resolution, 2)
        Points on circle boundary
    """
    VisualGeometryCore.ensure_initialized()
    
    center = np.asarray(center)
    
    # Create Julia Circle using GeometryBasics
    julia_center = VisualGeometryCore.numpy_to_julia(center)
    julia_circle = jl.seval(f"GeometryBasics.Circle(Point2f({center[0]}, {center[1]}), {radius}f0)")
    
    # Use GeometryBasics decompose
    julia_points = jl.seval(f"GeometryBasics.decompose(Point2f, circle; resolution={resolution})", circle=julia_circle)
    
    # Convert back to NumPy
    points = []
    for point in julia_points:
        points.append([float(point[0]), float(point[1])])
    
    return np.array(points)


def decompose_ellipse(center, semi_axes, angle, resolution=32):
    """
    Decompose an ellipse into boundary points using Julia GeometryBasics
    
    Parameters:
    -----------
    center : array-like, shape (2,)
        Ellipse center
    semi_axes : array-like, shape (2,)
        Semi-major and semi-minor axes
    angle : float
        Rotation angle in radians
    resolution : int
        Number of points to generate
        
    Returns:
    --------
    ndarray, shape (resolution, 2)
        Points on ellipse boundary
    """
    VisualGeometryCore.ensure_initialized()
    
    center = np.asarray(center)
    semi_axes = np.asarray(semi_axes)
    
    # Create Julia Ellipse - need to convert center to Point2f
    julia_ellipse = jl.seval(f"Ellipse(Point2f({center[0]}, {center[1]}), {semi_axes[0]}f0, {semi_axes[1]}f0, {angle}f0)")
    
    # Use GeometryBasics decompose
    julia_points = jl.seval(f"GeometryBasics.decompose(Point2f, ellipse; resolution={resolution})", ellipse=julia_ellipse)
    
    # Convert back to NumPy
    points = []
    for point in julia_points:
        points.append([float(point[0]), float(point[1])])
    
    return np.array(points)


def decompose(geometry, resolution=32):
    """
    Generic decompose function that works with different geometry types
    
    Parameters:
    -----------
    geometry : Circle, Ellipse, or dict
        Geometry object to decompose
    resolution : int
        Number of points to generate
        
    Returns:
    --------
    ndarray, shape (resolution, 2)
        Points on geometry boundary
    """
    from .circles import Circle
    from .conics import Ellipse
    
    if isinstance(geometry, Circle):
        return decompose_circle(geometry.center, geometry.radius, resolution)
    elif isinstance(geometry, Ellipse):
        return decompose_ellipse(geometry.center, geometry.semi_axes, geometry.angle, resolution)
    elif isinstance(geometry, dict):
        # Handle dictionary specification
        if 'radius' in geometry:
            # Circle
            return decompose_circle(
                geometry['center'], 
                geometry['radius'], 
                resolution
            )
        elif 'semi_axes' in geometry:
            # Ellipse
            return decompose_ellipse(
                geometry['center'],
                geometry['semi_axes'],
                geometry.get('angle', 0.0),
                resolution
            )
        else:
            raise ValueError("Dictionary must contain either 'radius' (circle) or 'semi_axes' (ellipse)")
    else:
        raise TypeError(f"Unsupported geometry type: {type(geometry)}")