"""
Python interface for coordinate transformations

This module provides coordinate transformation utilities that bridge between
Python NumPy arrays and Julia's high-performance transformation functions.

# Key Functions
- `to_homogeneous()`: Convert Euclidean to homogeneous coordinates
- `to_euclidean()`: Convert homogeneous to Euclidean coordinates  
- `apply_transform()`: Apply transformation matrices to points

# Integration with Julia Backend
All functions use Julia's VisualGeometryCore for numerical computations,
providing high precision and performance while maintaining NumPy compatibility.
"""

import numpy as np
from .core import VisualGeometryCore, jl


def to_homogeneous(points):
    """
    Convert Euclidean coordinates to homogeneous coordinates
    
    Appends a coordinate of 1 to convert from Euclidean to homogeneous representation.
    Uses Julia backend for high-precision computation.
    
    Parameters:
    -----------
    points : array-like
        Points in Euclidean coordinates
        - 1D array of shape (N,) -> homogeneous (N+1,)
        - 2D array of shape (M, N) -> homogeneous (M, N+1)
        
    Returns:
    --------
    points_homogeneous : ndarray
        Points in homogeneous coordinates
        
    Examples:
    ---------
    >>> # Single 2D point
    >>> euclidean = np.array([1.0, 2.0])
    >>> homogeneous = to_homogeneous(euclidean)
    >>> homogeneous
    array([1., 2., 1.])
    
    >>> # Multiple 2D points
    >>> euclidean = np.array([[1, 2], [3, 4]])
    >>> homogeneous = to_homogeneous(euclidean)
    >>> homogeneous.shape
    (2, 3)
    >>> homogeneous
    array([[1., 2., 1.],
           [3., 4., 1.]])
    
    Notes:
    ------
    - Uses Julia's to_homogeneous function for numerical stability
    - Handles both single points and arrays of points
    - Preserves input array structure and data types
    """
    VisualGeometryCore.ensure_initialized()
    
    points = np.asarray(points)
    
    if points.ndim == 1:
        # Single point
        julia_point = VisualGeometryCore.numpy_to_julia(points)
        julia_result = jl.to_homogeneous(julia_point)
        return VisualGeometryCore.julia_to_numpy(julia_result)
    
    elif points.ndim == 2:
        # Multiple points - process each row
        result = []
        for point in points:
            julia_point = VisualGeometryCore.numpy_to_julia(point)
            julia_result = jl.to_homogeneous(julia_point)
            result.append(VisualGeometryCore.julia_to_numpy(julia_result))
        return np.array(result)
    
    else:
        raise ValueError("Points must be 1D or 2D array")


def to_euclidean(points):
    """
    Convert homogeneous coordinates to Euclidean coordinates
    
    Performs perspective division by the last coordinate and removes it.
    Uses Julia backend for high-precision computation and proper handling
    of edge cases (e.g., points at infinity).
    
    Parameters:
    -----------
    points : array-like
        Points in homogeneous coordinates
        - 1D array of shape (N,) -> Euclidean (N-1,)
        - 2D array of shape (M, N) -> Euclidean (M, N-1)
        
    Returns:
    --------
    points_euclidean : ndarray
        Points in Euclidean coordinates
        
    Raises:
    -------
    ValueError
        If homogeneous coordinate (last element) is zero
        
    Examples:
    ---------
    >>> # Single 3D homogeneous point
    >>> homogeneous = np.array([2.0, 4.0, 2.0])
    >>> euclidean = to_euclidean(homogeneous)
    >>> euclidean
    array([1., 2.])
    
    >>> # Multiple 3D homogeneous points
    >>> homogeneous = np.array([[2, 4, 2], [6, 9, 3]])
    >>> euclidean = to_euclidean(homogeneous)
    >>> euclidean
    array([[1., 2.],
           [2., 3.]])
    
    Notes:
    ------
    - Uses Julia's to_euclidean function for numerical stability
    - Automatically handles perspective division
    - Raises error for points at infinity (w = 0)
    - Preserves relative precision during division
    """
    VisualGeometryCore.ensure_initialized()
    
    points = np.asarray(points)
    
    if points.ndim == 1:
        # Single point
        julia_point = VisualGeometryCore.numpy_to_julia(points)
        julia_result = jl.to_euclidean(julia_point)
        return VisualGeometryCore.julia_to_numpy(julia_result)
    
    elif points.ndim == 2:
        # Multiple points - process each row
        result = []
        for point in points:
            julia_point = VisualGeometryCore.numpy_to_julia(point)
            julia_result = jl.to_euclidean(julia_point)
            result.append(VisualGeometryCore.julia_to_numpy(julia_result))
        return np.array(result)
    
    else:
        raise ValueError("Points must be 1D or 2D array")


def apply_transform(transform_matrix, points):
    """
    Apply homogeneous transformation to points
    
    Parameters:
    -----------
    transform_matrix : array-like, shape (3, 3)
        Homogeneous transformation matrix
    points : array-like
        Points to transform (homogeneous coordinates)
        
    Returns:
    --------
    ndarray
        Transformed points
    """
    transform_matrix = np.asarray(transform_matrix)
    points = np.asarray(points)
    
    if transform_matrix.shape != (3, 3):
        raise ValueError("Transform matrix must be 3x3")
    
    if points.ndim == 1:
        # Single point
        if points.shape[0] != 3:
            raise ValueError("Points must be in homogeneous coordinates (3D)")
        return transform_matrix @ points
    
    elif points.ndim == 2:
        # Multiple points - each row is a point
        if points.shape[1] != 3:
            raise ValueError("Points must be in homogeneous coordinates (3D)")
        return (transform_matrix @ points.T).T
    
    else:
        raise ValueError("Points must be 1D or 2D array")