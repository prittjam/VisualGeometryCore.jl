"""
Python interface for coordinate transformations
"""

import numpy as np
from .core import VisualGeometryCore, jl


def to_homogeneous(points):
    """
    Convert euclidean coordinates to homogeneous
    
    Parameters:
    -----------
    points : array-like
        Points in euclidean coordinates
        - 1D array of shape (N,) -> homogeneous (N+1,)
        - 2D array of shape (M, N) -> homogeneous (M, N+1)
        
    Returns:
    --------
    ndarray
        Points in homogeneous coordinates
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
    Convert homogeneous coordinates to euclidean
    
    Parameters:
    -----------
    points : array-like
        Points in homogeneous coordinates
        - 1D array of shape (N,) -> euclidean (N-1,)
        - 2D array of shape (M, N) -> euclidean (M, N-1)
        
    Returns:
    --------
    ndarray
        Points in euclidean coordinates
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