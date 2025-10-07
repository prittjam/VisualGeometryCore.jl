"""
VisualGeometry Python Interface

A Python wrapper for VisualGeometryCore.jl providing NumPy-based
interfaces for circles, ellipses, and conic sections.
"""

from .core import VisualGeometryCore
from .conics import Ellipse, HomogeneousConic
from .circles import Circle
from .transforms import to_homogeneous, to_euclidean
from .decompose import decompose, decompose_circle, decompose_ellipse

__version__ = "0.1.0"
__all__ = [
    "VisualGeometryCore",
    "Ellipse", 
    "HomogeneousConic",
    "Circle",
    "to_homogeneous",
    "to_euclidean",
    "decompose",
    "decompose_circle", 
    "decompose_ellipse",
]