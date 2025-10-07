"""
VisualGeometry Python Interface

A high-performance Python interface to VisualGeometryCore.jl, providing seamless access 
to Julia's computational geometry capabilities with Python's ecosystem convenience.

This package combines:
- Julia's high-performance GeometryBasics.decompose for optimal point generation
- Automatic Julia-Python type conversion with NumPy arrays  
- Complete geometry suite: circles, ellipses, and coordinate transformations
- Direct integration with matplotlib and scientific Python ecosystem

# Quick Start
```python
from visualgeometry import Circle, Ellipse
import numpy as np
import matplotlib.pyplot as plt

# Create circle with Julia backend
circle = Circle([0, 0], 1.0)
points = circle.points(64)  # Uses Julia GeometryBasics.decompose

# Create ellipse
ellipse = Ellipse([0, 0], [2, 1], np.pi/4)
ellipse_points = ellipse.points(64)

# Plot with matplotlib
plt.plot(points[:, 0], points[:, 1], label='Circle')
plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], label='Ellipse')
plt.axis('equal')
plt.legend()
plt.show()
```

# Features
- High-performance Julia backend with Python convenience
- Seamless NumPy array integration
- Automatic fallback to pure Python when Julia unavailable
- Complete geometric primitive support
- Coordinate transformation utilities
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