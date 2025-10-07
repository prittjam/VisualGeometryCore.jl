# VisualGeometry Python Interface

A Python wrapper for VisualGeometryCore.jl providing NumPy-based interfaces for circles, ellipses, and conic sections.

## Installation

1. **Install Julia dependencies:**
   ```bash
   # From the main VisualGeometryCore.jl directory
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

2. **Install Python package:**
   ```bash
   cd python
   pip install -e .
   ```

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- JuliaCall ≥ 0.9.0
- Julia ≥ 1.6 with VisualGeometryCore.jl

## Quick Start

```python
import numpy as np
from visualgeometry import Circle, Ellipse, to_homogeneous, to_euclidean

# Create a circle
circle = Circle(center=[2.0, 3.0], radius=1.5)
points = circle.points(100)  # Generate 100 points on boundary

# Create an ellipse
ellipse = Ellipse(
    center=[0.0, 0.0], 
    semi_axes=[3.0, 2.0], 
    angle=np.pi/4
)
ellipse_points = ellipse.points(100)

# Coordinate transformations
euclidean = np.array([[1.0, 2.0], [3.0, 4.0]])
homogeneous = to_homogeneous(euclidean)
recovered = to_euclidean(homogeneous)
```

## API Reference

### Circle

```python
Circle(center, radius)
```

**Methods:**
- `points(n_points=100)` - Generate points on circle boundary
- `contains_point(point)` - Check if point is inside circle
- `distance_to_point(point)` - Distance from point to boundary
- `to_ellipse()` - Convert to Ellipse representation
- `to_homogeneous_conic()` - Convert to HomogeneousConic

### Ellipse

```python
Ellipse(center, semi_axes, angle=0.0)
```

**Properties:**
- `center` - Center point as NumPy array
- `semi_axes` - Semi-major and semi-minor axes
- `angle` - Rotation angle in radians

**Methods:**
- `points(n_points=100)` - Generate points on ellipse boundary
- `to_homogeneous_conic()` - Convert to HomogeneousConic

### HomogeneousConic

```python
HomogeneousConic(matrix)
```

**Properties:**
- `matrix` - 3×3 conic matrix as NumPy array

### Coordinate Transformations

```python
to_homogeneous(points)  # Euclidean → Homogeneous
to_euclidean(points)    # Homogeneous → Euclidean
```

## Examples

See `examples/basic_usage.py` for comprehensive examples:

```bash
cd python/examples
python basic_usage.py
```

## Integration with Julia

The Python interface uses JuliaCall to communicate with VisualGeometryCore.jl. All computations are performed in Julia for maximum performance, with seamless NumPy array conversion.

## Performance Notes

- Array conversions between Python and Julia have some overhead
- For performance-critical applications, consider batching operations
- Large arrays are handled efficiently through JuliaCall's memory sharing

## Troubleshooting

**Julia not found:**
```bash
# Make sure Julia is in PATH or set JULIA_BINDIR
export JULIA_BINDIR=/path/to/julia/bin
```

**Package not found:**
```bash
# Ensure VisualGeometryCore.jl is properly installed
julia --project=. -e "using VisualGeometryCore"
```

**Import errors:**
```bash
# Reinstall with dependencies
pip install -e . --force-reinstall
```