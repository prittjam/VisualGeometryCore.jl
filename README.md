# VisualGeometryCore.jl

[![Build Status](https://github.com/prittjam/VisualGeometryCore.jl/workflows/CI/badge.svg)](https://github.com/prittjam/VisualGeometryCore.jl/actions)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://prittjam.github.io/VisualGeometryCore.jl/stable)

A high-performance Julia package for computational geometry, computer vision, and geometric transformations. VisualGeometryCore.jl provides efficient implementations of geometric primitives, coordinate transformations, and visualization tools with seamless Python integration.

## 🚀 Features

### Core Geometry
- **Circles and Ellipses**: High-precision geometric primitives with efficient point generation
- **Homogeneous Coordinates**: Full support for projective geometry and transformations
- **Conic Sections**: Complete implementation of homogeneous conics with conversion utilities

### Transformations
- **Euclidean Transformations**: Translation, rotation, and scaling
- **Similarity Transformations**: Uniform scaling with rotation and translation
- **Affine Transformations**: General linear transformations with translation
- **Homogeneous Transformations**: Projective transformations in homogeneous coordinates

### Units and Measurements
- **Physical Units**: Support for millimeters, inches, points, pixels, and DPI conversions
- **Logical Units**: Density-independent measurements for consistent scaling
- **Unit Conversions**: Seamless conversion between different measurement systems

### Visualization
- **Makie Integration**: Native plotting support with GLMakie backend
- **Interactive Plots**: Real-time visualization of geometric transformations
- **Export Capabilities**: High-quality output for publications and presentations

### Python Integration
- **Seamless Interoperability**: Full Python interface with automatic type conversion
- **NumPy Compatibility**: Direct integration with NumPy arrays and matplotlib
- **High Performance**: Julia backend with Python convenience

## 📦 Installation

### Julia Package
```julia
using Pkg
Pkg.add("VisualGeometryCore")
```

### Development Installation
```julia
using Pkg
Pkg.develop(url="https://github.com/prittjam/VisualGeometryCore.jl")
```

### Python Interface
For the Python interface, see [`python/README.md`](python/README.md).

## 🎯 Quick Start

### Basic Geometry
```julia
using VisualGeometryCore
using GeometryBasics

# Create a circle using GeometryBasics
center = Point2f(2.0, 3.0)
radius = 1.5f0
circle = GeometryBasics.Circle(center, radius)

# Generate boundary points using high-performance decompose
points = GeometryBasics.decompose(Point2f, circle; resolution=64)
println("Generated $(length(points)) points")

# Create an ellipse using VisualGeometryCore
ellipse = Ellipse(Point2f(0, 0), 3.0f0, 2.0f0, π/4)
ellipse_points = GeometryBasics.decompose(Point2f, ellipse; resolution=32)
println("Generated $(length(ellipse_points)) ellipse points")
```

### Coordinate Transformations
```julia
using VisualGeometryCore
using CoordinateTransformations
using Rotations

# Create transformations using CoordinateTransformations
rotation = LinearMap(RotMatrix{2}(π/4))  # 45° rotation
translation = Translation([2.0, 1.0])    # translation
euclidean = translation ∘ rotation       # compose transformations

# Convert to typed homogeneous matrices for efficiency
euclidean_mat = to_homogeneous(euclidean)  # Returns EuclideanMat{Float64}
println("Transform type: $(typeof(euclidean_mat))")

# Apply transformations to points
original_points = [Point2f(1, 0), Point2f(0, 1), Point2f(-1, 0)]
transformed = [euclidean(p) for p in original_points]
```

### Homogeneous Conics
```julia
using VisualGeometryCore
using GeometryBasics

# Create a circle and convert to homogeneous conic
circle = GeometryBasics.Circle(Point2f(2, 3), 1.5f0)
circle_conic = HomogeneousConic(circle)
println("Circle conic matrix:")
println(SMatrix{3,3}(circle_conic))

# Create an ellipse and convert to homogeneous conic
ellipse = Ellipse(Point2f(0, 0), 3.0f0, 2.0f0, π/4)
ellipse_conic = HomogeneousConic(ellipse)

# Convert back from conic to geometric primitives
recovered_ellipse = Ellipse(ellipse_conic)
println("Recovered ellipse: center=$(recovered_ellipse.center), axes=$(recovered_ellipse.a, recovered_ellipse.b)")
```

### Units and Measurements
```julia
using VisualGeometryCore

# Physical measurements
width_mm = 210.0mm        # A4 width
height_in = 11.0inch      # US Letter height
font_size = 12.0pt        # Typography

# Convert between units
width_inches = to_physical_units(width_mm, inch)
height_mm = to_physical_units(height_in, mm)

# Logical units for DPI-independent layouts
logical_width = to_logical_units(width_mm, 300dpi)
pixel_width = to_physical_units(logical_width, px, 300dpi)
```

### Visualization
```julia
using VisualGeometryCore
using GLMakie

# Create geometric objects
circle = GeometryBasics.Circle(Point2f(0, 0), 1.0f0)
ellipse = Ellipse(Point2f(2, 0), 1.5f0, 0.8f0, π/3)

# Plot with custom styling
fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1], aspect=DataAspect())

# Plot geometries
plotellipses!(ax, [circle, ellipse], 
             colors=[:blue, :red], 
             linewidth=3,
             resolution=64)

# Add labels and styling
ax.title = "Geometric Primitives"
ax.xlabel = "X"
ax.ylabel = "Y"

display(fig)
```

## 📚 Examples

### Circle to Ellipse Transformation
```julia
using VisualGeometryCore
using GeometryBasics
using LinearAlgebra

# Create original circle
circle = GeometryBasics.Circle(Point2f(0, 0), 1.0f0)
circle_points = GeometryBasics.decompose(Point2f, circle)

# Define transformation matrix (scale + rotate)
scale_x, scale_y = 2.5, 1.2
rotation = π/4
T = [scale_x*cos(rotation) -scale_x*sin(rotation);
     scale_y*sin(rotation)  scale_y*cos(rotation)]

# Apply transformation
ellipse_points = [Point2f(T * [p[1], p[2]]) for p in circle_points]

println("Transformed $(length(circle_points)) points")
println("Transformation determinant: $(det(T))")
```

### Advanced Geometric Operations
```julia
using VisualGeometryCore

# Create multiple ellipses with different parameters
ellipses = [
    Ellipse(Point2f(0, 0), 2.0f0, 1.0f0, 0.0f0),
    Ellipse(Point2f(3, 1), 1.5f0, 0.8f0, π/4),
    Ellipse(Point2f(-2, 2), 1.2f0, 2.0f0, π/2)
]

# Generate points for all ellipses
all_points = [decompose(Point2f, ellipse; resolution=32) for ellipse in ellipses]

# Calculate properties
for (i, ellipse) in enumerate(ellipses)
    area = π * ellipse.a * ellipse.b
    println("Ellipse $i: area = $(round(area, digits=2))")
end
```

## 🔧 Advanced Usage

### Custom Transformations
```julia
# Create custom transformation pipeline
function create_transform_pipeline(scale, rotation, translation)
    # Scale matrix
    S = SimilarityMat(scale, 0.0, [0.0, 0.0])
    
    # Rotation matrix  
    R = EuclideanMat(rotation, [0.0, 0.0])
    
    # Translation matrix
    T = EuclideanMat(0.0, translation)
    
    # Compose transformations (applied right to left)
    return T * R * S
end

# Apply to geometry
transform = create_transform_pipeline(1.5, π/6, [2.0, 1.0])
circle = GeometryBasics.Circle(Point2f(0, 0), 1.0f0)
points = GeometryBasics.decompose(Point2f, circle)
transformed_points = [transform * p for p in points]
```

### Performance Optimization
```julia
using BenchmarkTools

# Benchmark point generation
circle = GeometryBasics.Circle(Point2f(0, 0), 1.0f0)

@benchmark GeometryBasics.decompose(Point2f, $circle) samples=1000
# Typical result: ~10-50 μs for 64 points

# Benchmark transformations
transform = EuclideanMat(π/4, [1.0, 2.0])
points = GeometryBasics.decompose(Point2f, circle)

@benchmark [$transform * p for p in $points] samples=1000
# Typical result: ~5-20 μs for 64 points
```

## 🐍 Python Integration

VisualGeometryCore.jl provides a seamless Python interface that combines Julia's performance with Python's ecosystem:

```python
from visualgeometry import Circle, Ellipse
import numpy as np
import matplotlib.pyplot as plt

# Create circle using Julia backend
circle = Circle([0, 0], 1.0)
points = circle.points(64)  # Uses Julia GeometryBasics.decompose

# Transform to ellipse
ellipse = circle.to_ellipse()
ellipse_points = ellipse.points(64)

# Plot with matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
plt.title('Circle (Julia Backend)')
plt.axis('equal')

plt.subplot(1, 2, 2)  
plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2)
plt.title('Ellipse (Julia Backend)')
plt.axis('equal')

plt.show()
```

For complete Python documentation, see [`python/README.md`](python/README.md).

## 📖 Documentation

- **API Reference**: Complete function and type documentation
- **Examples**: Comprehensive examples in `examples/` directory
- **Python Interface**: Full Python integration guide in `python/`
- **Troubleshooting**: OpenSSL and integration issues in `OPENSSL_TROUBLESHOOTING.md`

## 🧪 Testing

```julia
using Pkg
Pkg.test("VisualGeometryCore")
```

For Python interface testing:
```bash
cd python
python -m pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup
```julia
using Pkg
Pkg.develop(url="https://github.com/prittjam/VisualGeometryCore.jl")
Pkg.instantiate()
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GeometryBasics.jl**: Foundation for geometric primitives
- **Makie.jl**: Powerful visualization capabilities  
- **PythonCall.jl**: Seamless Python integration
- **Julia Community**: Excellent ecosystem and support

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/prittjam/VisualGeometryCore.jl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/prittjam/VisualGeometryCore.jl/discussions)
- **Documentation**: [Online Docs](https://prittjam.github.io/VisualGeometryCore.jl/stable)

---

**VisualGeometryCore.jl** - High-performance computational geometry for Julia and Python 🚀