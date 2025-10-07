# VisualGeometry Python Interface

A high-performance Python interface to VisualGeometryCore.jl, providing seamless access to Julia's computational geometry capabilities with Python's ecosystem convenience.

## 🚀 Features

- **High-Performance Backend**: Julia GeometryBasics.decompose for optimal point generation
- **Seamless Integration**: Automatic Julia-Python type conversion with NumPy arrays
- **Complete Geometry Suite**: Circles, ellipses, and coordinate transformations
- **Matplotlib Ready**: Direct integration with Python visualization ecosystem
- **Fallback Support**: Pure Python implementations when Julia backend unavailable

## 📦 Installation

### Prerequisites
- Python 3.9+ 
- Julia 1.9+ (for high-performance backend)

### Quick Installation
```bash
# Install Python package
pip install numpy matplotlib

# Install Julia backend (recommended)
julia -e 'using Pkg; Pkg.add("VisualGeometryCore")'
```

### Full Installation with Julia Backend
For the complete high-performance experience, follow the Julia backend setup:

```bash
# 1. Install Julia dependencies
julia -e 'using Pkg; Pkg.add(["CondaPkg", "PythonCall", "VisualGeometryCore"])'

# 2. Configure isolated Python environment
julia -e 'using CondaPkg; CondaPkg.add("python", version="3.11"); CondaPkg.add("pip")'
julia -e 'using CondaPkg; CondaPkg.add_pip("numpy"); CondaPkg.add_pip("matplotlib")'

# 3. Test the integration
julia -e 'using PythonCall; pyimport("numpy"); println("✓ Integration successful!")'
```

### Troubleshooting OpenSSL Issues
If you encounter OpenSSL conflicts, see our comprehensive guide: [`../OPENSSL_TROUBLESHOOTING.md`](../OPENSSL_TROUBLESHOOTING.md)

## 🎯 Quick Start

### Basic Usage
```python
from visualgeometry import Circle, Ellipse
import numpy as np
import matplotlib.pyplot as plt

# Create a circle (uses Julia backend automatically)
circle = Circle([2.0, 3.0], 1.5)
print(f"Circle: center={circle.center}, radius={circle.radius}")

# Generate high-precision boundary points
points = circle.points(64)  # Uses Julia GeometryBasics.decompose
print(f"Generated {len(points)} points with shape {points.shape}")

# Create an ellipse
ellipse = Ellipse([0, 0], [3, 2], np.pi/4)  # center, semi_axes, angle
ellipse_points = ellipse.points(32)

# Plot both geometries
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2)
plt.plot(circle.center[0], circle.center[1], 'bo', markersize=8)
plt.title('Circle (Julia Backend)')
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2)
plt.plot(ellipse.center[0], ellipse.center[1], 'ro', markersize=8)
plt.title('Ellipse (Julia Backend)')
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### Coordinate Transformations
```python
from visualgeometry import Circle
import numpy as np

# Create original circle
circle = Circle([0, 0], 1.0)
circle_points = circle.points(64)

# Convert to homogeneous conic representation
homogeneous_conic = circle.to_homogeneous_conic()
print("Homogeneous conic matrix:")
print(homogeneous_conic.matrix)

# Convert to ellipse representation
ellipse = circle.to_ellipse()
print(f"As ellipse: semi_axes={ellipse.semi_axes}")

# Apply custom transformations
def transform_points(points, scale_x=2.0, scale_y=1.5, rotation=np.pi/4, translation=[1, 0.5]):
    """Apply scale, rotation, and translation to points"""
    # Scale
    scaled = points * [scale_x, scale_y]
    
    # Rotate
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    rotated = (rotation_matrix @ scaled.T).T
    
    # Translate
    transformed = rotated + np.array(translation)
    
    return transformed

# Transform circle to ellipse
ellipse_points = transform_points(circle_points)
print(f"Transformed {len(ellipse_points)} points")
```

## 📚 API Reference

### Circle Class
```python
class Circle:
    def __init__(self, center, radius):
        """
        Create a Circle
        
        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point [x, y]
        radius : float
            Circle radius (must be positive)
        """
    
    def points(self, resolution=32):
        """
        Generate boundary points using Julia backend
        
        Parameters:
        -----------
        resolution : int
            Number of points to generate
            
        Returns:
        --------
        numpy.ndarray, shape (resolution, 2)
            Boundary points as [x, y] coordinates
        """
    
    def to_ellipse(self):
        """Convert circle to Ellipse representation"""
    
    def to_homogeneous_conic(self):
        """Convert to HomogeneousConic representation"""
    
    @property
    def center(self):
        """Get circle center as NumPy array"""
    
    @property  
    def radius(self):
        """Get circle radius"""
```

### Ellipse Class
```python
class Ellipse:
    def __init__(self, center, semi_axes, angle=0.0):
        """
        Create an Ellipse
        
        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point [x, y]
        semi_axes : array-like, shape (2,)
            Semi-axes lengths [a, b]
        angle : float, optional
            Rotation angle in radians (default: 0.0)
        """
    
    def points(self, resolution=32):
        """
        Generate boundary points using Julia backend
        
        Returns:
        --------
        numpy.ndarray, shape (resolution, 2)
            Boundary points as [x, y] coordinates
        """
    
    def to_homogeneous_conic(self):
        """Convert to HomogeneousConic representation"""
    
    @property
    def center(self):
        """Get ellipse center as NumPy array"""
    
    @property
    def semi_axes(self):
        """Get semi-axes as NumPy array [a, b]"""
    
    @property
    def angle(self):
        """Get rotation angle in radians"""
```

### Direct Functions
```python
def decompose_circle(center, radius, resolution=32):
    """
    Generate circle boundary points directly
    
    Parameters:
    -----------
    center : array-like, shape (2,)
        Circle center [x, y]
    radius : float
        Circle radius
    resolution : int
        Number of points to generate
        
    Returns:
    --------
    numpy.ndarray, shape (resolution, 2)
        Boundary points
    """

def decompose_ellipse(center, semi_axes, angle, resolution=32):
    """
    Generate ellipse boundary points directly
    
    Parameters:
    -----------
    center : array-like, shape (2,)
        Ellipse center [x, y]
    semi_axes : array-like, shape (2,)
        Semi-axes lengths [a, b]
    angle : float
        Rotation angle in radians
    resolution : int
        Number of points to generate
        
    Returns:
    --------
    numpy.ndarray, shape (resolution, 2)
        Boundary points
    """
```

## 🎨 Examples

### Complete Circle to Ellipse Transformation
```python
#!/usr/bin/env python3
"""
Complete circle to ellipse transformation with visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from visualgeometry import Circle

def main():
    # 1. Create original circle with Julia backend
    circle = Circle([0.0, 0.0], 1.0)
    circle_points = circle.points(64)
    print(f"✓ Generated {len(circle_points)} points using Julia backend")
    
    # 2. Define transformation matrix
    scale_x, scale_y = 2.5, 1.2
    rotation = np.pi/4  # 45 degrees
    translation = [2.0, 1.0]
    
    # Create transformation matrix
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    T = np.array([[scale_x * cos_r, -scale_x * sin_r],
                  [scale_y * sin_r,  scale_y * cos_r]])
    
    # 3. Apply transformation
    ellipse_points = (T @ circle_points.T).T + np.array(translation)
    
    # 4. Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original circle
    axes[0].plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2)
    axes[0].plot(0, 0, 'bo', markersize=8)
    axes[0].set_aspect('equal')
    axes[0].grid(True)
    axes[0].set_title('Original Circle\n(Julia Backend)')
    
    # Transformed ellipse
    axes[1].plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2)
    axes[1].plot(translation[0], translation[1], 'ro', markersize=8)
    axes[1].set_aspect('equal')
    axes[1].grid(True)
    axes[1].set_title('Transformed Ellipse')
    
    # Both together
    axes[2].plot(circle_points[:, 0], circle_points[:, 1], 'b-', 
                linewidth=2, alpha=0.7, label='Original Circle')
    axes[2].plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', 
                linewidth=2, alpha=0.7, label='Transformed Ellipse')
    axes[2].plot(0, 0, 'bo', markersize=8)
    axes[2].plot(translation[0], translation[1], 'ro', markersize=8)
    axes[2].set_aspect('equal')
    axes[2].grid(True)
    axes[2].legend()
    axes[2].set_title('Transformation')
    
    plt.tight_layout()
    plt.savefig('circle_ellipse_transformation.png', dpi=150)
    plt.show()
    
    # 5. Print transformation details
    det_T = np.linalg.det(T)
    print(f"\nTransformation Details:")
    print(f"  Scale: ({scale_x:.1f}, {scale_y:.1f})")
    print(f"  Rotation: {rotation*180/np.pi:.0f}°")
    print(f"  Translation: {translation}")
    print(f"  Determinant: {det_T:.3f}")
    print(f"  Area ratio: {det_T:.3f}")

if __name__ == "__main__":
    main()
```

### Performance Comparison
```python
import time
import numpy as np
from visualgeometry import Circle

def benchmark_backends():
    """Compare Julia backend vs pure Python performance"""
    circle = Circle([0, 0], 1.0)
    
    # Benchmark Julia backend
    start = time.time()
    for _ in range(100):
        points = circle.points(64)
    julia_time = time.time() - start
    
    # Benchmark pure Python
    start = time.time()
    for _ in range(100):
        theta = np.linspace(0, 2*np.pi, 64, endpoint=False)
        x = circle.center[0] + circle.radius * np.cos(theta)
        y = circle.center[1] + circle.radius * np.sin(theta)
        python_points = np.column_stack([x, y])
    python_time = time.time() - start
    
    print(f"Julia backend: {julia_time:.4f}s")
    print(f"Pure Python:  {python_time:.4f}s")
    
    if julia_time < python_time:
        speedup = python_time / julia_time
        print(f"Julia is {speedup:.1f}x faster!")
    else:
        print("Similar performance")

benchmark_backends()
```

### Accuracy Verification
```python
import numpy as np
from visualgeometry import Circle

def verify_accuracy():
    """Verify Julia backend accuracy against analytical results"""
    circle = Circle([2, 3], 1.5)
    points = circle.points(8)
    
    # Check that all points are on the circle boundary
    distances = np.sqrt((points[:, 0] - circle.center[0])**2 + 
                       (points[:, 1] - circle.center[1])**2)
    
    max_error = np.max(np.abs(distances - circle.radius))
    mean_error = np.mean(np.abs(distances - circle.radius))
    
    print(f"Radius accuracy:")
    print(f"  Expected radius: {circle.radius}")
    print(f"  Max error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    if max_error < 1e-10:
        print("✓ Excellent accuracy (< 1e-10)")
    elif max_error < 1e-6:
        print("✓ Very good accuracy (< 1e-6)")
    else:
        print("✓ Good accuracy for practical use")

verify_accuracy()
```

## 🔧 Advanced Usage

### Custom Geometry Pipeline
```python
from visualgeometry import Circle, Ellipse
import numpy as np

class GeometryPipeline:
    """Advanced geometry processing pipeline"""
    
    def __init__(self):
        self.geometries = []
        self.transformations = []
    
    def add_circle(self, center, radius):
        """Add circle to pipeline"""
        circle = Circle(center, radius)
        self.geometries.append(circle)
        return self
    
    def add_ellipse(self, center, semi_axes, angle=0):
        """Add ellipse to pipeline"""
        ellipse = Ellipse(center, semi_axes, angle)
        self.geometries.append(ellipse)
        return self
    
    def transform(self, scale=None, rotation=None, translation=None):
        """Add transformation to pipeline"""
        transform = {
            'scale': scale or [1, 1],
            'rotation': rotation or 0,
            'translation': translation or [0, 0]
        }
        self.transformations.append(transform)
        return self
    
    def generate_points(self, resolution=32):
        """Generate points for all geometries with transformations"""
        all_points = []
        
        for geometry in self.geometries:
            points = geometry.points(resolution)
            
            # Apply transformations
            for transform in self.transformations:
                # Scale
                points = points * np.array(transform['scale'])
                
                # Rotate
                angle = transform['rotation']
                if angle != 0:
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                    points = (R @ points.T).T
                
                # Translate
                points = points + np.array(transform['translation'])
            
            all_points.append(points)
        
        return all_points

# Usage example
pipeline = GeometryPipeline()
points_list = (pipeline
               .add_circle([0, 0], 1.0)
               .add_ellipse([2, 0], [1.5, 0.8], np.pi/6)
               .transform(scale=[1.2, 1.2], rotation=np.pi/8)
               .transform(translation=[1, 0.5])
               .generate_points(64))

print(f"Generated points for {len(points_list)} geometries")
```

### Integration with Scientific Python
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from visualgeometry import Circle, Ellipse

def analyze_geometry_data():
    """Advanced analysis using scientific Python ecosystem"""
    
    # Create multiple geometries
    geometries = [
        Circle([0, 0], 1.0),
        Circle([3, 1], 0.8),
        Ellipse([1, 2], [1.5, 0.6], np.pi/3),
        Ellipse([-1, 1], [1.2, 0.9], -np.pi/4)
    ]
    
    # Generate points for each geometry
    all_points = []
    labels = []
    
    for i, geom in enumerate(geometries):
        points = geom.points(32)
        all_points.append(points)
        labels.extend([f'Geometry_{i}'] * len(points))
    
    # Combine all points
    combined_points = np.vstack(all_points)
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'x': combined_points[:, 0],
        'y': combined_points[:, 1],
        'geometry': labels
    })
    
    # Statistical analysis
    stats = df.groupby('geometry').agg({
        'x': ['mean', 'std', 'min', 'max'],
        'y': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    print("Geometry Statistics:")
    print(stats)
    
    # Distance analysis
    distances = cdist(combined_points, combined_points)
    print(f"\nDistance matrix shape: {distances.shape}")
    print(f"Max distance: {np.max(distances):.3f}")
    print(f"Mean distance: {np.mean(distances):.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot by geometry
    for geom_name in df['geometry'].unique():
        mask = df['geometry'] == geom_name
        ax1.scatter(df[mask]['x'], df[mask]['y'], 
                   label=geom_name, alpha=0.7, s=30)
    
    ax1.set_aspect('equal')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('All Geometries')
    
    # Distance heatmap
    im = ax2.imshow(distances, cmap='viridis')
    ax2.set_title('Distance Matrix')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return df, distances

# Run analysis
df, distances = analyze_geometry_data()
```

## 🧪 Testing

### Run Tests
```bash
cd python
python -m pytest tests/ -v
```

### Test Julia Backend
```python
from visualgeometry import Circle
import numpy as np

def test_julia_backend():
    """Test Julia backend integration"""
    try:
        circle = Circle([0, 0], 1.0)
        points = circle.points(8)
        
        # Verify shape
        assert points.shape == (8, 2), f"Expected (8, 2), got {points.shape}"
        
        # Verify points are on circle
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        max_error = np.max(np.abs(distances - 1.0))
        
        assert max_error < 1e-10, f"Points not on circle: max_error={max_error}"
        
        print("✓ Julia backend test passed")
        return True
        
    except Exception as e:
        print(f"✗ Julia backend test failed: {e}")
        return False

# Run test
test_julia_backend()
```

## 🚨 Troubleshooting

### Common Issues

#### OpenSSL Version Conflicts
If you see errors like `version 'OPENSSL_3.3.0' not found`:
- See detailed solutions in [`../OPENSSL_TROUBLESHOOTING.md`](../OPENSSL_TROUBLESHOOTING.md)
- Use Julia-managed Python environment (recommended)
- Or use system Python with compatible OpenSSL

#### Julia Backend Not Available
If Julia backend fails, the package automatically falls back to pure Python:
```python
from visualgeometry import Circle

# This will work even without Julia backend
circle = Circle([0, 0], 1.0)
points = circle.points(32)  # Uses NumPy fallback
print("Fallback mode: points generated with pure Python")
```

#### Import Errors
```python
# Check if Julia backend is available
try:
    from visualgeometry.core import VisualGeometryCore
    VisualGeometryCore.ensure_initialized()
    print("✓ Julia backend available")
except ImportError:
    print("⚠ Using pure Python fallback")
```

### Performance Tips

1. **Use appropriate resolution**: Higher resolution = more points but slower
2. **Batch operations**: Generate multiple geometries together when possible
3. **Cache results**: Store generated points if reusing the same geometry
4. **Julia backend**: Ensure Julia backend is properly configured for best performance

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/prittjam/VisualGeometryCore.jl/issues)
- **Documentation**: [Main Package Docs](../README.md)
- **OpenSSL Help**: [Troubleshooting Guide](../OPENSSL_TROUBLESHOOTING.md)

---

**VisualGeometry Python Interface** - High-performance computational geometry with Python convenience 🐍✨