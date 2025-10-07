# VisualGeometry Python Interface

## Overview

This Python interface provides NumPy-based access to VisualGeometryCore.jl's circle and conic functionality. It uses JuliaCall to bridge between Python and Julia, offering the performance of Julia with the convenience of Python.

## Project Structure

```
python/
├── visualgeometry/           # Main Python package
│   ├── __init__.py          # Package exports
│   ├── core.py              # Julia interface core
│   ├── circles.py           # Circle class
│   ├── conics.py            # Ellipse and HomogeneousConic classes
│   └── transforms.py        # Coordinate transformations
├── examples/                # Usage examples
│   ├── basic_usage.py       # Complete examples with plotting
│   └── demo_structure.py    # API demonstration
├── setup.py                 # Package installation
├── requirements.txt         # Python dependencies
├── README.md               # User documentation
├── TROUBLESHOOTING.md      # Common issues and solutions
└── test_*.py               # Test files
```

## Key Features

### 1. Circle Operations
- Create circles with center and radius
- Generate boundary points
- Test point containment
- Calculate distances to boundary
- Convert to ellipse/conic representations

### 2. Ellipse Operations
- Create ellipses with center, semi-axes, and rotation
- Generate boundary points with proper rotation
- Convert to homogeneous conic representation
- Full parametric control

### 3. Homogeneous Conics
- Work with 3×3 conic matrices
- Represent circles, ellipses, and general conics
- Mathematical operations on conic sections

### 4. Coordinate Transformations
- Convert between Euclidean and homogeneous coordinates
- Handle single points or arrays of points
- Proper type promotion and numerical accuracy

## API Design Principles

### NumPy Integration
- All inputs/outputs use NumPy arrays
- Seamless integration with existing NumPy workflows
- Proper broadcasting and vectorization support

### Julia Backend
- Computations performed in Julia for performance
- Automatic array conversion between Python and Julia
- Lazy initialization to avoid startup overhead

### Pythonic Interface
- Clean, intuitive class-based API
- Properties and methods follow Python conventions
- Comprehensive error handling and validation

## Usage Patterns

### Basic Geometry
```python
import numpy as np
from visualgeometry import Circle, Ellipse

# Create and manipulate circles
circle = Circle([0, 0], 1.0)
points = circle.points(100)
inside = circle.contains_point([0.5, 0.5])

# Create and manipulate ellipses
ellipse = Ellipse([0, 0], [2, 1], np.pi/4)
ellipse_points = ellipse.points(100)
conic = ellipse.to_homogeneous_conic()
```

### Coordinate Transformations
```python
from visualgeometry import to_homogeneous, to_euclidean

# Convert coordinate systems
euclidean = np.array([[1, 2], [3, 4]])
homogeneous = to_homogeneous(euclidean)
recovered = to_euclidean(homogeneous)
```

### Integration with Matplotlib
```python
import matplotlib.pyplot as plt

circle = Circle([0, 0], 1)
points = circle.points(100)

plt.plot(points[:, 0], points[:, 1])
plt.axis('equal')
plt.show()
```

## Performance Characteristics

### Strengths
- Julia backend provides high-performance computations
- Efficient handling of large point arrays
- Optimized coordinate transformations

### Considerations
- Array conversion overhead for small operations
- Julia startup time on first use
- Memory usage for large datasets

### Optimization Tips
- Batch operations when possible
- Use larger arrays to amortize conversion costs
- Cache results for repeated computations

## Current Status

### ✅ Completed
- Complete Python package structure
- All core classes and functions implemented
- Comprehensive documentation and examples
- Error handling and validation
- NumPy integration

### ⚠️ Known Issues
- OpenSSL version conflict with conda environments
- Julia initialization requires environment setup
- JuliaCall dependency management

### 🔄 Next Steps
1. Resolve OpenSSL compatibility issues
2. Add comprehensive test suite
3. Performance benchmarking
4. Additional geometric primitives
5. Integration with scientific Python ecosystem

## Testing

### Structure Verification
```bash
cd python
python examples/demo_structure.py
```

### Full Integration Test (requires Julia setup)
```bash
# Test Julia backend
julia -e 'using Pkg; Pkg.test()'

# Test Python examples
cd python/examples
python circle_to_ellipse_julia.py
```

### Installation Test
```bash
cd python
pip install -e .
python -c "from visualgeometry import Circle; print('Success!')"
```

## Deployment

### For Development
```bash
cd python
pip install -e .
```

### For Production
```bash
cd python
pip install .
```

### With Dependencies
```bash
pip install -r requirements.txt
cd python
pip install .
```

## Integration Examples

### With SciPy
```python
from scipy.optimize import minimize
from visualgeometry import Circle

def fit_circle_to_points(points):
    def objective(params):
        center, radius = params[:2], params[2]
        circle = Circle(center, radius)
        distances = [circle.distance_to_point(p) for p in points]
        return sum(d**2 for d in distances)
    
    result = minimize(objective, [0, 0, 1])
    return Circle(result.x[:2], result.x[2])
```

### With Pandas
```python
import pandas as pd
from visualgeometry import Circle

df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 2, 3]})
circle = Circle([0, 0], 2)

df['inside_circle'] = [
    circle.contains_point([row.x, row.y]) 
    for row in df.itertuples()
]
```

This Python interface provides a solid foundation for integrating VisualGeometryCore.jl's powerful geometric capabilities into Python-based scientific computing workflows.