# Troubleshooting VisualGeometry Python Interface

## Common Issues and Solutions

### OpenSSL Version Conflict

**Problem:** Error message about `OPENSSL_3.3.0' not found` when using JuliaCall.

**Cause:** This is a common issue when using Julia with conda environments. The conda environment provides a different OpenSSL version than what Julia's artifacts expect.

**Solutions:**

#### Option 1: Use a Virtual Environment (Recommended)
```bash
# Create a clean Python virtual environment
python -m venv venv_visualgeometry
source venv_visualgeometry/bin/activate  # Linux/Mac
# or: venv_visualgeometry\Scripts\activate  # Windows

# Install dependencies
pip install numpy juliacall matplotlib

# Install the package
cd python
pip install -e .
```

#### Option 2: Set Environment Variables
```bash
# Before running Python, set these environment variables
export LD_LIBRARY_PATH=""
export JULIA_SSL_NO_VERIFY_HOSTS="*"
export JULIA_PKG_USE_CLI_GIT=true

python your_script.py
```

#### Option 3: Use System Python
```bash
# Use system Python instead of conda
/usr/bin/python3 -m pip install --user numpy juliacall matplotlib
/usr/bin/python3 your_script.py
```

#### Option 4: Julia Environment Setup
```bash
# In Julia, rebuild packages to match system libraries
julia -e 'using Pkg; Pkg.build()'
```

### Julia Package Not Found

**Problem:** `VisualGeometryCore` package not found.

**Solution:**
```bash
# Ensure you're in the correct directory structure
cd /path/to/VisualGeometryCore.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
julia --project=. -e "using VisualGeometryCore"  # Test it works

# Then run Python from the same directory or subdirectory
cd python
python your_script.py
```

### JuliaCall Installation Issues

**Problem:** `juliacall` package not installing or working.

**Solutions:**
```bash
# Method 1: Direct pip install
pip install juliacall

# Method 2: Install from conda-forge
conda install -c conda-forge juliacall

# Method 3: Install specific version
pip install "juliacall>=0.9.0,<1.0"
```

### Performance Issues

**Problem:** Slow array conversions between Python and Julia.

**Solutions:**
- Batch operations when possible
- Use larger arrays to amortize conversion overhead
- Consider pure Julia implementation for performance-critical code

### Memory Issues

**Problem:** High memory usage or memory leaks.

**Solutions:**
- Explicitly delete large Julia objects when done
- Use context managers for resource management
- Monitor memory usage with Julia's `@time` macro

## Testing Your Installation

### Basic Test (No Julia)
```python
import numpy as np
from visualgeometry.core import VisualGeometryCore

# This should work without Julia initialization
print("Imports successful!")
```

### Full Test (With Julia)
```python
import numpy as np
from visualgeometry import Circle, to_homogeneous

# This will initialize Julia
circle = Circle([0, 0], 1.0)
points = circle.points(10)
print(f"Generated {len(points)} points")
```

## Getting Help

1. **Check Julia Installation:**
   ```bash
   julia --version
   julia -e "using VisualGeometryCore"
   ```

2. **Check Python Environment:**
   ```python
   import sys, numpy, juliacall
   print(f"Python: {sys.version}")
   print(f"NumPy: {numpy.__version__}")
   print("JuliaCall: OK")
   ```

3. **Environment Information:**
   ```bash
   echo $CONDA_DEFAULT_ENV
   which python
   which julia
   ldd $(which julia)  # Check library dependencies
   ```

4. **Create Minimal Reproduction:**
   ```python
   # Save as test_minimal.py
   import sys
   sys.path.insert(0, '/path/to/VisualGeometry/python')
   
   try:
       from visualgeometry import Circle
       circle = Circle([0, 0], 1)
       print("SUCCESS: Circle created")
   except Exception as e:
       print(f"ERROR: {e}")
       import traceback
       traceback.print_exc()
   ```

## Alternative Approaches

If JuliaCall continues to cause issues, consider:

1. **Pure Python Implementation:** Implement basic circle/ellipse operations in pure Python/NumPy
2. **Julia Script Interface:** Call Julia scripts via subprocess
3. **Different Julia-Python Bridge:** Try PyJulia instead of JuliaCall
4. **Docker Container:** Use a containerized environment with known-good versions