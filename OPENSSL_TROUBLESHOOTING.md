# OpenSSL Troubleshooting Guide for Julia-Python Integration

This guide helps fix OpenSSL version conflicts between **PythonCall.jl** and **juliacall** without modifying system OpenSSL installations.

## 🎯 Quick Diagnosis

If you see errors like:
- `libssl.so.1.1 not found`
- `version 'OPENSSL_3.3.0' not found`
- `could not load library libssl.so`
- `CERTIFICATE_VERIFY_FAILED`

You have an OpenSSL version mismatch. **Solution: Use isolated environments.**

## 🧠 Choose Your Strategy

**Option A — Julia → PythonCall (Recommended)**
Let Julia manage Python through CondaPkg. Everything stays in Julia's depot.

**Option B — Python → juliacall**
Let Python manage Julia through conda/micromamba environments.

**Default: Choose Option A unless you specifically need to start from Python.**

---

## 🅰️ Option A: Let Julia Manage Python Safely

### Step 1: Clean Julia Environment (Optional)
```bash
export JULIA_DEPOT_PATH="$HOME/.julia-clean"   # optional clean depot
julia
```

### Step 2: Install Julia Packages
Inside Julia:
```julia
using Pkg
Pkg.add(["CondaPkg", "PythonCall"])
```

### Step 3: Configure Conda Environment
```julia
using CondaPkg
CondaPkg.add("python=3.11")
CondaPkg.add("pip")
```

### Step 4: Configure PythonCall
```julia
using PythonCall
PythonCall.configure()
PythonCall.pyversioninfo()    # ensure Conda env + OpenSSL 3
```

### Step 5: Install Python Dependencies
```julia
CondaPkg.pip("install", ["numpy", "juliacall"])
```

### Step 6: Verify Installation
```julia
using PythonCall
pyimport("numpy")
pyimport("juliacall")
```

---

## 🅱️ Option B: Let Python Manage Julia Safely

### Step 1: Create Clean Python Environment
```bash
micromamba create -n jlpy python=3.11 openssl certifi -c conda-forge -y
micromamba activate jlpy
pip install juliacall
```

### Step 2: Configure SSL Certificates
```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

### Step 3: Test Julia Connection
In Python:
```python
from juliacall import Main as jl
jl.seval("using Pkg; Pkg.status()")
```

### Step 4: Connect Julia → PythonCall (Optional)
Get Python path:
```bash
which python   # copy this path
```

Inside Julia:
```julia
using Pkg
Pkg.add("PythonCall")
import PythonCall
PythonCall.setpython!("/path/to/jlpy/bin/python"; force=true)
PythonCall.pyversioninfo()
```

---

## 🔧 Common Fixes

| Error | Fix |
|-------|-----|
| `libssl.so.1.1 not found` | Use Conda/micromamba env (pins Python + OpenSSL 3) |
| `CERTIFICATE_VERIFY_FAILED` | `export JULIA_PKG_SERVER=""` and set `SSL_CERT_FILE` via certifi |
| Wrong Python shown | `PythonCall.setpython!(...; force=true)` and restart Julia |
| `version 'OPENSSL_3.3.0' not found` | Use isolated environment with compatible OpenSSL |
| `could not load library libssl.so` | Avoid system Python, use conda-managed Python |

---

## 📦 Environment Reproducibility

### For Option A (Julia-managed):
Export your CondaPkg configuration:
```julia
using CondaPkg
CondaPkg.status()  # shows current environment
```

The configuration is saved in `CondaPkg.toml` in your project.

### For Option B (Python-managed):
Export your conda environment:
```bash
micromamba env export -n jlpy > environment.yml
```

To recreate:
```bash
micromamba env create -f environment.yml
```

---

## 🚀 Platform-Specific Commands

### Linux (Ubuntu/Debian)
```bash
# Option A - Julia-managed (recommended)
julia -e 'using Pkg; Pkg.add(["CondaPkg", "PythonCall"])'
julia -e 'using CondaPkg; CondaPkg.add("python=3.11"); CondaPkg.add("pip")'
julia -e 'using PythonCall; PythonCall.configure()'

# Option B - Python-managed
curl -Ls https://micro.mamba.pm/api/latest/linux-64 | tar -xvj bin/micromamba
./bin/micromamba create -n jlpy python=3.11 openssl certifi -c conda-forge -y
./bin/micromamba activate jlpy
pip install juliacall
```

### macOS
```bash
# Option A - Julia-managed (recommended)
julia -e 'using Pkg; Pkg.add(["CondaPkg", "PythonCall"])'
julia -e 'using CondaPkg; CondaPkg.add("python=3.11"); CondaPkg.add("pip")'
julia -e 'using PythonCall; PythonCall.configure()'

# Option B - Python-managed
brew install micromamba
micromamba create -n jlpy python=3.11 openssl certifi -c conda-forge -y
micromamba activate jlpy
pip install juliacall
```

### Windows
```powershell
# Option A - Julia-managed (recommended)
julia -e "using Pkg; Pkg.add([\"CondaPkg\", \"PythonCall\"])"
julia -e "using CondaPkg; CondaPkg.add(\"python=3.11\"); CondaPkg.add(\"pip\")"
julia -e "using PythonCall; PythonCall.configure()"

# Option B - Python-managed
# Install micromamba from https://mamba.readthedocs.io/en/latest/installation.html
micromamba create -n jlpy python=3.11 openssl certifi -c conda-forge -y
micromamba activate jlpy
pip install juliacall
```

---

## ⚠️ What NOT to Do

- **Don't** attempt to rebuild system OpenSSL
- **Don't** patch system libraries
- **Don't** modify `/usr/lib` or `/usr/local/lib`
- **Don't** use `sudo` to install Python packages
- **Don't** mix system Python with Julia's OpenSSL requirements

**Isolation is the only supported solution.**

---

## 🧪 Testing Your Setup

### Test Julia → Python
```julia
using PythonCall
py"""
import numpy as np
print("NumPy version:", np.__version__)
print("Python executable:", __import__('sys').executable)
"""
```

### Test Python → Julia
```python
from juliacall import Main as jl
jl.seval('println("Julia version: ", VERSION)')
jl.seval('using Pkg; println("Julia depot: ", first(DEPOT_PATH))')
```

### Test Bidirectional
```julia
using PythonCall
py"""
from juliacall import Main as jl
jl.seval('println("Bidirectional communication working!")')
"""
```

---

## 📞 Getting Help

If you're still having issues:

1. **Check your environment**: Run the test commands above
2. **Verify isolation**: Ensure you're not mixing system and managed environments
3. **Check versions**: Make sure OpenSSL versions are compatible (3.x)
4. **Clean start**: Try with a fresh Julia depot or Python environment

**Remember: The goal is isolation, not fixing system conflicts.**