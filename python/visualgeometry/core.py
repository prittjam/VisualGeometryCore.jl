"""
Core Julia interface for VisualGeometryCore.jl
"""

import os
import sys
from pathlib import Path
import numpy as np

try:
    from juliacall import Main as jl
except ImportError:
    raise ImportError(
        "juliacall is required. Install with: pip install juliacall"
    )


class VisualGeometryCore:
    """Main interface to VisualGeometryCore.jl"""
    
    _initialized = False
    _julia_pkg_path = None
    
    @classmethod
    def initialize(cls, julia_pkg_path=None):
        """Initialize the Julia environment and load VisualGeometryCore.jl"""
        if cls._initialized:
            return
            
        # Find the Julia package path
        if julia_pkg_path is None:
            # Assume we're in python/ subdirectory of the Julia package
            current_dir = Path(__file__).parent.parent.parent
            cls._julia_pkg_path = str(current_dir)
        else:
            cls._julia_pkg_path = julia_pkg_path
            
        # Activate the Julia project
        jl.seval(f'using Pkg; Pkg.activate("{cls._julia_pkg_path}")')
        
        # Load VisualGeometryCore
        jl.seval("using VisualGeometryCore")
        
        # Import commonly used Julia modules
        jl.seval("using StaticArrays")
        jl.seval("using LinearAlgebra")
        
        cls._initialized = True
    
    @classmethod
    def ensure_initialized(cls):
        """Ensure Julia environment is initialized"""
        if not cls._initialized:
            cls.initialize()
    
    @staticmethod
    def numpy_to_julia(arr):
        """Convert NumPy array to Julia array"""
        VisualGeometryCore.ensure_initialized()
        if arr.ndim == 1:
            if len(arr) == 2:
                return jl.seval(f"SVector({arr[0]}, {arr[1]})")
            else:
                return jl.seval(f"SVector({', '.join(map(str, arr))})")
        else:
            return jl.SMatrix[arr.T]  # Julia is column-major
    
    @staticmethod
    def julia_to_numpy(jarr):
        """Convert Julia array to NumPy array"""
        # Convert Julia array to Python list, then to NumPy
        return np.array(jarr)


# Don't initialize on import - wait until first use
# VisualGeometryCore.initialize()