#!/usr/bin/env python3
"""
Test script for Julia backend with proper environment isolation
"""
import os
import sys
import subprocess

def setup_julia_env():
    """Setup Julia environment with proper OpenSSL isolation"""
    
    # Clear problematic environment variables
    env = os.environ.copy()
    
    # Remove conda/miniconda paths that might interfere
    if 'LD_LIBRARY_PATH' in env:
        del env['LD_LIBRARY_PATH']
    
    # Set Julia project
    env['JULIA_PROJECT'] = 'julia_test_env'
    
    return env

def test_julia_direct():
    """Test Julia directly first"""
    print("Testing Julia directly...")
    
    env = setup_julia_env()
    
    julia_code = '''
    using VisualGeometryCore
    println("VisualGeometryCore loaded successfully!")
    println("Available functions:")
    for name in names(VisualGeometryCore)
        if !startswith(string(name), "#")
            println("  ", name)
        end
    end
    '''
    
    try:
        result = subprocess.run([
            'julia', '--project=julia_test_env', '-e', julia_code
        ], env=env, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✓ Julia test successful!")
            print(result.stdout)
            return True
        else:
            print("✗ Julia test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Julia test timed out!")
        return False
    except Exception as e:
        print(f"✗ Julia test error: {e}")
        return False

def test_python_julia():
    """Test Python-Julia interface"""
    print("\nTesting Python-Julia interface...")
    
    env = setup_julia_env()
    
    python_code = '''
import os
os.environ["JULIA_PROJECT"] = "julia_test_env"

import juliacall
jl = juliacall.Main

# Load the package
jl.seval("using VisualGeometryCore")
print("✓ VisualGeometryCore loaded from Python!")

# Test basic functionality
try:
    # Test if we can access the module
    vgc = jl.VisualGeometryCore
    print("✓ Module accessible from Python!")
    
    # List available functions
    print("Available functions from Python:")
    names = jl.seval("names(VisualGeometryCore)")
    for name in names:
        name_str = str(name)
        if not name_str.startswith("#"):
            print(f"  {name_str}")
            
except Exception as e:
    print(f"✗ Error accessing module: {e}")
'''
    
    try:
        result = subprocess.run([
            sys.executable, '-c', python_code
        ], env=env, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Python-Julia test successful!")
            print(result.stdout)
            return True
        else:
            print("✗ Python-Julia test failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Python-Julia test timed out!")
        return False
    except Exception as e:
        print(f"✗ Python-Julia test error: {e}")
        return False

if __name__ == "__main__":
    print("Setting up Julia backend test environment...")
    print("=" * 50)
    
    # Test Julia directly first
    julia_success = test_julia_direct()
    
    if julia_success:
        # Test Python-Julia interface
        python_success = test_python_julia()
        
        if python_success:
            print("\n" + "=" * 50)
            print("✓ All tests passed! Julia backend is ready.")
            print("You can now use the Python interface with:")
            print("  source venv_julia_test/bin/activate")
            print("  export JULIA_PROJECT=julia_test_env")
            print("  python your_script.py")
        else:
            print("\n" + "=" * 50)
            print("✗ Python-Julia interface failed.")
    else:
        print("\n" + "=" * 50)
        print("✗ Julia direct test failed. Check Julia installation.")