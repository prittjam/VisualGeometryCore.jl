#!/usr/bin/env python3
"""
Circle to Ellipse Transformation with Julia Backend

This example demonstrates geometric transformations using the VisualGeometry
Julia backend through the Python interface.

Workflow:
1. Create a Circle using VisualGeometry
2. Convert to HomogeneousConic 
3. Apply affine transformation to the conic matrix
4. Convert back to Ellipse using VisualGeometry

This showcases the integration between Python NumPy and Julia VisualGeometryCore.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - plots will be skipped")

def main():
    """Demonstrate circle to ellipse transformation using Julia backend"""
    print("Circle to Ellipse Transformation with Julia Backend")
    print("=" * 55)
    
    # Import VisualGeometry components
    from visualgeometry import Circle, Ellipse, HomogeneousConic
    
    print("✓ Successfully imported VisualGeometry with Julia backend")
    
    # Step 1: Create a circle
    print("\n1. Creating Circle with VisualGeometry")
    print("-" * 38)
    
    circle = Circle(center=[0.0, 0.0], radius=1.0)
    print(f"Created circle: {circle}")
    
    # Generate points on circle using Julia decompose
    circle_points = circle.points(100)
    print(f"Generated {len(circle_points)} points on circle boundary using Julia decompose")
    
    # Step 2: Convert to HomogeneousConic
    print("\n2. Converting to HomogeneousConic")
    print("-" * 33)
    
    conic = circle.to_homogeneous_conic()
    conic_matrix = conic.matrix
    
    print("Conic matrix from Julia:")
    print(conic_matrix)
    
    # Step 3: Apply affine transformation
    print("\n3. Applying Affine Transformation")
    print("-" * 33)
    
    # Create transformation matrix: scale, rotate, translate
    scale_x, scale_y = 2.0, 1.5
    angle = np.pi / 6  # 30 degrees
    tx, ty = 1.0, 2.0
    
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    transform_matrix = np.array([
        [scale_x * cos_a, -scale_x * sin_a, tx],
        [scale_y * sin_a,  scale_y * cos_a, ty],
        [0.0, 0.0, 1.0]
    ])
    
    print("Transformation matrix (scale, rotate, translate):")
    print(transform_matrix)
    print(f"Scale: ({scale_x}, {scale_y}), Rotation: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
    
    # Transform the conic matrix
    # For conic C and transform T: C' = (T^-1)^T * C * T^-1
    T_inv = np.linalg.inv(transform_matrix)
    transformed_conic_matrix = T_inv.T @ conic_matrix @ T_inv
    
    print("\nTransformed conic matrix:")
    print(transformed_conic_matrix)
    
    # Step 4: Create new HomogeneousConic and convert to Ellipse
    print("\n4. Converting to Ellipse")
    print("-" * 23)
    
    transformed_conic = HomogeneousConic(matrix=transformed_conic_matrix)
    ellipse = transformed_conic.to_ellipse()
    
    print(f"Extracted ellipse: {ellipse}")
    print(f"Center: {ellipse.center}")
    print(f"Semi-axes: {ellipse.semi_axes}")
    print(f"Rotation: {ellipse.rotation:.3f} rad ({np.degrees(ellipse.rotation):.1f}°)")
    
    # Generate points on the ellipse
    ellipse_points = ellipse.points(100)
    
    # Step 5: Verification
    print("\n5. Verification")
    print("-" * 14)
    
    # Transform original circle points directly and compare
    circle_points_homogeneous = np.column_stack([
        circle_points, 
        np.ones(len(circle_points))
    ])
    
    transformed_points_homogeneous = (transform_matrix @ circle_points_homogeneous.T).T
    transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
    
    # Check if transformed circle points lie on the ellipse
    max_error = np.max(np.abs(ellipse.evaluate_points(transformed_points)))
    print(f"Max ellipse equation error for transformed points: {max_error:.2e}")
    
    if max_error < 1e-10:
        print("✓ Verification successful - transformed points lie on ellipse!")
    else:
        print("✗ Verification failed - numerical issues detected")
    
    # Step 6: Visualization
    if HAS_MATPLOTLIB:
        print("\n6. Visualization")
        print("-" * 15)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original circle
        ax1.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Circle')
        ax1.plot(0, 0, 'bo', markersize=8, label='Center')
        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()
        ax1.set_title('Original Circle')
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        
        # Transformed ellipse (from ellipse object)
        ax2.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2, label='Ellipse')
        ax2.plot(ellipse.center[0], ellipse.center[1], 'ro', markersize=8, label='Center')
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        ax2.set_title('Transformed Ellipse')
        
        # Comparison: both methods
        ax3.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2, label='Ellipse (Julia)')
        ax3.plot(transformed_points[:, 0], transformed_points[:, 1], 'b--', linewidth=1, alpha=0.7, label='Direct transform')
        ax3.plot(ellipse.center[0], ellipse.center[1], 'ro', markersize=8, label='Center')
        ax3.set_aspect('equal')
        ax3.grid(True)
        ax3.legend()
        ax3.set_title('Verification')
        
        plt.tight_layout()
        plt.savefig('julia_circle_to_ellipse.png', dpi=150, bbox_inches='tight')
        print("Saved visualization as 'julia_circle_to_ellipse.png'")
        plt.show()
    
    return {
        'circle': circle,
        'ellipse': ellipse,
        'transform_matrix': transform_matrix,
        'max_error': max_error
    }



if __name__ == "__main__":
    print("Circle to Ellipse Transformation - Julia Backend Demo")
    print("This example shows integration with VisualGeometryCore.jl")
    print()
    
    try:
        result = main()
        
        print("\n" + "=" * 55)
        print("SUCCESS")
        print("=" * 55)
        print("✓ Julia backend is working!")
        print("✓ Circle to ellipse transformation completed successfully")
        print(f"✓ Numerical accuracy: {result['max_error']:.2e}")
        print("✓ Full VisualGeometry Python interface is available")
        
    except ImportError as e:
        print(f"✗ Julia backend not available: {e}")
        print("\nThis is expected if:")
        print("- JuliaCall has OpenSSL conflicts (see TROUBLESHOOTING.md)")
        print("- Julia environment is not properly set up")
        print("- VisualGeometryCore.jl is not installed")
        print("\nNext steps:")
        print("1. Resolve Julia/OpenSSL setup (see TROUBLESHOOTING.md)")
        print("2. Test Julia backend: python test_interface.py")
        print("3. Run this example again once Julia backend works")
        
    except Exception as e:
        print(f"✗ Error with Julia backend: {e}")
        print("\nThis indicates an issue with the Julia implementation.")
        print("Check that all required Julia functions are implemented.")
    
    print("\nThe Python interface provides:")
    print("- Familiar NumPy arrays and Pythonic API")
    print("- High-performance Julia computations under the hood")
    print("- Seamless integration with scientific Python ecosystem")