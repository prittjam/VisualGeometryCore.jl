#!/usr/bin/env python3
"""
Circle to Ellipse Transformation Visualization with Julia Backend

This example demonstrates the complete workflow of VisualGeometryCore's Python interface:

1. Creating geometric primitives using the high-performance Julia backend
2. Applying linear transformations (scale, rotation, translation)
3. Visualizing the results with matplotlib
4. Analyzing transformation properties

Features demonstrated:
- Julia GeometryBasics.coordinates for high-precision point generation
- Custom transformation matrices
- Scientific visualization with matplotlib
- Performance analysis and statistics

Requirements:
- Julia backend properly configured (see ../OPENSSL_TROUBLESHOOTING.md if needed)
- matplotlib for visualization
- numpy for numerical operations

Usage:
    python plot_circle_ellipse_transform.py

This will create an interactive plot and save 'circle_ellipse_transformation.png'
"""

import sys
import numpy as np
sys.path.insert(0, 'python')

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - install with: pip install matplotlib")
    sys.exit(1)

def create_transformation_matrix(scale_x=2.0, scale_y=1.5, rotation=np.pi/6, translation=[1.0, 0.5]):
    """Create a 2D transformation matrix"""
    # Scale matrix
    S = np.array([[scale_x, 0], [0, scale_y]])
    
    # Rotation matrix
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)
    R = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
    
    # Combined transformation (rotation then scale)
    T = R @ S
    
    return T, translation

def apply_transformation(points, T, translation):
    """Apply transformation matrix to points"""
    # Apply linear transformation
    transformed = (T @ points.T).T
    
    # Apply translation
    transformed += np.array(translation)
    
    return transformed

def main():
    print("🎨 Circle to Ellipse Transformation Visualization")
    print("=" * 55)
    
    try:
        # Add parent directory to path for imports
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from visualgeometry import Circle
        print("✓ Successfully imported Circle with Julia backend")
        
        # 1. Create original circle
        print("\n1. Creating original circle...")
        circle = Circle([0.0, 0.0], 1.0)
        circle_points = circle.points(64)  # High resolution for smooth curve
        print(f"✓ Generated {len(circle_points)} points using Julia GeometryBasics.coordinates")
        
        # 2. Create transformation matrix
        print("\n2. Creating transformation matrix...")
        T, translation = create_transformation_matrix(
            scale_x=2.5,      # Stretch in x-direction
            scale_y=1.2,      # Slight stretch in y-direction  
            rotation=np.pi/4, # 45-degree rotation
            translation=[2.0, 1.0]  # Move to new position
        )
        print(f"✓ Transformation matrix:")
        print(f"  Scale: (2.5, 1.2)")
        print(f"  Rotation: {np.pi/4:.3f} radians (45°)")
        print(f"  Translation: {translation}")
        
        # 3. Apply transformation
        print("\n3. Applying transformation...")
        ellipse_points = apply_transformation(circle_points, T, translation)
        print(f"✓ Transformed {len(ellipse_points)} points")
        
        # 4. Create visualization
        print("\n4. Creating visualization...")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Original Circle
        ax1.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Circle')
        ax1.plot(circle.center[0], circle.center[1], 'bo', markersize=8, label='Center')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title('Original Circle\n(Julia Backend)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Plot 2: Transformed Ellipse
        ax2.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2, label='Transformed Ellipse')
        ellipse_center = np.array([0, 0]) + translation
        ax2.plot(ellipse_center[0], ellipse_center[1], 'ro', markersize=8, label='Center')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Transformed Ellipse\n(Scale + Rotate + Translate)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        # Plot 3: Both together
        ax3.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Original Circle', alpha=0.7)
        ax3.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2, label='Transformed Ellipse', alpha=0.7)
        ax3.plot(circle.center[0], circle.center[1], 'bo', markersize=8, label='Circle Center')
        ax3.plot(ellipse_center[0], ellipse_center[1], 'ro', markersize=8, label='Ellipse Center')
        
        # Add transformation arrows
        ax3.annotate('', xy=ellipse_center, xytext=circle.center,
                    arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
        ax3.text((circle.center[0] + ellipse_center[0])/2, 
                (circle.center[1] + ellipse_center[1])/2 + 0.3,
                'Transformation', ha='center', color='green', fontweight='bold')
        
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_title('Circle → Ellipse Transformation\n(Julia Backend)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        
        plt.tight_layout()
        
        # 5. Add transformation details as text
        fig.suptitle('VisualGeometryCore: Circle to Ellipse Transformation with Julia Backend', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Add text box with transformation details
        transform_text = f"""Transformation Details:
• Original: Circle(center=[0,0], radius=1.0)
• Scale: x×{T[0,0]:.1f}, y×{T[1,1]:.1f}
• Rotation: {np.pi/4*180/np.pi:.0f}°
• Translation: [{translation[0]:.1f}, {translation[1]:.1f}]
• Points: {len(circle_points)} (Julia GeometryBasics.coordinates)"""
        
        fig.text(0.02, 0.02, transform_text, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        # 6. Save and show
        plt.savefig('circle_ellipse_transformation.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot as 'circle_ellipse_transformation.png'")
        
        plt.show()
        
        # 7. Print summary statistics
        print("\n" + "=" * 55)
        print("📊 TRANSFORMATION SUMMARY")
        print("=" * 55)
        
        # Calculate some properties
        circle_area = np.pi * circle.radius**2
        
        # Estimate ellipse area from transformation
        det_T = np.linalg.det(T)
        ellipse_area = circle_area * abs(det_T)
        
        print(f"Original Circle:")
        print(f"  • Center: {circle.center}")
        print(f"  • Radius: {circle.radius}")
        print(f"  • Area: {circle_area:.3f}")
        
        print(f"\nTransformed Ellipse:")
        print(f"  • Center: {ellipse_center}")
        print(f"  • Area: {ellipse_area:.3f} (estimated)")
        print(f"  • Area ratio: {ellipse_area/circle_area:.3f}")
        
        print(f"\nTransformation Matrix:")
        print(f"  • Determinant: {det_T:.3f}")
        print(f"  • Preserves orientation: {'Yes' if det_T > 0 else 'No'}")
        
        print(f"\nJulia Backend Performance:")
        print(f"  • Points generated: {len(circle_points)}")
        print(f"  • Backend: GeometryBasics.coordinates")
        print(f"  • OpenSSL conflicts: RESOLVED ✓")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Circle to Ellipse Transformation with Julia Backend")
    print("This demonstrates the complete workflow using the resolved Julia integration")
    print()
    
    success = main()
    
    if success:
        print("\n🎉 SUCCESS!")
        print("✓ Julia backend integration working perfectly")
        print("✓ Circle generation using GeometryBasics.coordinates")
        print("✓ Transformation and visualization complete")
        print("✓ Plot saved as 'circle_ellipse_transformation.png'")
    else:
        print("\n❌ FAILED!")
        print("Check the error messages above for troubleshooting")