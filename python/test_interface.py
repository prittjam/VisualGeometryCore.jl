#!/usr/bin/env python3
"""
Simple test of the VisualGeometry Python interface
"""

import sys
import numpy as np

# Add the package to path for testing
sys.path.insert(0, '.')

try:
    from visualgeometry import Circle, Ellipse, to_homogeneous, to_euclidean
    print("✓ Successfully imported VisualGeometry modules")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def test_coordinate_transforms():
    """Test coordinate transformations"""
    print("\n=== Testing Coordinate Transforms ===")
    
    # Test single point
    point_2d = np.array([1.0, 2.0])
    point_3d = to_homogeneous(point_2d)
    recovered = to_euclidean(point_3d)
    
    print(f"Original 2D: {point_2d}")
    print(f"Homogeneous: {point_3d}")
    print(f"Recovered:   {recovered}")
    
    # Check accuracy
    diff = np.abs(point_2d - recovered)
    if np.max(diff) < 1e-10:
        print("✓ Single point conversion accurate")
    else:
        print(f"✗ Single point conversion error: {np.max(diff)}")
    
    # Test multiple points
    points_2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    points_3d = to_homogeneous(points_2d)
    recovered_2d = to_euclidean(points_3d)
    
    print(f"\nMultiple points test:")
    print(f"Original shape: {points_2d.shape}")
    print(f"Homogeneous shape: {points_3d.shape}")
    print(f"Recovered shape: {recovered_2d.shape}")
    
    diff = np.abs(points_2d - recovered_2d)
    if np.max(diff) < 1e-10:
        print("✓ Multiple points conversion accurate")
    else:
        print(f"✗ Multiple points conversion error: {np.max(diff)}")

def test_circle():
    """Test Circle functionality"""
    print("\n=== Testing Circle ===")
    
    try:
        circle = Circle([2.0, 3.0], 1.5)
        print(f"✓ Created circle: {circle}")
        
        # Test properties
        center = circle.center
        radius = circle.radius
        print(f"Center: {center}, Radius: {radius}")
        
        # Test point generation
        points = circle.points(10)
        print(f"✓ Generated {len(points)} points, shape: {points.shape}")
        
        # Test point containment
        inside_point = np.array([2.1, 3.1])
        outside_point = np.array([5.0, 5.0])
        
        inside = circle.contains_point(inside_point)
        outside = circle.contains_point(outside_point)
        
        print(f"Point {inside_point} inside: {inside}")
        print(f"Point {outside_point} inside: {outside}")
        
        if inside and not outside:
            print("✓ Point containment test passed")
        else:
            print("✗ Point containment test failed")
            
    except Exception as e:
        print(f"✗ Circle test failed: {e}")

def test_ellipse():
    """Test Ellipse functionality"""
    print("\n=== Testing Ellipse ===")
    
    try:
        ellipse = Ellipse([0.0, 0.0], [3.0, 2.0], np.pi/4)
        print(f"✓ Created ellipse: {ellipse}")
        
        # Test properties
        center = ellipse.center
        semi_axes = ellipse.semi_axes
        angle = ellipse.angle
        
        print(f"Center: {center}")
        print(f"Semi-axes: {semi_axes}")
        print(f"Angle: {angle:.3f} rad ({np.degrees(angle):.1f}°)")
        
        # Test point generation
        points = ellipse.points(20)
        print(f"✓ Generated {len(points)} points, shape: {points.shape}")
        
        # Test conversion to conic
        conic = ellipse.to_homogeneous_conic()
        print(f"✓ Converted to HomogeneousConic")
        print(f"Conic matrix shape: {conic.matrix.shape}")
        
    except Exception as e:
        print(f"✗ Ellipse test failed: {e}")

if __name__ == "__main__":
    print("Testing VisualGeometry Python Interface")
    print("=" * 40)
    
    test_coordinate_transforms()
    test_circle()
    test_ellipse()
    
    print("\n" + "=" * 40)
    print("Test completed!")