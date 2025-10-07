#!/usr/bin/env python3
"""
Basic usage examples for VisualGeometry Python interface
"""

import numpy as np
import matplotlib.pyplot as plt
from visualgeometry import Circle, Ellipse, HomogeneousConic, to_homogeneous, to_euclidean


def example_circle():
    """Example using Circle class"""
    print("=== Circle Example ===")
    
    # Create a circle
    center = np.array([2.0, 3.0])
    radius = 1.5
    circle = Circle(center, radius)
    
    print(f"Circle: {circle}")
    
    # Generate points on circle
    points = circle.points(50)
    print(f"Generated {len(points)} points on circle boundary")
    
    # Test point containment
    test_point = np.array([2.1, 3.1])
    inside = circle.contains_point(test_point)
    distance = circle.distance_to_point(test_point)
    print(f"Point {test_point} inside circle: {inside}")
    print(f"Distance to boundary: {distance:.3f}")
    
    return circle, points


def example_ellipse():
    """Example using Ellipse class"""
    print("\n=== Ellipse Example ===")
    
    # Create an ellipse
    center = np.array([0.0, 0.0])
    semi_axes = np.array([3.0, 2.0])  # a=3, b=2
    angle = np.pi / 4  # 45 degrees
    
    ellipse = Ellipse(center, semi_axes, angle)
    print(f"Ellipse: {ellipse}")
    
    # Generate points on ellipse
    points = ellipse.points(100)
    print(f"Generated {len(points)} points on ellipse boundary")
    
    # Convert to homogeneous conic
    conic = ellipse.to_homogeneous_conic()
    print(f"Homogeneous conic matrix:\n{conic.matrix}")
    
    return ellipse, points


def example_coordinate_transforms():
    """Example using coordinate transformations"""
    print("\n=== Coordinate Transform Example ===")
    
    # Euclidean points
    euclidean_points = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    print(f"Euclidean points:\n{euclidean_points}")
    
    # Convert to homogeneous
    homogeneous_points = to_homogeneous(euclidean_points)
    print(f"Homogeneous points:\n{homogeneous_points}")
    
    # Convert back to euclidean
    recovered_points = to_euclidean(homogeneous_points)
    print(f"Recovered euclidean points:\n{recovered_points}")
    
    # Check if conversion is exact
    difference = np.abs(euclidean_points - recovered_points)
    print(f"Max difference: {np.max(difference):.2e}")


def plot_examples():
    """Plot the geometric objects"""
    print("\n=== Plotting Examples ===")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot circle
    circle, circle_points = example_circle()
    ax1.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Circle')
    ax1.plot(circle.center[0], circle.center[1], 'ro', markersize=8, label='Center')
    ax1.set_aspect('equal')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title('Circle')
    
    # Plot ellipse
    ellipse, ellipse_points = example_ellipse()
    ax2.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'g-', linewidth=2, label='Ellipse')
    ax2.plot(ellipse.center[0], ellipse.center[1], 'ro', markersize=8, label='Center')
    ax2.set_aspect('equal')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title('Ellipse (rotated)')
    
    plt.tight_layout()
    plt.savefig('python_examples.png', dpi=150, bbox_inches='tight')
    print("Saved plot as 'python_examples.png'")
    plt.show()


if __name__ == "__main__":
    # Run examples
    example_circle()
    example_ellipse()
    example_coordinate_transforms()
    
    # Create plots if matplotlib is available
    try:
        plot_examples()
    except ImportError:
        print("Matplotlib not available - skipping plots")
        print("Install with: pip install matplotlib")