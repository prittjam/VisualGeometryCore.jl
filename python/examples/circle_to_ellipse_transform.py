#!/usr/bin/env python3
"""
Circle to Ellipse Transformation Example

This example demonstrates the complete workflow:
1. Create a Circle
2. Convert to HomogeneousConic representation
3. Apply an affine transformation
4. Convert the transformed conic back to an Ellipse

This showcases the power of homogeneous representations for geometric transformations.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - plots will be skipped")

def create_affine_transform(scale_x=1.0, scale_y=1.0, rotation=0.0, translation=(0.0, 0.0)):
    """
    Create a 3x3 homogeneous affine transformation matrix
    
    Parameters:
    -----------
    scale_x, scale_y : float
        Scaling factors in x and y directions
    rotation : float
        Rotation angle in radians
    translation : tuple
        Translation (tx, ty)
        
    Returns:
    --------
    ndarray, shape (3, 3)
        Homogeneous transformation matrix
    """
    cos_r = np.cos(rotation)
    sin_r = np.sin(rotation)
    tx, ty = translation
    
    # Combined transformation matrix: T * R * S
    # Where T = translation, R = rotation, S = scaling
    transform = np.array([
        [scale_x * cos_r, -scale_y * sin_r, tx],
        [scale_x * sin_r,  scale_y * cos_r, ty],
        [0.0,              0.0,             1.0]
    ])
    
    return transform

def transform_conic(conic_matrix, transform_matrix):
    """
    Apply affine transformation to a conic section
    
    For a conic C and transformation T, the transformed conic is:
    C' = (T^-1)^T * C * T^-1
    
    Parameters:
    -----------
    conic_matrix : ndarray, shape (3, 3)
        Original conic matrix
    transform_matrix : ndarray, shape (3, 3)
        Affine transformation matrix
        
    Returns:
    --------
    ndarray, shape (3, 3)
        Transformed conic matrix
    """
    # Compute inverse transformation
    T_inv = np.linalg.inv(transform_matrix)
    
    # Apply transformation: C' = (T^-1)^T * C * T^-1
    transformed_conic = T_inv.T @ conic_matrix @ T_inv
    
    return transformed_conic

def conic_to_ellipse_parameters(conic_matrix):
    """
    Extract ellipse parameters from homogeneous conic matrix
    
    Uses eigenvalue decomposition for robust parameter extraction.
    
    Parameters:
    -----------
    conic_matrix : ndarray, shape (3, 3)
        Homogeneous conic matrix
        
    Returns:
    --------
    dict
        Dictionary with keys: center, semi_axes, angle
    """
    # Ensure matrix is symmetric
    C = (conic_matrix + conic_matrix.T) / 2
    
    # Extract quadratic form and linear parts
    A_quad = C[:2, :2]  # 2x2 quadratic part
    b_lin = C[:2, 2]    # Linear part
    c_const = C[2, 2]   # Constant part
    
    # Check if quadratic form is positive definite (ellipse)
    eigenvals_quad = np.linalg.eigvals(A_quad)
    if np.any(eigenvals_quad <= 0):
        raise ValueError("Not an ellipse - quadratic form is not positive definite")
    
    # Find center by solving: A_quad * center + b_lin = 0
    try:
        center = -np.linalg.solve(A_quad, b_lin)
    except np.linalg.LinAlgError:
        raise ValueError("Cannot find ellipse center - singular quadratic form")
    
    # Translate conic to center: evaluate at center
    center_value = center.T @ A_quad @ center + 2 * b_lin.T @ center + c_const
    
    if center_value >= 0:
        raise ValueError("Not an ellipse - discriminant has wrong sign")
    
    # Normalize: divide by -center_value to get standard form
    A_normalized = A_quad / (-center_value)
    
    # Eigenvalue decomposition to find axes
    eigenvals, eigenvecs = np.linalg.eigh(A_normalized)
    
    # Semi-axes are reciprocals of square roots of eigenvalues
    semi_axes = 1.0 / np.sqrt(eigenvals)
    
    # Sort by semi-major axis first
    sort_idx = np.argsort(semi_axes)[::-1]
    semi_axes = semi_axes[sort_idx]
    eigenvecs = eigenvecs[:, sort_idx]
    
    # Calculate rotation angle from first eigenvector (semi-major axis direction)
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
    
    # Normalize angle to [0, π) for ellipse representation
    if angle < 0:
        angle += np.pi
    if angle >= np.pi:
        angle -= np.pi
    
    return {
        'center': center,
        'semi_axes': semi_axes,
        'angle': angle
    }

def demo_circle_to_ellipse_transform():
    """Main demonstration function"""
    print("Circle to Ellipse Transformation Demo")
    print("=" * 40)
    
    # Step 1: Create a unit circle at origin
    print("\n1. Creating Unit Circle")
    print("-" * 25)
    
    circle_center = np.array([0.0, 0.0])
    circle_radius = 1.0
    
    print(f"Circle: center={circle_center}, radius={circle_radius}")
    
    # Generate points on original circle for visualization
    theta = np.linspace(0, 2*np.pi, 100)
    circle_points = np.column_stack([
        circle_center[0] + circle_radius * np.cos(theta),
        circle_center[1] + circle_radius * np.sin(theta)
    ])
    
    # Step 2: Convert to homogeneous conic representation
    print("\n2. Converting to Homogeneous Conic")
    print("-" * 35)
    
    # Unit circle: x² + y² - 1 = 0
    # In homogeneous form: x² + y² - z² = 0
    circle_conic = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0]
    ])
    
    print("Circle conic matrix:")
    print(circle_conic)
    
    # Verify some points satisfy the conic equation
    test_points_homogeneous = np.array([
        [1.0, 0.0, 1.0],  # (1, 0)
        [0.0, 1.0, 1.0],  # (0, 1)
        [0.707, 0.707, 1.0]  # (√2/2, √2/2)
    ])
    
    print("\nVerifying points on circle:")
    for i, point in enumerate(test_points_homogeneous):
        value = point.T @ circle_conic @ point
        euclidean = point[:2] / point[2]
        print(f"  Point {euclidean}: conic value = {value:.6f}")
    
    # Step 3: Create and apply affine transformation
    print("\n3. Applying Affine Transformation")
    print("-" * 33)
    
    # Create a transformation that:
    # - Scales by 2x in x-direction, 1.5x in y-direction
    # - Rotates by 30 degrees
    # - Translates by (1, 2)
    scale_x, scale_y = 2.0, 1.5
    rotation_angle = np.pi / 6  # 30 degrees
    translation = (1.0, 2.0)
    
    transform = create_affine_transform(
        scale_x=scale_x,
        scale_y=scale_y, 
        rotation=rotation_angle,
        translation=translation
    )
    
    print("Affine transformation matrix:")
    print(transform)
    print(f"Scale: ({scale_x}, {scale_y})")
    print(f"Rotation: {rotation_angle:.3f} rad ({np.degrees(rotation_angle):.1f}°)")
    print(f"Translation: {translation}")
    
    # Apply transformation to conic
    transformed_conic = transform_conic(circle_conic, transform)
    
    print("\nTransformed conic matrix:")
    print(transformed_conic)
    
    # Step 4: Extract ellipse parameters from transformed conic
    print("\n4. Converting Transformed Conic to Ellipse")
    print("-" * 42)
    
    try:
        ellipse_params = conic_to_ellipse_parameters(transformed_conic)
        
        print("Extracted ellipse parameters:")
        print(f"  Center: {ellipse_params['center']}")
        print(f"  Semi-axes: {ellipse_params['semi_axes']}")
        print(f"  Rotation: {ellipse_params['angle']:.3f} rad ({np.degrees(ellipse_params['angle']):.1f}°)")
        
        # Generate points on transformed ellipse
        ellipse_center = ellipse_params['center']
        semi_axes = ellipse_params['semi_axes']
        ellipse_angle = ellipse_params['angle']
        
        # Generate ellipse points
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Points on canonical ellipse
        x_canonical = semi_axes[0] * np.cos(theta)
        y_canonical = semi_axes[1] * np.sin(theta)
        
        # Apply rotation
        cos_angle = np.cos(ellipse_angle)
        sin_angle = np.sin(ellipse_angle)
        
        x_rotated = x_canonical * cos_angle - y_canonical * sin_angle
        y_rotated = x_canonical * sin_angle + y_canonical * cos_angle
        
        # Translate to center
        ellipse_points = np.column_stack([
            x_rotated + ellipse_center[0],
            y_rotated + ellipse_center[1]
        ])
        
        # Step 5: Verification - transform original circle points directly
        print("\n5. Verification")
        print("-" * 14)
        
        # Transform original circle points using the affine transformation
        circle_points_homogeneous = np.column_stack([
            circle_points, 
            np.ones(len(circle_points))
        ])
        
        # Apply transformation to points
        transformed_points_homogeneous = (transform @ circle_points_homogeneous.T).T
        
        # Convert back to euclidean
        direct_transformed_points = transformed_points_homogeneous[:, :2] / transformed_points_homogeneous[:, 2:]
        
        # Compare with ellipse points (use closest point matching for verification)
        # Since the parameterization might be different, we check if points lie on the same curve
        
        # Check if direct transformed points satisfy the ellipse equation
        verification_errors = []
        for point in direct_transformed_points[:10]:  # Check first 10 points
            # Translate to ellipse center
            p_centered = point - ellipse_center
            
            # Rotate to ellipse coordinate system
            cos_angle = np.cos(-ellipse_angle)  # Negative for inverse rotation
            sin_angle = np.sin(-ellipse_angle)
            
            x_rot = p_centered[0] * cos_angle - p_centered[1] * sin_angle
            y_rot = p_centered[0] * sin_angle + p_centered[1] * cos_angle
            
            # Check ellipse equation: (x/a)² + (y/b)² = 1
            ellipse_value = (x_rot / semi_axes[0])**2 + (y_rot / semi_axes[1])**2
            verification_errors.append(abs(ellipse_value - 1.0))
        
        max_error = max(verification_errors)
        print(f"Max ellipse equation error for transformed points: {max_error:.2e}")
        
        if max_error < 1e-6:
            print("✓ Verification successful - transformed points lie on ellipse!")
        else:
            print("⚠ Some numerical differences detected")
        
        # Step 6: Visualization
        if HAS_MATPLOTLIB:
            print("\n6. Visualization")
            print("-" * 15)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot original circle
            ax1.plot(circle_points[:, 0], circle_points[:, 1], 'b-', linewidth=2, label='Original Circle')
            ax1.plot(circle_center[0], circle_center[1], 'bo', markersize=8, label='Center')
            ax1.set_aspect('equal')
            ax1.grid(True)
            ax1.legend()
            ax1.set_title('Original Circle')
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            
            # Plot transformed ellipse
            ax2.plot(ellipse_points[:, 0], ellipse_points[:, 1], 'r-', linewidth=2, label='Transformed Ellipse')
            ax2.plot(direct_transformed_points[:, 0], direct_transformed_points[:, 1], 'g--', alpha=0.7, label='Direct Transform')
            ax2.plot(ellipse_center[0], ellipse_center[1], 'ro', markersize=8, label='Center')
            ax2.set_aspect('equal')
            ax2.grid(True)
            ax2.legend()
            ax2.set_title('Transformed Ellipse')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            
            plt.tight_layout()
            plt.savefig('circle_to_ellipse_transform.png', dpi=150, bbox_inches='tight')
            print("Saved visualization as 'circle_to_ellipse_transform.png'")
            plt.show()
        
        return {
            'original_circle': {'center': circle_center, 'radius': circle_radius},
            'transform_matrix': transform,
            'transformed_conic': transformed_conic,
            'ellipse_parameters': ellipse_params,
            'verification_error': max_error
        }
        
    except Exception as e:
        print(f"Error extracting ellipse parameters: {e}")
        return None

def demo_multiple_transformations():
    """Demonstrate multiple different transformations"""
    print("\n" + "=" * 50)
    print("Multiple Transformation Examples")
    print("=" * 50)
    
    transformations = [
        {
            'name': 'Pure Scaling',
            'scale_x': 3.0, 'scale_y': 0.5, 'rotation': 0.0, 'translation': (0, 0)
        },
        {
            'name': 'Pure Rotation', 
            'scale_x': 1.0, 'scale_y': 1.0, 'rotation': np.pi/3, 'translation': (0, 0)
        },
        {
            'name': 'Shear-like Transform',
            'scale_x': 2.0, 'scale_y': 1.0, 'rotation': np.pi/4, 'translation': (-1, 1)
        }
    ]
    
    # Original circle conic
    circle_conic = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, -1.0]
    ])
    
    results = []
    
    for i, params in enumerate(transformations):
        print(f"\n{i+1}. {params['name']}")
        print("-" * (len(params['name']) + 3))
        
        # Create transformation
        transform = create_affine_transform(**{k: v for k, v in params.items() if k != 'name'})
        
        # Apply to conic
        transformed_conic = transform_conic(circle_conic, transform)
        
        # Extract ellipse parameters
        try:
            ellipse_params = conic_to_ellipse_parameters(transformed_conic)
            
            print(f"  Center: ({ellipse_params['center'][0]:.2f}, {ellipse_params['center'][1]:.2f})")
            print(f"  Semi-axes: ({ellipse_params['semi_axes'][0]:.2f}, {ellipse_params['semi_axes'][1]:.2f})")
            print(f"  Rotation: {np.degrees(ellipse_params['angle']):.1f}°")
            
            results.append({
                'name': params['name'],
                'transform': transform,
                'ellipse': ellipse_params
            })
            
        except Exception as e:
            print(f"  Error: {e}")
    
    return results

if __name__ == "__main__":
    print("Circle to Ellipse Transformation Example")
    print("This demonstrates the mathematical workflow:")
    print("Circle → HomogeneousConic → AffineTransform → Ellipse")
    print()
    
    # Run main demonstration
    result = demo_circle_to_ellipse_transform()
    
    if result:
        # Run additional examples
        demo_multiple_transformations()
        
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print("✓ Successfully demonstrated complete transformation pipeline")
        print("✓ Circle converted to homogeneous conic representation")
        print("✓ Affine transformation applied to conic matrix")
        print("✓ Transformed conic converted back to ellipse parameters")
        print("✓ Verification shows numerical accuracy")
        
        if HAS_MATPLOTLIB:
            print("✓ Visualization saved as 'circle_to_ellipse_transform.png'")
        
        print("\nThis example shows the power of homogeneous representations")
        print("for geometric transformations in computer vision and graphics!")
    
    else:
        print("Demo encountered errors - check the implementation")