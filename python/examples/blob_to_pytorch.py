#!/usr/bin/env python3
"""
Blob Extraction with PyTorch and Matplotlib Integration

This example demonstrates the complete workflow for blob detection and conversion:

1. Extract blobs from JSON files (from BlobBoards or detection results)
2. Convert to PyTorch tensors for ML workflows
3. Convert to matplotlib patches for visualization
4. Convert to full Circle/Ellipse wrappers for geometric operations

Features:
- IsoBlob Python wrapper for blob manipulation
- PyTorch tensor conversion [x, y, r] for neural networks
- Matplotlib patches for quick visualization
- Full geometric functionality via Circle wrapper

Requirements:
- Julia backend with VisualGeometryCore.jl
- PyTorch (pip install torch)
- Matplotlib (pip install matplotlib)
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def example_isoblob_creation():
    """Create IsoBlob directly in Python"""
    print("=" * 60)
    print("Example 1: Creating IsoBlob in Python")
    print("=" * 60)

    from visualgeometry import IsoBlob

    # Create blob with center and sigma
    blob = IsoBlob(center=[100, 200], sigma=5.0)

    print(f"Created blob: {blob}")
    print(f"  Center: {blob.center}")
    print(f"  Sigma: {blob.sigma}")
    print()

def example_blob_to_circle():
    """Convert blob to Circle wrapper for geometric operations"""
    print("=" * 60)
    print("Example 2: Blob to Circle Conversion")
    print("=" * 60)

    from visualgeometry import IsoBlob

    blob = IsoBlob(center=[100, 200], sigma=5.0)

    # Convert to Circle with 3σ radius
    circle = blob.to_circle(cutoff=3.0)

    print(f"Original blob: {blob}")
    print(f"Converted circle: {circle}")
    print(f"  Circle radius: {circle.radius} (= 3.0 * {blob.sigma})")

    # Circle has full geometric functionality
    points = circle.points(64)
    print(f"  Generated {len(points)} boundary points using Julia backend")

    # Test containment
    test_point = [100, 200]
    print(f"  Point {test_point} inside circle: {circle.contains_point(test_point)}")
    print()

def example_blob_to_pytorch():
    """Convert blob to PyTorch tensor for ML workflows"""
    print("=" * 60)
    print("Example 3: Blob to PyTorch Tensor")
    print("=" * 60)

    try:
        import torch
    except ImportError:
        print("PyTorch not installed - skipping PyTorch examples")
        print("Install with: pip install torch")
        print()
        return

    from visualgeometry import IsoBlob

    # Create multiple blobs
    blobs = [
        IsoBlob(center=[100, 200], sigma=5.0),
        IsoBlob(center=[150, 250], sigma=7.5),
        IsoBlob(center=[200, 300], sigma=10.0),
    ]

    # Convert to PyTorch tensors
    cutoff = 3.0
    tensors = [blob.to_torch_tensor(cutoff=cutoff) for blob in blobs]

    # Stack into batch tensor
    batch = torch.stack(tensors)

    print(f"Created {len(blobs)} blobs")
    print(f"Batch tensor shape: {batch.shape}  # (N, 3) format: [x, y, r]")
    print(f"Batch tensor:\n{batch}")

    # Extract components
    centers = batch[:, :2]
    radii = batch[:, 2]

    print(f"\nCenters:\n{centers}")
    print(f"Radii: {radii}")
    print(f"Mean radius: {radii.mean().item():.2f}")
    print()

def example_blob_to_matplotlib():
    """Convert blob to matplotlib Circle patch for visualization"""
    print("=" * 60)
    print("Example 4: Blob to Matplotlib Visualization")
    print("=" * 60)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not installed - skipping matplotlib examples")
        print("Install with: pip install matplotlib")
        print()
        return

    from visualgeometry import IsoBlob

    # Create blobs at different scales
    blobs = [
        IsoBlob(center=[1, 1], sigma=0.2),
        IsoBlob(center=[2, 1.5], sigma=0.3),
        IsoBlob(center=[1.5, 2.2], sigma=0.25),
    ]

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))

    # Add blob circles to plot
    cutoff = 3.0
    for i, blob in enumerate(blobs):
        circle_patch = blob.to_mpl_circle(
            cutoff=cutoff,
            fill=False,
            edgecolor=f'C{i}',
            linewidth=2,
            label=f'Blob {i+1} (σ={blob.sigma:.2f})'
        )
        ax.add_patch(circle_patch)

        # Mark center
        ax.plot(blob.center[0], blob.center[1], 'o', color=f'C{i}', markersize=8)

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Blob Visualization (cutoff={cutoff}σ)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    output_file = 'blob_matplotlib_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization as '{output_file}'")
    plt.close()
    print()

def example_circle_ellipse_conversions():
    """Demonstrate Circle and Ellipse PyTorch/matplotlib conversions"""
    print("=" * 60)
    print("Example 5: Circle & Ellipse Conversions")
    print("=" * 60)

    from visualgeometry import Circle, Ellipse
    import torch

    # Circle conversions
    circle = Circle(center=[100, 200], radius=15)

    print("Circle conversions:")
    print(f"  PyTorch tensor: {circle.to_torch_tensor()}")
    print(f"  Matplotlib patch: {circle.to_mpl_circle(fill=False, edgecolor='r')}")

    # Ellipse conversions
    ellipse = Ellipse(center=[100, 200], semi_axes=[20, 10], angle=np.pi/4)

    print("\nEllipse conversions:")
    print(f"  PyTorch tensor: {ellipse.to_torch_tensor()}")
    print(f"  Matplotlib patch: {ellipse.to_mpl_ellipse(fill=False, edgecolor='b')}")
    print()

def example_batch_processing():
    """Demonstrate batch processing of blobs for ML"""
    print("=" * 60)
    print("Example 6: Batch Processing for ML")
    print("=" * 60)

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch not installed - skipping batch processing example")
        print()
        return

    from visualgeometry import IsoBlob

    # Simulate detected blobs
    np.random.seed(42)
    n_blobs = 10
    blobs = [
        IsoBlob(
            center=[np.random.uniform(0, 800), np.random.uniform(0, 600)],
            sigma=np.random.uniform(3, 15)
        )
        for _ in range(n_blobs)
    ]

    # Convert to batch tensor
    cutoff = 3.0
    batch = torch.stack([blob.to_torch_tensor(cutoff=cutoff) for blob in blobs])

    print(f"Batch of {n_blobs} blobs: {batch.shape}")

    # Normalize for neural network input
    # Normalize centers to [0, 1]
    centers = batch[:, :2]
    centers_norm = centers / torch.tensor([800.0, 600.0])

    # Normalize radii
    radii = batch[:, 2]
    radii_norm = radii / radii.max()

    # Combine
    normalized_batch = torch.cat([centers_norm, radii_norm.unsqueeze(1)], dim=1)

    print(f"Normalized batch:\n{normalized_batch}")
    print(f"\nNormalized batch stats:")
    print(f"  Center range: [{centers_norm.min().item():.3f}, {centers_norm.max().item():.3f}]")
    print(f"  Radii range: [{radii_norm.min().item():.3f}, {radii_norm.max().item():.3f}]")
    print()

def main():
    print("🎨 Blob Extraction with PyTorch & Matplotlib Integration")
    print("=" * 60)
    print()

    try:
        # Run all examples
        example_isoblob_creation()
        example_blob_to_circle()
        example_blob_to_pytorch()
        example_blob_to_matplotlib()
        example_circle_ellipse_conversions()
        example_batch_processing()

        print("=" * 60)
        print("✓ All examples completed successfully!")
        print()
        print("Summary of conversions:")
        print("  • IsoBlob → Circle:     Full geometric functionality")
        print("  • IsoBlob → PyTorch:    [x, y, r] tensor for ML")
        print("  • IsoBlob → Matplotlib: Patch for visualization")
        print("  • Circle → PyTorch:     [x, y, r] tensor")
        print("  • Circle → Matplotlib:  Circle patch")
        print("  • Ellipse → PyTorch:    [x, y, a, b, θ] tensor")
        print("  • Ellipse → Matplotlib: Ellipse patch")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
