#!/usr/bin/env python3
"""
Tests for blob conversions (PyTorch, matplotlib, Circle)
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualgeometry import IsoBlob, Circle, Ellipse


def test_isoblob_creation():
    """Test IsoBlob creation"""
    blob = IsoBlob(center=[100, 200], sigma=5.0)

    assert np.allclose(blob.center, [100, 200])
    assert np.isclose(blob.sigma, 5.0)
    print("✓ IsoBlob creation test passed")


def test_blob_to_circle():
    """Test blob to Circle conversion"""
    blob = IsoBlob(center=[100, 200], sigma=5.0)
    circle = blob.to_circle(cutoff=3.0)

    assert np.allclose(circle.center, [100, 200])
    assert np.isclose(circle.radius, 15.0)  # 3.0 * 5.0
    print("✓ Blob to Circle conversion test passed")


def test_blob_to_torch():
    """Test blob to PyTorch tensor conversion"""
    try:
        import torch
    except ImportError:
        print("⊘ PyTorch not installed - skipping torch tests")
        return

    blob = IsoBlob(center=[100, 200], sigma=5.0)
    tensor = blob.to_torch_tensor(cutoff=3.0)

    assert tensor.shape == (3,)
    assert torch.allclose(tensor, torch.tensor([100.0, 200.0, 15.0]))
    print("✓ Blob to PyTorch tensor test passed")


def test_blob_to_matplotlib():
    """Test blob to matplotlib patch conversion"""
    try:
        from matplotlib.patches import Circle as MPLCircle
    except ImportError:
        print("⊘ Matplotlib not installed - skipping matplotlib tests")
        return

    blob = IsoBlob(center=[100, 200], sigma=5.0)
    patch = blob.to_mpl_circle(cutoff=3.0)

    assert isinstance(patch, MPLCircle)
    assert np.allclose(patch.center, [100, 200])
    assert np.isclose(patch.radius, 15.0)
    print("✓ Blob to matplotlib Circle patch test passed")


def test_circle_to_torch():
    """Test Circle to PyTorch tensor conversion"""
    try:
        import torch
    except ImportError:
        print("⊘ PyTorch not installed - skipping torch tests")
        return

    circle = Circle(center=[100, 200], radius=15.0)
    tensor = circle.to_torch_tensor()

    assert tensor.shape == (3,)
    assert torch.allclose(tensor, torch.tensor([100.0, 200.0, 15.0]))
    print("✓ Circle to PyTorch tensor test passed")


def test_circle_to_matplotlib():
    """Test Circle to matplotlib patch conversion"""
    try:
        from matplotlib.patches import Circle as MPLCircle
    except ImportError:
        print("⊘ Matplotlib not installed - skipping matplotlib tests")
        return

    circle = Circle(center=[100, 200], radius=15.0)
    patch = circle.to_mpl_circle()

    assert isinstance(patch, MPLCircle)
    assert np.allclose(patch.center, [100, 200])
    assert np.isclose(patch.radius, 15.0)
    print("✓ Circle to matplotlib Circle patch test passed")


def test_ellipse_to_torch():
    """Test Ellipse to PyTorch tensor conversion"""
    try:
        import torch
    except ImportError:
        print("⊘ PyTorch not installed - skipping torch tests")
        return

    ellipse = Ellipse(center=[100, 200], semi_axes=[20, 10], angle=np.pi/4)
    tensor = ellipse.to_torch_tensor()

    assert tensor.shape == (5,)
    expected = torch.tensor([100.0, 200.0, 20.0, 10.0, np.pi/4])
    assert torch.allclose(tensor, expected)
    print("✓ Ellipse to PyTorch tensor test passed")


def test_ellipse_to_matplotlib():
    """Test Ellipse to matplotlib patch conversion"""
    try:
        from matplotlib.patches import Ellipse as MPLEllipse
    except ImportError:
        print("⊘ Matplotlib not installed - skipping matplotlib tests")
        return

    ellipse = Ellipse(center=[100, 200], semi_axes=[20, 10], angle=np.pi/4)
    patch = ellipse.to_mpl_ellipse()

    assert isinstance(patch, MPLEllipse)
    assert np.allclose(patch.center, [100, 200])
    assert np.isclose(patch.width, 40.0)  # 2 * 20
    assert np.isclose(patch.height, 20.0)  # 2 * 10
    assert np.isclose(patch.angle, 45.0)  # pi/4 in degrees
    print("✓ Ellipse to matplotlib Ellipse patch test passed")


def test_batch_processing():
    """Test batch processing of blobs"""
    try:
        import torch
    except ImportError:
        print("⊘ PyTorch not installed - skipping batch processing tests")
        return

    blobs = [
        IsoBlob(center=[100, 200], sigma=5.0),
        IsoBlob(center=[150, 250], sigma=7.5),
        IsoBlob(center=[200, 300], sigma=10.0),
    ]

    tensors = [blob.to_torch_tensor(cutoff=3.0) for blob in blobs]
    batch = torch.stack(tensors)

    assert batch.shape == (3, 3)
    assert torch.allclose(batch[0], torch.tensor([100.0, 200.0, 15.0]))
    assert torch.allclose(batch[1], torch.tensor([150.0, 250.0, 22.5]))
    assert torch.allclose(batch[2], torch.tensor([200.0, 300.0, 30.0]))
    print("✓ Batch processing test passed")


def main():
    print("Running blob conversion tests...")
    print("=" * 60)

    try:
        test_isoblob_creation()
        test_blob_to_circle()
        test_blob_to_torch()
        test_blob_to_matplotlib()
        test_circle_to_torch()
        test_circle_to_matplotlib()
        test_ellipse_to_torch()
        test_ellipse_to_matplotlib()
        test_batch_processing()

        print("=" * 60)
        print("✓ All tests passed!")
        return True

    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
