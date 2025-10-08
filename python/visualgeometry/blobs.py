"""
Python interface for blob features (IsoBlob, IsoBlobDetection)
"""

import numpy as np
from .core import VisualGeometryCore, jl


class IsoBlob:
    """Python wrapper for IsoBlob from VisualGeometryCore.jl"""

    def __init__(self, center=None, sigma=None, julia_blob=None):
        """
        Create an IsoBlob

        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point (x, y)
        sigma : float
            Scale parameter (standard deviation)
        julia_blob : Julia object
            Existing Julia IsoBlob object
        """
        VisualGeometryCore.ensure_initialized()

        if julia_blob is not None:
            self._julia_obj = julia_blob
        elif center is not None and sigma is not None:
            center = np.asarray(center)

            if center.shape != (2,):
                raise ValueError("Center must be 2D point")

            # Create Julia IsoBlob using Point2 constructor
            # Note: Using pd (pixel/dot) units for logical coordinates
            self._julia_obj = jl.seval(f"IsoBlob(Point2({center[0]}pd, {center[1]}pd), {sigma}pd)")
        else:
            raise ValueError("Either (center, sigma) or julia_blob must be provided")

    @property
    def center(self):
        """Get blob center as NumPy array"""
        center_julia = self._julia_obj.center
        # Strip units for NumPy array
        return np.array([float(jl.ustrip(center_julia[1])),
                        float(jl.ustrip(center_julia[2]))])

    @property
    def sigma(self):
        """Get scale parameter (sigma) as float"""
        return float(jl.ustrip(self._julia_obj.σ))

    def to_circle(self, cutoff=3.0):
        """
        Convert blob to Circle wrapper with radius = cutoff * σ

        Parameters:
        -----------
        cutoff : float
            Radius multiplier (default: 3.0 for 3σ radius)

        Returns:
        --------
        Circle
            Circle wrapper with full geometric functionality

        Example:
        --------
        >>> blob = IsoBlob(center=[100, 200], sigma=5)
        >>> circle = blob.to_circle(cutoff=3.0)
        >>> circle.radius
        15.0
        """
        from .circles import Circle
        # Use Julia Circle constructor from blob
        julia_circle = jl.Circle(self._julia_obj, cutoff)
        return Circle(julia_circle=julia_circle)

    def to_torch_tensor(self, cutoff=3.0):
        """
        Convert blob to PyTorch tensor [x, y, r] for ML workflows

        Parameters:
        -----------
        cutoff : float
            Radius multiplier (default: 3.0 for 3σ radius)

        Returns:
        --------
        torch.Tensor
            Tensor with shape (3,) containing [x, y, radius]

        Example:
        --------
        >>> blob = IsoBlob(center=[100, 200], sigma=5)
        >>> tensor = blob.to_torch_tensor(cutoff=3.0)
        >>> tensor
        tensor([100., 200., 15.])
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_torch_tensor(). Install with: pip install torch")

        return torch.tensor([self.center[0], self.center[1], cutoff * self.sigma])

    def to_mpl_circle(self, cutoff=3.0, **kwargs):
        """
        Convert blob to matplotlib Circle patch for quick plotting

        Parameters:
        -----------
        cutoff : float
            Radius multiplier (default: 3.0 for 3σ radius)
        **kwargs
            Additional arguments passed to matplotlib.patches.Circle
            (e.g., facecolor, edgecolor, alpha, fill)

        Returns:
        --------
        matplotlib.patches.Circle
            Circle patch ready for adding to matplotlib axes

        Example:
        --------
        >>> import matplotlib.pyplot as plt
        >>> blob = IsoBlob(center=[100, 200], sigma=5)
        >>> fig, ax = plt.subplots()
        >>> circle = blob.to_mpl_circle(cutoff=3.0, fill=False, edgecolor='r')
        >>> ax.add_patch(circle)
        >>> plt.show()
        """
        try:
            from matplotlib.patches import Circle as MPLCircle
        except ImportError:
            raise ImportError("Matplotlib is required for to_mpl_circle(). Install with: pip install matplotlib")

        return MPLCircle(self.center, cutoff * self.sigma, **kwargs)

    def __repr__(self):
        return f"IsoBlob(center={self.center}, σ={self.sigma:.3f})"


class IsoBlobDetection:
    """Python wrapper for IsoBlobDetection from VisualGeometryCore.jl"""

    def __init__(self, center=None, sigma=None, response=None, polarity=None, julia_detection=None):
        """
        Create an IsoBlobDetection

        Parameters:
        -----------
        center : array-like, shape (2,)
            Center point (x, y)
        sigma : float
            Scale parameter (standard deviation)
        response : float
            Detection response score
        polarity : str
            Feature polarity - "PositiveFeature" or "NegativeFeature"
        julia_detection : Julia object
            Existing Julia IsoBlobDetection object
        """
        VisualGeometryCore.ensure_initialized()

        if julia_detection is not None:
            self._julia_obj = julia_detection
        elif center is not None and sigma is not None and response is not None and polarity is not None:
            center = np.asarray(center)

            if center.shape != (2,):
                raise ValueError("Center must be 2D point")
            if polarity not in ["PositiveFeature", "NegativeFeature"]:
                raise ValueError("Polarity must be 'PositiveFeature' or 'NegativeFeature'")

            # Create Julia IsoBlobDetection
            self._julia_obj = jl.seval(
                f"IsoBlobDetection(Point2({center[0]}pd, {center[1]}pd), {sigma}pd, {response}, {polarity})"
            )
        else:
            raise ValueError("Either all parameters or julia_detection must be provided")

    @property
    def center(self):
        """Get blob center as NumPy array"""
        center_julia = self._julia_obj.center
        return np.array([float(jl.ustrip(center_julia[1])),
                        float(jl.ustrip(center_julia[2]))])

    @property
    def sigma(self):
        """Get scale parameter (sigma) as float"""
        return float(jl.ustrip(self._julia_obj.σ))

    @property
    def response(self):
        """Get detection response score"""
        return float(self._julia_obj.response)

    @property
    def polarity(self):
        """Get feature polarity as string"""
        return str(self._julia_obj.polarity)

    def to_circle(self, cutoff=3.0):
        """Convert to Circle (inherits from IsoBlob functionality)"""
        from .circles import Circle
        julia_circle = jl.Circle(self._julia_obj, cutoff)
        return Circle(julia_circle=julia_circle)

    def to_torch_tensor(self, cutoff=3.0):
        """Convert to PyTorch tensor [x, y, r] (inherits from IsoBlob functionality)"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for to_torch_tensor(). Install with: pip install torch")

        return torch.tensor([self.center[0], self.center[1], cutoff * self.sigma])

    def to_mpl_circle(self, cutoff=3.0, **kwargs):
        """Convert to matplotlib Circle patch (inherits from IsoBlob functionality)"""
        try:
            from matplotlib.patches import Circle as MPLCircle
        except ImportError:
            raise ImportError("Matplotlib is required for to_mpl_circle(). Install with: pip install matplotlib")

        return MPLCircle(self.center, cutoff * self.sigma, **kwargs)

    def __repr__(self):
        return f"IsoBlobDetection(center={self.center}, σ={self.sigma:.3f}, response={self.response:.3f}, polarity={self.polarity})"
