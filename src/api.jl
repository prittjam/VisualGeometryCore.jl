# Essential types for function arguments
export Size2                    # For specifying pattern dimensions
export IsoBlob                  # Blob objects returned by pattern functions

# Essential units for measurements
export pd, mm, inch, dpi, px, pt, μm # Core units: pixels/dots, millimeters, inches, dots-per-inch, pixels, points

# Unit conversion functions
export to_logical_units, to_physical_units  # Convert IsoBlob between logical and physical units

# Additional types and constants needed by BlobBoards
export PixelCount, Len, LogicalCount, LogicalDensity, LogicalWidth
export SIGMA_CUTOFF, CLEAN_TOL
export filter_kwargs
export FeaturePolarity, PositiveFeature, NegativeFeature, ImageFeature, AbstractBlob, IsoBlobDetection
export extract_field, ScalarOrQuantity

export StructTypes
