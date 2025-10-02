# Essential types for function arguments
export Size2                    # For specifying pattern dimensions
export IsoBlob                  # Blob objects returned by pattern functions
export ImageBlobs               # Compound plot type for image + blobs overlay

# Essential units for measurements
export pd, mm, inch, dpi, px, pt, μm # Core units: pixels/dots, millimeters, inches, dots-per-inch, pixels, points

# Unit conversion functions
export to_logical_units, to_physical_units  # Convert IsoBlob between logical and physical units

# Additional types and constants needed by BlobBoards
export PixelCount, Len, LogicalCount, LogicalDensity, LogicalWidth
export SIGMA_CUTOFF, CLEAN_TOL
export filter_kwargs, validate_dir
export FeaturePolarity, PositiveFeature, NegativeFeature, ImageFeature, AbstractBlob, IsoBlobDetection
export extract_field, ScalarOrQuantity
export intersects

export StructTypes

# Plotting functionality using Makie Spec API with convert_arguments
# Users can directly call:
# - plot(pattern, blobs; color=:red, scale_factor=2.0)
# - plot(pattern, detected_blobs, ground_truth_blobs; detected_color=:red, ground_truth_color=:green)
