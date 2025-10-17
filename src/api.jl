# Essential types for function arguments
export Size2                    # For specifying pattern dimensions
export IsoBlob                  # Blob objects returned by pattern functions

# Essential units for measurements
export pd, mm, cm, m, inch, ft, dpi, px, pt, μm # Core units: pixels/dots, length units (mm, cm, m, inch, ft), density (dpi), pixels, points, micrometers

# Unit conversion functions
export to_logical_units, to_physical_units  # Convert IsoBlob between logical and physical units

# Coordinate convention conversion
export apply_pixel_convention, shift_origin  # Generic pixel coordinate convention conversion

# Additional types and constants needed by BlobBoards
export PixelCount, Len, LogicalCount, LogicalDensity, LogicalWidth
export CLEAN_TOL
export filter_kwargs, validate_dir
export FeaturePolarity, PositiveFeature, NegativeFeature, ImageFeature, AbstractBlob, IsoBlobDetection
export extract_field, ScalarOrQuantity
export intersects

# Scale space exports
export OctaveGeometry, ScaleLevel, ScaleSpace
export get_octave_geometry, get_octave_levels, get_sigma
export memory_usage, valid_octave_range, valid_scale_range, level_exists, iterate_levels, iterate_scale_levels
export filter_by_octave, filter_by_sigma_range, apply_to_all_levels!
export scale_coordinates, octave_to_input_coordinates, input_to_octave_coordinates
export gaussian_kernel, populate_octave!, populate_scale_space!
export compute_scale_normalized_laplacian!, apply_gaussian_to_all!, filter_high_sigma_levels

export StructTypes

# Plotting functionality using Makie Spec API with convert_arguments
# Users can directly call:
# - plot(pattern, blobs; color=:red, scale_factor=2.0)
# - plot(pattern, detected_blobs, ground_truth_blobs; detected_color=:red, ground_truth_color=:green)
export imshow
