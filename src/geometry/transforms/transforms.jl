# =============================================================================
# Coordinate Transformations and Mappings
# =============================================================================
#
# This module provides coordinate transformation functionality:
# - homogeneous.jl: Homogeneous 2D transform matrices
# - conversions.jl: Coordinate system conversions
# - coord_maps.jl:  Generic coord_map API for mapping between geometries
# - logpolar.jl:    Log-polar coordinate transformations

include("homogeneous.jl")
include("conversions.jl")
include("coord_maps.jl")
include("logpolar.jl")
