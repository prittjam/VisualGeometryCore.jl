# =============================================================================
# Core Utilities
# =============================================================================
#
# This module coordinates core utility functionality by including specialized files:
# - json.jl:            JSON3/StructTypes serialization support
# - geometry_utils.jl:  Geometry utilities (center, ranges, coordinate conversions)
# - sampling.jl:        Random sampling from geometric primitives
# - image_utils.jl:     Image processing utilities
# =============================================================================

# =============================================================================
# Type Utilities
# =============================================================================

"""
    neltype(x)
    neltype(::Type{T})

Get the nested element type by recursively descending through aggregate types.

For arrays and other container types, this recursively calls `eltype` until
reaching a non-container (atomic) type. For non-container types, returns the type itself.

# Examples
```julia
# Vector of vectors
neltype(Vector{Vector{Float64}}) # Float64

# Vector of static arrays with units
blob_positions = [SVector(1.0px, 2.0px), SVector(3.0px, 4.0px)]
neltype(blob_positions) # Unitful.Quantity{Float64, 𝐍, Unitful.FreeUnits{(px,), 𝐍, nothing}}

# Get zero with proper units
z = zero(neltype(blob_positions)) # 0.0 px
```
"""
neltype(x) = neltype(typeof(x))
neltype(::Type{T}) where T <: AbstractArray = neltype(eltype(T))
neltype(::Type{T}) where T = T

# =============================================================================
# Helper Functions
# =============================================================================

"""
    filter_kwargs(T; kwargs...) -> (allowed, unknown)

Split keyword args into those that match fieldnames of `T` and those that don't.
"""
function filter_kwargs(::Type{T}; kwargs...) where {T}
    fns = Set(fieldnames(T))
    ks  = keys(kwargs)
    keep = Tuple(k for k in ks if k in fns)
    drop = Tuple(k for k in ks if k ∉ fns)

    # Create NamedTuples with only the relevant keys and values
    keep_values = Tuple(kwargs[k] for k in keep)
    drop_values = Tuple(kwargs[k] for k in drop)

    return NamedTuple{keep}(keep_values), NamedTuple{drop}(drop_values)
end

"""
    validate_dir(dir::AbstractString; create::Bool=false)

Validate that a directory exists, optionally creating it if it doesn't.

# Arguments
- `dir`: Path to the directory to validate
- `create=false`: If true, create the directory if it doesn't exist

# Returns
- The validated directory path

# Throws
- `ArgumentError` if the directory doesn't exist and `create=false`
"""
function validate_dir(dir::AbstractString; create::Bool=false)
    if create && !isdir(dir)
        mkpath(dir)
    end
    if !isdir(dir)
        throw(ArgumentError("Directory does not exist: $dir"))
    end
    return dir
end

# =============================================================================
# UID Generation Constants
# =============================================================================

# Choose one:
const ALPHABET_BASE64 = collect("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
const ALPHABET_BASE58 = collect("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

# =============================================================================
# Include Specialized Utility Modules
# =============================================================================

include("json.jl")
include("geometry_utils.jl")
include("sampling.jl")
include("image_utils.jl")
