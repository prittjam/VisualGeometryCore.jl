"""
Scale space construction for multi-scale blob detection.

This module implements Gaussian scale space similar to VLFeat, with:
- Multi-octave pyramid structure
- Preallocated image storage with proper grayscale types
- Flexible octave geometry
- Efficient broadcasting operations via StructArrays
- Type-safe grayscale processing with intensities guaranteed in [0,1]
- Support for both 8-bit (N0f8) and 16-bit (N0f16) fixed-point precision
- Proper handling of derivative operations using floating point types
"""


# =============================================================================
# Core Types
# =============================================================================

"""
    ScaleLevel{T}

A single level in the scale space pyramid with associated metadata.
"""
struct ScaleLevel{T}
    data::T
    octave::Int
    subdivision::Int
    sigma::Float64
end

"""
    ScaleSpace{T}

Preallocated scale space pyramid with efficient StructArray storage and 2D indexing.

T can be:
- `GaussianImage{N0f8}` for grayscale smoothed images with 8-bit precision
- `GaussianImage{N0f16}` for grayscale smoothed images with 16-bit precision  
- `HessianImages{Float32}` for Hessian second derivative components
- `LaplacianImage{Float32}` for Laplacian components
- Any other NamedTuple data structure needed for the scale space

All image data uses proper color types to ensure intensities are in [0,1] for grayscale
and appropriate floating point types for derivatives that can be negative.

# Indexing
ScaleSpace supports 2D indexing with `ss[octave, subdivision]`:
- `ss[0, 1]` - Access level at octave 0, subdivision 1
- `ss[CartesianIndex(0, 1)]` - CartesianIndex support
- Standard Julia iteration: `for level in ss`, `map(f, ss)`, etc.
- `size(ss)` returns `(num_octaves, num_subdivisions)`
- `axes(ss)` returns the actual octave and subdivision ranges
"""
struct ScaleSpace{T}
    first_octave::Int
    last_octave::Int
    octave_resolution::Int
    octave_first_subdivision::Int
    octave_last_subdivision::Int
    base_sigma::Float64
    camera_psf::Float64
    levels::StructArray{ScaleLevel{T}}
    input_size::Size2{Int}

    # Inner constructor for building scale space with image type specification
    function ScaleSpace(first_octave::Int, last_octave::Int, octave_resolution::Int,
                       octave_first_subdivision::Int, octave_last_subdivision::Int,
                       base_sigma::Float64, camera_psf::Float64,
                       input_size::Size2{Int}, image_type::Type{T}) where T
        octave_range = first_octave:last_octave
        subdivision_range = octave_first_subdivision:octave_last_subdivision

        # Create combinations in octave-major order:
        # (o0,s0), (o0,s1), (o0,s2), (o1,s0), (o1,s1), (o1,s2), ...
        octaves = Int[]
        subdivisions = Int[]
        for o in octave_range
            for s in subdivision_range
                push!(octaves, o)
                push!(subdivisions, s)
            end
        end

        sigmas = base_sigma .* 2.0 .^ (octaves .+ subdivisions ./ octave_resolution)
        sizes = [Size2(max(1, round(Int, input_size.width * 2.0^(-o))),
                       max(1, round(Int, input_size.height * 2.0^(-o)))) for o in octaves]

        # Create data arrays based on image type
        field_names = fieldnames(image_type)
        field_types = fieldtypes(image_type)
        data_arrays = [NamedTuple{field_names}(ntuple(i -> field_types[i](undef, sz.height, sz.width), 
                                                      length(field_names))) for sz in sizes]

        levels = StructArray{ScaleLevel{T}}((
            data = data_arrays,
            octave = octaves,
            subdivision = subdivisions,
            sigma = sigmas
        ))

        return new{T}(first_octave, last_octave, octave_resolution,
                     octave_first_subdivision, octave_last_subdivision,
                     base_sigma, camera_psf, levels, input_size)
    end


end

# =============================================================================
# Property Accessors
# =============================================================================

# =============================================================================
# Custom Indexing Interface
# =============================================================================

"""
    Base.size(ss::ScaleSpace)

Return the size of the scale space as (num_octaves, num_subdivisions).
This enables 2D indexing with ss[octave, subdivision].
"""
function Base.size(ss::ScaleSpace)
    num_octaves = ss.last_octave - ss.first_octave + 1
    num_subdivisions = ss.octave_last_subdivision - ss.octave_first_subdivision + 1
    return (num_octaves, num_subdivisions)
end

"""
    Base.axes(ss::ScaleSpace)

Return the axes ranges for the scale space indexing.
Enables indexing with actual octave and subdivision values.
"""
function Base.axes(ss::ScaleSpace)
    octave_range = ss.first_octave:ss.last_octave
    subdivision_range = ss.octave_first_subdivision:ss.octave_last_subdivision
    return (octave_range, subdivision_range)
end

"""
    Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int)

Access a scale level using 2D indexing: ss[octave, subdivision].

# Arguments
- `ss`: ScaleSpace instance
- `octave`: Octave number (within ss.first_octave:ss.last_octave)
- `subdivision`: Subdivision number (within ss.octave_first_subdivision:ss.octave_last_subdivision)

# Returns
- ScaleLevel at the specified octave and subdivision

# Example
```julia
ss = ScaleSpace(Size2(256, 256))
level = ss[0, 1]  # Get level at octave 0, subdivision 1
data = ss[0, 1].data  # Access the data directly
```
"""
function Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int)
    octave_range, subdivision_range = axes(ss)
    @boundscheck begin
        octave in octave_range || throw(BoundsError(ss, (octave, subdivision)))
        subdivision in subdivision_range || throw(BoundsError(ss, (octave, subdivision)))
    end
    
    subdivisions_per_octave = length(subdivision_range)
    octave_offset = octave - first(octave_range)
    subdivision_offset = subdivision - first(subdivision_range)
    
    linear_idx = octave_offset * subdivisions_per_octave + subdivision_offset + 1
    return ss.levels[linear_idx]
end

"""
    Base.getindex(ss::ScaleSpace, I::CartesianIndex{2})

Access a scale level using CartesianIndex.
"""
Base.getindex(ss::ScaleSpace, I::CartesianIndex{2}) = ss[I[1], I[2]]

"""
    Base.setindex!(ss::ScaleSpace, value, octave::Int, subdivision::Int)

Set a scale level using 2D indexing: ss[octave, subdivision] = value.
"""
function Base.setindex!(ss::ScaleSpace, value, octave::Int, subdivision::Int)
    octave_range, subdivision_range = axes(ss)
    @boundscheck begin
        octave in octave_range || throw(BoundsError(ss, (octave, subdivision)))
        subdivision in subdivision_range || throw(BoundsError(ss, (octave, subdivision)))
    end
    
    subdivisions_per_octave = length(subdivision_range)
    octave_offset = octave - first(octave_range)
    subdivision_offset = subdivision - first(subdivision_range)
    
    linear_idx = octave_offset * subdivisions_per_octave + subdivision_offset + 1
    ss.levels[linear_idx] = value
    return value
end

"""
    Base.setindex!(ss::ScaleSpace, value, I::CartesianIndex{2})

Set a scale level using CartesianIndex.
"""
Base.setindex!(ss::ScaleSpace, value, I::CartesianIndex{2}) = (ss[I[1], I[2]] = value)

"""
    Base.getindex(ss::ScaleSpace, i::Int)

Linear indexing support for compatibility with Base functions like first(), last().
"""
Base.getindex(ss::ScaleSpace, i::Int) = ss.levels[i]

"""
    Base.setindex!(ss::ScaleSpace, value, i::Int)

Linear indexing support for setting values.
"""
Base.setindex!(ss::ScaleSpace, value, i::Int) = (ss.levels[i] = value)

# Enable iteration over ScaleSpace
Base.iterate(ss::ScaleSpace) = iterate(ss.levels)
Base.iterate(ss::ScaleSpace, state) = iterate(ss.levels, state)
Base.length(ss::ScaleSpace) = length(ss.levels)
Base.eltype(::Type{ScaleSpace{T}}) where T = ScaleLevel{T}
Base.firstindex(ss::ScaleSpace) = firstindex(ss.levels)
Base.lastindex(ss::ScaleSpace) = lastindex(ss.levels)

"""
    level_step(level::ScaleLevel)

Calculate the sampling step size for a scale level.

The step size represents how much the image is downsampled at this octave level.
For octave 0, step = 1 (original resolution). For octave 1, step = 2 (half resolution), etc.

# Arguments
- `level`: ScaleLevel instance

# Returns
- Step size as a Float64 (2^octave)
"""
level_step(level::ScaleLevel) = 2.0^level.octave

"""
    level_size(level::ScaleLevel{T}) where T

Get the image dimensions for a scale level.

Extracts the size from the first field of the level's data NamedTuple,
returning it as a Size2 with (width, height) ordering.

# Arguments
- `level`: ScaleLevel instance

# Returns
- Size2{Int} with (width, height) dimensions
"""
function level_size(level::ScaleLevel{T}) where T
    data = level.data
    # Get size from first field of NamedTuple
    first_field = values(data)[1]
    return Size2(first_field)
end

# =============================================================================
# Constructors
# =============================================================================

# Public constructors
"""
    ScaleSpace(size::Size2{Int}; octave_resolution=3, base_sigma=1.6, 
               camera_psf=0.5, image_type=GaussianImage{N0f8})

Create a scale space with specified size and optional parameters.

# Arguments
- `size`: Image dimensions as Size2{Int}
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `base_sigma`: Base smoothing scale (default: 1.6)
- `camera_psf`: Camera point spread function sigma (default: 0.5)
- `image_type`: NamedTuple type specifying the data structure (default: Gaussian with N0f8 grayscale)

# Examples
```julia
using Colors, FixedPointNumbers

# Gaussian scale space (default with N0f8)
ss = ScaleSpace(Size2(256, 256))
# or explicitly:
ss = GaussianScaleSpace(Size2(256, 256))

# Hessian scale space
hess_ss = HessianScaleSpace(Size2(256, 256))
# or explicitly:
hess_ss = ScaleSpace(Size2(256, 256), image_type=HessianImages{Float32})

# Laplacian scale space
lap_ss = LaplacianScaleSpace(Size2(256, 256))
# or explicitly:
lap_ss = ScaleSpace(Size2(256, 256), image_type=LaplacianImage{Float32})

# Higher precision with N0f16
ss_16 = GaussianScaleSpace(Size2(256, 256), element_type=N0f16)

# Custom multi-channel (still uses explicit NamedTuple)
custom_ss = ScaleSpace(Size2(256, 256),
    image_type=NamedTuple{(:magnitude, :phase), Tuple{Matrix{Float32}, Matrix{Float32}}})

```
"""
function ScaleSpace(size::Size2{Int}; 
                   octave_resolution::Int=3, base_sigma::Float64=1.6,
                   camera_psf::Float64=0.5,
                   image_type::Type{<:NamedTuple}=GaussianImage{N0f8})
    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(0, last_octave, octave_resolution, 0, octave_resolution - 1,
                     adjusted_base_sigma, camera_psf, size, image_type)
end

"""
    ScaleSpace(image::Matrix{Gray{T}}; octave_resolution=3, base_sigma=1.6, camera_psf=0.5) where T<:FixedPoint

Convenience constructor that creates and populates a Gaussian scale space from a grayscale image.
Automatically determines the size from the image dimensions and builds the complete scale pyramid.

# Arguments
- `image`: Input grayscale image matrix with intensities in [0,1] using FixedPoint types
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `base_sigma`: Base smoothing scale (default: 1.6)
- `camera_psf`: Camera point spread function sigma (default: 0.5)

# Examples
```julia
# Create and populate scale space directly from grayscale image
using Colors, FixedPointNumbers
img = rand(Gray{N0f8}, 256, 256)  # Grayscale with 8-bit fixed point
ss = ScaleSpace(img)  # Returns fully populated scale space

# With custom parameters
ss = ScaleSpace(img, octave_resolution=4, base_sigma=1.2)

# Higher precision grayscale
img16 = rand(Gray{N0f16}, 256, 256)  # 16-bit fixed point
ss = ScaleSpace(img16)  # Returns fully populated scale space
```
"""
function ScaleSpace(image::Matrix{Gray{T}}; 
                   octave_resolution::Int=3, base_sigma::Float64=1.6,
                   camera_psf::Float64=0.5) where T<:FixedPoint
    img_size = Size2(image)
    ss = ScaleSpace(img_size; octave_resolution=octave_resolution, 
                   base_sigma=base_sigma, camera_psf=camera_psf,
                   image_type=GaussianImage{T})
    # Populate the scale space with the image data
    ss(image)
    return ss
end

# =============================================================================
# Type Aliases for Common Image Types
# =============================================================================

"""
    GaussianImage{T}

Type alias for Gaussian scale space data with smoothed grayscale image field (:g).
T should be a FixedPoint type (N0f8, N0f16) to ensure intensities are in [0,1].
"""
const GaussianImage{T} = NamedTuple{(:g,), Tuple{Matrix{Gray{T}}}}

"""
    HessianImages{T}

Type alias for Hessian scale space data with second derivative fields (:Ixx, :Iyy, :Ixy).
T should be a floating point type (Float32, Float64) since derivatives can be negative.
"""
const HessianImages{T} = NamedTuple{(:Ixx, :Iyy, :Ixy), Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}

"""
    LaplacianImage{T}

Type alias for Laplacian scale space data with Laplacian field (:L).
T should be a floating point type (Float32, Float64) since Laplacian can be negative.
"""
const LaplacianImage{T} = NamedTuple{(:L,), Tuple{Matrix{T}}}

# Convenience constructors using type aliases
"""
    GaussianScaleSpace(size::Size2{Int}; element_type=N0f8, kwargs...)

Create a Gaussian scale space with GaussianImage{T} structure.
Uses FixedPoint types to ensure intensities are in [0,1].
"""
GaussianScaleSpace(size::Size2{Int}; element_type::Type{T}=N0f8, kwargs...) where T<:FixedPoint = 
    ScaleSpace(size; image_type=GaussianImage{T}, kwargs...)

"""
    GaussianScaleSpace(image::Matrix{Gray{T}}; kwargs...) where T<:FixedPoint

Create a Gaussian scale space from a grayscale image with GaussianImage{T} structure.
Automatically populates the scale pyramid from the input image.
"""
GaussianScaleSpace(image::Matrix{Gray{T}}; kwargs...) where T<:FixedPoint = 
    ScaleSpace(image; kwargs...)  # Image constructor already uses GaussianImage

"""
    HessianScaleSpace(size::Size2{Int}; element_type=Float32, kwargs...)

Create a Hessian scale space with HessianImages{T} structure.
Uses floating point types since derivatives can be negative.
"""
HessianScaleSpace(size::Size2{Int}; element_type::Type{T}=Float32, kwargs...) where T<:AbstractFloat = 
    ScaleSpace(size; image_type=HessianImages{T}, kwargs...)

"""
    LaplacianScaleSpace(size::Size2{Int}; element_type=Float32, kwargs...)

Create a Laplacian scale space with LaplacianImage{T} structure.
Uses floating point types since Laplacian can be negative.
"""
LaplacianScaleSpace(size::Size2{Int}; element_type::Type{T}=Float32, kwargs...) where T<:AbstractFloat = 
    ScaleSpace(size; image_type=LaplacianImage{T}, kwargs...)

# =============================================================================
# Base.similar Methods
# =============================================================================

"""
    Base.similar(ss::ScaleSpace, filters::NamedTuple)

Create a new ScaleSpace for multi-channel output based on named filters.
The output type is inferred from the filter names and source element type.

# Arguments
- `ss`: Source scale space for geometry
- `filters`: NamedTuple of filters that determines output field names

# Example
```julia
using ImageFiltering: centered
hess_ss = similar(smooth_ss, (
    Ixx = centered([0 0 0; 1 -2 1; 0 0 0]),
    Iyy = centered([0 1 0; 0 -2 0; 0 1 0]),
    Ixy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])
))
```
"""
function Base.similar(ss::ScaleSpace{T}, filters::NamedTuple) where T
    # Get element type from source scale space - type-stable version
    element_type = eltype(fieldtype(T, 1))  # Get element type from first field of T
    
    # Create NamedTuple type with same field names as filters
    filter_names = keys(filters)
    field_types = ntuple(i -> Matrix{element_type}, length(filter_names))
    image_type = NamedTuple{filter_names, typeof(field_types)}
    
    return ScaleSpace(ss.first_octave, ss.last_octave, ss.octave_resolution,
                     ss.octave_first_subdivision, ss.octave_last_subdivision,
                     ss.base_sigma, ss.camera_psf, ss.input_size, image_type)
end

"""
    Base.similar(ss::ScaleSpace, filter_func::Function)

Create a new ScaleSpace with data type inferred from the filter function return type.

# Arguments
- `ss`: Source scale space for geometry and input type inference
- `filter_func`: Function that will process the data

# Example
```julia
# Create Laplacian scale space from Hessian (returns single Matrix)
lap_ss = similar(hess_ss, hess_data -> hess_data.Ixx + hess_data.Iyy)

# Create custom multi-channel output (returns NamedTuple)
custom_ss = similar(smooth_ss, img -> (mag = sqrt.(img.g.^2), phase = atan.(img.g)))
```
"""
function Base.similar(ss::ScaleSpace{T}, filter_func::Function) where T
    # Infer return type from function and source data type - type-stable version
    image_type = Core.Compiler.return_type(filter_func, Tuple{T})
    
    return ScaleSpace(ss.first_octave, ss.last_octave, ss.octave_resolution,
                     ss.octave_first_subdivision, ss.octave_last_subdivision,
                     ss.base_sigma, ss.camera_psf, ss.input_size, image_type)
end

# =============================================================================
# Base Methods
# =============================================================================

function Base.summary(ss::ScaleSpace{T}) where T
    num_octaves = ss.last_octave - ss.first_octave + 1
    println("ScaleSpace{$T} Summary:" *
          "\n  Input size: $(ss.input_size.width) × $(ss.input_size.height)" *
          "\n  Octaves: $(ss.first_octave) to $(ss.last_octave) ($num_octaves total)" *
          "\n  Subdivisions: $(ss.octave_first_subdivision) to $(ss.octave_last_subdivision)" *
          "\n  Octave resolution: $(ss.octave_resolution)" *
          "\n  Base sigma: $(ss.base_sigma)" *
          "\n  Camera PSF: $(ss.camera_psf)" *
          "\n  Total levels: $(length(ss.levels))" *
          "\n  Sigma range: $(minimum(ss.levels.sigma)) to $(maximum(ss.levels.sigma))")
end

# =============================================================================
# Utility Functions
# =============================================================================

"""
    effective_sigma(target_sigma::Real, current_sigma::Real)

Calculate the effective sigma needed for incremental smoothing.

Given a current smoothing level and a target smoothing level, computes the
additional sigma needed to achieve the target. Returns 0 if target is already
achieved or exceeded.

# Arguments
- `target_sigma`: Desired final smoothing level
- `current_sigma`: Current smoothing level

# Returns
- Effective sigma for additional smoothing, or 0 if no additional smoothing needed
"""
function effective_sigma(target_sigma::Real, current_sigma::Real)
    target_sigma <= current_sigma && return 0.0
    return sqrt(target_sigma^2 - current_sigma^2)
end



# =============================================================================
# Scale Space Population - Functor Interface
# =============================================================================

"""
    (ss::ScaleSpace)(input::Matrix{Gray{T}}, kernel_func=Kernel.gaussian) where T<:FixedPoint

Build scale space pyramid from a grayscale image using octave-based construction with smoothing.

This method builds the multi-octave Gaussian pyramid by:
1. Downsampling the input to each octave resolution
2. Applying incremental smoothing to reach target sigma values
3. Building each octave from the previous via decimation

# Arguments
- `input`: Input grayscale image with intensities in [0,1] (must match scale space dimensions)
- `kernel_func`: Kernel function that takes sigma and returns a kernel (default: Kernel.gaussian)
  - Can be Kernel.gaussian, Kernel.laplacian, or any function with signature `(sigma) -> kernel`

# Example
```julia
using Colors, FixedPointNumbers
img = rand(Gray{N0f8}, 512, 512)
ss = ScaleSpace(Size2(512, 512))
ss(img)  # Uses default Gaussian kernel
ss(img, Kernel.gaussian)  # Explicitly use Gaussian
ss(img, sigma -> Kernel.DoG(sigma))  # Use custom kernel function
```
"""
function (ss::ScaleSpace)(input::Matrix{Gray{T}}, kernel_func=Kernel.gaussian) where T<:FixedPoint
    # Verify input dimensions
    input_h, input_w = size(input)
    @assert input_h == ss.input_size.height && input_w == ss.input_size.width
    
    # Verify this is a Gaussian scale space (single :g field)
    first_level = first(ss)
    dst_type = typeof(first_level.data)
    @assert fieldcount(dst_type) == 1 && haskey(first_level.data, :g) "Scale space must be Gaussian (have single :g field) for image input"

    # Inner function for incremental smoothing computation
    function apply_incremental_smoothing!(dst_data, src_data, target_sigma, prev_sigma, level)
        if target_sigma > prev_sigma
            delta_sigma = sqrt(target_sigma^2 - prev_sigma^2)
            kernel_sigma = delta_sigma / level_step(level)
            # Extract numeric values for filtering, then convert back to target type
            numeric_src = channelview(src_data)
            filtered_numeric = imfilter(numeric_src, kernel_func(kernel_sigma), "reflect")
            # Convert back to Gray type with proper element type
            dst_data .= Gray{eltype(eltype(dst_data))}.(filtered_numeric)
        end
    end

    # Populate first octave from input image
    o = ss.first_octave
    first_level = ss[o, ss.octave_first_subdivision]

    # Downsample/upsample input image to this octave's resolution
    # Store in the :g field (already verified to exist)
    if o >= 0
        step = 2^o
        # Convert Gray values to target element type
        sampled_input = input[1:step:end, 1:step:end]
        first_level.data.g .= eltype(first_level.data.g).(sampled_input)
    else
        scale_factor = 2.0^(-o)
        h, w = size(input)
        new_size = (round(Int, h * scale_factor), round(Int, w * scale_factor))
        resized_input = imresize(input, new_size)
        first_level.data.g .= eltype(first_level.data.g).(resized_input)
    end

    # Apply smoothing to reach target sigma
    apply_incremental_smoothing!(first_level.data.g, first_level.data.g, 
                                first_level.sigma, ss.camera_psf, first_level)

    # Fill rest of first octave by incremental smoothing using broadcasting
    subdivision_range = (ss.octave_first_subdivision + 1):ss.octave_last_subdivision
    prev_range = (ss.octave_first_subdivision):(ss.octave_last_subdivision - 1)
    # Broadcast the smoothing operation across all subdivisions in this octave
    apply_incremental_smoothing!.(getfield.(getfield.(ss[o, subdivision_range], :data), :g),
                                 getfield.(getfield.(ss[o, prev_range], :data), :g),
                                 getfield.(ss[o, subdivision_range], :sigma),
                                 getfield.(ss[o, prev_range], :sigma),
                                 ss[o, subdivision_range])

    # Populate remaining octaves from previous octave
    for o in (ss.first_octave + 1):ss.last_octave
        # Pick the level from previous octave to downsample
        prev_subdivision_idx = min(ss.octave_first_subdivision + ss.octave_resolution,
                                  ss.octave_last_subdivision)

        # Downsample by simple decimation
        decimated = ss[o - 1, prev_subdivision_idx].data.g[1:2:end, 1:2:end]
        ss[o, ss.octave_first_subdivision].data.g .= eltype(ss[o, ss.octave_first_subdivision].data.g).(decimated)

        # Apply additional smoothing if needed
        apply_incremental_smoothing!(ss[o, ss.octave_first_subdivision].data.g, ss[o, ss.octave_first_subdivision].data.g,
                                    ss[o, ss.octave_first_subdivision].sigma, ss[o - 1, prev_subdivision_idx].sigma, ss[o, ss.octave_first_subdivision])

        # Fill rest of octave using broadcasting
        subdivision_range = (ss.octave_first_subdivision + 1):ss.octave_last_subdivision
        prev_range = (ss.octave_first_subdivision):(ss.octave_last_subdivision - 1)
        # Broadcast the smoothing operation across all subdivisions in this octave
        apply_incremental_smoothing!.(getfield.(getfield.(ss[o, subdivision_range], :data), :g),
                                     getfield.(getfield.(ss[o, prev_range], :data), :g),
                                     getfield.(ss[o, subdivision_range], :sigma),
                                     getfield.(ss[o, prev_range], :sigma),
                                     ss[o, subdivision_range])
    end

    return ss
end

"""
    (ss::ScaleSpace)(input::ScaleSpace, filters::NamedTuple)

Transform from one scale space to another using named filters for multi-channel output.

# Arguments
- `input`: Source scale space (must have matching geometry)
- `filters`: NamedTuple of ImageFiltering kernels matching output field names

# Example
```julia
# Compute Hessian from Gaussian scale space
using ImageFiltering: centered
hess_ss = ScaleSpace(Size2(512, 512), image_type=HessianImages{Float32})
hess_ss(smooth_ss, (
    Ixx = centered([0 0 0; 1 -2 1; 0 0 0]),
    Iyy = centered([0 1 0; 0 -2 0; 0 1 0]),
    Ixy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])
))
```
"""
function (ss::ScaleSpace)(input::ScaleSpace, filters::NamedTuple)
    first_dst_level = first(ss)
    dst_type = typeof(first_dst_level.data)
    @assert dst_type <: NamedTuple "Named filters require NamedTuple output type"
    
    # Verify field names match
    dst_fields = fieldnames(dst_type)
    filter_fields = keys(filters)
    @assert dst_fields == filter_fields "Filter fields $filter_fields must match output fields $dst_fields"

    # Assert that input is Gaussian scale space (has :g field)
    first_src_level = first(input)
    src_type = typeof(first_src_level.data)
    @assert fieldcount(src_type) == 1 && haskey(first_src_level.data, :g) "Input scale space must be Gaussian (have :g field)"

    # Apply filters using broadcasting across all levels
    for (field_name, filter_kernel) in pairs(filters)
        # Create a function that applies the filter to a single level's data
        apply_filter = data -> begin
            numeric_src = channelview(data.g)
            filtered_result = imfilter(numeric_src, filter_kernel, "reflect")
            return eltype(getfield(first(ss.levels.data), field_name)).(filtered_result)
        end
        
        # Broadcast the filter application and store results
        filtered_results = apply_filter.(input.levels.data)
        for (i, result) in enumerate(filtered_results)
            getfield(ss.levels.data[i], field_name) .= result
        end
    end

    return ss
end

"""
    (ss::ScaleSpace)(input::ScaleSpace, filter_func::Function)

Transform from one scale space to another using a function that processes each level.

The function return type is inferred at compile time to ensure type stability.

# Arguments
- `input`: Source scale space (must have matching geometry)
- `filter_func`: Function with signature `(src_data) -> dst_data`

# Example
```julia
# Custom processing function
lap_ss = ScaleSpace(Size2(512, 512), image_type=LaplacianImage{Float32})
lap_ss(hess_ss, hess_data -> (L = hess_data.Ixx + hess_data.Iyy,))
```
"""
function (ss::ScaleSpace)(input::ScaleSpace, filter_func::Function)
    # Infer return type at compile time
    first_src_level = first(input)
    src_type = typeof(first_src_level.data)
    expected_dst_type = Core.Compiler.return_type(filter_func, Tuple{src_type})
    first_dst_level = first(ss)
    dst_type = typeof(first_dst_level.data)
    
    # Handle Matrix return type by wrapping in NamedTuple
    if expected_dst_type <: Matrix
        # Function returns Matrix, but we need NamedTuple - check if dst has single field
        if fieldcount(dst_type) == 1
            field_name = fieldnames(dst_type)[1]
            # Apply function and wrap result using broadcasting
            results = filter_func.(input.levels.data)
            ss.levels.data .= [NamedTuple{(field_name,)}((r,)) for r in results]
        else
            error("Function returns Matrix but destination has multiple fields")
        end
    else
        @assert dst_type == expected_dst_type "Function return type $expected_dst_type doesn't match scale space type $dst_type"
        # Apply function to each level using broadcasting
        ss.levels.data .= filter_func.(input.levels.data)
    end

    return ss
end

"""
    (ss::ScaleSpace)(input::ScaleSpace, kernels::Vector)

Transform from one scale space to another by broadcasting kernel filtering over all levels.

This method performs derivative computations on already-smoothed scale space levels
using direct kernel convolution. Works with both single-channel (Matrix) and
multi-channel (NamedTuple) outputs.

# Arguments
- `input`: Source scale space (must have matching geometry)
- `kernels`: Vector of ImageFiltering kernels
  - Number of kernels must match output channels (1 for Matrix, N for NamedTuple with N fields)

# Example
```julia
# Compute Hessian from Gaussian scale space
using ImageFiltering: centered
kernel_xx = centered([0 0 0; 1 -2 1; 0 0 0])
kernel_yy = centered([0 1 0; 0 -2 0; 0 1 0])
kernel_xy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])

hess_ss = ScaleSpace(Size2(512, 512), image_type=HessianImages{Float32})
hess_ss(smooth_ss, [kernel_xx, kernel_yy, kernel_xy])
```
"""
function (ss::ScaleSpace)(input::ScaleSpace, kernels::Vector)
    dst_type = eltype(ss.levels.data)
    src_type = eltype(input.levels.data)

    # Both source and destination are always NamedTuple now
    num_dst_fields = fieldcount(dst_type)
    @assert length(kernels) == num_dst_fields "NamedTuple with $num_dst_fields fields requires $num_dst_fields kernels, got $(length(kernels))"

    # Assert that input is Gaussian scale space (has :g field)
    @assert fieldcount(src_type) == 1 && haskey(first(input.levels.data), :g) "Input scale space must be Gaussian (have :g field)"

    # Apply kernels using broadcasting across all levels
    for (j, kernel) in enumerate(kernels)
        field_name = fieldnames(dst_type)[j]
        
        # Create a function that applies the kernel to a single level's data
        apply_kernel = data -> begin
            numeric_src = channelview(data.g)
            filtered_result = imfilter(numeric_src, kernel, "reflect")
            return eltype(getfield(first(ss.levels.data), field_name)).(filtered_result)
        end
        
        # Broadcast the kernel application and store results
        filtered_results = apply_kernel.(input.levels.data)
        for (i, result) in enumerate(filtered_results)
            getfield(ss.levels.data[i], field_name) .= result
        end
    end

    return ss
end


