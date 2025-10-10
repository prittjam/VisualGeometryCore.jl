"""
Scale space construction for multi-scale blob detection.

This module implements Gaussian scale space similar to VLFeat, with:
- Multi-octave pyramid structure
- Preallocated image storage
- Flexible octave geometry
- Efficient broadcasting operations via StructArrays
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

Preallocated scale space pyramid with efficient StructArray storage for broadcasting.

T can be:
- `Matrix{Float32}` for smoothed images
- `NamedTuple{(:Ixx, :Iyy, :Ixy)}` for Hessian components
- Any other data structure needed for the scale space
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

        # Create combinations in the order expected by level_index:
        # (o0,s0), (o0,s1), (o0,s2), (o1,s0), (o1,s1), (o1,s2), ...
        # NOT (o0,s0), (o1,s0), (o2,s0), (o0,s1), ... which is what product(octave, subdivision) gives
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

# Type-stable accessor for ScaleSpace
function level_index(ss::ScaleSpace, octave::Int, subdivision::Int)
    @assert ss.first_octave <= octave <= ss.last_octave
    @assert ss.octave_first_subdivision <= subdivision <= ss.octave_last_subdivision

    subdivisions_per_octave = ss.octave_last_subdivision - ss.octave_first_subdivision + 1
    octave_offset = octave - ss.first_octave
    subdivision_offset = subdivision - ss.octave_first_subdivision

    return octave_offset * subdivisions_per_octave + subdivision_offset + 1
end

# Type-stable property accessors for ScaleLevel
level_step(level::ScaleLevel) = 2.0^level.octave

function level_size(level::ScaleLevel{T}) where T
    data = level.data
    # Get size from first field of NamedTuple
    first_field = values(data)[1]
    h, w = size(first_field)
    return Size2(w, h)
end

# =============================================================================
# Constructors
# =============================================================================

# Public constructors
"""
    ScaleSpace(; size=nothing, width=0, height=0, octave_resolution=3,
               base_sigma=1.6, camera_psf=0.5, image_type=NamedTuple{(:g,), Tuple{Matrix{Float32}}})

Create a scale space with default geometry and specified image type.

# Arguments
- `size` or `width`/`height`: Image dimensions
- `octave_resolution`: Number of subdivisions per octave (default: 3)
- `base_sigma`: Base smoothing scale (default: 1.6)
- `camera_psf`: Camera point spread function sigma (default: 0.5)
- `image_type`: NamedTuple type specifying the data structure (default: Gaussian with Float32)

# Examples
```julia
# Gaussian scale space (default)
ss = ScaleSpace(width=256, height=256)
# or explicitly:
ss = GaussianScaleSpace(width=256, height=256)

# Hessian scale space
hess_ss = HessianScaleSpace(width=256, height=256)
# or explicitly:
hess_ss = ScaleSpace(width=256, height=256, image_type=HessianImages{Float32})

# Laplacian scale space
lap_ss = LaplacianScaleSpace(width=256, height=256)
# or explicitly:
lap_ss = ScaleSpace(width=256, height=256, image_type=LaplacianImage{Float32})

# Custom multi-channel (still uses explicit NamedTuple)
custom_ss = ScaleSpace(width=256, height=256,
    image_type=NamedTuple{(:magnitude, :phase), Tuple{Matrix{Float32}, Matrix{Float32}}})
```
"""
function ScaleSpace(; size::Union{Size2{Int}, Nothing}=nothing, width::Int=0, height::Int=0,
                   octave_resolution::Int=3, base_sigma::Float64=1.6,
                   camera_psf::Float64=0.5,
                   image_type::Type{<:NamedTuple}=GaussianImage{Float32})
    if size === nothing
        size = Size2(width, height)
    end

    min_dim = min(size.width, size.height)
    last_octave = max(floor(Int, log2(min_dim)) - 3, 0)

    # VLFeat applies base_sigma scaling: baseScale = 1.6 * 2^(1/octaveResolution)
    adjusted_base_sigma = base_sigma * 2.0^(1.0 / octave_resolution)

    return ScaleSpace(0, last_octave, octave_resolution, 0, octave_resolution - 1,
                     adjusted_base_sigma, camera_psf, size, image_type)
end

# =============================================================================
# Type Aliases for Common Image Types
# =============================================================================

"""
    GaussianImage{T}

Type alias for Gaussian scale space data with smoothed image field (:g).
"""
const GaussianImage{T} = NamedTuple{(:g,), Tuple{Matrix{T}}}

"""
    HessianImages{T}

Type alias for Hessian scale space data with second derivative fields (:Ixx, :Iyy, :Ixy).
"""
const HessianImages{T} = NamedTuple{(:Ixx, :Iyy, :Ixy), Tuple{Matrix{T}, Matrix{T}, Matrix{T}}}

"""
    LaplacianImage{T}

Type alias for Laplacian scale space data with Laplacian field (:L).
"""
const LaplacianImage{T} = NamedTuple{(:L,), Tuple{Matrix{T}}}

# Convenience constructors using type aliases
"""
    GaussianScaleSpace(; size=nothing, width=0, height=0, element_type=Float32, kwargs...)

Create a Gaussian scale space with GaussianImage{T} structure.
"""
GaussianScaleSpace(; element_type::Type{T}=Float32, kwargs...) where T = 
    ScaleSpace(; image_type=GaussianImage{T}, kwargs...)

"""
    HessianScaleSpace(; size=nothing, width=0, height=0, element_type=Float32, kwargs...)

Create a Hessian scale space with HessianImages{T} structure.
"""
HessianScaleSpace(; element_type::Type{T}=Float32, kwargs...) where T = 
    ScaleSpace(; image_type=HessianImages{T}, kwargs...)

"""
    LaplacianScaleSpace(; size=nothing, width=0, height=0, element_type=Float32, kwargs...)

Create a Laplacian scale space with LaplacianImage{T} structure.
"""
LaplacianScaleSpace(; element_type::Type{T}=Float32, kwargs...) where T = 
    ScaleSpace(; image_type=LaplacianImage{T}, kwargs...)

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

function effective_sigma(target_sigma::Real, current_sigma::Real)
    target_sigma <= current_sigma && return 0.0
    return sqrt(target_sigma^2 - current_sigma^2)
end



# =============================================================================
# Scale Space Population - Functor Interface
# =============================================================================

"""
    (ss::ScaleSpace)(input::AbstractMatrix, kernel_func=Kernel.gaussian)

Build scale space pyramid from an image using octave-based construction with smoothing.

This method builds the multi-octave Gaussian pyramid by:
1. Downsampling the input to each octave resolution
2. Applying incremental smoothing to reach target sigma values
3. Building each octave from the previous via decimation

# Arguments
- `input`: Input image (must match scale space dimensions)
- `kernel_func`: Kernel function that takes sigma and returns a kernel (default: Kernel.gaussian)
  - Can be Kernel.gaussian, Kernel.laplacian, or any function with signature `(sigma) -> kernel`

# Example
```julia
ss = ScaleSpace(width=512, height=512)
ss(image)  # Uses default Gaussian kernel
ss(image, Kernel.gaussian)  # Explicitly use Gaussian
ss(image, sigma -> Kernel.DoG(sigma))  # Use custom kernel function
```
"""
function (ss::ScaleSpace)(input::AbstractMatrix, kernel_func=Kernel.gaussian)
    # Verify input dimensions
    input_h, input_w = size(input)
    @assert input_h == ss.input_size.height && input_w == ss.input_size.width
    
    # Verify this is a Gaussian scale space (single :g field)
    dst_type = eltype(ss.levels.data)
    @assert fieldcount(dst_type) == 1 && haskey(first(ss.levels.data), :g) "Scale space must be Gaussian (have single :g field) for image input"

    # Inner function for incremental smoothing computation
    function apply_incremental_smoothing!(dst_data, src_data, target_sigma, prev_sigma, level)
        if target_sigma > prev_sigma
            delta_sigma = sqrt(target_sigma^2 - prev_sigma^2)
            kernel_sigma = delta_sigma / level_step(level)
            filtered_result = imfilter(src_data, kernel_func(kernel_sigma), "reflect")
            dst_data .= eltype(dst_data).(filtered_result)
        end
    end

    # Populate first octave from input image
    o = ss.first_octave
    first_level = ss.levels[level_index(ss, o, ss.octave_first_subdivision)]

    # Downsample/upsample input image to this octave's resolution
    # Store in the :g field (already verified to exist)
    if o >= 0
        step = 2^o
        first_level.data.g .= input[1:step:end, 1:step:end]
    else
        scale_factor = 2.0^(-o)
        h, w = size(input)
        new_size = (round(Int, h * scale_factor), round(Int, w * scale_factor))
        first_level.data.g .= imresize(input, new_size)
    end

    # Apply smoothing to reach target sigma
    apply_incremental_smoothing!(first_level.data.g, first_level.data.g, 
                                first_level.sigma, ss.camera_psf, first_level)

    # Fill rest of first octave by incremental smoothing
    for s in (ss.octave_first_subdivision + 1):ss.octave_last_subdivision
        curr_level = ss.levels[level_index(ss, o, s)]
        prev_level = ss.levels[level_index(ss, o, s - 1)]

        apply_incremental_smoothing!(curr_level.data.g, prev_level.data.g,
                                    curr_level.sigma, prev_level.sigma, curr_level)
    end

    # Populate remaining octaves from previous octave
    for o in (ss.first_octave + 1):ss.last_octave
        # Pick the level from previous octave to downsample
        prev_subdivision_idx = min(ss.octave_first_subdivision + ss.octave_resolution,
                                  ss.octave_last_subdivision)

        prev_level = ss.levels[level_index(ss, o - 1, prev_subdivision_idx)]
        curr_level = ss.levels[level_index(ss, o, ss.octave_first_subdivision)]

        # Downsample by simple decimation
        curr_level.data.g .= prev_level.data.g[1:2:end, 1:2:end]

        # Apply additional smoothing if needed
        apply_incremental_smoothing!(curr_level.data.g, curr_level.data.g,
                                    curr_level.sigma, prev_level.sigma, curr_level)

        # Fill rest of octave
        for s in (ss.octave_first_subdivision + 1):ss.octave_last_subdivision
            curr_level = ss.levels[level_index(ss, o, s)]
            prev_level = ss.levels[level_index(ss, o, s - 1)]

            apply_incremental_smoothing!(curr_level.data.g, prev_level.data.g,
                                        curr_level.sigma, prev_level.sigma, curr_level)
        end
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
hess_ss = ScaleSpace(width=512, height=512, data_type=:hessian)
hess_ss(smooth_ss, (
    Ixx = centered([0 0 0; 1 -2 1; 0 0 0]),
    Iyy = centered([0 1 0; 0 -2 0; 0 1 0]),
    Ixy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])
))
```
"""
function (ss::ScaleSpace)(input::ScaleSpace, filters::NamedTuple)
    dst_type = eltype(ss.levels.data)
    @assert dst_type <: NamedTuple "Named filters require NamedTuple output type"
    
    # Verify field names match
    dst_fields = fieldnames(dst_type)
    filter_fields = keys(filters)
    @assert dst_fields == filter_fields "Filter fields $filter_fields must match output fields $dst_fields"

    # Assert that input is Gaussian scale space (has :g field)
    src_type = eltype(input.levels.data)
    @assert fieldcount(src_type) == 1 && haskey(first(input.levels.data), :g) "Input scale space must be Gaussian (have :g field)"

    # Apply filters to each level using broadcasting
    for i in eachindex(ss.levels.data)
        src_img = input.levels.data[i].g  # Access the .g field correctly
        dst = ss.levels.data[i]
        
        # Broadcast imfilter! across destination fields
        imfilter!.(values(dst), Ref(src_img), values(filters))
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
lap_ss = ScaleSpace(width=512, height=512, data_type=:laplacian)
lap_ss(hess_ss, hess_data -> (L = hess_data.Ixx + hess_data.Iyy,))
```
"""
function (ss::ScaleSpace)(input::ScaleSpace, filter_func::Function)
    # Infer return type at compile time
    src_type = eltype(input.levels.data)
    expected_dst_type = Core.Compiler.return_type(filter_func, Tuple{src_type})
    dst_type = eltype(ss.levels.data)
    
    # Handle Matrix return type by wrapping in NamedTuple
    if expected_dst_type <: Matrix
        # Function returns Matrix, but we need NamedTuple - check if dst has single field
        if fieldcount(dst_type) == 1
            field_name = fieldnames(dst_type)[1]
            # Apply function and wrap result
            for i in eachindex(ss.levels.data)
                result = filter_func(input.levels.data[i])
                ss.levels.data[i] = NamedTuple{(field_name,)}((result,))
            end
        else
            error("Function returns Matrix but destination has multiple fields")
        end
    else
        @assert dst_type == expected_dst_type "Function return type $expected_dst_type doesn't match scale space type $dst_type"
        # Apply function to each level
        for i in eachindex(ss.levels.data)
            ss.levels.data[i] = filter_func(input.levels.data[i])
        end
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

hess_ss = ScaleSpace(width=512, height=512, data_type=:hessian)
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

    # Apply kernels to each level using broadcasting
    for i in eachindex(ss.levels.data)
        src_img = input.levels.data[i].g  # Access the .g field correctly
        dst = ss.levels.data[i]
        
        # Broadcast imfilter! across destination fields
        imfilter!.(values(dst), Ref(src_img), kernels)
    end

    return ss
end


