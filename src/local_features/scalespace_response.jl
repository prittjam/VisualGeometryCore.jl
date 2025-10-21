"""
ScaleSpaceResponse for computing derivatives and filter responses from ScaleSpace.
"""

# ScaleSpaceResponse - using statements handled by main module

# =============================================================================
# SCALESPACERESPONSE - DERIVATIVE AND FILTER RESPONSES
# =============================================================================

# ResponseOctave removed - we reuse ScaleOctave from the main module


"""
    ScaleSpaceResponse

Preallocated response structure for computing derivatives and filters from ScaleSpace.

This type stores a single response pyramid (e.g., Ixx, Iyy, Ixy, etc.) with 3D cube storage
and the same geometry as the source. Levels are SubArray views into octave cubes.

All response data uses Float32 for consistency and performance.

# Fields
- `levels`: StructArray of ScaleLevelView for response storage
- `transform`: Transform (kernel, factored kernel, or function) for computing responses
- `octaves`: Vector of ScaleOctave objects with 3D cubes

# Transform Types
- **2D kernels**: Applied level-by-level (e.g., DERIVATIVE_KERNELS.xy)
- **Factored 2D kernels**: Applied level-by-level with separation (e.g., DERIVATIVE_KERNELS.xx)
- **3D kernels**: Applied to octave cubes (e.g., DERIVATIVE_KERNELS_3D.dx)
- **Factored 3D kernels**: Applied to octave cubes with separation (all 3D kernels are factored)
- **Functions**: Applied level-by-level with custom logic

# Usage
```julia
# Create response from ScaleSpace with factored kernels (efficient)
ss = ScaleSpace(img, first_octave=-1, octave_resolution=3)
ixx_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
ixx_resp(ss)  # Compute Ixx using factored kernel

# Create response from another response with 3D derivatives
∇x_resp = ScaleSpaceResponse(ixx_resp, DERIVATIVE_KERNELS_3D.dx)
∇x_resp(ixx_resp)  # Compute ∂/∂x using 3D factored kernel

# Custom function transforms
custom_resp = ScaleSpaceResponse(ss, (dst, src) -> dst .= src .^ 2)
custom_resp(ss)  # Apply custom function
```
"""
struct ScaleSpaceResponse{T} <: AbstractScaleSpace
    levels::StructArray{ScaleLevelView}
    transform::T
    octaves::Vector{ScaleOctave}

    function ScaleSpaceResponse(template::Union{ScaleSpace, ScaleSpaceResponse}, transform::T) where T
        # Validate template is not empty
        @assert !isempty(template.levels) "Template must have levels defined"

        # Get octaves from template (works for both ScaleSpace and ScaleSpaceResponse)
        template_octaves = if template isa ScaleSpace
            template.octaves
        else
            template.octaves
        end

        # Create response octave objects with 3D cubes matching template octaves
        response_octaves = ScaleOctave[]

        for template_octave in template_octaves
            # Create response cube with same dimensions as template
            H_o, W_o, num_levels = size(template_octave.G)
            # Use Gray{Float32} to match ScaleOctave interface
            cube = Array{Gray{Float32},3}(undef, H_o, W_o, num_levels)

            response_octave = ScaleOctave(cube, template_octave.octave,
                                        template_octave.subdivisions,
                                        template_octave.sigmas,
                                        template_octave.step)
            push!(response_octaves, response_octave)
        end

        # Create SubArray views into response octave cubes matching template structure
        data_arrays = SubArray{Gray{Float32},2,Array{Gray{Float32},3}}[]

        for (i, template_level) in enumerate(template.levels)
            # Find corresponding response octave
            response_octave = response_octaves[findfirst(oct -> oct.octave == template_level.octave, response_octaves)]

            # Create view with same slice index as template
            first_sub = first(response_octave.subdivisions)
            slice_idx = (template_level.subdivision - first_sub) + 1
            view_data = @view response_octave.G[:, :, slice_idx]
            push!(data_arrays, view_data)
        end

        # Create StructArray reusing template's metadata
        levels = StructArray{ScaleLevelView}((
            data = data_arrays,
            octave = template.levels.octave,
            subdivision = template.levels.subdivision,
            sigma = template.levels.sigma
        ))

        return new{T}(levels, transform, response_octaves)
    end
end

# =============================================================================
# BASE INTERFACE EXTENSIONS
# =============================================================================

Base.eltype(::Type{ScaleSpaceResponse{T}}) where T = ScaleLevelView

# Add indexing support similar to ScaleSpace
"""
    resp[octave] -> ResponseOctave

Access octave by number, returning ResponseOctave with 3D cube and metadata.
"""
function Base.getindex(resp::ScaleSpaceResponse{T}, octave::Int) where T
    # Find the octave object in the vector
    for octave_obj in resp.octaves
        if octave_obj.octave == octave
            return octave_obj
        end
    end
    throw(BoundsError(resp, (octave,)))
end

"""
    resp[octave, subdivision] -> ScaleLevelView

Access level by octave and subdivision, returning ScaleLevelView with SubArray data.
"""
function Base.getindex(resp::ScaleSpaceResponse{T}, octave::Int, subdivision::Int) where T
    # Use StructArray filtering for efficient lookup
    mask = (resp.levels.octave .== octave) .& (resp.levels.subdivision .== subdivision)
    indices = findall(mask)
    
    if isempty(indices)
        throw(BoundsError(resp, (octave, subdivision)))
    end
    
    return resp.levels[first(indices)]
end

Base.getindex(resp::ScaleSpaceResponse{T}, I::CartesianIndex{2}) where T = resp[I[1], I[2]]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

function level_size(level::ScaleLevelView)
    return Size2(level.data)
end



# =============================================================================
# TRANSFORM APPLICATION - TWO-LEVEL DISPATCH
# =============================================================================

"""
    apply_one!(dst, src, kernel::Union{AbstractArray, Tuple}, border_policy)

Apply kernel to data using imfilter. Handles both dense arrays and factored kernels.

# Arguments
- `dst`: Destination array (modified in-place)
- `src`: Source array
- `kernel`: Kernel (2D/3D array or factored tuple from kernelfactors)
- `border_policy`: Border handling policy (e.g., "replicate")

Factored kernels from `kernelfactors` are automatically separated by imfilter for efficiency.
"""
apply_one!(dst, src, kernel::Union{AbstractArray, Tuple}, border_policy) =
    imfilter!(dst, src, kernel, border_policy)

"""
    apply_one!(dst, src, func::Function, border_policy)

Apply function transform directly to data.

# Arguments
- `dst`: Destination array (modified in-place)
- `src`: Source array
- `func`: Function with signature `func(dst, src)`
- `border_policy`: Ignored for function transforms

The function should perform the transform in-place, modifying `dst`.
"""
apply_one!(dst, src, func::Function, border_policy) =
    func(dst, src)

# =============================================================================
# DISPATCH ON DIMENSIONALITY (2D = LEVELS, 3D = OCTAVES)
# =============================================================================

"""
    apply_transform!(response, source, transform::Union{Tuple{T1,T2}, AbstractArray{T,2}, Function})

Apply 2D transform by looping over levels independently.

Dispatches to this method for:
- **Factored 2D kernels**: `Tuple{T1,T2}` from kernelfactors (efficient)
- **Dense 2D kernels**: `AbstractArray{T,2}` (non-separable like xy)
- **Functions**: Custom transform functions

Each level is processed independently using `apply_one!` which dispatches on kernel vs function.
"""
function apply_transform!(response::ScaleSpaceResponse, source::AbstractScaleSpace,
                         transform::Union{Tuple{T1,T2}, AbstractArray{T,2}, Function}) where {T1,T2,T}
    border_pol = hasproperty(source, :border_policy) ? source.border_policy : "replicate"

    for idx in 1:length(response.levels)
        apply_one!(response.levels[idx].data, source.levels[idx].data, transform, border_pol)
    end

    return response
end

"""
    apply_transform!(response, source, transform::Union{Tuple{T1,T2,T3}, AbstractArray{T,3}})

Apply 3D transform by looping over octave cubes.

Dispatches to this method for:
- **Factored 3D kernels**: `Tuple{T1,T2,T3}` from kernelfactors (efficient, all 3D kernels)
- **Dense 3D kernels**: `AbstractArray{T,3}` (for custom non-separable 3D kernels)

Each octave cube is processed as a whole (spatial + scale dimensions) using `apply_one!`.
"""
function apply_transform!(response::ScaleSpaceResponse, source::AbstractScaleSpace,
                         transform::Union{Tuple{T1,T2,T3}, AbstractArray{T,3}}) where {T1,T2,T3,T}
    border_pol = hasproperty(source, :border_policy) ? source.border_policy : "replicate"

    for octave_num in unique(response.levels.octave)
        apply_one!(response[octave_num].G, source[octave_num].G, transform, border_pol)
    end

    return response
end

# =============================================================================
# SCALESPACERESPONSE COMPUTATION - FUNCTOR INTERFACE
# =============================================================================

"""
    (response::ScaleSpaceResponse)(source::Union{ScaleSpace, ScaleSpaceResponse})

Compute responses from a source using the stored transform.

# Arguments
- `source`: Source ScaleSpace or ScaleSpaceResponse (must have compatible geometry and be populated)

# Returns
- The ScaleSpaceResponse instance with computed response data

# Example
```julia
# Create response structure and compute 2D responses with factored kernels
ixx_resp = ScaleSpaceResponse(ss, DERIVATIVE_KERNELS.xx)
ixx_resp(ss)  # Compute Ixx responses from ScaleSpace (uses factored kernel)

# Compute 3D derivative of a response with factored 3D kernel
dx_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)
dx_resp(hessian_resp)  # Compute ∂/∂x from Hessian determinant response

# Custom function transforms
custom_resp = ScaleSpaceResponse(ss, (dst, src) -> dst .= src .^ 2)
custom_resp(ss)  # Apply custom function level-by-level
```
"""
function (response::ScaleSpaceResponse{T})(source::Union{ScaleSpace, ScaleSpaceResponse}) where T
    # Check compatibility between response and source
    check_compatibility(response, source)

    # Apply transform using multiple dispatch based on transform type
    apply_transform!(response, source, response.transform)

    return response
end

# =============================================================================
# COMPATIBILITY CHECKING
# =============================================================================

"""
    check_compatibility(response::ScaleSpaceResponse, source::Union{ScaleSpace, ScaleSpaceResponse})

Check that a ScaleSpaceResponse and source are compatible for computation.

# Arguments
- `response`: ScaleSpaceResponse instance
- `source`: Source ScaleSpace or ScaleSpaceResponse instance

# Throws
- `ArgumentError` if the structures are incompatible

# Compatibility Requirements
- Source must have levels defined
- Response and source must have the same number of levels
- Levels must have matching geometry (octaves, subdivisions, sigmas)
"""
function check_compatibility(response::ScaleSpaceResponse{T}, source::Union{ScaleSpace, ScaleSpaceResponse}) where T
    # Check that source has levels
    if isempty(source.levels)
        throw(ArgumentError("Source must have levels defined."))
    end

    # Check geometry compatibility
    if length(response.levels) != length(source.levels)
        throw(ArgumentError("Response and source must have same number of levels. " *
              "Response has $(length(response.levels)) levels, source has $(length(source.levels)) levels. " *
              "Ensure the ScaleSpaceResponse was created from a compatible template."))
    end

    # Use StructArray vectorized comparisons for efficient checking
    resp_levels = response.levels
    src_levels = source.levels
    
    # Check octaves match
    octave_mismatch = resp_levels.octave .!= src_levels.octave
    if any(octave_mismatch)
        idx = findfirst(octave_mismatch)
        throw(ArgumentError("Level $idx octave mismatch: response=$(resp_levels.octave[idx]), source=$(src_levels.octave[idx])"))
    end
    
    # Check subdivisions match
    subdivision_mismatch = resp_levels.subdivision .!= src_levels.subdivision
    if any(subdivision_mismatch)
        idx = findfirst(subdivision_mismatch)
        throw(ArgumentError("Level $idx subdivision mismatch: response=$(resp_levels.subdivision[idx]), source=$(src_levels.subdivision[idx])"))
    end
    
    # Check sigmas match (with tolerance for floating point)
    sigma_mismatch = abs.(resp_levels.sigma .- src_levels.sigma) .> 1e-10
    if any(sigma_mismatch)
        idx = findfirst(sigma_mismatch)
        throw(ArgumentError("Level $idx sigma mismatch: response=$(resp_levels.sigma[idx]), source=$(src_levels.sigma[idx])"))
    end

    return true
end

# =============================================================================
# BASE METHODS
# =============================================================================

function Base.summary(response::ScaleSpaceResponse{T}) where T
    transform_info = "Transform: $(response.transform)"
    sigma_min, sigma_max = extrema(response.levels.sigma)
    octaves = unique(response.levels.octave)
    
    println("ScaleSpaceResponse Summary:" *
          "\n  $transform_info" *
          "\n  Total levels: $(length(response.levels))" *
          "\n  Octaves: $(length(octaves)) ($(minimum(octaves)) to $(maximum(octaves)))" *
          "\n  Sigma range: $sigma_min to $sigma_max")
end