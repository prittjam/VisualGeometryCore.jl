"""
ScaleSpaceResponse for computing derivatives and filter responses from ScaleSpace.
"""

# ScaleSpaceResponse - using statements handled by main module

# =============================================================================
# SCALESPACERESPONSE - DERIVATIVE AND FILTER RESPONSES
# =============================================================================

# ResponseOctave removed - we reuse ScaleOctave from the main module

"""
    ResponseLevelView

Type alias for SubArray-based response levels that are views into 3D cubes.
"""
const ResponseLevelView = ScaleLevel{SubArray{Gray{Float32},2,Array{Gray{Float32},3}}}

"""
    ScaleSpaceResponse

Preallocated response structure for computing derivatives and filters from ScaleSpace.

This type stores a single response pyramid (e.g., Ixx, Iyy, Ixy, etc.) with 3D cube storage
and the same geometry as the source ScaleSpace. Levels are SubArray views into octave cubes.

All response data uses Float32 for consistency and performance.

# Fields
- `levels`: StructArray of ResponseLevelView for response storage
- `transform`: Transform function or filter for computing responses
- `octaves`: Vector of ResponseOctave objects with 3D cubes

# Usage
```julia
# Create separate response pyramids for each derivative
template = ScaleSpace(Size2(256, 256))

ixx_resp = ScaleSpaceResponse(template, compute_ixx)
iyy_resp = ScaleSpaceResponse(template, compute_iyy)
ixy_resp = ScaleSpaceResponse(template, compute_ixy)

# Compute responses
ixx_resp(scalespace)
iyy_resp(scalespace)
ixy_resp(scalespace)
```
"""
struct ScaleSpaceResponse{T} <: AbstractScaleSpace
    levels::StructArray{ResponseLevelView}
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
        levels = StructArray{ResponseLevelView}((
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

Base.eltype(::Type{ScaleSpaceResponse{T}}) where T = ResponseLevelView

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
    resp[octave, subdivision] -> ResponseLevelView

Access level by octave and subdivision, returning ResponseLevelView with SubArray data.
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

function level_size(level::ResponseLevelView)
    return Size2(level.data)
end



# =============================================================================
# HELPER - DISPATCH ON HOW TO APPLY (KERNEL VS FUNCTION)
# =============================================================================

"""
    apply_one!(dst, src, kernel::Union{AbstractArray, Tuple}, border_policy)

Apply kernel (array or factored tuple) using imfilter.
"""
apply_one!(dst, src, kernel::Union{AbstractArray, Tuple}, border_policy) =
    imfilter!(dst, src, kernel, border_policy)

"""
    apply_one!(dst, src, func::Function, border_policy)

Apply function transform directly (ignores border_policy).
"""
apply_one!(dst, src, func::Function, border_policy) =
    func(dst, src)

# =============================================================================
# DISPATCH ON DIMENSIONALITY (2D = LEVELS, 3D = OCTAVES)
# =============================================================================

"""
    apply_transform!(response, source, transform::Union{Tuple{T1,T2}, AbstractArray{T,2}, Function})

Process 2D transforms by looping over levels.
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

Process 3D transforms by looping over octaves.
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
# Create response structure and compute responses
ixx_resp = ScaleSpaceResponse(template, compute_ixx)
ixx_resp(gaussian_scalespace)  # Compute Ixx responses from ScaleSpace

# Or compute derivative of a response
dx_resp = ScaleSpaceResponse(hessian_resp, DERIVATIVE_KERNELS_3D.dx)
dx_resp(hessian_resp)  # Compute ∇x from Hessian determinant response
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