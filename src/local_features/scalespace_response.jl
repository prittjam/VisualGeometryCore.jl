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

    function ScaleSpaceResponse(template::ScaleSpace, transform::T) where T
        # Validate template is not empty
        @assert !isempty(template.levels) "Template ScaleSpace must have levels defined"
        
        # Create response octave objects with 3D cubes matching template octaves
        response_octaves = ScaleOctave[]
        
        for template_octave in template.octaves
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



"""
    apply_transform_batch!(response::ScaleSpaceResponse, source::ScaleSpace, octave_filter=nothing)

Apply transform to multiple levels efficiently using StructArray operations.
Optionally filter by octave for partial computation.
"""
function apply_transform_batch!(response::ScaleSpaceResponse{T}, source::ScaleSpace, octave_filter=nothing) where T
    # Determine which indices to process
    indices = if octave_filter !== nothing
        findall(response.levels.octave .== octave_filter)
    else
        1:length(response.levels)
    end

    # Process each level
    for idx in indices
        src_level = source.levels[idx]
        dst_level = response.levels[idx]

        # Apply transform (kernel or function)
        if response.transform isa AbstractArray
            # Kernel: use imfilter with border policy
            imfilter!(dst_level.data, src_level.data, response.transform, source.border_policy)
        else
            # Function: apply directly (transform handles data format)
            response.transform(dst_level.data, src_level.data)
        end
    end

    return response
end

# =============================================================================
# SCALESPACERESPONSE COMPUTATION - FUNCTOR INTERFACE
# =============================================================================

"""
    (response::ScaleSpaceResponse)(source::ScaleSpace)

Compute responses from a source ScaleSpace using the stored transform.

# Arguments
- `source`: Source ScaleSpace (must have compatible geometry and be populated)

# Returns
- The ScaleSpaceResponse instance with computed response data

# Example
```julia
# Create response structure and compute responses
ixx_resp = ScaleSpaceResponse(template, compute_ixx)
ixx_resp(gaussian_scalespace)  # Compute Ixx responses
```
"""
function (response::ScaleSpaceResponse{T})(source::ScaleSpace) where T
    # Check compatibility between response and source
    check_compatibility(response, source)

    # Apply transform to all levels using batch processing
    apply_transform_batch!(response, source)

    return response
end

# =============================================================================
# COMPATIBILITY CHECKING
# =============================================================================

"""
    check_compatibility(response::ScaleSpaceResponse, source::ScaleSpace)

Check that a ScaleSpaceResponse and ScaleSpace are compatible for computation.

# Arguments
- `response`: ScaleSpaceResponse instance
- `source`: Source ScaleSpace instance

# Throws
- `ArgumentError` if the structures are incompatible

# Compatibility Requirements
- Source must have levels defined
- Response and source must have the same number of levels
- Levels must have matching geometry (octaves, subdivisions, sigmas)
"""
function check_compatibility(response::ScaleSpaceResponse{T}, source::ScaleSpace) where T
    # Check that source has levels
    if isempty(source.levels)
        throw(ArgumentError("Source ScaleSpace must have levels. Create ScaleSpace first using ScaleSpace constructor."))
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
# SAVE FUNCTIONALITY
# =============================================================================

function save_responses(ss::AbstractScaleSpace, output_dir::String; prefix::String="level")
    mkpath(output_dir)
    saved_count = 0
    skipped_count = 0

    for level in ss.levels
        saved, skipped = save_level_data(level.data, level, output_dir, prefix)
        saved_count += saved
        skipped_count += skipped
    end

    println("\n✓ Saved $saved_count images to $output_dir")
    if skipped_count > 0
        println("⚠ Skipped $skipped_count uninitialized levels")
    end
end

function save_level_data(data::Matrix{Gray{Float32}}, level, output_dir::String, prefix::String)
    if any(isnan, channelview(data))
        println("Skipped: o=$(level.octave), s=$(level.subdivision) (contains NaN values - uninitialized)")
        return 0, 1
    end

    min_val, max_val = extrema(channelview(data))
    filename = joinpath(output_dir, "$(prefix)_o$(level.octave)_s$(level.subdivision).tif")
    save(filename, data)

    sz = level_size(level)
    println("Saved: $filename (σ=$(round(level.sigma, digits=3)), size=$(sz.width)×$(sz.height), range=[$min_val, $max_val])")
    return 1, 0
end

function save_level_data(data::SubArray{Gray{Float32},2,Array{Gray{Float32},3}}, level, output_dir::String, prefix::String)
    if any(isnan, channelview(data))
        println("Skipped: o=$(level.octave), s=$(level.subdivision) (contains NaN values - uninitialized)")
        return 0, 1
    end

    min_val, max_val = extrema(channelview(data))
    filename = joinpath(output_dir, "$(prefix)_o$(level.octave)_s$(level.subdivision).tif")
    save(filename, data)

    sz = level_size(level)
    println("Saved: $filename (σ=$(round(level.sigma, digits=3)), size=$(sz.width)×$(sz.height), range=[$min_val, $max_val])")
    return 1, 0
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