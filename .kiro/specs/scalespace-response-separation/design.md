# Design Document

## Overview

This design separates the current monolithic ScaleSpace into two focused types:

1. **ScaleSpace**: Handles only Gaussian scale space construction and storage
2. **ScaleSpaceResponse**: Handles response computations (derivatives, filters) from ScaleSpace

This separation follows the Single Responsibility Principle and creates cleaner, more maintainable code. The current implementation mixes Gaussian smoothing with derivative computations in a single struct, leading to architectural complexity and unclear responsibilities.

**Design Rationale**: The current ScaleSpace{T, F} allows T to be any NamedTuple type (GaussianImage, HessianImages, LaplacianImage) and F to be any transform type. This creates confusion about what a ScaleSpace actually represents and makes the API unclear. By constraining ScaleSpace to only handle Gaussian smoothing and creating ScaleSpaceResponse for all other computations, we achieve clear separation of concerns.

## Architecture

### Core Type Hierarchy

```julia
# Base scale space - only Gaussian smoothing
struct ScaleSpace{T <: GaussianImage}
    # Geometry fields (unchanged)
    first_octave::Int
    last_octave::Int
    octave_resolution::Int
    octave_first_subdivision::Int
    octave_last_subdivision::Int
    base_sigma::Float64
    camera_psf::Float64
    input_size::Size2{Int}
    
    # Data storage (constrained to Gaussian only)
    levels::StructArray{ScaleLevel{T}}
    
    # Gaussian kernel function (defaults to VLFeat)
    kernel_func::Function  # vlfeat_gaussian (default), Kernel.gaussian, etc.
end

# Response computations from scale space
struct ScaleSpaceResponse{T, F}
    # Response data storage (preallocated)
    levels::StructArray{ScaleLevel{T}}
    
    # Transform for computing responses
    transform::F  # NamedTuple of filters OR Function
end
```

**Design Decision**: ScaleSpaceResponse is a reusable, preallocated structure that only stores levels and transform. No source reference, no geometry fields. ScaleSpace is passed as parameter to computation functions for maximum reusability.

### Type Constraints

- **ScaleSpace{T}**: T is always `GaussianImage{ElementType}` where ElementType is a grayscale type (N0f8, N0f16, Float32, etc.)
- **ScaleSpaceResponse{T, F}**: 
  - T can be `HessianImages{Float32}`, `LaplacianImage{Float32}`, or any other response NamedTuple type
  - F can be `NamedTuple` (for filter-based responses) or `Function` (for function-based responses)

**Type Safety Rationale**: By constraining ScaleSpace to only GaussianImage types, we prevent accidental creation of "scale spaces" that contain derivative data. The type system enforces that ScaleSpace is only for Gaussian smoothing, while ScaleSpaceResponse handles all other computations.

## Indexing Strategy

### Handling Non-1-Based Octave and Subdivision Indexing

The scale space uses VLFeat-compatible indexing where both octaves and subdivisions can start from 0 or negative values:
- **Octaves**: Can be negative (upsampling), 0, or positive (e.g., -1, 0, 1, 2)
- **Subdivisions**: Typically start from 0 (e.g., 0, 1, 2 for octave_resolution=3)

This conflicts with Julia's 1-based indexing. The current implementation solves this through:

```julia
# Custom indexing that maps octave/subdivision to linear indices
function Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int)
    octave_range, subdivision_range = axes(ss)
    
    # Bounds checking against actual ranges (not 1-based)
    @boundscheck begin
        octave in octave_range || throw(BoundsError(ss, (octave, subdivision)))
        subdivision in subdivision_range || throw(BoundsError(ss, (octave, subdivision)))
    end
    
    # Map to linear index: octave-major order (VLFeat compatible)
    subdivisions_per_octave = length(subdivision_range)
    octave_offset = octave - first(octave_range)          # Handle negative/0-based octaves
    subdivision_offset = subdivision - first(subdivision_range)  # Handle 0-based subdivisions
    
    linear_idx = octave_offset * subdivisions_per_octave + subdivision_offset + 1
    return ss.levels[linear_idx]
end

# Return actual VLFeat-compatible ranges, not 1-based
Base.axes(ss::ScaleSpace) = (ss.first_octave:ss.last_octave, 
                            ss.octave_first_subdivision:ss.octave_last_subdivision)

# Example: VLFeat default is octaves 0:N, subdivisions 0:(octave_resolution-1)
# So ss[0, 0] maps to linear index 1, ss[0, 1] to index 2, ss[1, 0] to index 4, etc.
```

**Indexing Rationale**: This approach preserves the VLFeat-compatible octave/subdivision coordinate system used in scale space literature while providing efficient linear storage and Julia-compatible iteration. Both octaves and subdivisions can start from arbitrary values (including 0 or negative), which is essential for VLFeat compatibility.

**VLFeat Consistency**: The default VLFeat configuration uses:
- Octaves: 0 to `max_octave` (positive integers)
- Subdivisions: 0 to `octave_resolution - 1` (e.g., 0, 1, 2 for resolution=3)
- But negative octaves are supported for upsampling beyond input resolution

## Components and Interfaces

### ScaleSpace (Simplified)

```julia
# Constructors (simplified - only Gaussian)
ScaleSpace(size::Size2; kernel_func=vlfeat_gaussian, kwargs...)     # Empty Gaussian scale space, defaults to VLFeat
ScaleSpace(image::Matrix{Gray}; kernel_func=vlfeat_gaussian, kwargs...)  # Auto-populated, defaults to VLFeat

# Functors (Gaussian smoothing only)
ss(image::Matrix{Gray})                               # Populate with stored kernel function (VLFeat default)
ss(image::Matrix{Gray}, kernel_func::Function)       # Override stored kernel function

# Properties (unchanged interface with custom indexing)
ss[octave, subdivision]                               # 2D indexing returns ScaleLevel{GaussianImage}
                                                      # Handles negative octaves via custom getindex
level_size(level), level_step(level)                 # Utilities work unchanged
size(ss), axes(ss)                                    # Returns (num_octaves, num_subdivisions) and ranges
```

**API Simplification Rationale**: The simplified ScaleSpace constructor removes the `image_type` parameter (always GaussianImage) but keeps `kernel_func` parameter that defaults to `vlfeat_gaussian`. This enforces the single responsibility of Gaussian smoothing while allowing different Gaussian kernel implementations.

**Indexing Solution**: The current implementation already handles negative octaves through custom `Base.getindex` methods that:
1. Use `axes(ss)` to return the actual octave/subdivision ranges (e.g., -1:2, 0:2)
2. Map `ss[octave, subdivision]` to linear indices via offset calculation
3. Provide bounds checking against the actual ranges, not 1-based indices

### ScaleSpaceResponse (New)

```julia
# Core constructor (preallocated, no source reference)
ScaleSpaceResponse(geometry_template::ScaleSpace, transform::F) where {F}  # Type inferred from transform

# Functors (compute responses from provided ScaleSpace)
response(source::ScaleSpace)                          # Compute all responses from source using stored transform
response(source::ScaleSpace, octave_range, subdivision_range)  # Compute subset of responses

# Properties (indexing requires source context)
response[octave, subdivision, source]                 # 2D indexing with source context
level_size(level), level_step(level)                 # Same utilities as ScaleSpace
size(response.levels)                                 # Size of response levels array
# Note: No direct size/axes methods since no geometry stored
```

**Reusability Rationale**: By removing the source reference, ScaleSpaceResponse becomes a reusable, preallocated structure that can be used with multiple ScaleSpace instances of the same geometry. This is particularly beneficial for:
1. **Performance**: Avoid repeated allocations when processing multiple images
2. **Memory efficiency**: Reuse the same response buffers
3. **Flexibility**: Same response structure works with different scale spaces

**Indexing Challenge**: Without storing geometry, ScaleSpaceResponse indexing becomes more complex. Two approaches:
1. **Require source context**: `response[octave, subdivision, source]` or `response(source)[octave, subdivision]`
2. **Store geometry template**: Add geometry fields but lose some reusability across different geometries

**Final Approach**: Store NO geometry in ScaleSpaceResponse. Pass ScaleSpace as parameter to all operations for maximum reusability across different geometries.

### Predefined Filter Constants

```julia
# Predefined filter constants for common response types
const HESSIAN_FILTERS = (
    Ixx = centered([0 0 0; 1 -2 1; 0 0 0]),
    Iyy = centered([0 1 0; 0 -2 0; 0 1 0]),
    Ixy = centered([0.25 0 -0.25; 0 0 0; -0.25 0 0.25])
)

const LAPLACIAN_FUNCTION = hess_data -> (L = hess_data.Ixx + hess_data.Iyy,)



# Usage: types can be inferred from filters or specified explicitly
hess_resp = ScaleSpaceResponse(template, HESSIAN_FILTERS)  # Type inferred from filter names
lap_resp = ScaleSpaceResponse(template, LAPLACIAN_FUNCTION)  # Type inferred from function return

# Reuse multiple times
hess_resp(scalespace1)  # Compute responses for first image
hess_resp(scalespace2)  # Reuse same structure for second image
```

**Predefined Filter Rationale**: Instead of convenience functions, we provide predefined filter constants (`HESSIAN_FILTERS`, `LAPLACIAN_FUNCTION`) that users can pass to the core `ScaleSpaceResponse` constructor. This is cleaner, more explicit, and allows easy customization by defining new filter constants.

## Data Models

### ScaleSpace Data Model

**Data Model Rationale**: The constraint `T <: GaussianImage` enforces at the type level that ScaleSpace can only contain Gaussian smoothed grayscale images. This prevents the current mixed-use pattern where the same struct stores both Gaussian and derivative data.

### ScaleSpaceResponse Data Model

**Response Data Model Rationale**: ScaleSpaceResponse stores ONLY levels and transform. No geometry, no source reference. Maximum reusability across any ScaleSpace geometry by passing source as parameter to operations.



## Error Handling

### Type Safety

**Type Safety Rationale**: The type system prevents common mistakes like passing a ScaleSpaceResponse where a ScaleSpace is expected, since they are distinct types. The constraint `ScaleSpace{T <: GaussianImage}` ensures ScaleSpace can only contain Gaussian data.

### Runtime Validation

```julia
# Validate geometry compatibility during computation
function (response::ScaleSpaceResponse)(source::ScaleSpace)
    @assert !isempty(source.levels) "Source ScaleSpace must be populated"
    
    # Validate geometry compatibility
    @assert response.first_octave == source.first_octave "Octave ranges must match"
    @assert response.last_octave == source.last_octave "Octave ranges must match"
    @assert response.octave_first_subdivision == source.octave_first_subdivision "Subdivision ranges must match"
    @assert response.octave_last_subdivision == source.octave_last_subdivision "Subdivision ranges must match"
    
    # Validate that source contains actual image data
    first_level = first(source.levels)
    @assert !isempty(first_level.data.g) "Source ScaleSpace must contain image data"
    
    # ... compute responses
end

# Helper function for geometry compatibility
function compatible_geometry(response::ScaleSpaceResponse, source::ScaleSpace)
    return (response.first_octave == source.first_octave &&
            response.last_octave == source.last_octave &&
            response.octave_first_subdivision == source.octave_first_subdivision &&
            response.octave_last_subdivision == source.octave_last_subdivision)
end
```

**Runtime Validation Rationale**: These checks ensure that responses are only created from properly populated scale spaces and that the source data remains valid throughout the response computation process.

## Testing Strategy

### Unit Tests

1. **ScaleSpace Tests (Requirement 1 & 3)**
   - Gaussian smoothing only (verify no derivative functionality)
   - Kernel function storage and application (vlfeat_gaussian, Kernel.gaussian)
   - 2D indexing with non-1-based coordinates (ss[0, 0], ss[-1, 2], etc.)
   - VLFeat indexing compatibility (octaves 0:N, subdivisions 0:(resolution-1))
   - Utilities work with custom indexing (level_size, level_step)
   - Type constraints (only GaussianImage{T} types accepted)
   - Constructor validation (reject non-Gaussian image types)

2. **ScaleSpaceResponse Tests (Requirement 2 & 3)**
   - Response computation from populated ScaleSpace
   - Filter NamedTuple and Function storage and application
   - Type safety (prevent ScaleSpaceResponse where ScaleSpace expected)
   - Source validation (require populated ScaleSpace)
   - Performance (broadcasting, memory efficiency)

3. **Integration Tests (Requirement 4 & 5)**
   - Complete pipeline: Image → ScaleSpace → ScaleSpaceResponse → ScaleSpaceResponse
   - Predefined filter constant workflows (HESSIAN_FILTERS, LAPLACIAN_FUNCTION)
   - Backward compatibility (existing indexing, utilities work unchanged)
   - API clarity (clear distinction between scale space and response operations)

### Performance Tests (Requirement 6)

1. **Memory Allocation**
   - Verify no unnecessary copying between ScaleSpace and ScaleSpaceResponse
   - Benchmark response computation efficiency
   - Ensure minimal memory allocation during response creation

2. **Computational Performance**
   - Compare performance before/after refactoring (no regression)
   - Ensure broadcasting optimizations are preserved in both types
   - Verify 2D indexing performance maintained
   - Benchmark chained operations (ScaleSpace → Response → Response)

### Type Safety Tests (Requirement 3)

1. **Compile-time Checks**
   - Verify ScaleSpace{T} only accepts T <: GaussianImage types
   - Verify ScaleSpaceResponse prevents passing where ScaleSpace expected
   - Test function signatures reject wrong types at compile time

2. **Runtime Validation**
   - Test clear error messages for invalid operations
   - Verify transform type constraints (Function for ScaleSpace, NamedTuple/Function for ScaleSpaceResponse)
   - Test source validation (populated ScaleSpace required)

## Migration Strategy

### Phase 1: Add ScaleSpaceResponse (Additive)
- Add new ScaleSpaceResponse type alongside existing ScaleSpace
- Add convenience functions (hessian_response, laplacian_response)
- Maintain full backward compatibility

### Phase 2: Deprecate Mixed-Use ScaleSpace (Gradual)
- Add deprecation warnings for non-Gaussian ScaleSpace usage
- Update documentation to recommend new approach
- Provide migration examples

### Phase 3: Simplify ScaleSpace (Breaking)
- Remove non-Gaussian functionality from ScaleSpace
- Constrain ScaleSpace type parameter to GaussianImage only
- Update all internal code to use new architecture

This phased approach ensures smooth migration while maintaining backward compatibility during the transition.