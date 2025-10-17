# ScaleSpace 3D Cube Refactor Requirements

## Introduction

This specification addresses refactoring the scale space implementation to use octave-backed 3D cubes with per-level views instead of the current flattened level storage. This change will improve memory locality, enable more efficient batch operations, and provide better support for 3D finite differences while preserving VLFeat terminology and current API ergonomics.

## Requirements

### Requirement 1: 3D Cube Storage Architecture

**User Story:** As a computer vision developer, I want each octave to store a single 3D Gaussian cube so that I can perform efficient batch operations and have better memory locality for scale-space processing.

#### Acceptance Criteria

1. WHEN creating a ScaleSpace THEN each octave SHALL store data as a single 3D array `G::Array{T,3}` with dimensions `(H^o, W^o, Lg)`
2. WHEN accessing subdivision levels THEN each level SHALL be a view `@view G[:,:,ℓ]` into the 3D cube
3. WHEN storing Gaussian levels THEN each octave SHALL contain `Lg = S + 3` levels where `S` is the octave resolution
4. WHEN mapping subdivisions to slices THEN the slice index SHALL be `ℓ = (s - first(subdivisions)) + 1` where `s` is the subdivision number
5. WHEN allocating octave storage THEN dimensions SHALL be computed as `(H^o, W^o) = (H_input / 2^o, W_input / 2^o)`

### Requirement 2: New Type System

**User Story:** As a developer using the ScaleSpace API, I want clear type distinctions between octaves and level views so that the code is more maintainable and type-safe.

#### Acceptance Criteria

1. WHEN defining types THEN the system SHALL provide `ScaleLevelView{T}` containing:
   - `data::SubArray{T,2,Array{T,3}}` (view into the 3D cube)
   - `octave::Int` (octave number)
   - `subdivision::Int` (VLFeat subdivision/level within octave)
   - `sigma::Float64` (absolute σ for this level)
   - `step::Int` (pixel stride `2^octave`)

2. WHEN defining types THEN the system SHALL provide `ScaleOctave{T}` containing:
   - `G::Array{T,3}` (Gaussian cube with dims `(H^o, W^o, Lg)`)
   - `octave::Int` (octave number)
   - `subdivisions::UnitRange{Int}` (e.g., `0:(S+2)` for Gaussian levels)
   - `sigmas::Vector{Float64}` (length `Lg`, absolute σ per level)
   - `step::Int` (`2^octave`)

3. WHEN defining types THEN the system SHALL provide `ScaleSpace{T}` containing:
   - `base_sigma::Float64`
   - `camera_psf::Float64`
   - `octaves::Vector{ScaleOctave{T}}`
   - `octave_resolution::Int` (S, VLFeat octaveResolution parameter)
   - `input_size::Tuple{Int,Int}`
   - `border_policy::String`

### Requirement 3: Indexing and Access Patterns

**User Story:** As a developer using the ScaleSpace API, I want convenient `ss[o,s]` access patterns so that I can efficiently access octaves and subdivisions.

#### Acceptance Criteria

1. WHEN accessing octaves THEN `ss[o]` SHALL return a `ScaleOctave` for octave `o`
2. WHEN accessing levels THEN `ss[o, s]` SHALL return a `ScaleLevelView` for subdivision `s` within octave `o`
3. WHEN using ergonomic access THEN `ss[o].level[s]` SHALL be equivalent to `ss[o,s]`
4. WHEN indexing with external octave numbers THEN the system SHALL map external `o` to internal storage index `oi = o - first_octave + 1`
5. WHEN accessing out-of-bounds indices THEN the system SHALL provide clear error messages mentioning octave number and subdivision

### Requirement 4: VLFeat-Compatible Gaussian Schedule

**User Story:** As a computer vision researcher, I want the sigma schedule to remain compatible with VLFeat so that my results are consistent with established SIFT implementations.

#### Acceptance Criteria

1. WHEN computing sigma values THEN the system SHALL use `sigma(o,s) = base_sigma * 2.0^(o + s/S)` for absolute σ per level
2. WHEN creating Gaussian levels THEN each octave SHALL have `Lg = S + 3` levels with `s ∈ 0:(S+2)`
3. WHEN setting pixel stride THEN `step = 2^o` SHALL apply to all levels in octave `o`
4. WHEN building the scale space THEN the system SHALL preserve existing incremental Gaussian smoothing and border policy

### Requirement 5: DoG and Derivative Support

**User Story:** As a feature detection developer, I want efficient DoG computation and 3D finite differences so that I can perform keypoint localization and extrema detection.

#### Acceptance Criteria

1. WHEN computing DoG THEN the system SHALL support `DoG[:,:,ℓ] = G[:,:,ℓ+1] - G[:,:,ℓ]` for `ℓ = 1:(Lg-1)`
2. WHEN computing DoG levels THEN there SHALL be `Ld = S + 2` DoG levels corresponding to `s ∈ 1:(S+1)`
3. WHEN performing 3D finite differences THEN the system SHALL compute gradient/Hessian on demand at candidate voxels using 3×3×3 central differences
4. WHEN processing dense spatial derivatives THEN the system SHALL treat scale as batch dimension for efficient convolution across all slices

### Requirement 6: Clean API Design

**User Story:** As a developer using the new ScaleSpace API, I want a clean and efficient interface that takes advantage of the 3D cube architecture so that I can write performant scale-space processing code.

#### Acceptance Criteria

1. WHEN using response functions THEN they SHALL accept `AbstractMatrix` views for efficient processing
2. WHEN iterating levels THEN the pattern `for o in ss.octaves, s in o.subdivisions` SHALL work efficiently
3. WHEN accessing level data THEN consumers SHALL receive `SubArray{T,2}` views for zero-copy access
4. WHEN using kernels THEN they SHALL work efficiently with the new view-based data
5. WHEN designing the API THEN it SHALL prioritize performance and clarity over backward compatibility

### Requirement 7: Performance and Memory Efficiency

**User Story:** As a performance-conscious developer, I want the 3D cube storage to provide better memory locality and enable batch operations so that my scale-space processing is faster.

#### Acceptance Criteria

1. WHEN accessing adjacent scale levels THEN memory locality SHALL be improved compared to flattened storage
2. WHEN performing batch operations THEN the system SHALL enable efficient processing across all levels in an octave
3. WHEN computing derivatives THEN dense operations SHALL be optimizable using reshape and batch convolution
4. WHEN storing data THEN memory usage SHALL not increase significantly compared to the current implementation
5. WHEN accessing views THEN there SHALL be no unnecessary copying of data

### Requirement 8: Validation and Testing

**User Story:** As a quality assurance engineer, I want comprehensive tests to ensure the refactored implementation maintains correctness so that existing functionality is preserved.

#### Acceptance Criteria

1. WHEN constructing with `S=3` THEN `ss[o] isa ScaleOctave` and `ss[o,s] isa ScaleLevelView` SHALL be true
2. WHEN verifying sigma schedule THEN `abs(ss[o,s].sigma - base_sigma * 2^(o + s/S)) < 1e-6` SHALL hold for any valid `(o,s)`
3. WHEN testing view identity THEN writing `ss[o,s].data[i,j] = v` SHALL result in `o.G[i,j,ℓ] == v` where `ℓ = (s - first(o.subdivisions)) + 1`
4. WHEN computing DoG THEN results SHALL match the previous flattened implementation within numerical tolerance
5. WHEN running existing response functions THEN outputs SHALL be numerically identical to pre-refactor results
6. WHEN performing 3×3×3 localization THEN gradient/Hessian computations SHALL yield identical offsets to the old implementation