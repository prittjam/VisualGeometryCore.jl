# ScaleSpace Abstract Base Type Implementation Tasks

- [x] 1. Define abstract base type and update type hierarchy
  - Add `abstract type AbstractScaleSpace{T} end` at the top of the file
  - Update `ScaleSpace{T <: GaussianImage}` to inherit from `AbstractScaleSpace{T}`
  - Update `ScaleSpaceResponse{T, F}` to inherit from `AbstractScaleSpace{T}`
  - Verify type hierarchy works correctly with simple instantiation tests
  - _Requirements: R1.1, R1.2, R4.3_

- [x] 2. Implement shared geometry interface
  - Create `geometry_info(ss::ScaleSpace)` method that returns geometry tuple
  - Create `geometry_info(ss::ScaleSpaceResponse, source::ScaleSpace)` method
  - Add helper function `levels(ss::AbstractScaleSpace) = ss.levels` for field access
  - Test that geometry extraction works for both types
  - _Requirements: R1.3, R4.1_

- [x] 3. Move shared indexing methods to abstract base type
  - Implement `Base.getindex(ss::AbstractScaleSpace, i::Int)` for linear indexing
  - Implement `Base.setindex!(ss::AbstractScaleSpace, value, i::Int)` for linear indexing
  - Implement `Base.getindex(ss::AbstractScaleSpace, I::CartesianIndex{2})` with delegation
  - Remove duplicated linear indexing code from concrete types
  - _Requirements: R2.1, R2.3, R6.1_

- [x] 4. Move shared iteration interface to abstract base type
  - Implement `Base.iterate(ss::AbstractScaleSpace)` and `Base.iterate(ss::AbstractScaleSpace, state)`
  - Implement `Base.length(ss::AbstractScaleSpace)`, `Base.eltype`, `Base.firstindex`, `Base.lastindex`
  - Remove duplicated iteration code from concrete types
  - Test that iteration works identically for both types
  - _Requirements: R3.1, R3.2, R3.3, R6.2_

- [x] 5. Implement specialized 2D indexing for each concrete type
  - Keep `Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int)` as ScaleSpace-specific
  - Keep `Base.getindex(ss::ScaleSpaceResponse, octave::Int, subdivision::Int, source::ScaleSpace)` as ScaleSpaceResponse-specific
  - Implement `Base.getindex(ss::ScaleSpace, octave::Int, subdivisions::AbstractRange)`
  - Implement `Base.getindex(ss::ScaleSpaceResponse, octave::Int, subdivisions::AbstractRange, source::ScaleSpace)`
  - _Requirements: R2.1, R2.2, R4.2_

- [x] 6. Implement specialized size and axes methods
  - Keep `Base.size(ss::ScaleSpace)` and `Base.axes(ss::ScaleSpace)` as ScaleSpace-specific
  - Keep `Base.size(ss::ScaleSpaceResponse, source::ScaleSpace)` and `Base.axes(ss::ScaleSpaceResponse, source::ScaleSpace)` as ScaleSpaceResponse-specific
  - Ensure both use the geometry_info interface consistently
  - Test that size/axes work correctly for both types
  - _Requirements: R3.2, R4.2_

- [x] 7. Create comprehensive inheritance and shared functionality tests
  - Test that both types are proper subtypes of AbstractScaleSpace
  - Test that shared methods work identically on both types
  - Test that iteration, length, indexing work consistently
  - Test that type parameters are properly propagated
  - _Requirements: R3.4, R4.1, R4.3_

- [x] 8. Run full regression testing suite
  - Execute all existing ScaleSpace tests to ensure no functionality breaks
  - Execute all existing ScaleSpaceResponse tests to ensure no functionality breaks
  - Verify all existing APIs continue to work without modification
  - Test performance to ensure no regressions from abstraction
  - _Requirements: R5.1, R5.2, R5.3, R5.4_

- [x] 9. Update documentation and add usage examples
  - Add docstring for AbstractScaleSpace explaining the shared interface
  - Update ScaleSpace and ScaleSpaceResponse docstrings to mention inheritance
  - Add examples showing how shared methods work on both types
  - Document the geometry_info interface for future extensions
  - _Requirements: R1.4, R6.4_