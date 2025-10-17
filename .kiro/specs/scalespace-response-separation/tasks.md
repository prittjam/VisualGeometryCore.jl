# Implementation Plan

- [x] 1. Create ScaleSpaceResponse type and basic infrastructure
  - Create new ScaleSpaceResponse{T, F} struct with only levels and transform fields
  - Implement basic constructor that takes geometry template and transform
  - Add type inference from transform (NamedTuple filters or Function)
  - Create preallocated levels matching template geometry
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 2. Implement ScaleSpaceResponse indexing and utilities
  - Implement custom getindex methods that require source ScaleSpace context
  - Add support for response[octave, subdivision, source] indexing pattern
  - Ensure level_size() and level_step() utilities work with response levels
  - Add iteration support and basic array interface methods
  - _Requirements: 2.4, 5.1, 5.2_

- [x] 3. Create predefined filter constants
  - Define HESSIAN_FILTERS constant with standard second derivative filters
  - Define LAPLACIAN_FUNCTION constant for Laplacian computation from Hessian
  - Ensure filters use ImageFiltering.centered() for proper kernel definition
  - Add documentation for filter definitions and usage patterns
  - _Requirements: 4.2, 4.3_

- [x] 4. Implement ScaleSpaceResponse functor for response computation
  - Create (response::ScaleSpaceResponse)(source::ScaleSpace) method
  - Add geometry compatibility validation between response and source
  - Implement filter application for NamedTuple transforms (broadcasting)
  - Implement function application for Function transforms
  - Add support for partial computation (octave/subdivision ranges)
  - _Requirements: 2.3, 2.4, 6.1, 6.2_

- [x] 5. Constrain ScaleSpace to Gaussian-only operations
  - Add type constraint T <: GaussianImage to ScaleSpace struct definition
  - Update ScaleSpace constructor to only accept Gaussian image types
  - Remove image_type parameter from constructor (always GaussianImage)
  - Ensure kernel_func parameter defaults to vlfeat_gaussian
  - Add validation to reject non-Gaussian image types at construction
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2_

- [x] 6. Update ScaleSpace functor to use stored kernel function
  - Modify (ss::ScaleSpace)(image) to use stored kernel_func by default
  - Maintain override capability: ss(image, custom_kernel_func)
  - Ensure Gaussian smoothing operations only (no derivative computation)
  - Validate that input images match expected grayscale format
  - _Requirements: 1.3, 4.1, 5.3_

- [x] 7. Implement comprehensive validation and error handling
  - Add geometry compatibility checks in ScaleSpaceResponse constructor
  - Implement clear error messages for type mismatches
  - Add validation for populated ScaleSpace requirement in response computation
  - Ensure bounds checking works with custom octave/subdivision indexing
  - Add helpful error messages for common API misuse patterns
  - _Requirements: 3.3, 3.4, 6.3_

- [x] 8. Create comprehensive test suite for separated types
  - Write unit tests for ScaleSpace Gaussian-only functionality
  - Write unit tests for ScaleSpaceResponse creation and computation
  - Test custom indexing with negative octaves and 0-based subdivisions
  - Test predefined filter constants (HESSIAN_FILTERS, LAPLACIAN_FUNCTION)
  - Test type safety and validation error messages
  - Test performance and memory allocation patterns
  - _Requirements: 5.4, 6.1, 6.2, 6.4_

- [ ] 9. Test integration and backward compatibility
  - Test complete pipeline: Image → ScaleSpace → ScaleSpaceResponse
  - Verify existing utility functions work unchanged (level_size, level_step)
  - Test reusability: same ScaleSpaceResponse with multiple ScaleSpace instances
  - Verify VLFeat indexing compatibility (octaves 0:N, subdivisions 0:resolution-1)
  - Test broadcasting operations and performance characteristics
  - _Requirements: 5.1, 5.2, 5.3, 6.1, 6.2_

- [ ] 10. Update documentation and examples
  - Update docstrings for simplified ScaleSpace API
  - Add documentation for ScaleSpaceResponse usage patterns
  - Create examples showing predefined filter usage
  - Document the reusability pattern for ScaleSpaceResponse
  - Add migration guide from old mixed-use ScaleSpace
  - _Requirements: 4.1, 4.2, 4.3, 4.4_