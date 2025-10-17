# Requirements Document

## Introduction

The current ScaleSpace implementation tries to handle both Gaussian scale space construction and response computations (Hessian, Laplacian, etc.) in a single struct. This creates architectural complexity and mixing of concerns. We need to separate these into two distinct types:

1. **ScaleSpace**: Focused solely on Gaussian scale space construction and storage
2. **ScaleSpaceResponse**: Focused on computing responses (derivatives, filters) from ScaleSpace

This separation will create cleaner code, better type safety, and more intuitive APIs.

## Requirements

### Requirement 1: ScaleSpace Simplification

**User Story:** As a developer, I want ScaleSpace to only handle Gaussian smoothing, so that it has a single, clear responsibility.

#### Acceptance Criteria

1. WHEN creating a ScaleSpace THEN it SHALL only store Gaussian smoothed images
2. WHEN ScaleSpace is created THEN it SHALL store only Gaussian kernel functions (like vlfeat_gaussian)
3. WHEN ScaleSpace is populated THEN it SHALL only perform Gaussian smoothing operations
4. WHEN accessing ScaleSpace data THEN it SHALL only contain grayscale image data (GaussianImage{T})

### Requirement 2: ScaleSpaceResponse Creation

**User Story:** As a developer, I want a separate ScaleSpaceResponse type to compute derivatives and filters, so that response computations are cleanly separated from scale space construction.

#### Acceptance Criteria

1. WHEN creating a ScaleSpaceResponse THEN it SHALL accept a source ScaleSpace as input
2. WHEN ScaleSpaceResponse is created THEN it SHALL store either filter NamedTuples or processing functions
3. WHEN ScaleSpaceResponse is populated THEN it SHALL compute responses from the source ScaleSpace
4. WHEN accessing ScaleSpaceResponse data THEN it SHALL contain computed response data (HessianImages, LaplacianImage, etc.)

### Requirement 3: Type Safety and Consistency

**User Story:** As a developer, I want clear type distinctions between scale spaces and responses, so that I cannot accidentally mix them up.

#### Acceptance Criteria

1. WHEN working with ScaleSpace THEN it SHALL have type ScaleSpace{T} where T is always GaussianImage
2. WHEN working with ScaleSpaceResponse THEN it SHALL have type ScaleSpaceResponse{T, F} where T is response type and F is transform type
3. WHEN passing arguments THEN the type system SHALL prevent passing ScaleSpaceResponse where ScaleSpace is expected
4. WHEN creating responses THEN the API SHALL clearly distinguish between scale space and response operations

### Requirement 4: Clean API Design

**User Story:** As a developer, I want intuitive APIs for both scale spaces and responses, so that the code is easy to understand and use.

#### Acceptance Criteria

1. WHEN creating Gaussian scale space THEN I SHALL use ScaleSpace(image) or gaussian_scalespace(image)
2. WHEN creating Hessian responses THEN I SHALL use HessianResponse(scalespace) or hessian_response(scalespace)
3. WHEN creating Laplacian responses THEN I SHALL use LaplacianResponse(scalespace) or laplacian_response(scalespace)
4. WHEN chaining operations THEN I SHALL be able to create responses from scale spaces: response = HessianResponse(scalespace)

### Requirement 5: Backward Compatibility

**User Story:** As a developer, I want the refactoring to maintain existing functionality, so that current code continues to work.

#### Acceptance Criteria

1. WHEN using existing ScaleSpace indexing THEN it SHALL continue to work: ss[octave, subdivision]
2. WHEN using existing utility functions THEN they SHALL continue to work: level_size(), level_step()
3. WHEN using existing broadcasting operations THEN they SHALL continue to work efficiently
4. WHEN using existing Size2 optimizations THEN they SHALL continue to work correctly

### Requirement 6: Performance Maintenance

**User Story:** As a developer, I want the separated design to maintain current performance, so that there's no regression in computational efficiency.

#### Acceptance Criteria

1. WHEN computing responses THEN broadcasting operations SHALL be preserved
2. WHEN accessing data THEN 2D indexing performance SHALL be maintained
3. WHEN creating responses THEN memory allocation SHALL be minimized
4. WHEN chaining operations THEN unnecessary data copying SHALL be avoided