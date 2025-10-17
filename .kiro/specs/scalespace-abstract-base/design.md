# ScaleSpace Abstract Base Type Design

## Overview

This design introduces an abstract base type `AbstractScaleSpace{T}` to eliminate code duplication between `ScaleSpace` and `ScaleSpaceResponse`. Both types share identical structure for the `levels` field and indexing patterns, making them perfect candidates for inheritance-based code reuse.

## Architecture

### Type Hierarchy

```julia
abstract type AbstractScaleSpace{T} end

struct ScaleSpace{T <: GaussianImage} <: AbstractScaleSpace{T}
    # Geometry fields
    first_octave::Int
    last_octave::Int
    octave_resolution::Int
    octave_first_subdivision::Int
    octave_last_subdivision::Int
    base_sigma::Float64
    camera_psf::Float64
    
    # Shared field
    levels::StructArray{ScaleLevel{T}}
    
    # ScaleSpace-specific fields
    input_size::Size2{Int}
    kernel_func::Function
    populated::Ref{Bool}
end

struct ScaleSpaceResponse{T, F} <: AbstractScaleSpace{T}
    # Shared field
    levels::StructArray{ScaleLevel{T}}
    
    # ScaleSpaceResponse-specific field
    transform::F
end
```

### Shared Interface

The abstract base type defines the interface for accessing the common `levels` field:

```julia
# Abstract interface - must be implemented by concrete types
levels(ss::AbstractScaleSpace) = ss.levels

# Geometry interface - different implementations for each type
geometry_info(ss::ScaleSpace) = (ss.first_octave, ss.last_octave, 
                                ss.octave_first_subdivision, ss.octave_last_subdivision)
geometry_info(ss::ScaleSpaceResponse, source::ScaleSpace) = geometry_info(source)
```

## Components and Interfaces

### Abstract Base Type Methods

All shared functionality will be implemented once on `AbstractScaleSpace{T}`:

#### Indexing Methods
```julia
# 2D indexing: ss[octave, subdivision]
function Base.getindex(ss::AbstractScaleSpace, octave::Int, subdivision::Int)
    # Implementation using geometry_info(ss) for bounds and conversion
end

# CartesianIndex support
Base.getindex(ss::AbstractScaleSpace, I::CartesianIndex{2}) = ss[I[1], I[2]]

# Range indexing
function Base.getindex(ss::AbstractScaleSpace, octave::Int, subdivisions::AbstractRange)
    return [ss[octave, s] for s in subdivisions]
end

# Linear indexing
Base.getindex(ss::AbstractScaleSpace, i::Int) = levels(ss)[i]
Base.setindex!(ss::AbstractScaleSpace, value, i::Int) = (levels(ss)[i] = value)
```

#### Iteration Interface
```julia
# Iterator protocol
Base.iterate(ss::AbstractScaleSpace) = iterate(levels(ss))
Base.iterate(ss::AbstractScaleSpace, state) = iterate(levels(ss), state)

# Collection interface
Base.length(ss::AbstractScaleSpace) = length(levels(ss))
Base.eltype(::Type{<:AbstractScaleSpace{T}}) where T = ScaleLevel{T}
Base.firstindex(ss::AbstractScaleSpace) = firstindex(levels(ss))
Base.lastindex(ss::AbstractScaleSpace) = lastindex(levels(ss))
```

#### Size and Axes Methods
```julia
# Size interface - requires geometry information
function Base.size(ss::ScaleSpace)
    first_oct, last_oct, first_sub, last_sub = geometry_info(ss)
    num_octaves = last_oct - first_oct + 1
    num_subdivisions = last_sub - first_sub + 1
    return (num_octaves, num_subdivisions)
end

function Base.axes(ss::ScaleSpace)
    first_oct, last_oct, first_sub, last_sub = geometry_info(ss)
    return (first_oct:last_oct, first_sub:last_sub)
end

# ScaleSpaceResponse needs source context for geometry
Base.size(ss::ScaleSpaceResponse, source::ScaleSpace) = size(source)
Base.axes(ss::ScaleSpaceResponse, source::ScaleSpace) = axes(source)
```

### Concrete Type Specializations

#### ScaleSpace-Specific Methods
```julia
# Geometry extraction (no source needed)
geometry_info(ss::ScaleSpace) = (ss.first_octave, ss.last_octave,
                                ss.octave_first_subdivision, ss.octave_last_subdivision)

# Size/axes without source context
Base.size(ss::ScaleSpace) = # implementation using internal geometry
Base.axes(ss::ScaleSpace) = # implementation using internal geometry

# ScaleSpace-specific indexing (no source needed)
Base.getindex(ss::ScaleSpace, octave::Int, subdivision::Int) = # direct implementation
```

#### ScaleSpaceResponse-Specific Methods
```julia
# Geometry extraction (requires source)
geometry_info(ss::ScaleSpaceResponse, source::ScaleSpace) = geometry_info(source)

# Indexing with source context
function Base.getindex(ss::ScaleSpaceResponse, octave::Int, subdivision::Int, source::ScaleSpace)
    # Use source geometry for bounds checking and indexing
end

# Size/axes with source context
Base.size(ss::ScaleSpaceResponse, source::ScaleSpace) = size(source)
Base.axes(ss::ScaleSpaceResponse, source::ScaleSpace) = axes(source)
```

## Data Models

### Shared Structure

Both types share the core data structure:
```julia
levels::StructArray{ScaleLevel{T}}
```

This enables all iteration and linear indexing to work identically.

### Geometry Handling

The key difference is geometry access:
- **ScaleSpace**: Stores geometry fields directly
- **ScaleSpaceResponse**: Requires source ScaleSpace for geometry context

This is handled through the `geometry_info()` function with different signatures.

## Error Handling

### Bounds Checking
- Shared bounds checking logic in abstract base methods
- Geometry-specific validation in concrete type methods
- Consistent error messages across both types

### Type Safety
- Abstract type parameter `T` ensures type consistency
- Concrete types can add additional constraints (e.g., `T <: GaussianImage` for ScaleSpace)
- Method dispatch ensures correct specialization

## Testing Strategy

### Inheritance Testing
```julia
@testset "Abstract Base Type" begin
    # Test that both types are subtypes of AbstractScaleSpace
    @test ScaleSpace{GaussianImage{N0f8}} <: AbstractScaleSpace
    @test ScaleSpaceResponse{HessianImages{Float32}, Function} <: AbstractScaleSpace
    
    # Test shared interface works for both types
    ss = create_test_scalespace()
    ssr = create_test_response()
    
    @test length(ss) == length(ssr)  # If same geometry
    @test eltype(ss) == eltype(ssr)  # If same T parameter
end
```

### Shared Functionality Testing
```julia
@testset "Shared Methods" begin
    # Test that iteration works identically
    ss = create_test_scalespace()
    ssr = create_test_response()
    
    # Both should support same iteration patterns
    @test collect(ss) isa Vector{ScaleLevel}
    @test collect(ssr) isa Vector{ScaleLevel}
    
    # Both should support same linear indexing
    @test ss[1] isa ScaleLevel
    @test ssr[1] isa ScaleLevel
end
```

### Specialization Testing
```julia
@testset "Type-Specific Methods" begin
    # Test ScaleSpace-specific methods
    ss = create_test_scalespace()
    @test size(ss) isa Tuple{Int, Int}
    @test axes(ss) isa Tuple{UnitRange, UnitRange}
    @test ss[0, 1] isa ScaleLevel
    
    # Test ScaleSpaceResponse-specific methods
    ssr = create_test_response()
    @test size(ssr, ss) isa Tuple{Int, Int}
    @test axes(ssr, ss) isa Tuple{UnitRange, UnitRange}
    @test ssr[0, 1, ss] isa ScaleLevel
end
```

## Implementation Details

### Migration Strategy

1. **Define abstract type**: Add `abstract type AbstractScaleSpace{T} end`
2. **Update concrete types**: Add `<: AbstractScaleSpace{T}` to both structs
3. **Move shared methods**: Migrate common indexing/iteration to abstract type
4. **Add geometry interface**: Implement `geometry_info()` for both types
5. **Specialize where needed**: Override methods that need type-specific behavior
6. **Update tests**: Ensure all existing functionality still works

### Method Resolution

Julia's multiple dispatch will handle method resolution:
- Shared methods dispatch on `AbstractScaleSpace`
- Type-specific methods dispatch on concrete types
- Most specific method wins (concrete type methods override abstract ones)

### Performance Considerations

- **No runtime overhead**: Abstract types in Julia have zero runtime cost
- **Compile-time specialization**: Methods are specialized at compile time
- **Inlining**: Shared methods can be inlined just like concrete methods
- **Type stability**: Abstract type parameters maintain type stability

## Benefits

### Code Quality
- **DRY principle**: Eliminates duplicated indexing/iteration code
- **Consistency**: Ensures both types behave identically for shared operations
- **Maintainability**: Single source of truth for shared functionality

### Type Safety
- **Inheritance**: Proper type hierarchy with shared interface
- **Specialization**: Ability to override methods when needed
- **Dispatch**: Leverages Julia's multiple dispatch system

### API Design
- **Unified interface**: Common operations work the same way
- **Extensibility**: Easy to add new types that share the same structure
- **Backward compatibility**: Existing code continues to work unchanged

## Migration Notes

This refactoring maintains full backward compatibility:
- All existing method signatures remain unchanged
- All existing functionality continues to work
- Performance characteristics are preserved or improved
- No user code changes required

The abstract base type is purely an internal implementation detail that improves code organization without affecting the public API.