# Performance Notes for backproject

## Summary

The `backproject` function currently allocates ~400 bytes per call when using unitful (px) coordinates, with 37% of execution time spent in garbage collection for large batches. This is due to Unitful.jl type operations not being fully optimized away.

## Current Implementation

### Design
- K stored with pixel units in both `LogicalIntrinsics` and `PhysicalIntrinsics`
- K_inv precomputed during construction (stored as unitless `SMatrix{3,3,Float64}`)
- Unit consistency enforced via assertion: `typeof(K[1,1]) == typeof(h[1])`
- Unit algebra: `K_inv * ustrip(h) * (unit(h[1]) / unit(K[1,1]))`

### Performance Characteristics

**Single call:**
- Allocations: ~400 bytes
- Time: ~6 microseconds
- Breakdown:
  - `ustrip(h)`: ~64 bytes
  - `unit(h[1]) / unit(K[1,1])`: ~32 bytes
  - Matrix operations: ~128 bytes
  - Assertion and overhead: ~176 bytes

**Batch operations (1M calls):**
- Total allocations: 17M allocations, 405 MB
- GC time: 37% of total execution time
- Per-call overhead is consistent

## Investigation Results

### What We Tested

1. **Pure StaticArrays (no Unitful):** 64 bytes/call
   - Even unitless operations show some allocation
   - This appears to be measurement overhead or Julia internals

2. **Unitful operations separately:**
   - `unit(x)`: 0 bytes (optimized away)
   - `ustrip(x)`: 16 bytes
   - `unit(x) / unit(y)`: 32 bytes
   - `ustrip(SVector{3})`: 64 bytes
   - Matrix multiply with units: 128+ bytes

3. **Combined operations:** 288 bytes
   - Julia optimizes away some intermediate allocations
   - But not enough to eliminate GC pressure

### Root Cause

Unitful.jl allocates during:
- Type arithmetic (`unit(x) / unit(y)`)
- `ustrip` on container types (SVector)
- Multiplying results by unit ratios

These are likely temporary objects created during type propagation that can't be fully optimized away by the compiler.

## Optimization Options (Future Work)

### Option 1: Accept Current Performance
- **Pros:** Simple, correct, type-safe
- **Cons:** 37% GC overhead on large batches
- **Verdict:** Acceptable for typical use (small batches of points for P3P, calibration)

### Option 2: Allocation-Free Unitless Fast Path
Store K without units for performance-critical applications:

```julia
struct FastLogicalIntrinsics <: AbstractIntrinsics
    K::SMatrix{3,3,Float64}
    K_inv::SMatrix{3,3,Float64}
end

function backproject(intrinsics::FastLogicalIntrinsics, ::PinholeProjection, u::StaticVector{2,Float64})
    h = to_affine(u)
    ray = intrinsics.K_inv * h
    return normalize(ray)
end
```

- **Pros:** Zero allocations, maximum performance
- **Cons:** Loses unit safety, need to maintain two code paths
- **Verdict:** Implement if profiling shows backproject is a bottleneck

### Option 3: Upstream Fix in Unitful.jl
File an issue with Unitful.jl requesting allocation-free unit arithmetic.

- **Pros:** Benefits entire ecosystem
- **Cons:** Outside our control, may not be possible
- **Verdict:** Worth investigating if this becomes critical

## Design Decisions Made

### 1. Precompute K_inv (✅ Implemented)
Store `K_inv` in intrinsics structs to avoid recomputing `inv(K)` for every point.

- **Memory cost:** 72 bytes (9 Float64 values)
- **Performance benefit:** Eliminates one matrix inversion per call
- **Verdict:** Clear win, no downsides

### 2. Store K with Units (✅ Implemented)
K stored with pixel units enables natural unit algebra: (px^-1) * px → dimensionless

- **Memory cost:** Negligible (same size as Float64)
- **Runtime cost:** ~400 bytes allocation per call
- **Verdict:** Correctness > performance for now

### 3. Unit Consistency Assertion (✅ Implemented)
Enforce that K and pixel coordinates have matching unit types.

```julia
@assert typeof(K[1,1]) == typeof(h[1]) "Unit mismatch: ..."
```

- **Cost:** ~32 bytes, helps with error messages
- **Verdict:** Keep for safety, can disable with `--check-bounds=no` if needed

## Benchmarks

### Baseline (1000 points × 1000 iterations)
```
1.039 seconds (17.01 M allocations: 404.871 MiB, 37.18% gc time)
```

### Per-call metrics
- Time: 6 μs
- Allocations: 400 bytes
- GC overhead: 37% (significant!)

## Recommendations

1. **Current implementation is acceptable** for typical use cases:
   - P3P solver: 3-10 points
   - Camera calibration: 10-100 points
   - SLAM feature tracking: 100-1000 points per frame

2. **Monitor in real applications:**
   - Profile actual usage patterns
   - If GC time > 10% of total, revisit optimization

3. **Future optimization path:**
   - Implement `FastLogicalIntrinsics` without units
   - Use for performance-critical tight loops
   - Keep unit-safe version as default

## Related Code

- `src/geometry/cameras/intrinsics.jl`: Struct definitions with K_inv
- `src/geometry/cameras/camera_models.jl`: backproject implementations
- Tests: `test/test_p3p_poselib.jl`, `src/geometry/example.jl`

## Date
2025-01-29
