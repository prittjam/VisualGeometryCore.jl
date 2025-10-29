# World → Image Transform Sequence for Pinhole Camera

This document traces the complete sequence of function calls and transformations when constructing and using a pinhole camera model with extrinsics.

## Construction Phase

### Step 1: Build PhysicalIntrinsics
```julia
# User code:
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
physical = PhysicalIntrinsics(f, pitch, pp)

# Function call sequence:
PhysicalIntrinsics(f, pitch, pp)
├─> CameraCalibrationMatrix(f, pitch, pp)  # cameras.jl:136
│   ├─> fx = f / pitch.width              # 16.0mm / (5.86μm/px) = 2730.4 px
│   ├─> fy = f / pitch.height             # 16.0mm / (5.86μm/px) = 2730.4 px
│   └─> Returns K::SMatrix{3,3}:
│       [fx  0  ppx]   [2730.4   0     320.0]
│       [0  fy  ppy] = [  0    2730.4  240.0]
│       [0   0   1 ]   [  0      0       1  ]
└─> PhysicalIntrinsics{Float64}(K, f, pitch)  # cameras.jl:214
```

### Step 2: Build CameraModel with Cached Transform
```julia
# User code:
model = CameraModel(physical, PinholeProjection())

# Function call sequence:
CameraModel(physical::PhysicalIntrinsics, PinholeProjection())  # cameras.jl:375
├─> build_projection_transform(physical, PinholeProjection())  # cameras.jl:318
│   ├─> Extract K matrix components:
│   │   A = K[1:2, 1:2] = [2730.4    0   ]
│   │                     [  0    2730.4 ]
│   │   t = K[1:2, 3] = [320.0, 240.0]
│   │
│   ├─> Build AffineMap:
│   │   affine = AffineMap(A, t)
│   │   # Applies: u = A * x + t
│   │
│   ├─> Build PerspectiveMap:
│   │   perspective = PerspectiveMap()
│   │   # Applies: x_norm = [X/Z, Y/Z]
│   │
│   └─> Compose:
│       transform = affine ∘ perspective
│       # Combined: u = A * [X/Z, Y/Z] + t
│
└─> CameraModel(physical, PinholeProjection(), transform)  # cameras.jl:377
    # Cached transform stored in struct
```

### Step 3: Build Camera with Extrinsics
```julia
# User code:
camera_pos = SVector(0.0, 0.0, 200.0)  # unitless (mm)
target = SVector(0.0, 0.0, 0.0)
up = SVector(0.0, 1.0, 0.0)
extrinsics = lookat(camera_pos, target, up)
camera = Camera(model, extrinsics)

# Function call sequence:
lookat(camera_pos, target, up)  # cameras.jl:541
├─> Computes rotation R and translation t
│   # For camera at (0,0,200) looking at origin:
│   # R = I (identity, camera aligned with world axes)
│   # t = [0, 0, 200]
└─> EuclideanMap(R, t)

Camera(model, extrinsics)  # cameras.jl:680
└─> Camera{CameraModel{...}, EuclideanMap{...}}(model, extrinsics)
```

## Projection Phase: World → Image

### Complete Transform Pipeline
```julia
# User code:
X_world = SVector(20.0mm, 10.0mm, 50.0mm)
u = project(camera, X_world)

# Function call sequence:
project(camera::Camera{<:CameraModel}, X_world)  # cameras.jl:717
│
├─> Step 1: Strip units
│   X_world_unitless = ustrip.(X_world)
│   # [20.0, 10.0, 50.0] (unitless, represents mm)
│
├─> Step 2: Apply extrinsics (World → Camera)
│   X_cam_unitless = camera.extrinsics(X_world_unitless)  # types.jl:78
│   │
│   │   EuclideanMap call operator:
│   │   (::EuclideanMap)(x)
│   │   ├─> R * x + t
│   │   ├─> For camera at (0,0,200) looking at origin:
│   │   │   R = I, t = [0, 0, 200]
│   │   └─> Result: [20.0, 10.0, 50.0] + [0, 0, 200]
│   │                = [20.0, 10.0, 250.0]
│   │
│   └─> X_cam_unitless = [20.0, 10.0, 250.0]
│
├─> Step 3: Restore units
│   X_cam = X_cam_unitless .* mm
│   # [20.0mm, 10.0mm, 250.0mm]
│
└─> Step 4: Project Camera → Pixels
    project(model::CameraModel, X_cam)  # cameras.jl:434
    │
    ├─> Strip units again
    │   X_cam_unitless = ustrip.(X_cam)
    │   # [20.0, 10.0, 250.0]
    │
    ├─> Apply cached composed transform
    │   u_unitless = model.transform(X_cam_unitless)
    │   │
    │   │   Composed transform call:
    │   │   (affine ∘ perspective)(X_cam_unitless)
    │   │   │
    │   │   ├─> perspective([20.0, 10.0, 250.0])
    │   │   │   └─> [20.0/250.0, 10.0/250.0] = [0.08, 0.04]
    │   │   │
    │   │   └─> affine([0.08, 0.04])
    │   │       ├─> A * [0.08, 0.04] + t
    │   │       ├─> [2730.4    0   ] * [0.08] + [320.0]
    │   │       │   [  0    2730.4 ]   [0.04]   [240.0]
    │   │       ├─> [218.4, 109.2] + [320.0, 240.0]
    │   │       └─> [538.4, 349.2]
    │   │
    │   └─> u_unitless = [538.4, 349.2]
    │
    └─> Add pixel units
        u = Point2(u_unitless...) * px
        # [538.4px, 349.2px]
```

## Mathematical Breakdown

### Complete Transform Chain

For a point **X**_world = [X_w, Y_w, Z_w]^T in world coordinates:

1. **Extrinsics (World → Camera):**
   ```
   X_cam = R * X_world + t
   ```
   Where R is rotation matrix, t is translation vector

2. **Perspective Division (Camera → Normalized):**
   ```
   x_norm = [X_cam / Z_cam]
            [Y_cam / Z_cam]
   ```

3. **Intrinsics (Normalized → Pixels):**
   ```
   u = K[1:2,1:2] * x_norm + K[1:2,3]
     = [fx  0 ] * [X_cam/Z_cam] + [cx]
       [0  fy]   [Y_cam/Z_cam]   [cy]
   ```

### Single Matrix Form (Traditional)

The traditional projection matrix P combines K and [R|t]:
```
P = K * [R | t]  (3×4 matrix)

u_h = P * X_world_h  (homogeneous coordinates)
u = u_h[1:2] / u_h[3]  (perspective division)
```

### Composed Transform Form (Our Implementation)

Our implementation achieves the same result through composition:
```
project = intrinsics ∘ perspective ∘ extrinsics

Where:
- extrinsics: X_cam = R * X_world + t
- perspective: x_norm = [X_cam/Z_cam, Y_cam/Z_cam]
- intrinsics: u = A * x_norm + t  (where K = [A t; 0 1])
```

## Key Design Points

1. **Lazy Composition**: Transforms are composed using `∘` operator at construction
2. **Caching**: The `intrinsics ∘ perspective` transform is cached in CameraModel
3. **Unit Handling**:
   - Units stripped before EuclideanMap (operates on unitless values)
   - Units restored for projection (operates on physical quantities)
   - Final result has pixel units
4. **Type Safety**: Julia compiler can optimize the composed transforms
5. **Extensibility**: Easy to add new projection models via dispatch on `build_projection_transform`

## Performance Characteristics

- **Construction**: O(1) - builds and caches transform once
- **Projection**: O(1) - applies cached transform directly
- **No allocation**: StaticArrays and composed transforms avoid heap allocation
- **Type stability**: All types known at compile time for optimal performance
