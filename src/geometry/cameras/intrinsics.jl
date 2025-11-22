# ==============================================================================
# Camera Intrinsics - Calibration matrices and intrinsics models
# ==============================================================================

abstract type AbstractIntrinsics end

# Define CameraCalibrationMatrix as a 3×3 static matrix type
@smatrix_wrapper CameraCalibrationMatrix 3 3

"""
    CameraCalibrationMatrix(f::PixelWidth, pp::AbstractVector{<:PixelWidth}) -> CameraCalibrationMatrix{Float64}
    CameraCalibrationMatrix(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth}) where T<:PixelWidth -> CameraCalibrationMatrix{Float64}

Construct camera intrinsics matrix K from focal length(s) and principal point in pixel units.

Units are enforced at construction but stored internally as unitless Float64 for performance.

# Arguments
- `f`: Focal length in pixels (PixelWidth) - either scalar for isotropic or tuple `(fx, fy)` for anisotropic
- `pp`: Principal point in pixels as 2D vector [cx, cy] (PixelWidth elements)

# Returns
3×3 unitless Float64 camera intrinsics matrix in column-major order:
```
[fx  0  cx]
[ 0 fy  cy]
[ 0  0   1]
```

# Examples
```julia
# Isotropic focal length (fx = fy)
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])

# Anisotropic focal length (fx ≠ fy)
K = CameraCalibrationMatrix((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
function CameraCalibrationMatrix(f::PixelWidth, pp::AbstractVector{<:PixelWidth})
    # Strip units and store as Float64
    f_val = ustrip(Float64, px, f)
    pp_val = ustrip.(Float64, px, pp)
    K = SMatrix{3,3,Float64}(f_val, 0.0, 0.0,
                              0.0, f_val, 0.0,
                              pp_val[1], pp_val[2], 1.0)
    return CameraCalibrationMatrix{Float64}(Tuple(K))
end

function CameraCalibrationMatrix(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth}) where T<:PixelWidth
    # Strip units and store as Float64
    fx_val = ustrip(Float64, px, f[1])
    fy_val = ustrip(Float64, px, f[2])
    pp_val = ustrip.(Float64, px, pp)
    K = SMatrix{3,3,Float64}(fx_val, 0.0, 0.0,
                              0.0, fy_val, 0.0,
                              pp_val[1], pp_val[2], 1.0)
    return CameraCalibrationMatrix{Float64}(Tuple(K))
end

"""
    CameraCalibrationMatrix(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth}) -> CameraCalibrationMatrix{Float64}

Construct camera intrinsics matrix K from physical focal length, pixel pitch, and principal point.

Units are enforced at construction but stored internally as unitless Float64 for performance.

This constructor converts from metric units to pixel units using the factorization:
```
K = [1  0  cx] [1/px   0   0] [f  0  0]
    [0  1  cy] [0   1/py  0] [0  f  0]
    [0  0   1] [0    0    1] [0  0  1]
```

where f is in mm, px/py are pixel pitch in mm/px, and (cx,cy) are in pixels.

# Arguments
- `f::Len`: Focal length in physical units (e.g., mm)
- `pitch::Size2{<:LogicalPitch}`: Pixel pitch (e.g., `Size2(width=5.86μm/px, height=5.86μm/px)`)
- `pp::AbstractVector{<:PixelWidth}`: Principal point in pixels [cx, cy]

# Returns
3×3 unitless Float64 camera intrinsics matrix with focal length in pixels:
```
[αx  0  cx]
[ 0  αy cy]   where αx = f/px, αy = f/py
[ 0  0   1]
```

# Examples
```julia
# From sensor datasheet and lens specification
f = 16.0mm                                    # Lens focal length
pitch = Size2(width=5.86μm/px, height=5.86μm/px)  # Sensor pixel pitch
pp = [320.0px, 240.0px]                       # Principal point (image center)
K = CameraCalibrationMatrix(f, pitch, pp)
# K[1,1] ≈ 2730.38 (focal length in pixels)
```
"""
function CameraCalibrationMatrix(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth})
    # Convert focal length from physical units to pixels
    # αx = f / pitch.width, αy = f / pitch.height
    # The division f/pitch gives dimension 𝐍 but mixed units, so convert to px
    fx_px = uconvert(px, f / pitch.width)
    fy_px = uconvert(px, f / pitch.height)

    # Now construct matrix using the pixel-based constructor (which strips units)
    # Need to convert to tuple for the anisotropic case
    return CameraCalibrationMatrix((fx_px, fy_px), pp)
end

"""
    ImageCalibrationMatrix(f::Len, K::CameraCalibrationMatrix) -> SMatrix{3,3}

Construct image calibration matrix by normalizing camera calibration matrix by focal length.

This divides the dimensionless camera matrix K by the focal length to obtain a matrix
with dimensions of inverse length (𝐋^-1), suitable for metric ray back-projection.

# Arguments
- `f::Len`: Focal length in physical units (e.g., mm)
- `K::CameraCalibrationMatrix`: Dimensionless camera calibration matrix

# Returns
3×3 matrix with inverse length units (e.g., mm^-1)

# Example
```julia
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
f = 16.0mm
K_img = ImageCalibrationMatrix(f, K)  # Units: mm^-1
```
"""
ImageCalibrationMatrix(f::Len, K::CameraCalibrationMatrix) = K / ustrip(f)

"""
    ImageCalibrationMatrix(f::Len, pp::AbstractVector{<:PixelWidth}, pitch::Size2{<:LogicalPitch}) -> SMatrix{3,3}

Construct image calibration matrix from focal length, principal point, and pixel pitch.

This is a convenience constructor that:
1. Builds the camera calibration matrix K from f, pp, and pitch
2. Normalizes by focal length to get the image calibration matrix

# Arguments
- `f::Len`: Focal length in physical units (e.g., mm)
- `pp::AbstractVector{<:PixelWidth}`: Principal point in pixels [cx, cy]
- `pitch::Size2{<:LogicalPitch}`: Pixel pitch (e.g., `Size2(width=5.86μm/px, height=5.86μm/px)`)

# Returns
3×3 matrix with inverse length units (e.g., mm^-1)

# Example
```julia
f = 16.0mm
pp = [320.0px, 240.0px]
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
K_img = ImageCalibrationMatrix(f, pp, pitch)  # Units: mm^-1
```
"""
ImageCalibrationMatrix(f::Len, pp::AbstractVector{<:PixelWidth}, pitch::Size2{<:LogicalPitch}) =
    CameraCalibrationMatrix(f, pitch, pp) / ustrip(f)

# ============================================================================
# Intrinsics Models: LogicalIntrinsics and PhysicalIntrinsics
# ============================================================================

"""
    LogicalIntrinsics

Camera intrinsics containing only the calibration matrix K (dimensionless).

Units are enforced at construction but stored internally as unitless Float64 for performance.

Used when physical parameters cannot be disentangled from K, such as when K is
recovered from autocalibration, Zhang's method, or other calibration techniques
that don't provide metric information about the sensor.

# Fields
- `K::CameraCalibrationMatrix{Float64}`: 3×3 calibration matrix (unitless Float64)
- `K_inv::SMatrix{3,3,Float64}`: Precomputed inverse of K for efficient backprojection

# Examples
```julia
# From autocalibration
K = CameraCalibrationMatrix(800.0px, [320.0px, 240.0px])
intrinsics = LogicalIntrinsics(K)
```
"""
struct LogicalIntrinsics <: AbstractIntrinsics
    K::CameraCalibrationMatrix{Float64}
    K_inv::SMatrix{3,3,Float64}

    # Inner constructor to compute K_inv automatically
    function LogicalIntrinsics(K::CameraCalibrationMatrix{Float64})
        K_inv = inv(SMatrix{3,3,Float64}(K))
        new(K, K_inv)
    end
end

"""
    PhysicalIntrinsics

Camera intrinsics containing the calibration matrix K plus physical parameters.

Units are enforced at construction but stored internally as unitless Float64 for performance:
- Focal length stored in mm
- Pixel pitch stored in μm/px
- Calibration matrix K stored as unitless Float64

Used when the focal length and pixel pitch are known from sensor datasheets,
allowing metric operations like 3D point reconstruction from depth.

# Fields
- `K::CameraCalibrationMatrix{Float64}`: Calibration matrix (unitless Float64)
- `K_inv::SMatrix{3,3,Float64}`: Precomputed inverse of K for efficient backprojection
- `f::Float64`: Focal length in mm (unitless, canonical storage)
- `pitch::Size2{LogicalPitch{Float64}}`: Pixel pitch (stored with units for now)

# Accessors
- `focal_length_mm(intrinsics)`: Get focal length with mm units attached

# Examples
```julia
# From sensor datasheet
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
intrinsics = PhysicalIntrinsics(f, pitch, pp)
```
"""
struct PhysicalIntrinsics <: AbstractIntrinsics
    K::CameraCalibrationMatrix{Float64}
    K_inv::SMatrix{3,3,Float64}
    f::Float64  # Stored unitless in mm
    pitch::Size2{LogicalPitch{Float64}}  # TODO: could also store unitless

    # Inner constructor to compute K_inv automatically
    function PhysicalIntrinsics(K::CameraCalibrationMatrix{Float64}, f::Len, pitch::Size2{<:LogicalPitch})
        K_inv = inv(SMatrix{3,3,Float64}(K))
        f_mm = ustrip(Float64, mm, f)
        new(K, K_inv, f_mm, pitch)
    end
end

# Accessor to reconstruct with units
"""
    focal_length_mm(intrinsics::PhysicalIntrinsics) -> Quantity{Float64, mm}

Get focal length with mm units attached.
"""
focal_length_mm(intrinsics::PhysicalIntrinsics) = intrinsics.f * mm

"""
    focal_length(intrinsics::PhysicalIntrinsics) -> Float64

Get focal length from PhysicalIntrinsics (unitless, in mm).

Returns the physical focal length as a unitless Float64 value in millimeters.

# Example
```julia
intrinsics = PhysicalIntrinsics(...)
f = focal_length(intrinsics)  # Returns Float64 (mm)
```
"""
focal_length(intrinsics::PhysicalIntrinsics) = intrinsics.f

"""
    focal_length(intrinsics::LogicalIntrinsics) -> Tuple{Float64, Float64}

Get focal lengths (fx, fy) from LogicalIntrinsics K matrix (unitless, in pixels).

Returns a tuple (fx, fy) extracted from the calibration matrix K.
For square pixels, fx ≈ fy.

# Example
```julia
intrinsics = LogicalIntrinsics(K)
fx, fy = focal_length(intrinsics)  # Returns (Float64, Float64) in pixels
```
"""
focal_length(intrinsics::LogicalIntrinsics) = (intrinsics.K[1,1], intrinsics.K[2,2])

# ============================================================================
# Convenience Constructors for Intrinsics
# ============================================================================

"""
    LogicalIntrinsics(f::PixelWidth, pp::AbstractVector{<:PixelWidth}) -> LogicalIntrinsics

Convenience constructor for LogicalIntrinsics from focal length and principal point (both in pixels).

This constructor builds the calibration matrix K internally. Use this when you have
calibrated intrinsics but no physical sensor parameters.

# Example
```julia
logical = LogicalIntrinsics(800.0px, [320.0px, 240.0px])
```
"""
function LogicalIntrinsics(f::PixelWidth, pp::AbstractVector{<:PixelWidth})
    K = CameraCalibrationMatrix(f, pp)
    return LogicalIntrinsics(K)
end

"""
    LogicalIntrinsics(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth}) where T<:PixelWidth -> LogicalIntrinsics

Convenience constructor for LogicalIntrinsics with anisotropic focal lengths (fx ≠ fy).

# Example
```julia
# Different focal lengths in x and y
logical = LogicalIntrinsics((800.0px, 805.0px), [320.0px, 240.0px])
```
"""
function LogicalIntrinsics(f::Tuple{T,T}, pp::AbstractVector{<:PixelWidth}) where T<:PixelWidth
    K = CameraCalibrationMatrix(f, pp)
    return LogicalIntrinsics(K)
end

"""
    PhysicalIntrinsics(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth}) -> PhysicalIntrinsics

Convenience constructor for PhysicalIntrinsics from focal length, pixel pitch, and principal point.

This constructor builds the calibration matrix K internally using the intrinsic matrix factorization.

# Example
```julia
f = 16.0mm
pitch = Size2(width=5.86μm/px, height=5.86μm/px)
pp = [320.0px, 240.0px]
intrinsics = PhysicalIntrinsics(f, pitch, pp)
```
"""
function PhysicalIntrinsics(f::Len, pitch::Size2{<:LogicalPitch}, pp::AbstractVector{<:PixelWidth})
    K = CameraCalibrationMatrix(f, pitch, pp)
    return PhysicalIntrinsics(K, f, pitch)
end
