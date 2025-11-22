struct Sensor{TR<:PixelCount,TP<:LogicalPitch}
    resolution::Size2{TR}   # in px (logical width/height)
    pitch::Size2{TP}        # in LogicalPitch (e.g., μm/px)

    # Inner constructor to control creation
    function Sensor{TR,TP}(resolution::Size2{TR}, pitch::Size2{TP}) where {TR<:PixelCount,TP<:LogicalPitch}
        new{TR,TP}(resolution, pitch)
    end
end

# Constructors
function Sensor(res::Size2{<:Integer}, pitch::Len)
    pixel_res = Size2(width=res.width * px, height=res.height * px)
    # Convert length to LogicalPitch (length per pixel)
    pixel_pitch = Size2(width=pitch / px, height=pitch / px)
    return Sensor{typeof(pixel_res.width),typeof(pixel_pitch.width)}(pixel_res, pixel_pitch)
end

function Sensor(res::Size2{<:PixelCount}, pitch::Len)
    # Convert length to LogicalPitch (length per pixel)
    pixel_pitch = Size2(width=pitch / px, height=pitch / px)
    return Sensor{typeof(res.width),typeof(pixel_pitch.width)}(res, pixel_pitch)
end

function Sensor(res::Size2{TR}, pitch::Size2{TP}) where {TR<:PixelCount,TP<:LogicalPitch}
    return Sensor{TR,TP}(res, pitch)
end

# Accessors
sensor_size(s::Sensor) = Size2(width=s.resolution.width * s.pitch.width,
    height=s.resolution.height * s.pitch.height)

pixel_density(s::Sensor) = Size2(width=inv(s.pitch.width),
    height=inv(s.pitch.height))

aspect_ratio(s::Sensor) = s.resolution.width / s.resolution.height

@inline Base.Tuple(sensor::Sensor) = (sensor.resolution, sensor.pitch)

const CMOS_SENSORS = ImmutableDict(
    "Sony" => ImmutableDict(
        "IMX249" => Sensor(
            Size2(width=1920px, height=1200px),
            5.86μm
        ),
        "IMX174" => Sensor(
            Size2(width=1920px, height=1200px),
            5.86μm
        ),
        "IMX250" => Sensor(
            Size2(width=2448px, height=2048px),
            3.45μm
        )
    ),
    "OmniVision" => ImmutableDict(
        "OV9281" => Sensor(
            Size2(width=1280px, height=800px),
            3.0μm
        ),
        "OV2311" => Sensor(
            Size2(width=1600px, height=1300px),
            3.0μm
        )
    ),
    "ON Semiconductor" => ImmutableDict(
        "XGS 5000" => Sensor(
            Size2(width=2592px, height=2048px),
            3.2μm
        ),
        "XGS 12000" => Sensor(
            Size2(width=4096px, height=3072px),
            3.2μm
        )
    )
)

# =============================================================================
# Pixel Convention Definitions
# =============================================================================

# Map pixel conventions to their corner offsets (in units of pixels)
# Corner offset is the position of the top-left corner of the image relative to (0,0)
const INTRINSICS_COORDINATE_OFFSET = ImmutableDict(
    # OpenCV/VLFeat: pixel center [0,0] at (0,0) → corner at (-0.5, -0.5)
    :opencv => Vec2(-0.5, -0.5),
    :vlfeat => Vec2(-0.5, -0.5),

    # Makie/Colmap: pixel center [0,0] at (0.5,0.5) → corner at (0,0)
    :makie => Vec2(0.0, 0.0),
    :colmap => Vec2(0.0, 0.0),

    # MATLAB/Julia: pixel center [1,1] at (1,1) → corner at (0.5, 0.5)
    :matlab => Vec2(0.5, 0.5),
    :julia => Vec2(0.5, 0.5)
)

# =============================================================================
# Rect Constructor for Sensor
# =============================================================================

"""
    Rect(sensor::Sensor; image_origin::Symbol=:opencv) -> Rect

Create a Rect representing the sensor bounds using the specified image origin convention.

The Rect represents the image area covered by the sensor pixels, from the top-left corner
to the bottom-right corner. The convention determines where pixel centers are located relative
to integer coordinates.

# Image Origin Conventions

- **`:opencv`** or **`:vlfeat`** (default): Top-left pixel center at (0, 0)
  - Rect origin (top-left corner): (-0.5, -0.5)
  - Rect extent: width × height in pixels
  - Bottom-right corner: (width-0.5, height-0.5)

- **`:makie`** or **`:colmap`**: Top-left pixel center at (0.5, 0.5)
  - Rect origin (top-left corner): (0, 0)
  - Rect extent: width × height in pixels
  - Bottom-right corner: (width, height)

- **`:matlab`** or **`:julia`**: Top-left pixel center at (1, 1)
  - Rect origin (top-left corner): (0.5, 0.5)
  - Rect extent: width × height in pixels
  - Bottom-right corner: (width+0.5, height+0.5)

# Arguments
- `sensor::Sensor`: Sensor with resolution in pixels
- `image_origin::Symbol=:opencv`: Image origin convention

# Returns
Rect with origin at top-left corner and widths equal to sensor resolution

# Examples
```julia
sensor = Sensor(Size2(width=1920px, height=1200px), 5.86μm)

# OpenCV convention (default): corner at (-0.5, -0.5)
rect_opencv = Rect(sensor)
# Rect origin: (-0.5px, -0.5px), widths: (1920px, 1200px)

# Makie convention: corner at (0, 0)
rect_makie = Rect(sensor; image_origin=:makie)
# Rect origin: (0px, 0px), widths: (1920px, 1200px)

# MATLAB convention: corner at (0.5, 0.5)
rect_matlab = Rect(sensor; image_origin=:matlab)
# Rect origin: (0.5px, 0.5px), widths: (1920px, 1200px)

# Use with CartesianIndices for warping
indices = CartesianIndices(round(Int, rect_makie))
output_axes = indices.indices
warped = warp(image, transform, output_axes)
```
"""
function Rect(sensor::Sensor; image_origin::Symbol=:opencv)
    # Get offset from Julia convention (1-based) to target convention
    offset = image_origin_offset(from=:julia, to=image_origin)

    # Create 1-based ranges and shift by offset
    w = ustrip(sensor.resolution.width)
    h = ustrip(sensor.resolution.height)

    # Create intervals in unitless coordinates
    ix_unitless = ClosedInterval((1:w) .+ ustrip(offset[1]); align_corners=false)
    iy_unitless = ClosedInterval((1:h) .+ ustrip(offset[2]); align_corners=false)

    # Apply units to interval endpoints
    unit_val = oneunit(sensor.resolution.width)
    ix = ClosedInterval(leftendpoint(ix_unitless) * unit_val, rightendpoint(ix_unitless) * unit_val)
    iy = ClosedInterval(leftendpoint(iy_unitless) * unit_val, rightendpoint(iy_unitless) * unit_val)

    return Rect((ix, iy))
end

"""
    Rect2(sensor::Sensor; image_origin::Symbol=:opencv) -> Rect2

Convenience alias for `Rect(sensor; image_origin)`.

Returns a `Rect2` (alias for `HyperRectangle{2}`) representing the sensor bounds
using the specified image origin convention.

See [`Rect(::Sensor)`](@ref) for full documentation of image origin conventions and usage.
"""
Rect2(sensor::Sensor; image_origin::Symbol=:opencv) = Rect(sensor; image_origin=image_origin)

# =============================================================================
# Pixel Center Ranges
# =============================================================================

"""
    pixel_centers(sensor::Sensor; image_origin::Symbol=:opencv) -> Tuple{UnitRange, UnitRange}

Return unitless ranges of pixel center coordinates for the sensor using the specified image origin convention.

This function leverages `Rect(sensor; image_origin)` and adjusts the extrema to represent the
actual coordinate range of pixel centers (from first pixel center to last pixel center).

# Arguments
- `sensor::Sensor`: Sensor with resolution in pixels
- `image_origin::Symbol=:opencv`: Image origin convention

# Returns
A tuple `(x_range, y_range)` of unitless `UnitRange` representing pixel center coordinates.

# Examples
```julia
sensor = Sensor(Size2(width=1920px, height=1200px), 5.86μm)

# OpenCV convention: pixel centers from (0, 0) to (1919, 1199)
x_range, y_range = pixel_centers(sensor)  # 0:1919, 0:1199

# Julia convention: pixel centers from (1, 1) to (1920, 1200)
x_range, y_range = pixel_centers(sensor; image_origin=:julia)  # 1:1920, 1:1200

# Makie convention: pixel centers from (0.5, 0.5) to (1919.5, 1199.5)
x_range, y_range = pixel_centers(sensor; image_origin=:makie)  # 0.5:1919.5, 0.5:1199.5

# Use with warp
x_range, y_range = pixel_centers(sensor; image_origin=:julia)
warped = warp(image, transform, reverse((x_range, y_range)))
```
"""
function pixel_centers(sensor::Sensor; image_origin::Symbol=:opencv)
    # Get the rect bounds (corner to corner)
    rect = Rect(sensor; image_origin=image_origin)

    # Adjust extrema: add 0.5 to top_left (first pixel center), subtract 0.5 from bottom_right (last pixel center)
    top_left, bottom_right = extrema(rect)
    offset = 0.5 .* (oneunit(top_left[1]), oneunit(top_left[2]))
    top_left_center = top_left .+ offset
    bottom_right_center = bottom_right .- offset

    # Strip units and create UnitRange
    x_start = ustrip(top_left_center[1])
    x_end = ustrip(bottom_right_center[1])
    y_start = ustrip(top_left_center[2])
    y_end = ustrip(bottom_right_center[2])

    # Convert to Int when endpoints are exactly integers, otherwise use Float64
    x_range = (isinteger(x_start) && isinteger(x_end)) ? UnitRange(Int(x_start), Int(x_end)) : UnitRange(x_start, x_end)
    y_range = (isinteger(y_start) && isinteger(y_end)) ? UnitRange(Int(y_start), Int(y_end)) : UnitRange(y_start, y_end)

    return (x_range, y_range)
end
