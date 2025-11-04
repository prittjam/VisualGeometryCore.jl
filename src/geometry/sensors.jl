struct Sensor{TR<:VisualGeometryCore.PixelCount,TP<:VisualGeometryCore.LogicalPitch}
    resolution::Size2{TR}   # in px (logical width/height)
    pitch::Size2{TP}        # in LogicalPitch (e.g., μm/px)

    # Inner constructor to control creation
    function Sensor{TR,TP}(resolution::Size2{TR}, pitch::Size2{TP}) where {TR<:VisualGeometryCore.PixelCount,TP<:VisualGeometryCore.LogicalPitch}
        new{TR,TP}(resolution, pitch)
    end
end

# Constructors
function Sensor(res::Size2{<:Integer}, pitch::VisualGeometryCore.Len)
    pixel_res = Size2(width=res.width * px, height=res.height * px)
    # Convert length to LogicalPitch (length per pixel)
    pixel_pitch = Size2(width=pitch / px, height=pitch / px)
    return Sensor{typeof(pixel_res.width),typeof(pixel_pitch.width)}(pixel_res, pixel_pitch)
end

function Sensor(res::Size2{<:VisualGeometryCore.PixelCount}, pitch::VisualGeometryCore.Len)
    # Convert length to LogicalPitch (length per pixel)
    pixel_pitch = Size2(width=pitch / px, height=pitch / px)
    return Sensor{typeof(res.width),typeof(pixel_pitch.width)}(res, pixel_pitch)
end

function Sensor(res::Size2{TR}, pitch::Size2{TP}) where {TR<:VisualGeometryCore.PixelCount,TP<:VisualGeometryCore.LogicalPitch}
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
    # Look up corner offset for this convention
    if !haskey(INTRINSICS_COORDINATE_OFFSET, image_origin)
        valid = join(sort(collect(keys(INTRINSICS_COORDINATE_OFFSET))), ", :")
        error("Unknown image origin :$image_origin. Valid options: :$valid")
    end

    offset_vec = INTRINSICS_COORDINATE_OFFSET[image_origin]

    # Get resolution widths (has units)
    widths = Vec2(sensor.resolution.width, sensor.resolution.height)

    # Create corner offset with proper units
    corner_offset = Point2(offset_vec[1] * oneunit(widths[1]), offset_vec[2] * oneunit(widths[2]))

    return Rect(corner_offset, widths)
end

"""
    Rect2(sensor::Sensor; image_origin::Symbol=:opencv) -> Rect2

Convenience alias for `Rect(sensor; image_origin)`.

Returns a `Rect2` (alias for `HyperRectangle{2}`) representing the sensor bounds
using the specified image origin convention.

See [`Rect(::Sensor)`](@ref) for full documentation of image origin conventions and usage.
"""
Rect2(sensor::Sensor; image_origin::Symbol=:opencv) = Rect(sensor; image_origin=image_origin)
