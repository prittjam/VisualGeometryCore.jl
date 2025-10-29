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
