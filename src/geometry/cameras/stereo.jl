# ==============================================================================
# StereoRig - Stereo camera system with epipolar geometry
# ==============================================================================

struct StereoRig{U <: Camera,
                 V <: Camera,
                 Rt <: EuclideanMap}
    source::U
    target::V

    extrinsics::Rt
    pose::Rt

    function StereoRig(source, target, extrinsics)
        pose = inv(extrinsics)
        new{typeof(source), typeof(target), typeof(extrinsics)}(source, target, extrinsics, pose)
    end
end

StereoRig(source::Camera, target::Camera) =
    StereoRig(source, target, target.extrinsics ∘ pose(source))

"""
    epipolarmap(source::Camera, target::Camera, extrinsics)

Create epipolar mapping from source camera to target camera.

Note: This function assumes points lie on a specific depth surface. For proper stereo
correspondence, depth must be known or estimated through matching.

# Arguments
- `source::Camera`: Source camera
- `target::Camera`: Target camera
- `extrinsics::EuclideanMap`: Relative transform from source to target

# Returns
- Function mapping pixel coordinates from source to target camera
"""
function epipolarmap(source::Camera, target::Camera, extrinsics)
    # WARNING: This implementation assumes a specific depth for the 3D point.
    # Without depth, we can only backproject to a ray, not a 3D point.
    # This is a simplified version that assumes unit depth.
    # For proper stereo correspondence, use triangulation with matched features.
    function map_pixel(u)
        # Backproject to normalized ray (no depth)
        ray_source = backproject(source.model, u)
        # Assume depth=1 in source camera frame
        X_source = ray_source  # SVector{3} with unit depth
        # Transform to target camera frame (extrinsics stores unitless internally)
        X_target = extrinsics(X_source)
        # Project to target image
        return project(target.model, X_target)
    end
    return map_pixel
end

epipolarmap(rig::StereoRig) = epipolarmap(rig.source, rig.target, rig.extrinsics)

# ============================================================================
# StereoRig Property Accessors
# ============================================================================

function Base.getproperty(c::StereoRig, s::Symbol)
    if s === :orientation
        return pose(c).R
    elseif s === :eye_position
        return Meshes.Point(Tuple(pose(c).t))
    elseif s === :forward
        return RotMatrix(pose(c).R)[:,3]
    elseif s === :up
        return -RotMatrix(pose(c).R)[:,2]
    else
        return getfield(c, s)
    end
end
