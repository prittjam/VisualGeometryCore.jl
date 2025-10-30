#!/usr/bin/env julia
# P3P Camera Pose Estimation Example

using VisualGeometryCore
using VisualGeometryCore: Camera  # Explicit import to avoid Makie.Camera conflict
using JSON3, LinearAlgebra, Random, Rotations, Statistics
using GeometryBasics: Vec3, Point3, origin
using StaticArrays: SVector
using Unitful: ustrip, mm, °
using Printf: @sprintf
using Colors: Gray
using FileIO: load
using ImageTransformations: warp
import Makie.SpecApi as S
using GLMakie: plot, resize_to_layout!
using Makie: Fixed

println("="^70)
println("P3P Camera Pose Estimation Example")
println("="^70)

# Load blobs - these are in pixel coordinates on the pattern (logical coordinates)
data_path = joinpath(@__DIR__, "..", "..", "test", "data", "blob_pattern_eBd.json")
json_data = JSON3.read(read(data_path, String))
blobs = [JSON3.read(JSON3.write(blob), IsoBlob) for blob in json_data.blobs]

# For this synthetic example, treat pixel coordinates as mm world coordinates
# (i.e., pattern where 1px = 1mm spacing). Add z=0 for planar pattern.
# All coordinates are unitless Float64 (assumed to be in mm)
blobs = origin.(blobs)
X = vcat.(ustrip.(blobs), Ref(0.0))

println("\nLoaded $(length(X)) blob centers (pattern at z=0)")

# Setup camera
sensor = CMOS_SENSORS["Sony"]["IMX174"]
f = focal_length(40.0°, sensor; dimension=:horizontal)
pp = sensor.resolution ./ 2
model = CameraModel(f, sensor.pitch, pp)

println("Camera: fx=$(round(ustrip(model.intrinsics.K[1,1]), digits=1))px, f=$(round(typeof(1.0mm), f, digits=1))")

Random.seed!(42)

# Ground truth pose: camera at (400mm, 300mm, 1000mm) looking down at pattern
camera_position = Point3(400., 300., 1000.)
extrinsics = lookat(camera_position, Point3(400., 300., 0.), Vec3(0.0, -1.0, 0.0))
camera = Camera(model, extrinsics)
u = project.(Ref(camera), X)

# Sample 3 points and project to image
sampled_idx = randperm(length(X))[1:3]
X3 = X[sampled_idx]
u3 = u[sampled_idx]

println("\nSampled points (unitless, in mm and px):")
println.(["  $i: X=$(X3[i]), u=($(@sprintf("%.1f", ustrip(u3[i][1]))), $(@sprintf("%.1f", ustrip(u3[i][2]))))" for i in 1:3])

# Backproject and solve P3P (backproject expects unitless Float64)
rays = backproject.(Ref(model), ustrip.(u3))
# P3P expects unitless coordinates (X3 already unitless in mm)
Rs, ts = p3p(rays, X3)

println("\nP3P found $(length(Rs)) solution(s)")

# Validate solutions
R_gt = Rotations.RotMatrix(extrinsics.R)
t_gt = extrinsics.t

recovered_cameras = Camera.(Ref(model), EuclideanMap.(Rotations.RotMatrix.(Rs), ts))
#X_sv = SVector.(getindex.(X_sampled, 1), getindex.(X_sampled, 2), getindex.(X_sampled, 3))
u_proj = [project.(Ref(cam), X) for cam in recovered_cameras]
errors_sq = [sum.(abs2, ustrip.(u2 .- u)) for u2 in u_proj]
rms_error = sqrt.(mean.(errors_sq))
max_error = sqrt.(maximum.(errors_sq))

min_rms_value, best_rms_idx = findmin(rms_error)
min_error_value, best_error_idx = findmin(max_error)

println("\nValidation:")
println("  Best RMS error: $(min_rms_value) px" * (min_rms_value < 1e-10 ? " ✓" : ""))
println("  Best max error: $(min_error_value) px" * (min_error_value < 1e-10 ? " ✓" : ""))
if min_rms_value < 1e-10 && min_error_value < 1e-10
    println("  ✓ Valid solution (zero reprojection error)")
end

# Load and render the blob board image
println("\n" * "="^70)
println("Board Rendering with Homography")
println("="^70)

# Load the blob board PNG image
image_path = joinpath(@__DIR__, "..", "..", "test", "data", "blob_pattern_eBd.png")
board_image = load(image_path)

println("\nLoaded board image: $(size(board_image))")

# Warp the board as seen from camera using HomographyTransform
H_transform = HomographyTransform(camera)
output_height = Int(ceil(ustrip(sensor.resolution.height)))
output_width = Int(ceil(ustrip(sensor.resolution.width)))
output_axes = (1:output_height, 1:output_width)
camera_view = warp(board_image, H_transform, output_axes)

println("Rendered camera view: $(size(camera_view)) (sensor resolution: $(sensor.resolution.width)×$(sensor.resolution.height))")

# Display the results using Spec API
lscene1 = Spec.Imshow(board_image)
lscene2 = Spec.Imshow(camera_view)

# Create grid layout
layout = S.GridLayout([lscene1 lscene2])

println("\nDisplaying board and camera view...")
fig = plot(layout)
resize_to_layout!(fig.figure)


println("\n" * "="^70)
println("Example Complete")
println("="^70)
