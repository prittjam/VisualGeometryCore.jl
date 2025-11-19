module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob, INTRINSICS_COORDINATE_OFFSET, Ellipse, Cspond, pose, canonical_basis, center
using ..VisualGeometryCore: Camera, CameraModel, PhysicalIntrinsics, unproject
using GeometryBasics
using GeometryBasics: Circle, Rect, Point2, Point3, Point3f, Vec2, Vec3f, TriangleFace, origin, coordinates
using StaticArrays
using StructArrays
using Unitful: ustrip
using LinearAlgebra: norm, inv
import Makie.SpecApi as MakieSpec

"""
    Imshow(img; width=nothing, height=nothing, scenekw=(camera=campixel!,), imagekw=(interpolate=false, rasterize=true), image_origin=:julia) -> LScene
    Imshow(x_bounds, y_bounds, img; kwargs...) -> LScene

Create an LScene spec for displaying an image with proper orientation (y-axis flipped).

Bounds can be specified as positional arguments (matching Makie API conventions):
- Ranges: `Imshow(1:100, 1:200, img)`
- Tuples: `Imshow((1, 100), (1, 200), img)`
- Intervals: `Imshow(1..100, 1..200, img)` (if IntervalSets is available)

# Arguments
- `img`: Image matrix (AbstractMatrix{<:Colorant})
- `x_bounds`: X-axis bounds (range, tuple, interval, or any type Makie.Image accepts)
- `y_bounds`: Y-axis bounds (range, tuple, interval, or any type Makie.Image accepts)

# Keyword Arguments
- `width::Union{Nothing,Int}=nothing`: LScene width. If nothing, uses image width.
- `height::Union{Nothing,Int}=nothing`: LScene height. If nothing, uses image height.
- `scenekw::NamedTuple=(camera=campixel!,)`: Scene keyword arguments (e.g., camera, backgroundcolor, limits).
  Defaults to pixel-aligned camera. Users can override by passing a complete NamedTuple.
- `imagekw::NamedTuple=(interpolate=false, rasterize=true)`: Image keyword arguments passed to Makie.Image
  (e.g., interpolate, rasterize, colormap, colorrange, alpha, visible).
  Defaults to non-interpolated, rasterized images for clean rendering.

Returns an LScene spec that can be used with SpecApi for composable plotting.

# Examples
```julia
import VisualGeometryCore.Spec as S

# Basic usage - image fills scene
lscene = S.Imshow(img)

# With custom scene background
lscene = S.Imshow(img; scenekw=(backgroundcolor=:lightblue,))

# With interpolation enabled
lscene = S.Imshow(img; imagekw=(interpolate=true, rasterize=true))

# With bounds as tuples
lscene = S.Imshow((50, 150), (50, 150), img; width=200, height=200)

# Custom coordinate system (normalized [0,1])
lscene = S.Imshow((0.2, 0.8), (0.2, 0.8), img;
                  width=200, height=200,
                  scenekw=(limits=(0, 1, 0, 1),))

# Add overlays using SpecApi
lscene = S.Imshow(img)
push!(lscene.plots, S.Scatter(x, y; color=:red))
append!(lscene.plots, S.Blobs(blobs))
plot(MakieSpec.GridLayout([lscene]))
```
"""
# Dispatch: bounds (ranges, tuples, intervals) - pass directly to Spec.Image
function Imshow(x_bounds, y_bounds, img;
                width=nothing, height=nothing,
                scenekw=(camera=campixel!,),
                imagekw=(interpolate=false, rasterize=true),
                plots=PlotSpec[])
    # Scene dimensions: use provided or image dimensions (width, height)
    img_width, img_height = reverse(size(img))
    scene_width = something(width, img_width)
    scene_height = something(height, img_height)

    # Pass bounds and imagekw directly to Spec.Image
    image_spec = MakieSpec.Image(x_bounds, y_bounds, transpose(img); imagekw...)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        scene.transformation.scale[] = (1.0, -1.0, 1.0)
        scene.transformation.translation[] = (0.0, Float64(scene_height), 0.0)
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots=vcat([image_spec], plots),
        show_axis=false,
        width=Makie.Fixed(scene_width),
        height=Makie.Fixed(scene_height),
        tellwidth=true,
        tellheight=true,
        scenekw=scenekw
    )

    push!(lscene.then_funcs, flip_y_axis)
    return lscene
end

# Dispatch: no bounds (image fills scene) - compute bounds from image_origin
function Imshow(img;
                width=nothing, height=nothing, scale=1.0,
                scenekw=(camera=campixel!,),
                imagekw=(interpolate=false, rasterize=true),
                image_origin::Symbol=:julia,
                plots=PlotSpec[])
    # Scene dimensions: use provided or image dimensions (width, height)
    img_width, img_height = reverse(size(img))
    scene_width = something(width, img_width)
    scene_height = something(height, img_height)

    # Get corner offset for the specified convention
    offset = INTRINSICS_COORDINATE_OFFSET[image_origin]

    # Compute bounds: offset to offset + (width, height)
    # For julia: (0.5, 0.5) to (0.5 + width, 0.5 + height)
    # For opencv: (-0.5, -0.5) to (-0.5 + width, -0.5 + height)
    # For makie: (0.0, 0.0) to (width, height)
    x_bounds = (offset[1], offset[1] + img_width)
    y_bounds = (offset[2], offset[2] + img_height)

    # Create image spec with computed bounds and pass imagekw
    image_spec = MakieSpec.Image(x_bounds, y_bounds, transpose(img); imagekw...)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        # Flip y-axis by scaling y by -1 and translating
        # For a scene of height H, point at y=0 maps to y=H, point at y=H maps to y=0
        # Transformation: y' = -y + H (scale by -1, translate by H)
        scene.transformation.scale[] = (scale, -scale, 1.0)
        scene.transformation.translation[] = (0.0, scale*Float64(scene_height), 0.0)

        # Update camera to ensure it doesn't clip
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots=vcat([image_spec], plots),
        show_axis=false,
        width=Makie.Fixed(scale*scene_width),
        height=Makie.Fixed(scale*scene_height),
        tellwidth=true,
        tellheight=true,
        scenekw=scenekw
    )

    # Add callback to flip y-axis after scene is created
    push!(lscene.then_funcs, flip_y_axis)

    return lscene
end

"""
    imshow_image(img; image_origin=:julia) -> Image

Create a Makie Image spec with proper y-axis orientation, without the LScene wrapper.
This is useful for adding images to existing LScenes or composing with other plots.

# Arguments
- `img`: Image matrix
- `image_origin::Symbol=:julia`: Coordinate convention (:julia, :opencv, :makie, etc.)

# Returns
- A Makie Image PlotSpec that can be added to an LScene

# Example
```julia
using GLMakie
import VisualGeometryCore.Spec as S

f = Figure()
ls = LScene(f[1,1], scenekw=(camera=campixel!,))

# Add image to the LScene
img_spec = S.imshow_image(img; image_origin=:julia)
plot!(ls.scene, img_spec)

# The scene needs y-flip transform
ls.scene.transformation.scale[] = (1.0, -1.0, 1.0)
ls.scene.transformation.translation[] = (0.0, Float64(size(img, 1)), 0.0)
```
"""
function imshow_image(img;
                      image_origin::Symbol=:julia,
                      imagekw=(interpolate=false, rasterize=true))
    img_width, img_height = reverse(size(img))

    # Get corner offset for the specified convention
    offset = INTRINSICS_COORDINATE_OFFSET[image_origin]

    # Compute bounds
    x_bounds = (offset[1], offset[1] + img_width)
    y_bounds = (offset[2], offset[2] + img_height)

    # Create image spec with computed bounds
    return MakieSpec.Image(x_bounds, y_bounds, transpose(img); imagekw...)
end

"""
    Imshow(position::GridPosition, args...; kwargs...)

Create an image display at a specific grid position.

Note: Since Makie's SpecApi doesn't support positional GridPosition arguments for LScene,
this function creates the LScene and wraps it in a GridPosition for use with GridLayout.

# Arguments
- `position::GridPosition`: Grid position (e.g., `f[1, 1]`)
- `args...`: Arguments passed to Imshow (img, or x_bounds, y_bounds, img)
- `kwargs...`: Keyword arguments passed to Imshow

# Returns
- A GridPosition wrapping the LScene, suitable for use in GridLayout

# Examples
```julia
using GLMakie
import Makie.SpecApi as Spec

# Create a GridLayout
layout = Spec.GridLayout([
    Imshow(Spec.GridPosition(1, 1), img1),
    Imshow(Spec.GridPosition(1, 2), img2)
])
```
"""
function Imshow(position::Makie.GridPosition, args...; kwargs...)
    # Create the LScene without layout
    lscene = Imshow(args...; kwargs...)
    # Wrap it in the GridPosition
    row_span = position.span.rows
    col_span = position.span.cols
    return MakieSpec.GridPosition(row_span, col_span, lscene)
end

"""
    cspond!(src_lscene, tgt_lscene, csponds; marker_size=10.0, strokewidth=2.0, colormap=:plasma)

Add correspondence visualizations to existing LScenes. Mutates the LScenes by adding colored circle overlays
showing which points correspond between the two views.

# Arguments
- `src_lscene`: Source LScene (from Spec.Imshow) - will be mutated
- `tgt_lscene`: Target LScene (from Spec.Imshow) - will be mutated
- `csponds`: StructArray of Cspond with source and target points

# Keyword Arguments
- `marker_size=10.0`: Radius of circles marking correspondences
- `strokewidth=2.0`: Width of circle outlines
- `colormap=:plasma`: Colormap for matching correspondences

# Returns
- Nothing (mutates the LScenes in place)

# Examples
```julia
import VisualGeometryCore.Spec as S

# Create image displays
lscene1 = S.Imshow(board_image; scale=0.5)
lscene2 = S.Imshow(camera_image; scale=0.5)

# Add correspondence overlays (mutates lscenes)
S.cspond!(lscene1, lscene2, csponds)

# Create layout
layout = S.GridLayout([(1,1) => lscene1, (1,2) => lscene2])
fig, ax, pl = plot(layout)
```
"""
# Tuple of two primitives - create Cspond and delegate
function cspond!(src_lscene, tgt_lscene,
                 primitives::Tuple{S, T};
                 kwargs...) where {S, T}
    # Use tuple constructor for Cspond
    return cspond!(src_lscene, tgt_lscene, Cspond(primitives); kwargs...)
end

# Single correspondence - wrap in vector and delegate
function cspond!(src_lscene, tgt_lscene,
                 cspond::Cspond;
                 kwargs...)
    # Wrap single cspond in vectors and create StructArray
    csponds = StructArrays.StructArray{typeof(cspond)}(([cspond.source], [cspond.target]))
    return cspond!(src_lscene, tgt_lscene, csponds; kwargs...)
end

# Tuple of correspondences - convert to StructArray and delegate
function cspond!(src_lscene, tgt_lscene,
                 csponds_tuple::Tuple{Vararg{Cspond}};
                 kwargs...)
    # Extract sources and targets
    sources = [cs.source for cs in csponds_tuple]
    targets = [cs.target for cs in csponds_tuple]

    # Create StructArray with concrete eltype from first element
    T = typeof(first(csponds_tuple))
    csponds = StructArrays.StructArray{T}((sources, targets))
    return cspond!(src_lscene, tgt_lscene, csponds; kwargs...)
end

# Point correspondences - draw marker circles
function cspond!(src_lscene, tgt_lscene,
                 csponds::StructArrays.StructArray{<:Cspond{<:StaticVector, <:StaticVector}};
                 marker_size=10.0,
                 strokewidth=2.0,
                 colormap=:plasma,
                 kwargs...)

    # Extract source and target points
    src_points = csponds.source
    tgt_points = csponds.target

    # Convert to Point2 (handles both Point2 and Point3)
    src_pts = [Point2(pt[1], pt[2]) for pt in src_points]
    tgt_pts = [Point2(pt[1], pt[2]) for pt in tgt_points]

    # Create circles around points for visualization
    src_circles = [Circle(pt, marker_size) for pt in src_pts]
    tgt_circles = [Circle(pt, marker_size) for pt in tgt_pts]

    # Add color-coded circles to LScenes
    # Set defaults, but kwargs will override if provided
    push!(src_lscene.plots, MakieSpec.Poly(src_circles,
                                           color=:transparent,
                                           strokecolor=1:length(src_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth;
                                           kwargs...))

    push!(tgt_lscene.plots, MakieSpec.Poly(tgt_circles,
                                           color=:transparent,
                                           strokecolor=1:length(tgt_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth;
                                           kwargs...))

    return nothing
end

# Blob correspondences - convert to circles and delegate to generic method
function cspond!(src_lscene, tgt_lscene,
                 csponds::StructArrays.StructArray{<:Cspond{<:AbstractBlob, <:AbstractBlob}},
                 sigma_cutoff;
                 kwargs...)

    # Extract source and target blobs
    src_blobs = csponds.source
    tgt_blobs = csponds.target

    # Create circles from blobs (using sigma_cutoff to determine radius)
    src_circles = [Circle(origin(b), sigma_cutoff * b.σ) for b in src_blobs]
    tgt_circles = [Circle(origin(b), sigma_cutoff * b.σ) for b in tgt_blobs]

    # Create circle correspondences and call through to generic method
    circle_csponds = StructArrays.StructArray{Cspond{eltype(src_circles), eltype(tgt_circles)}}((src_circles, tgt_circles))
    return cspond!(src_lscene, tgt_lscene, circle_csponds; kwargs...)
end

# Generic geometric primitive correspondences - works with Circle, Ellipse, etc.
# Fallback for any types that aren't StaticVectors or AbstractBlobs
function cspond!(src_lscene, tgt_lscene,
                 csponds::StructArrays.StructArray{<:Cspond{S, T}};
                 strokewidth=2.0,
                 colormap=:plasma,
                 kwargs...) where {S, T}

    # Extract source and target shapes
    src_shapes = csponds.source
    tgt_shapes = csponds.target

    # Add color-coded shapes directly to LScenes using Poly
    # Poly can handle Circle, Ellipse, Rect, etc.
    # Set defaults, but kwargs will override if provided
    push!(src_lscene.plots, MakieSpec.Poly(src_shapes,
                                           color=:transparent,
                                           strokecolor=1:length(src_shapes),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth;
                                           kwargs...))

    push!(tgt_lscene.plots, MakieSpec.Poly(tgt_shapes,
                                           color=:transparent,
                                           strokecolor=1:length(tgt_shapes),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth;
                                           kwargs...))

    return nothing
end

"""
    poly!(lscene, shape; kwargs...)
    poly!(lscene, shapes::AbstractVector; kwargs...)

Add geometric primitive(s) to an LScene as a Poly plot.

# Arguments
- `lscene`: The LScene to add the plot to
- `shape` or `shapes`: Geometric primitive(s) (Circle, Ellipse, Rect, etc.)
- `kwargs...`: Additional keyword arguments passed to MakieSpec.Poly

# Common Keyword Arguments
- `color=:transparent`: Fill color
- `strokecolor`: Stroke color
- `strokewidth=2.0`: Stroke width

# Examples
```julia
lscene = Spec.Imshow(img)
Spec.poly!(lscene, circle; strokecolor=:red, strokewidth=3.0)
Spec.poly!(lscene, [circle1, circle2]; strokecolor=:blue)
```
"""
function poly!(lscene, shape;
               color=:transparent,
               strokewidth=2.0,
               linestyle=nothing,
               kwargs...)
    # Convert linestyle symbol to Linestyle if provided
    linestyle_arg = isnothing(linestyle) ? NamedTuple() : (linestyle=Makie.Linestyle(linestyle),)
    push!(lscene.plots, MakieSpec.Poly(shape,
                                       color=color,
                                       strokewidth=strokewidth;
                                       linestyle_arg...,
                                       kwargs...))
    return nothing
end

function poly!(lscene, shapes::AbstractVector;
               color=:transparent,
               strokewidth=2.0,
               linestyle=nothing,
               kwargs...)
    # Convert linestyle symbol to Linestyle if provided
    linestyle_arg = isnothing(linestyle) ? NamedTuple() : (linestyle=Makie.Linestyle(linestyle),)
    push!(lscene.plots, MakieSpec.Poly(shapes,
                                       color=color,
                                       strokewidth=strokewidth;
                                       linestyle_arg...,
                                       kwargs...))
    return nothing
end

"""
    cspond_plot(src_image, tgt_image, csponds; kwargs...)
    cspond_plot(src_image, src_shapes, tgt_image, tgt_shapes; kwargs...)

Convenience function to create a side-by-side visualization of correspondences between two images with color-coded geometric overlays.

# Arguments
- `src_image`: Source image (left side)
- `tgt_image`: Target image (right side)
- `csponds`: StructArray of Cspond with source and target points

OR

- `src_image`: Source image (left side)
- `src_shapes`: Vector of geometric shapes (Circle, Ellipse, Rect) in source image
- `tgt_image`: Target image (right side)
- `tgt_shapes`: Vector of geometric shapes in target image (corresponding to src_shapes)

# Keyword Arguments
- `scale=0.5`: Scale factor for image display
- `image_origin=:julia`: Image coordinate convention
- `marker_size=10.0`: Size of markers for point correspondences
- `strokewidth=1.0`: Width of shape outlines
- `colormap=:plasma`: Colormap for corresponding points/shapes
- `strokecolormap=:viridis`: Colormap for shapes (when passing shapes)

# Returns
- `GridLayout`: 1x2 layout with source and target images side-by-side

# Examples
```julia
import VisualGeometryCore.Spec as S

# With Cspond (from P3P, etc.)
extrinsics, csponds = sample_p3p(model, X, sensor_bounds)
layout = S.cspond_plot(board_img, camera_img, csponds)

# With explicit shapes
circles = [Circle(Point2(i*10.0, i*10.0), 5.0) for i in 1:5]
ellipses = [Ellipse(Point2(i*15.0, i*12.0), 6.0, 4.0, π/6) for i in 1:5]
layout = S.cspond_plot(src_img, circles, tgt_img, ellipses)

fig, ax, pl = plot(layout)
```
"""
# Method 1: Accept csponds directly
function cspond_plot(src_image, tgt_image, csponds;
                     scale=0.5,
                     image_origin=:julia,
                     marker_size=10.0,
                     strokewidth=2.0,
                     colormap=:plasma)

    # Extract source and target points from csponds
    # For 3D source points, take only x,y (assume planar scene at z=0)
    src_points = [Point2(cs.source[1], cs.source[2]) for cs in csponds]
    tgt_points = [Point2(cs.target[1], cs.target[2]) for cs in csponds]

    # Create circles around points for visualization
    src_circles = [Circle(pt, marker_size) for pt in src_points]
    tgt_circles = [Circle(pt, marker_size) for pt in tgt_points]

    # Create source image scene with color-coded circles
    src_plots = [MakieSpec.Poly(src_circles,
                                color=:transparent,
                                strokecolor=1:length(src_circles),
                                strokecolormap=colormap,
                                strokewidth=strokewidth)]

    scene1 = Imshow(src_image; scale=scale, image_origin=image_origin,
                    plots=src_plots)

    # Create target image scene with matching color-coded circles
    tgt_plots = [MakieSpec.Poly(tgt_circles,
                                color=:transparent,
                                strokecolor=1:length(tgt_circles),
                                strokecolormap=colormap,
                                strokewidth=strokewidth)]

    scene2 = Imshow(tgt_image; scale=scale, image_origin=image_origin,
                    plots=tgt_plots)

    # Return 1x2 GridLayout
    return MakieSpec.GridLayout([
        (1, 1) => scene1,
        (1, 2) => scene2
    ])
end

# Method 2: Accept shapes directly (original interface)
function cspond_plot(src_image, src_shapes::AbstractVector, tgt_image, tgt_shapes::AbstractVector;
                     scale=0.5,
                     image_origin=:julia,
                     strokewidth=1.0,
                     strokecolormap=:viridis)

    # Create source image scene with color-coded shapes
    src_plots = [MakieSpec.Poly(src_shapes,
                                color=:transparent,
                                strokecolor=1:length(src_shapes),
                                strokecolormap=strokecolormap,
                                strokewidth=strokewidth)]

    scene1 = Imshow(src_image; scale=scale, image_origin=image_origin,
                    plots=src_plots)

    # Create target image scene with color-coded shapes
    tgt_plots = [MakieSpec.Poly(tgt_shapes,
                                color=:transparent,
                                strokecolor=1:length(tgt_shapes),
                                strokecolormap=strokecolormap,
                                strokewidth=strokewidth)]

    scene2 = Imshow(tgt_image; scale=scale, image_origin=image_origin,
                    plots=tgt_plots)

    # Return 1x2 GridLayout
    return MakieSpec.GridLayout([
        (1, 1) => scene1,
        (1, 2) => scene2
    ])
end

"""
    frustum!(lscene, camera::Camera{CameraModel{PhysicalIntrinsics,...}}, sensor_bounds; kwargs...)

Add camera frustum visualization for camera with physical intrinsics.
Uses focal length as default depth for natural frustum visualization.

# Arguments
- `lscene`: LScene with 3D camera
- `camera`: Camera object with PhysicalIntrinsics
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
- `depth=nothing`: Frustum depth (mm). If nothing, uses focal length
- `color=:orange`: Frustum color
- `linewidth=2.0`: Width of frustum edge lines

# Examples
```julia
S.frustum!(scene3d, camera, sensor_bounds)  # Uses focal length as depth
S.frustum!(scene3d, camera, sensor_bounds; depth=300.0)  # Custom depth
```
"""
function frustum!(lscene, camera::Camera{<:CameraModel{PhysicalIntrinsics}}, sensor_bounds;
                  depth=nothing,
                  kwargs...)
    # Use focal length as default depth for physical cameras
    focal_depth = camera.model.intrinsics.f
    frustum_depth = something(depth, focal_depth)
    return frustum!(lscene, camera, sensor_bounds, frustum_depth; kwargs...)
end

"""
    frustum!(lscene, camera, sensor_bounds, depth; kwargs...)

Add camera frustum visualization to a 3D LScene with explicit depth.

# Arguments
- `lscene`: LScene with 3D camera
- `camera`: Camera object
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates
- `depth`: Depth of frustum from camera center (mm)

# Keyword Arguments
- `color=:orange`: Frustum color
- `linewidth=2.0`: Width of frustum edge lines
- `show_near_plane=false`: Show near plane rectangle
- `near_depth=10.0`: Near plane depth (if show_near_plane=true)
- `show_up_indicator=true`: Show triangle indicator at top edge pointing in camera's up direction
- `indicator_size=40.0`: Height of the up indicator triangle in pixels (image coordinates)
- `image=nothing`: Optional image to display on the far plane (texture mapped to frustum rectangle)

# Examples
```julia
S.frustum!(scene3d, camera, sensor_bounds, 250.0; color=:cyan)
S.frustum!(scene3d, camera, sensor_bounds, 250.0; image=camera_view)  # Show image on far plane
```
"""
function frustum!(lscene, camera, sensor_bounds, depth;
                  color=:orange,
                  linewidth=2.0,
                  show_near_plane=false,
                  near_depth=10.0,
                  show_up_indicator=true,
                  indicator_size=40.0,
                  image=nothing)

    # Get camera pose and center
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)

    # Unproject sensor corners to 3D in camera space, then transform to world
    corners_2d = coordinates(sensor_bounds)
    corners_cam = unproject.(Ref(camera.model), corners_2d, depth)
    corners_world = Point3f.(cam_pose.(corners_cam))

    # Optional: show image on far plane (before drawing wireframe)
    if !isnothing(image)
        # Create a unit rectangle mesh with UV coordinates using GeometryBasics
        unit_rect_mesh = GeometryBasics.uv_normal_mesh(Rect(0, 0, 1, 1))

        # Transform the unit rectangle to the frustum far plane
        # Extract positions and transform them to world space
        rect_positions = GeometryBasics.coordinates(unit_rect_mesh)

        # Map [0,1]x[0,1] rectangle to frustum far plane corners
        # corners_world: [p1, p2, p3, p4] from coordinates(sensor_bounds)
        transformed_positions = map(rect_positions) do p
            u, v = p[1], p[2]
            # Bilinear interpolation: p = (1-u)(1-v)p1 + u(1-v)p2 + uv*p3 + (1-u)v*p4
            p1, p2, p3, p4 = corners_world[1], corners_world[2], corners_world[3], corners_world[4]
            Point3f((1-u)*(1-v)*p1 + u*(1-v)*p2 + u*v*p3 + (1-u)*v*p4)
        end

        # Create new mesh with transformed positions but keep UV coordinates and faces
        far_mesh = GeometryBasics.Mesh(
            transformed_positions,
            GeometryBasics.faces(unit_rect_mesh);
            uv=GeometryBasics.texturecoordinates(unit_rect_mesh)
        )

        # Display textured mesh
        # Use parent() to unwrap OffsetArrays (safe for regular arrays too)
        # Makie will use the UV coordinates from the mesh to map the image
        push!(lscene.plots, MakieSpec.Mesh(far_mesh;
            color=parent(image),
            shading=Makie.NoShading,
            interpolate=true))
    end

    # Create frustum mesh: pyramid from camera center to far plane
    vertices = [cam_center, corners_world...]
    faces = [
        TriangleFace(1, 2, 3),  # side 1
        TriangleFace(1, 3, 4),  # side 2
        TriangleFace(1, 4, 5),  # side 3
        TriangleFace(1, 5, 2),  # side 4
    ]
    frustum_mesh = GeometryBasics.Mesh(vertices, faces)

    # Draw frustum as wireframe mesh
    push!(lscene.plots, MakieSpec.Wireframe(frustum_mesh;
        color=color, linewidth=linewidth))

    # Optional: near plane as rectangular mesh
    if show_near_plane
        near_cam = unproject.(Ref(camera.model), corners_2d, near_depth)
        near_world = Point3f.(cam_pose.(near_cam))

        # Create near plane rectangle as two triangles
        near_faces = [
            TriangleFace(1, 2, 3),
            TriangleFace(1, 3, 4)
        ]
        near_mesh = GeometryBasics.Mesh(near_world, near_faces)

        push!(lscene.plots, MakieSpec.Wireframe(near_mesh;
            color=color, linewidth=linewidth))
    end

    # Add up indicator: small triangle at midpoint of top edge pointing in -y direction (camera up)
    if show_up_indicator
        # Use center function and adjust y to top edge
        sensor_center = center(sensor_bounds)
        top_y = sensor_bounds.origin[2]  # Top edge (minimum y, since y points down)
        base_half_width = sensor_bounds.widths[1] * 0.02  # 2% of sensor width

        # Construct triangle in 2D image space
        tip_2d = Point2(sensor_center[1], top_y - indicator_size)
        base_left_2d = Point2(sensor_center[1] - base_half_width, top_y)
        base_right_2d = Point2(sensor_center[1] + base_half_width, top_y)
        triangle_2d = GeometryBasics.Triangle(tip_2d, base_left_2d, base_right_2d)

        # Broadcast unproject over triangle vertices (convert to SVector for unproject) and transform to world
        vertices_world = Point3f.(cam_pose.(unproject.(Ref(camera.model), SVector{2}.(triangle_2d), Ref(depth))))

        # Create 3D triangle from unprojected vertices
        indicator_triangle = GeometryBasics.Triangle(vertices_world...)

        push!(lscene.plots, MakieSpec.Mesh(indicator_triangle; color=color))
    end

    return nothing
end

"""
    camera!(lscene, camera, sensor_bounds; axis_length=50.0, frustum_depth=200.0, kwargs...)

Add camera visualization to a 3D LScene, including camera center, coordinate axes, and frustum.

# Arguments
- `lscene`: LScene with 3D camera (cam3d!)
- `camera`: Camera object with model and extrinsics
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
- `axis_length=50.0`: Length of camera coordinate axes (mm)
- `frustum_depth=200.0`: Depth of frustum visualization (mm)
- `show_center=true`: Show camera center marker
- `show_axes=true`: Show camera coordinate frame axes
- `show_frustum=true`: Show camera frustum
- `center_color=:red`: Color for camera center marker
- `center_size=20.0`: Size of camera center marker
- `frustum_color=:orange`: Color for frustum
- `frustum_linewidth=2.0`: Width of frustum edges
- `frustum_image=nothing`: Optional image to display on frustum far plane

# Examples
```julia
import VisualGeometryCore.Spec as S

# Create 3D scene
scene3d = S.LScene3D()

# Add camera visualization
S.camera!(scene3d, camera, sensor_bounds)

# Customize appearance
S.camera!(scene3d, camera, sensor_bounds;
    axis_length=100.0,
    frustum_depth=300.0,
    frustum_color=:blue)
```
"""
function camera!(lscene, camera, sensor_bounds;
                 axis_length=50.0,
                 frustum_depth=200.0,
                 show_center=true,
                 show_axes=true,
                 show_frustum=true,
                 center_color=:red,
                 center_size=20.0,
                 frustum_color=:orange,
                 frustum_linewidth=2.0,
                 frustum_image=nothing)

    # Get camera pose (camera-to-world)
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)
    R_c2w = cam_pose.R  # Camera-to-world rotation (columns are camera axes in world coords)

    # 1. Camera center
    if show_center
        push!(lscene.plots, MakieSpec.Scatter([cam_center];
            color=center_color,
            markersize=center_size))
    end

    # 2. Camera coordinate axes
    if show_axes
        # Transform scaled canonical basis vectors to world coordinates via pose
        basis = canonical_basis(SVector{3,Float64})
        axis_ends = Point3f.(cam_pose.(basis .* axis_length))

        # X-axis (red), Y-axis (green), Z-axis/optical axis (blue)
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[1]];
            color=:red, linewidth=3))
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[2]];
            color=:green, linewidth=3))
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[3]];
            color=:blue, linewidth=3))

        # Add markers at endpoints to indicate direction
        push!(lscene.plots, MakieSpec.Scatter(
            collect(axis_ends);
            color=[:red, :green, :blue],
            markersize=8))
    end

    # 3. Camera frustum
    if show_frustum
        frustum!(lscene, camera, sensor_bounds, frustum_depth;
            color=frustum_color,
            linewidth=frustum_linewidth,
            image=frustum_image)
    end

    return nothing
end

"""
    LScene3D(; show_axis=true, width=600, height=600, kwargs...)

Create a 3D LScene with interactive Camera3D controls.

# Keyword Arguments
- `show_axis=true`: Show 3D axis
- `width=600`: Scene width
- `height=600`: Scene height
- `scenekw`: Additional scene keyword arguments

# Returns
- LScene configured for 3D visualization

# Examples
```julia
scene3d = S.LScene3D()

# Add visualizations
S.board!(scene3d, X)
S.camera!(scene3d, camera)

# Include in layout
layout = S.GridLayout([(1,1) => scene3d])
```
"""
function LScene3D(;
                  show_axis=true,
                  width=600,
                  height=600,
                  plots=PlotSpec[],
                  scenekw=(;))

    lscene = MakieSpec.LScene(
        plots=plots,
        show_axis=show_axis,
        width=Makie.Fixed(width),
        height=Makie.Fixed(height),
        scenekw=scenekw
    )

    # Add callback to enable Camera3D controls
    function setup_camera3d(lscene_block)
        cam3d!(lscene_block.scene)
        return
    end

    push!(lscene.then_funcs, setup_camera3d)

    return lscene
end

end # module Spec


