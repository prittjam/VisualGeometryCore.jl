module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob, INTRINSICS_COORDINATE_OFFSET, Ellipse, Cspond
using GeometryBasics
using GeometryBasics: Circle, Rect, Point2, origin
using StaticArrays
using StructArrays
using Unitful: ustrip
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
# Point correspondences - draw marker circles
function cspond!(src_lscene, tgt_lscene,
                 csponds::StructArrays.StructArray{<:Cspond{<:StaticVector, <:StaticVector}};
                 marker_size=10.0,
                 strokewidth=2.0,
                 colormap=:plasma)

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
    push!(src_lscene.plots, MakieSpec.Poly(src_circles,
                                           color=:transparent,
                                           strokecolor=1:length(src_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth))

    push!(tgt_lscene.plots, MakieSpec.Poly(tgt_circles,
                                           color=:transparent,
                                           strokecolor=1:length(tgt_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth))

    return nothing
end

# Blob correspondences - draw the blobs themselves as circles
function cspond!(src_lscene, tgt_lscene,
                 csponds::StructArrays.StructArray{<:Cspond{<:AbstractBlob, <:AbstractBlob}};
                 sigma_cutoff=3.0,
                 strokewidth=2.0,
                 colormap=:plasma)

    # Extract source and target blobs
    src_blobs = csponds.source
    tgt_blobs = csponds.target

    # Create circles from blobs (using sigma_cutoff to determine radius)
    src_circles = [Circle(origin(b), sigma_cutoff * b.σ) for b in src_blobs]
    tgt_circles = [Circle(origin(b), sigma_cutoff * b.σ) for b in tgt_blobs]

    # Add color-coded circles to LScenes
    push!(src_lscene.plots, MakieSpec.Poly(src_circles,
                                           color=:transparent,
                                           strokecolor=1:length(src_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth))

    push!(tgt_lscene.plots, MakieSpec.Poly(tgt_circles,
                                           color=:transparent,
                                           strokecolor=1:length(tgt_circles),
                                           strokecolormap=colormap,
                                           strokewidth=strokewidth))

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

end # module Spec


