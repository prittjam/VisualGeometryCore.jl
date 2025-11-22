module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob, INTRINSICS_COORDINATE_OFFSET, Ellipse, Cspond, pose, canonical_basis, center
using ..VisualGeometryCore: Camera, CameraModel, PhysicalIntrinsics, unproject
using GeometryBasics
using GeometryBasics: Circle, Rect, Point2, Point3, Point3f, Vec2, Vec3f, TriangleFace, origin, coordinates
using Infiltrator
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

# Camera plotting functions moved to cameras.jl
include("cameras.jl")

end # module Spec


