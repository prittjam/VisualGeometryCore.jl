module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob, INTRINSICS_COORDINATE_OFFSET, Ellipse
using GeometryBasics
using GeometryBasics: Circle, Rect
using Unitful: ustrip
import Makie.SpecApi as MakieSpec

"""
    Imshow(img; width=nothing, height=nothing, scenekw=(camera=campixel!,), imagekw=(interpolate=false, rasterize=true), image_origin=:makie) -> LScene
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
                image_origin::Symbol=:makie,
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
    imshow_image(img; image_origin=:makie) -> Image

Create a Makie Image spec with proper y-axis orientation, without the LScene wrapper.
This is useful for adding images to existing LScenes or composing with other plots.

# Arguments
- `img`: Image matrix
- `image_origin::Symbol=:makie`: Coordinate convention (:julia, :opencv, :makie, etc.)

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
                      image_origin::Symbol=:makie,
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
    Locus(geometry; color=:green, linewidth=1.0, linestyle=:solid, resolution=64)

Create a plot spec for the outline (locus) of a geometric object.

Uses multiple dispatch to handle different geometry types:
- `Circle`: circular outline
- `Ellipse`: elliptical outline
- `Rect`: rectangular outline

For multiple geometries, use broadcasting: `Locus.(geometries; kwargs...)`

# Arguments
- `geometry`: A geometric object (Circle, Ellipse, or Rect)

# Keyword Arguments
- `color=:green`: Color for the outline
- `linewidth=1.0`: Width of the outline
- `linestyle=:solid`: Line style (:solid, :dash, :dot, etc.)
- `resolution=64`: Number of points for curved geometries (Circle, Ellipse)

Returns a single PlotSpec. For multiple geometries, broadcast to get Vector{PlotSpec}.

# Examples
```julia
import VisualGeometryCore.Spec as S

# Single circle locus (returns PlotSpec, wrap in array for plots)
circle = Circle(Point2(100.0, 100.0), 50.0)
plots = [S.Locus(circle; color=:red, linewidth=2.0)]

# Multiple circles (broadcast to get Vector{PlotSpec})
circles = [Circle(Point2(100.0, 100.0), 50.0), Circle(Point2(200.0, 200.0), 30.0)]
plots = S.Locus.(circles; color=:blue)

# Per-element colors with broadcasting
plots = S.Locus.(circles; color=[:red, :green], linewidth=2.0)

# Compose with image
lscene = S.Imshow(img; plots=S.Locus.(circles; color=:yellow))
```
"""
function Locus(circle::Circle;
               color=:green,
               linewidth::Float64=1.0,
               linestyle=:solid,
               resolution::Int=64)
    # Generate circle boundary points using GeometryBasics
    points = GeometryBasics.coordinates(circle, resolution)

    # Close the path by adding the first point at the end
    closed_points = vcat(points, [points[1]])

    # Return Lines spec (unwrapped for broadcasting)
    return MakieSpec.Lines(closed_points; color=color, linewidth=linewidth, linestyle=linestyle)
end

function Locus(ellipse::Ellipse;
               color=:green,
               linewidth::Float64=1.0,
               linestyle=:solid,
               resolution::Int=64)
    # Generate ellipse boundary points using GeometryBasics
    points = GeometryBasics.coordinates(ellipse, resolution)

    # Close the path by adding the first point at the end
    closed_points = vcat(points, [points[1]])

    # Return Lines spec (unwrapped for broadcasting)
    return MakieSpec.Lines(closed_points; color=color, linewidth=linewidth, linestyle=linestyle)
end

function Locus(rect::Rect;
               color=:green,
               linewidth::Float64=1.0,
               linestyle=:solid,
               resolution::Int=64)  # Ignored for rectangles but kept for interface consistency
    # Extract rectangle corners
    min_corner = GeometryBasics.minimum(rect)
    max_corner = GeometryBasics.maximum(rect)

    # Generate rectangle boundary points (clockwise from bottom-left)
    points = [
        Point2(min_corner[1], min_corner[2]),  # bottom-left
        Point2(max_corner[1], min_corner[2]),  # bottom-right
        Point2(max_corner[1], max_corner[2]),  # top-right
        Point2(min_corner[1], max_corner[2]),  # top-left
        Point2(min_corner[1], min_corner[2])   # back to bottom-left (close path)
    ]

    # Return Lines spec (unwrapped for broadcasting)
    return MakieSpec.Lines(points; color=color, linewidth=linewidth, linestyle=linestyle)
end

end # module Spec

