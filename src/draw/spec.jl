module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob
using GeometryBasics
using Unitful: ustrip
import Makie.SpecApi as MakieSpec

"""
    Imshow(pattern; interpolate=false, rasterize=true, width=nothing, height=nothing) -> LScene
    Imshow(x_bounds, y_bounds, pattern; kwargs...) -> LScene

Create an LScene spec for displaying an image with proper orientation (y-axis flipped).

Bounds can be specified as positional arguments (matching Makie API conventions):
- Ranges: `Imshow(1:100, 1:200, img)`
- Tuples: `Imshow((1, 100), (1, 200), img)`
- Intervals: `Imshow(1..100, 1..200, img)` (if IntervalSets is available)

# Arguments
- `pattern`: Image matrix (AbstractMatrix{<:Colorant})
- `x_bounds`: X-axis bounds (range, tuple, interval, or any type Makie.Image accepts)
- `y_bounds`: Y-axis bounds (range, tuple, interval, or any type Makie.Image accepts)

# Keyword Arguments
- `interpolate::Bool=false`: Enable image interpolation
- `rasterize::Bool=true`: Rasterize image for efficient PDF serialization (prevents huge PDFs)
- `width::Union{Nothing,Int}=nothing`: LScene width. If nothing, uses pattern width.
- `height::Union{Nothing,Int}=nothing`: LScene height. If nothing, uses pattern height.

Returns an LScene spec that can be used with SpecApi for composable plotting.

# Examples
```julia
import VisualGeometryCore.Spec as S

# Basic usage - image fills scene
lscene = S.Imshow(img)

# With bounds as ranges
lscene = S.Imshow(50:150, 50:150, img; width=200, height=200)

# With bounds as tuples
lscene = S.Imshow((50, 150), (50, 150), img; width=200, height=200)

# Add overlays using SpecApi
lscene = S.Imshow(img)
push!(lscene.plots, S.Scatter(x, y; color=:red))
append!(lscene.plots, S.Blobs(blobs))
plot(MakieSpec.GridLayout([lscene]))
```
"""
# Dispatch: bounds (ranges, tuples, intervals) - pass directly to Spec.Image
function Imshow(x_bounds, y_bounds, pattern;
                interpolate=false, rasterize=true, width=nothing, height=nothing)
    # Scene dimensions: use provided or pattern dimensions (width, height)
    pattern_width, pattern_height = reverse(size(pattern))
    scene_width = something(width, pattern_width)
    scene_height = something(height, pattern_height)

    # Pass bounds directly to Spec.Image (accepts tuples, intervals, or EndPoints)
    image_spec = MakieSpec.Image(x_bounds, y_bounds, transpose(pattern);
                                 interpolate=interpolate, rasterize=rasterize)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        scene.transformation.scale[] = (1.0, -1.0, 1.0)
        scene.transformation.translation[] = (0.0, Float64(scene_height), 0.0)
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots=[image_spec],
        show_axis=false,
        width=Makie.Fixed(scene_width),
        height=Makie.Fixed(scene_height),
        tellwidth=true,
        tellheight=true,
        scenekw=(camera=campixel!,)
    )

    push!(lscene.then_funcs, flip_y_axis)
    return lscene
end

# Dispatch: no bounds (image fills scene)
function Imshow(pattern; interpolate=false, rasterize=true, width=nothing, height=nothing)
    # Scene dimensions: use provided or pattern dimensions (width, height)
    pattern_width, pattern_height = reverse(size(pattern))
    scene_width = something(width, pattern_width)
    scene_height = something(height, pattern_height)

    # Create image spec without bounds (fills scene)
    image_spec = MakieSpec.Image(transpose(pattern); interpolate=interpolate, rasterize=rasterize)

    # Use then_funcs to apply y-flip transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        # Flip y-axis by scaling y by -1 and translating
        # For a scene of height H, point at y=0 maps to y=H, point at y=H maps to y=0
        # Transformation: y' = -y + H (scale by -1, translate by H)
        scene.transformation.scale[] = (1.0, -1.0, 1.0)
        scene.transformation.translation[] = (0.0, Float64(scene_height), 0.0)

        # Update camera to ensure it doesn't clip
        notify(scene.camera.projectionview)
        return
    end

    lscene = MakieSpec.LScene(
        plots=[image_spec],
        show_axis=false,
        width=Makie.Fixed(scene_width),
        height=Makie.Fixed(scene_height),
        tellwidth=true,
        tellheight=true,
        scenekw=(camera=campixel!,)
    )

    # Add callback to flip y-axis after scene is created
    push!(lscene.then_funcs, flip_y_axis)

    return lscene
end

"""
    Blobs(blobs; color=:green, colormap=:viridis, scale_factor=3.0, marker=:cross, markersize=15.0, linewidth=1.0, linestyle=:solid)

Create plot specs for blob visualization (centers and scale circles).

Follows Makie conventions for color/colormap handling:
- By default, uses uniform green coloring (matching Makie's theme-based defaults)
- To visualize blob ordering with a colormap, explicitly set `color` to numeric indices
- Blobs are saved in column-major order, so colormapping reveals spatial structure

# Arguments
- `blobs`: Vector of blob features

# Keyword Arguments
- `color=:green`: Color specification for blobs. Can be:
  - A single color (e.g., `:red`, `:blue`) for uniform coloring (default)
  - Numeric values (e.g., `1:length(blobs)`) to map through the colormap
  - A vector of colors for per-blob custom coloring
- `colormap=:viridis`: Colormap to use when `color` contains numeric values.
  Common options: `:viridis`, `:plasma`, `:turbo`, `:inferno`.
  Only used when `color` is numeric; ignored when `color` is a symbolic color.
- `scale_factor=3.0`: Scaling factor for blob circles
- `marker=:cross`: Marker style for blob centers
- `markersize=15.0`: Size of center markers
- `linewidth=1.0`: Width of circle outlines
- `linestyle=:solid`: Line style for circles

Returns a vector of PlotSpec objects (scatter + circles) that can be composed.

# Examples
```julia
import Makie.SpecApi as S

# Default: uniform green (no color parameter)
S.Blobs(blobs)

# Uniform color
S.Blobs(blobs; color=:red)

# Color by index using default viridis colormap (shows column-major spatial ordering)
# Note: Must explicitly set color to numeric values to use colormap
S.Blobs(blobs; color=1:length(blobs))

# Color by index with custom colormap
S.Blobs(blobs; color=1:length(blobs), colormap=:turbo)

# Uniform color with colormap specified (colormap is ignored - Makie convention)
S.Blobs(blobs; color=:red, colormap=:turbo)  # uses uniform red

# Compose with image
lscene = S.Imshow(img)
append!(lscene.plots, S.Blobs(blobs; color=1:length(blobs), colormap=:turbo))
plot(S.GridLayout([lscene]))
```
"""
function Blobs(blobs;
                             color=:green,
                             colormap=:viridis,
                             sigma_cutoff::Float64=3.0,
                             marker=:cross,
                             markersize::Float64=15.0,
                             linewidth::Float64=1.0,
                             linestyle=:solid)
    plot_specs = Makie.PlotSpec[]
    if !isempty(blobs)
        # Handle color specification following Makie conventions:
        # - If color is numeric, let Makie map through colormap (pass colormap & colorrange)
        # - If color is a single symbolic color, use uniformly
        # - If color is a vector of colors, use directly
        use_colormap = false
        colorrange_val = nothing

        if color isa AbstractVector{<:Number}
            # Numeric values: let Makie handle the colormap mapping
            if length(color) != length(blobs)
                throw(ArgumentError("color vector length ($(length(color))) must match blobs length ($(length(blobs)))"))
            end
            use_colormap = true
            colorrange_val = extrema(color)
            colors = color  # Pass numeric values directly
        elseif color isa AbstractVector
            # Vector of colors: use directly
            if length(color) != length(blobs)
                throw(ArgumentError("color vector length ($(length(color))) must match blobs length ($(length(blobs)))"))
            end
            colors = color
        else
            # Single color: replicate for all blobs
            colors = fill(color, length(blobs))
        end

        # Extract centers (common to both branches)
        centers = ustrip.(origin.(blobs))

        # Add circles for blob scales using Circle constructor with broadcast
        if sigma_cutoff > 0
            # Create all circles at once using broadcast
            circles = Circle.(blobs, Ref(sigma_cutoff))

            # Convert circles to point arrays for Makie Lines plotting
            circle_points = GeometryBasics.coordinates.(circles)

            # Create individual Lines specs for each circle (Makie Lines needs one spec per line)
            for (i, points) in enumerate(circle_points)
                # Background circle (white outline)
                push!(plot_specs, MakieSpec.Lines(
                    points;
                    color=:white,
                    linewidth=linewidth + 2,
                    linestyle=linestyle
                ))

                # Foreground circle (colored outline)
                if use_colormap
                    push!(plot_specs, MakieSpec.Lines(
                        points;
                        color=colors[i],
                        linewidth=linewidth,
                        linestyle=linestyle,
                        colormap=colormap,
                        colorrange=colorrange_val
                    ))
                else
                    push!(plot_specs, MakieSpec.Lines(
                        points;
                        color=colors[i],
                        linewidth=linewidth,
                        linestyle=linestyle
                    ))
                end
            end
        end

        # Center markers with per-blob colors (always rendered)
        if use_colormap
            center_markers = MakieSpec.Scatter(
                centers;
                marker=marker,
                markersize=markersize,
                color=colors,
                colormap=colormap,
                colorrange=colorrange_val
            )
        else
            center_markers = MakieSpec.Scatter(
                centers;
                marker=marker,
                markersize=markersize,
                color=colors
            )
        end
        push!(plot_specs, center_markers)
    end

    return plot_specs
end

"""
    Ellipses(ellipses; color=:green, marker=:cross, markersize=15.0, linewidth=1.0, linestyle=:solid, fillcolor=nothing, fillalpha=0.2, resolution=64)

Create plot specs for ellipse visualization (outlines, optional fills, and optional center markers).

# Arguments
- `ellipses`: Vector of Ellipse objects

# Keyword Arguments
- `color=:green`: Color for ellipse outlines and center markers
- `marker=:cross`: Marker style for ellipse centers (:cross, :circle, :x, etc., or nothing for no markers)
- `markersize=15.0`: Size of center markers
- `linewidth=1.0`: Width of ellipse outlines
- `linestyle=:solid`: Line style (:solid, :dash, :dot, etc.)
- `fillcolor=nothing`: Fill color (nothing for no fill, or any color)
- `fillalpha=0.2`: Alpha transparency for fill (0.0 = transparent, 1.0 = opaque)
- `resolution=64`: Number of points to use for ellipse approximation

Returns a vector of PlotSpec objects that can be composed with SpecApi.

# Examples
```julia
import Makie.SpecApi as S

ellipses = [Ellipse(Point2(100.0, 100.0), 50.0, 30.0, π/4)]

# Default: green outlines with cross markers
specs = S.Ellipses(ellipses)

# Just outlines (no markers)
specs = S.Ellipses(ellipses; color=:red, linewidth=3.0, marker=nothing)

# With fill
specs = S.Ellipses(ellipses; color=:blue, fillcolor=:lightblue, fillalpha=0.3)

# Custom markers
specs = S.Ellipses(ellipses; color=:green, marker=:circle, markersize=8.0)

# Compose with image
lscene = S.Imshow(img)
append!(lscene.plots, S.Ellipses(ellipses))
plot(S.GridLayout([lscene]))
```
"""
function Ellipses(ellipses;
                                color=:green,
                                marker=:cross,
                                markersize::Float64=15.0,
                                linewidth::Float64=1.0,
                                linestyle=:solid,
                                fillcolor=nothing,
                                fillalpha::Float64=0.2,
                                resolution::Int=64)
    plot_specs = Makie.PlotSpec[]

    if !isempty(ellipses)
        for ellipse in ellipses
            # Generate ellipse boundary points using GeometryBasics coordinates
            points = GeometryBasics.coordinates(ellipse, resolution)

            # Add fill if requested
            if fillcolor !== nothing
                fill_spec = MakieSpec.Poly(points; color=fillcolor, alpha=fillalpha)
                push!(plot_specs, fill_spec)
            end

            # Add outline
            # Close the ellipse by adding the first point at the end
            closed_points = vcat(points, [points[1]])
            outline_spec = MakieSpec.Lines(closed_points; color=color, linewidth=linewidth, linestyle=linestyle)
            push!(plot_specs, outline_spec)
        end

        # Add center markers if requested
        if marker !== nothing
            centers = [e.center for e in ellipses]
            marker_spec = MakieSpec.Scatter(centers;
                                           marker=marker,
                                           markersize=markersize,
                                           color=color)
            push!(plot_specs, marker_spec)
        end
    end

    return plot_specs
end

end # module Spec

