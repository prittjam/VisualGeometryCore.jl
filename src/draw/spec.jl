module Spec

using Makie
using ..VisualGeometryCore: AbstractBlob
using GeometryBasics
using Unitful: ustrip
import Makie.SpecApi as MakieSpec

"""
    Imshow(pattern; interpolate=false, rasterize=true, bounds=nothing, scene_size=nothing) -> LScene

Create an LScene spec for displaying an image with proper orientation (y-axis flipped).

# Arguments
- `pattern`: Image matrix (AbstractMatrix{<:Colorant})

# Keyword Arguments
- `interpolate::Bool=false`: Enable image interpolation
- `rasterize::Bool=true`: Rasterize image for efficient PDF serialization (prevents huge PDFs)
- `bounds::Union{Nothing, NTuple{4, Number}}=nothing`: Optional (xmin, xmax, ymin, ymax) for positioning
  within a larger scene. If nothing, image fills the scene.
- `scene_size::Union{Nothing, Tuple{Int,Int}}=nothing`: Optional (width, height) for LScene size.
  If nothing, uses pattern dimensions. Use with bounds to center images in fixed-size scenes.

Returns an LScene spec that can be used with SpecApi for composable plotting.

# Examples
```julia
import Makie.SpecApi as S

# Basic usage - image fills scene
lscene = S.Imshow(img)

# Rasterized for PDF (default)
lscene = S.Imshow(img; rasterize=true)

# Center 100x100 image in 200x200 scene
lscene = S.Imshow(img; bounds=(50, 150, 50, 150), scene_size=(200, 200))

# Add overlays using SpecApi
lscene = S.Imshow(img)
push!(lscene.plots, S.Scatter(x, y; color=:red))
append!(lscene.plots, S.Blobs(blobs))
plot(S.GridLayout([lscene]))
```
"""
function Imshow(pattern; interpolate=false, rasterize=true, bounds=nothing, scene_size=nothing)
    pattern_height, pattern_width = size(pattern)

    # Determine scene dimensions
    scene_width, scene_height = if scene_size !== nothing
        scene_size
    else
        (pattern_width, pattern_height)
    end

    # Create image spec with optional bounds for positioning
    image_spec = if bounds !== nothing
        xmin, xmax, ymin, ymax = bounds
        MakieSpec.Image(
            Makie.EndPoints(xmin, xmax),
            Makie.EndPoints(ymin, ymax),
            transpose(pattern);
            interpolate=interpolate,
            rasterize=rasterize
        )
    else
        MakieSpec.Image(transpose(pattern); interpolate=interpolate, rasterize=rasterize)
    end

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
        scenekw=(camera=campixel!, limits=(0, scene_width, 0, scene_height))
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

            # Background circles (white outlines) - single PlotSpec for all
            background_circles = MakieSpec.Lines(
                circles;
                color=:white,
                linewidth=linewidth + 2,
                linestyle=linestyle
            )
            push!(plot_specs, background_circles)

            # Foreground circles (colored outlines) - single PlotSpec with per-circle colors
            if use_colormap
                foreground_circles = MakieSpec.Lines(
                    circles;
                    color=colors,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    colormap=colormap,
                    colorrange=colorrange_val
                )
            else
                foreground_circles = MakieSpec.Lines(
                    circles;
                    color=colors,
                    linewidth=linewidth,
                    linestyle=linestyle
                )
            end
            push!(plot_specs, foreground_circles)
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

