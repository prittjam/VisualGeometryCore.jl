# ========================================================================
# SpecApi Plotting Functions
# ========================================================================

"""
    imshow(pattern; interpolate=false, rasterize=true, bounds=nothing, scene_size=nothing) -> LScene

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
# Basic usage - image fills scene
lscene = imshow(img)

# Rasterized for PDF (default)
lscene = imshow(img; rasterize=true)

# Center 100x100 image in 200x200 scene
lscene = imshow(img; bounds=(50, 150, 50, 150), scene_size=(200, 200))

# Add overlays using SpecApi
lscene = imshow(img)
push!(lscene.plots, Spec.Scatter(x, y; color=:red))
append!(lscene.plots, plotblobs(blobs))
plot(Spec.GridLayout([lscene]))
```
"""
function imshow(pattern; interpolate=false, rasterize=true, bounds=nothing, scene_size=nothing)
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
        Spec.Image(
            Makie.EndPoints(xmin, xmax),
            Makie.EndPoints(ymin, ymax),
            transpose(pattern);
            interpolate=interpolate,
            rasterize=rasterize
        )
    else
        Spec.Image(transpose(pattern); interpolate=interpolate, rasterize=rasterize)
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

    lscene = Spec.LScene(
        plots=[image_spec],
        show_axis=false,
        width=MakieFixed(scene_width),
        height=MakieFixed(scene_height),
        tellwidth=false,
        tellheight=false,
        scenekw=(camera=campixel!, limits=(0, scene_width, 0, scene_height))
    )

    # Add callback to flip y-axis after scene is created
    push!(lscene.then_funcs, flip_y_axis)

    return lscene
end

"""
    plotblobs(blobs; color=:green, colormap=:viridis, scale_factor=3.0, marker=:cross, markersize=15.0, linewidth=3.0)

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
- `linewidth=3.0`: Width of circle outlines
- `linestyle=:solid`: Line style for circles

Returns a vector of PlotSpec objects (scatter + circles) that can be composed.

# Examples
```julia
# Default: uniform green (no color parameter)
plotblobs(blobs)

# Uniform color
plotblobs(blobs; color=:red)

# Color by index using default viridis colormap (shows column-major spatial ordering)
# Note: Must explicitly set color to numeric values to use colormap
plotblobs(blobs; color=1:length(blobs))

# Color by index with custom colormap
plotblobs(blobs; color=1:length(blobs), colormap=:turbo)

# Uniform color with colormap specified (colormap is ignored - Makie convention)
plotblobs(blobs; color=:red, colormap=:turbo)  # uses uniform red

# Compose with image
lscene = imshow(img)
append!(lscene.plots, plotblobs(blobs; color=1:length(blobs), colormap=:turbo))
plot(Spec.GridLayout([lscene]))
```
"""
function plotblobs(blobs;
                   color=:green,
                   colormap=:viridis,
                   scale_factor::Float64=3.0,
                   marker=:cross,
                   markersize::Float64=15.0,
                   linewidth::Float64=1.0,
                   linestyle=:solid)
    plot_specs = Makie.PlotSpec[]
    if !isempty(blobs)
        centers = [ustrip.(blob.center) for blob in blobs]
        scales = [ustrip(blob.σ) for blob in blobs]

        # Handle color specification following Makie conventions:
        # - If color is numeric, map through colormap
        # - If color is a single symbolic color, use uniformly
        # - If color is a vector of colors, use directly
        if color isa AbstractVector{<:Number}
            # Numeric values: map through colormap
            cmap = Makie.to_colormap(colormap)
            n = length(color)
            if n != length(blobs)
                throw(ArgumentError("color vector length ($(n)) must match blobs length ($(length(blobs)))"))
            end
            # Normalize color values to [0, 1] and map to colormap
            cmin, cmax = extrema(color)
            colors = if cmin == cmax
                fill(cmap[1], length(blobs))
            else
                [cmap[max(1, min(length(cmap), round(Int, (c - cmin) / (cmax - cmin) * (length(cmap) - 1) + 1)))] for c in color]
            end
        elseif color isa AbstractVector
            # Vector of colors: use directly
            if length(color) != length(blobs)
                throw(ArgumentError("color vector length ($(length(color))) must match blobs length ($(length(blobs)))"))
            end
            colors = color
        else
            # Single color: use uniformly
            colors = fill(color, length(blobs))
        end

        # Add circles for blob scales using proper geometric circles
        if scale_factor > 0
            for (i, (c, s)) in enumerate(zip(centers, scales))
                radius = s * scale_factor
                center_point = Point2f(c[1], c[2])
                circle_geom = GeometryBasics.Circle(center_point, radius)

                # Background circle (white outline)
                background_circle = Spec.Lines(
                    circle_geom;
                    color=:white,
                    linewidth=linewidth + 2
                )
                push!(plot_specs, background_circle)

                # Foreground circle (colored outline)
                foreground_circle = Spec.Lines(
                    circle_geom;
                    color=colors[i],
                    linewidth=linewidth
                )
                push!(plot_specs, foreground_circle)
            end

            # Center markers with per-blob colors
            center_markers = Spec.Scatter(
                [c[1] for c in centers], [c[2] for c in centers];
                marker=marker,
                markersize=markersize,
                color=colors
            )
            push!(plot_specs, center_markers)
        else
            # If no scale_factor, just plot center markers
            center_markers = Spec.Scatter(
                [c[1] for c in centers], [c[2] for c in centers];
                marker=marker,
                markersize=markersize,
                color=colors
            )
            push!(plot_specs, center_markers)
        end
    end

    return plot_specs
end

"""
    plotellipses(ellipses; color=:blue, linewidth=2.0, linestyle=:solid, fillcolor=nothing, fillalpha=0.2, resolution=64)

Create plot specs for ellipse visualization (outlines and optional fills).

# Arguments
- `ellipses`: Vector of Ellipse objects

# Keyword Arguments
- `color=:blue`: Color for ellipse outlines
- `linewidth=2.0`: Width of ellipse outlines
- `linestyle=:solid`: Line style (:solid, :dash, :dot, etc.)
- `fillcolor=nothing`: Fill color (nothing for no fill, or any color)
- `fillalpha=0.2`: Alpha transparency for fill (0.0 = transparent, 1.0 = opaque)
- `resolution=64`: Number of points to use for ellipse approximation

Returns a vector of PlotSpec objects that can be composed with SpecApi.

# Examples
```julia
ellipses = [Ellipse(Point2(100.0, 100.0), 50.0, 30.0, π/4)]

# Just outlines
specs = plotellipses(ellipses; color=:red, linewidth=3.0)

# With fill
specs = plotellipses(ellipses; color=:blue, fillcolor=:lightblue, fillalpha=0.3)

# Compose with image
lscene = imshow(img)
append!(lscene.plots, plotellipses(ellipses))
plot(Spec.GridLayout([lscene]))
```
"""
function plotellipses(ellipses;
                     color=:blue,
                     linewidth::Float64=2.0,
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
                fill_spec = Spec.Poly(points; color=fillcolor, alpha=fillalpha)
                push!(plot_specs, fill_spec)
            end
            
            # Add outline
            # Close the ellipse by adding the first point at the end
            closed_points = vcat(points, [points[1]])
            outline_spec = Spec.Lines(closed_points; color=color, linewidth=linewidth, linestyle=linestyle)
            push!(plot_specs, outline_spec)
        end
    end
    
    return plot_specs
end