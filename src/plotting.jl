# Plotting functionality for VisualGeometryCore types
# Uses Makie Spec API with convert_arguments for composable plotting

import Makie.SpecApi as Spec
using Makie: Fixed, campixel!
using GeometryBasics: Circle, Point2f

# Convert arguments for images using SpecApi
# Note: For image+blob overlays, use SpecApi composition:
#   lscene = imshow(image)
#   blob_specs = plotblobs(blobs; color=:red, scale_factor=3.0)
#   append!(lscene.plots, blob_specs)
#   fig, _, _ = plot(S.GridLayout([lscene]))

# PlottableImage wrapper for plot!/plot compatibility
Makie.used_attributes(::Type{<:Plot}, ::PlottableImage) = ()

function Makie.convert_arguments(::Type{<:AbstractPlot}, img::PlottableImage)
    image_spec = Spec.Image(transpose(img.data), interpolate=img.interpolate)
    return [image_spec]
end

# Convert arguments for PlotSpec integration with recipes
# Atomic convert_arguments for blobs only
Makie.used_attributes(::Type{<:AbstractPlot}, ::Vector{<:AbstractBlob}) = (:color, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob}; color=:green, scale_factor=3.0, ...)

Convert blobs to PlotSpec vector for Makie recipe integration.
Allows users to call: plot(blobs; color=:red, scale_factor=2.0) or plot!(blobs; ...)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob};
                                color=:green, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0,
                                linewidth::Float64=1.0, linestyle=:solid)
    # Return PlotSpec vector for SpecApi integration
    return plotblobs(blobs; color=color, scale_factor=scale_factor,
                    marker=marker, markersize=markersize,
                    linewidth=linewidth, linestyle=linestyle)
end

# Atomic convert_arguments for blob detections with dashed outlines
Makie.used_attributes(::Type{<:AbstractPlot}, ::Vector{IsoBlobDetection}) = (:color, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection}; color=:blue, linestyle=:dash, ...)

Convert blob detections to PlotSpec vector with dashed outlines for Makie recipe integration.
Allows users to call: plot(detections; color=:blue, linestyle=:dash) or plot!(detections; ...)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection};
                                color=:blue, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0,
                                linestyle=:dash)
    # Extract IsoBlobs from detections and call plotblobs
    blobs = [det.blob for det in detections]
    return plotblobs(blobs; color=color, scale_factor=scale_factor,
                    marker=marker, markersize=markersize,
                    linewidth=linewidth, linestyle=linestyle)
end

# Direct tuple conversion - returns data tuples (Makie standard pattern for multiple arguments)
# This enables: scatter(img, blobs) but blobs will be plotted as points, not overlaid on image
# For proper image+blob overlay, use ImageWithBlobs wrapper struct instead

"""
    Makie.convert_arguments(::PointBased, image::AbstractMatrix, blobs::Vector{<:AbstractBlob})

Convert image and blobs to point data for scatter-style plots.
Note: This plots blobs as points in a scatter plot, NOT overlaid on the image.
For image with blob overlay, use `ImageWithBlobs` wrapper instead.
"""
function Makie.convert_arguments(::Makie.PointBased, image::AbstractMatrix, blobs::Vector{<:AbstractBlob})
    # Extract blob centers as points (standard Makie pattern for multiple args)
    if isempty(blobs)
        return (Point2f[],)
    end

    centers = [ustrip.(blob.center) for blob in blobs]
    points = [Point2f(c[1], c[2]) for c in centers]
    return (points,)
end

# Wrapper struct for image with blobs overlay - following Makie's CustomMatrix pattern
"""
    ImageWithBlobs(image, blobs; color=:green, scale_factor=3.0, marker=:cross, markersize=15.0, linewidth=1.0)

Wrapper struct for plotting an image with blob overlays using convert_arguments.
Follows the Makie SpecApi pattern with a custom struct (like CustomMatrix in docs).

# Examples
```julia
img = testimage("cameraman")
blobs = [IsoBlob(Point2(100pd, 100pd), 20pd)]
obj = ImageWithBlobs(img, blobs; color=:red, scale_factor=3.0)
fig, _, _ = plot(obj)
```
"""
struct ImageWithBlobs{T<:AbstractMatrix, B<:Vector{<:AbstractBlob}}
    image::T
    blobs::B
    color::Symbol
    scale_factor::Float64
    marker::Symbol
    markersize::Float64
    linewidth::Float64
end

function ImageWithBlobs(image::AbstractMatrix, blobs::Vector{<:AbstractBlob};
                       color::Symbol=:green, scale_factor::Float64=3.0,
                       marker::Symbol=:cross, markersize::Float64=15.0, linewidth::Float64=1.0)
    return ImageWithBlobs(image, blobs, color, scale_factor, marker, markersize, linewidth)
end

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, obj::ImageWithBlobs)

Convert ImageWithBlobs to SpecApi PlotSpec vector for Makie recipe integration.
Returns a vector of PlotSpecs (Image + blob overlays) with proper y-axis orientation.
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, obj::ImageWithBlobs)
    # Create image PlotSpec with proper y-flip
    pattern_height, pattern_width = size(obj.image)
    image_spec = Spec.Image(transpose(obj.image), interpolate=false)

    # Create blob PlotSpecs
    blob_specs = plotblobs(obj.blobs; color=obj.color, scale_factor=obj.scale_factor,
                          marker=obj.marker, markersize=obj.markersize, linewidth=obj.linewidth)

    # Combine and return
    plots = Makie.PlotSpec[image_spec]
    append!(plots, blob_specs)

    return plots
end

# Composable plotting functions using Spec API

"""
    imshow(pattern; interpolate=false)

Create a standalone image display with proper orientation (y-axis flipped).

# Arguments
- `pattern`: Image pattern to display

# Keyword Arguments
- `interpolate=false`: Whether to interpolate between pixels

Returns an LScene BlockSpec with flipped y-limits to display image right-side up.
Additional plots can be added via `.plots`.

# Examples
```julia
# Standalone display (MATLAB-style)
plot(pattern)  # Uses imshow automatically via convert_arguments

# Manual display with imshow
lscene = imshow(pattern)

# Add overlays to the image
lscene = imshow(pattern)
append!(lscene.plots, plotblobs(blobs))  # Add blob overlays
```
"""
function imshow(pattern; interpolate=false)
    pattern_height, pattern_width = size(pattern)
    # Transpose to convert (row, col) to (x, y): pattern[row, col] -> pattern'[col, row]
    # This gives us (x, y) coordinates where x is column, y is row
    image_spec = Spec.Image(transpose(pattern), interpolate=interpolate)

    # We need to flip the y-axis so that row 1 (y=1) appears at the TOP
    # In Makie's default coordinate system, y increases upward (bottom to top)
    # In image coordinates, row index increases downward (top to bottom)
    # Solution: Apply transformation that flips y-axis

    # Use then_funcs to apply transformation after scene is created
    function flip_y_axis(lscene_block)
        scene = lscene_block.scene
        # Flip y-axis by scaling y by -1 and translating
        # For an image of height H, point at y=0 maps to y=H, point at y=H maps to y=0
        # Transformation: y' = -y + H (scale by -1, translate by H)
        scene.transformation.scale[] = (1.0, -1.0, 1.0)
        scene.transformation.translation[] = (0.0, Float64(pattern_height), 0.0)

        # Update camera to ensure it doesn't clip
        # The transformation is applied to the plotted data, so the camera sees
        # the transformed coordinates. We need to ensure the camera viewport is correct.
        notify(scene.camera.projectionview)
        return
    end

    lscene = Spec.LScene(plots=[image_spec], show_axis=false,
                        width=Fixed(pattern_width), height=Fixed(pattern_height),
                        tellwidth=false, tellheight=false,
                        scenekw=(camera=campixel!,
                                limits=(0, pattern_width, 0, pattern_height)))

    # Add callback to flip y-axis after scene is created
    push!(lscene.then_funcs, flip_y_axis)

    return lscene
end

"""
    plotblobs(blobs; color=:green, scale_factor=3.0, marker=:cross, markersize=15.0, linewidth=3.0)

Create plot specs for blob visualization (centers and scale circles).

# Arguments
- `blobs`: Vector of blob features

# Keyword Arguments
- `color=:green`: Color for blob visualization
- `scale_factor=3.0`: Scaling factor for blob circles
- `marker=:cross`: Marker style for blob centers
- `markersize=15.0`: Size of center markers
- `linewidth=3.0`: Width of circle outlines

Returns a vector of PlotSpec objects (scatter + circles) that can be composed.
"""
function plotblobs(blobs;
                   color=:green,
                   scale_factor::Float64=3.0,
                   marker=:cross,
                   markersize::Float64=15.0,
                   linewidth::Float64=1.0,
                   linestyle=:solid)
    plot_specs = Makie.PlotSpec[]

    if !isempty(blobs)
        centers = [ustrip.(blob.center) for blob in blobs]
        scales = [ustrip(blob.σ) for blob in blobs]

        # Add circles for blob scales using proper geometric circles
        if scale_factor > 0
            for (c, s) in zip(centers, scales)
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
                    color=color,
                    linewidth=linewidth
                )
                push!(plot_specs, foreground_circle)
            end

            # Center markers
            center_markers = Spec.Scatter(
                [c[1] for c in centers], [c[2] for c in centers];
                marker=marker,
                markersize=markersize,
                color=color
            )
            push!(plot_specs, center_markers)
        else
            # If no scale_factor, just plot center markers
            center_markers = Spec.Scatter(
                [c[1] for c in centers], [c[2] for c in centers];
                marker=marker,
                markersize=markersize,
                color=color
            )
            push!(plot_specs, center_markers)
        end
    end

    return plot_specs
end


using GLMakie



