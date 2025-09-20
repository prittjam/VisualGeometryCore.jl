# Plotting functionality for VisualGeometryCore types
# Uses Makie Spec API with convert_arguments for composable plotting

import Makie.SpecApi as Spec
using Makie: Fixed
using GeometryBasics: Circle, Point2f

# Convert arguments for PlotSpec integration with recipes
# Atomic convert_arguments for blobs only
Makie.used_attributes(::Vector{<:AbstractBlob}) = (:color, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{Plot{plot}}, blobs::Vector{<:AbstractBlob}; kwargs...)

Convert blobs to PlotSpec for Makie recipe integration.
Allows users to call: plot(blobs; color=:red, scale_factor=2.0)
"""
function Makie.convert_arguments(::Type{Plot{plot}}, blobs::Vector{<:AbstractBlob}) where {plot}
    # Extract centers and scales for standard Makie plotting
    if isempty(blobs)
        return (Float64[], Float64[])
    end

    centers = [ustrip.(blob.center) for blob in blobs]
    x_coords = [c[1] for c in centers]
    y_coords = [c[2] for c in centers]

    return (x_coords, y_coords)
end

# Atomic convert_arguments for blob detections with dashed outlines
Makie.used_attributes(::Vector{IsoBlobDetection}) = (:color, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{Plot{plot}}, detections::Vector{IsoBlobDetection}; kwargs...)

Convert blob detections to PlotSpec with dashed outlines for Makie recipe integration.
Allows users to call: plot(detections; color=:blue, linestyle=:dash)
"""
function Makie.convert_arguments(::Type{Plot{plot}}, detections::Vector{IsoBlobDetection};
                                color=:blue, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0,
                                linestyle=:dash) where {plot}
    # Extract IsoBlobs from detections and call through to AbstractBlob implementation
    blobs = [det.blob for det in detections]
    return Makie.convert_arguments(Plot{plot}, blobs; color=color, scale_factor=scale_factor,
                                  marker=marker, markersize=markersize, linewidth=linewidth,
                                  linestyle=linestyle)
end

# Tuple-based convert_arguments for pattern and blobs
Makie.used_attributes(::Tuple{Any, Vector{<:AbstractBlob}}) = (:color, :scale_factor, :marker, :markersize, :linewidth)

"""
    Makie.convert_arguments(::Type{Plot{plot}}, pattern, blobs::Vector{<:AbstractBlob}; kwargs...)

Convert pattern and blobs tuple to PlotSpec for Makie recipe integration.
Allows users to call: plot(pattern, blobs; color=:red, scale_factor=2.0)
"""
function Makie.convert_arguments(::Type{Plot{plot}}, pattern, blobs::Vector{<:AbstractBlob};
                                color=:green, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0) where {plot}
    # Get pattern dimensions
    pattern_height, pattern_width = size(pattern)

    # Create image spec (no kwargs needed)
    image_spec = imshow(pattern)
    # Use plotblobs function with blob-specific kwargs only
    blob_specs = plotblobs(blobs; color=color, scale_factor=scale_factor, marker=marker, markersize=markersize, linewidth=linewidth)
    all_specs = [image_spec, blob_specs...]
    return Spec.GridLayout([Spec.LScene(plots=all_specs, show_axis=false,
                                       width=Fixed(pattern_width), height=Fixed(pattern_height),
                                       scenekw=(camera=campixel!, yreversed=true))])
end

# Tuple-based convert_arguments for detection results (pattern, detected_blobs, ground_truth_blobs)
Makie.used_attributes(::Tuple{Any, Vector{<:AbstractBlob}, Union{Vector{<:AbstractBlob}, Nothing}}) = (:detected_color, :ground_truth_color, :scale_factor)

"""
    Makie.convert_arguments(::Type{Plot{plot}}, pattern, detected_blobs::Vector{<:AbstractBlob}, ground_truth_blobs; kwargs...)

Convert pattern, detected blobs, and ground truth blobs tuple to PlotSpec for Makie recipe integration.
Allows users to call: plot(pattern, detected_blobs, ground_truth_blobs; detected_color=:red, ground_truth_color=:green)
"""
function Makie.convert_arguments(::Type{Plot{plot}}, pattern, detected_blobs::Vector{<:AbstractBlob},
                                ground_truth_blobs::Union{Vector{<:AbstractBlob}, Nothing};
                                detected_color=:red, ground_truth_color=:green,
                                scale_factor::Float64=3.0) where {plot}
    # Get pattern dimensions
    pattern_height, pattern_width = size(pattern)

    # Use atomic functions
    all_specs = [imshow(pattern)]

    # Add ground truth blobs with alpha (use plotblobs function)
    if ground_truth_blobs !== nothing && !isempty(ground_truth_blobs)
        gt_specs = plotblobs(ground_truth_blobs;
                            color=(ground_truth_color, 0.7),
                            scale_factor=scale_factor)
        append!(all_specs, gt_specs)
    end

    # Add detected blobs (use plotblobs function)
    detected_specs = plotblobs(detected_blobs;
                              color=detected_color,
                              scale_factor=scale_factor)
    append!(all_specs, detected_specs)

    return Spec.GridLayout([Spec.LScene(plots=all_specs, show_axis=false,
                                       width=Fixed(pattern_width), height=Fixed(pattern_height),
                                       scenekw=(camera=campixel!, yreversed=true))])
end

# Composable plotting functions using Spec API

"""
    imshow(pattern; interpolate=false)

Create an image plot spec for displaying a pattern.

# Arguments
- `pattern`: Image pattern to display

# Keyword Arguments
- `interpolate=false`: Whether to interpolate between pixels

Returns a Spec.Image object that can be composed with other plot elements.
"""
function imshow(pattern; interpolate=false)
    return Spec.Image(transpose(pattern), interpolate=interpolate)
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



