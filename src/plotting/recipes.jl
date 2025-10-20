# ========================================================================
# Makie Recipe Integration - convert_arguments methods
# ========================================================================

# Convert arguments for images using SpecApi
# Note: For image+blob overlays, use SpecApi composition:
#   lscene = imshow(image)
#   blob_specs = plotblobs(blobs; color=:red, scale_factor=3.0)
#   append!(lscene.plots, blob_specs)
#   fig, _, _ = plot(S.GridLayout([lscene]))

# Direct convert_arguments for image matrices
# Returns GridLayout with y-flipped LScene for proper image display
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}) = (:interpolate,)

function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant};
                                 interpolate=false)
    # Use imshow to get proper y-axis flipping
    lscene = imshow(img; interpolate=interpolate)
    return Spec.GridLayout([lscene]; rowgaps=MakieFixed(0), colgaps=MakieFixed(0))
end

# ========================================================================
# Blob Plotting Recipes
# ========================================================================

# Multi-argument convert_arguments for image + blobs composition
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:AbstractBlob}) = (:interpolate, :color, :colormap, :scale_factor, :marker, :markersize, :linewidth)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, blobs::Vector{<:AbstractBlob}; kwargs...)

Convert image and blobs to composed GridLayout with overlay.
Composes by calling through to imshow() and plotblobs().

# Examples
```julia
img = testimage("cameraman")
blobs = [IsoBlob(Point2(100pd, 100pd), 20pd)]
fig, ax, pl = plot(img, blobs; color=:red, scale_factor=3.0)
fig, ax, pl = plot(img, blobs; colormap=:viridis, scale_factor=3.0)
```
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, blobs::Vector{<:AbstractBlob};
                                 interpolate=false, color=:green, colormap=nothing, scale_factor::Float64=3.0,
                                 marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0)
    # Compose by calling through to simpler functions
    lscene = imshow(img; interpolate=interpolate)
    blob_specs = plotblobs(blobs; color=color, colormap=colormap, scale_factor=scale_factor,
                          marker=marker, markersize=markersize, linewidth=linewidth)
    append!(lscene.plots, blob_specs)
    return Spec.GridLayout([lscene]; rowgaps=MakieFixed(0), colgaps=MakieFixed(0))
end

# Three-argument convert_arguments for image + detections + ground truth
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:AbstractBlob}, ::Vector{<:AbstractBlob}) =
    (:interpolate, :detection_color, :ground_truth_color, :scale_factor, :detection_marker, :ground_truth_marker,
     :detection_markersize, :ground_truth_markersize, :linewidth, :detection_linestyle, :ground_truth_linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, detections::Vector{<:AbstractBlob}, ground_truth::Vector{<:AbstractBlob}; kwargs...)

Convert image, detected blobs, and ground truth blobs to composed GridLayout with overlays.

# Examples
```julia
img = testimage("cameraman")
detections = [IsoBlob(Point2(100pd, 100pd), 20pd)]
ground_truth = [IsoBlob(Point2(105pd, 102pd), 18pd)]
fig, ax, pl = plot(img, detections, ground_truth; scale_factor=3.0)
```
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant},
                                 detections::Vector{<:AbstractBlob}, ground_truth::Vector{<:AbstractBlob};
                                 interpolate=false,
                                 detection_color=:blue, ground_truth_color=:green,
                                 scale_factor::Float64=3.0,
                                 detection_marker=:circle, ground_truth_marker=:cross,
                                 detection_markersize::Float64=8.0, ground_truth_markersize::Float64=10.0,
                                 linewidth::Float64=1.0,
                                 detection_linestyle=:dash, ground_truth_linestyle=:solid)
    # Create base image
    lscene = imshow(img; interpolate=interpolate)

    # Add ground truth blobs (green, solid)
    gt_specs = plotblobs(ground_truth; color=ground_truth_color, scale_factor=scale_factor,
                        marker=ground_truth_marker, markersize=ground_truth_markersize,
                        linewidth=linewidth, linestyle=ground_truth_linestyle)
    append!(lscene.plots, gt_specs)

    # Add detection blobs (blue, dashed)
    det_specs = plotblobs(detections; color=detection_color, scale_factor=scale_factor,
                         marker=detection_marker, markersize=detection_markersize,
                         linewidth=linewidth, linestyle=detection_linestyle)
    append!(lscene.plots, det_specs)

    return Spec.GridLayout([lscene]; rowgaps=MakieFixed(0), colgaps=MakieFixed(0))
end

# Convert arguments for PlotSpec integration with recipes
# Atomic convert_arguments for blobs only
Makie.used_attributes(::Type{<:Plot}, ::Vector{<:AbstractBlob}) = (:color, :colormap, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob}; color=:green, colormap=nothing, scale_factor=3.0, ...)

Convert blobs to PlotSpec vector for Makie recipe integration.
Allows users to call: plot(blobs; color=:red, scale_factor=2.0) or plot!(blobs; ...)
or with colormap: plot(blobs; colormap=:viridis, scale_factor=2.0)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob};
                                color=:green, colormap=nothing, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0,
                                linewidth::Float64=1.0, linestyle=:solid)
    # Return PlotSpec vector for SpecApi integration
    return plotblobs(blobs; color=color, colormap=colormap, scale_factor=scale_factor,
                    marker=marker, markersize=markersize,
                    linewidth=linewidth, linestyle=linestyle)
end

# Atomic convert_arguments for blob detections with dashed outlines
Makie.used_attributes(::Type{<:Plot}, ::Vector{IsoBlobDetection}) = (:color, :colormap, :scale_factor, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection}; color=:blue, colormap=nothing, linestyle=:dash, ...)

Convert blob detections to PlotSpec vector with dashed outlines for Makie recipe integration.
Allows users to call: plot(detections; color=:blue, linestyle=:dash) or plot!(detections; ...)
or with colormap: plot(detections; colormap=:viridis, linestyle=:dash)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection};
                                color=:blue, colormap=nothing, scale_factor::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0,
                                linestyle=:dash)
    # Extract IsoBlobs from detections and call plotblobs
    blobs = [det.blob for det in detections]
    return plotblobs(blobs; color=color, colormap=colormap, scale_factor=scale_factor,
                    marker=marker, markersize=markersize,
                    linewidth=linewidth, linestyle=linestyle)
end

# ========================================================================
# Ellipse Plotting Recipes
# ========================================================================

# Convert arguments for ellipses using SpecApi
Makie.used_attributes(::Type{<:Plot}, ::Vector{<:Ellipse}) = (:color, :linewidth, :linestyle, :fillcolor, :fillalpha, :resolution)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, ellipses::Vector{<:Ellipse}; kwargs...)

Convert ellipses to PlotSpec vector for Makie recipe integration.
Allows users to call: plot(ellipses; color=:red, linewidth=2.0) or plot!(ellipses; ...)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, ellipses::Vector{<:Ellipse};
                                color=:blue, linewidth::Float64=2.0, linestyle=:solid,
                                fillcolor=nothing, fillalpha::Float64=0.2,
                                resolution::Int=64)
    return plotellipses(ellipses; color=color, linewidth=linewidth, linestyle=linestyle,
                       fillcolor=fillcolor, fillalpha=fillalpha, resolution=resolution)
end

# Single ellipse convenience
Makie.used_attributes(::Type{<:Plot}, ::Ellipse) = (:color, :linewidth, :linestyle, :fillcolor, :fillalpha, :resolution)

function Makie.convert_arguments(::Type{<:AbstractPlot}, ellipse::Ellipse;
                                color=:blue, linewidth::Float64=2.0, linestyle=:solid,
                                fillcolor=nothing, fillalpha::Float64=0.2,
                                resolution::Int=64)
    return plotellipses([ellipse]; color=color, linewidth=linewidth, linestyle=linestyle,
                       fillcolor=fillcolor, fillalpha=fillalpha, resolution=resolution)
end

# Multi-argument convert_arguments for image + ellipses composition
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:Ellipse}) = 
    (:interpolate, :color, :linewidth, :linestyle, :fillcolor, :fillalpha, :resolution)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, ellipses::Vector{<:Ellipse}; kwargs...)

Convert image and ellipses to composed GridLayout with overlay.
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, ellipses::Vector{<:Ellipse};
                                interpolate=false, color=:red, linewidth::Float64=2.0, linestyle=:solid,
                                fillcolor=nothing, fillalpha::Float64=0.2, resolution::Int=64)
    # Compose by calling through to simpler functions
    lscene = imshow(img; interpolate=interpolate)
    ellipse_specs = plotellipses(ellipses; color=color, linewidth=linewidth, linestyle=linestyle,
                                fillcolor=fillcolor, fillalpha=fillalpha, resolution=resolution)
    append!(lscene.plots, ellipse_specs)
    return Spec.GridLayout([lscene]; rowgaps=MakieFixed(0), colgaps=MakieFixed(0))
end