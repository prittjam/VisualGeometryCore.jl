# Plotting functionality for VisualGeometryCore types
# Uses Makie Spec API with convert_arguments for composable plotting

import Makie.SpecApi as MakieSpec

# Convert arguments for images using SpecApi
# Note: For image+blob overlays, use SpecApi composition:
#   import VisualGeometryCore.Spec as S
#   lscene = S.Imshow(image)
#   circles = Circle.(blobs, Ref(3.0))  # Convert blobs to circles
#   blob_specs = S.Locus.(circles; color=:red, linewidth=2.0)
#   append!(lscene.plots, blob_specs)
#   fig, _, _ = plot(MakieSpec.GridLayout([lscene]))

# Direct convert_arguments for image matrices
# Returns GridLayout with y-flipped LScene for proper image display
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}) = (:interpolate,)

function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant};
                                 interpolate=false)
    # Use Spec.Imshow to get proper y-axis flipping
    lscene = Spec.Imshow(img; interpolate=interpolate)
    return MakieSpec.GridLayout([lscene]; rowgaps=Makie.Fixed(0), colgaps=Makie.Fixed(0))
end

# Multi-argument convert_arguments for image + blobs composition
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:AbstractBlob}) = (:interpolate, :color, :sigma_cutoff, :marker, :markersize, :linewidth)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, blobs::Vector{<:AbstractBlob}; kwargs...)

Convert image and blobs to composed GridLayout with overlay.
Composes by calling through to Spec.Imshow() and Spec.Locus() (after converting blobs to circles).

# Examples
```julia
img = testimage("cameraman")
blobs = [IsoBlob(Point2(100pd, 100pd), 20pd)]
fig, ax, pl = plot(img, blobs; color=:red, sigma_cutoff=3.0)
```
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, blobs::Vector{<:AbstractBlob};
                                 interpolate=false, color=:green, sigma_cutoff::Float64=3.0,
                                 marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0)
    # Compose by calling Spec functions
    lscene = Spec.Imshow(img; imagekw=(interpolate=interpolate,))
    # Convert blobs to circles for visualization
    circles = Circle.(blobs, Ref(sigma_cutoff))
    blob_specs = Spec.Locus.(circles; color=color, linewidth=linewidth)
    append!(lscene.plots, blob_specs)
    return MakieSpec.GridLayout([lscene]; rowgaps=Makie.Fixed(0), colgaps=Makie.Fixed(0))
end

# Three-argument convert_arguments for image + detections + ground truth
Makie.used_attributes(::Type{<:Plot}, ::AbstractMatrix{<:Colorant}, ::Vector{<:AbstractBlob}, ::Vector{<:AbstractBlob}) =
    (:interpolate, :detection_color, :ground_truth_color, :sigma_cutoff, :detection_marker, :ground_truth_marker,
     :detection_markersize, :ground_truth_markersize, :linewidth, :detection_linestyle, :ground_truth_linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant}, detections::Vector{<:AbstractBlob}, ground_truth::Vector{<:AbstractBlob}; kwargs...)

Convert image, detected blobs, and ground truth blobs to composed GridLayout with overlays.

# Examples
```julia
img = testimage("cameraman")
detections = [IsoBlob(Point2(100pd, 100pd), 20pd)]
ground_truth = [IsoBlob(Point2(105pd, 102pd), 18pd)]
fig, ax, pl = plot(img, detections, ground_truth; sigma_cutoff=3.0)
```
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, img::AbstractMatrix{<:Colorant},
                                 detections::Vector{<:AbstractBlob}, ground_truth::Vector{<:AbstractBlob};
                                 interpolate=false,
                                 detection_color=:blue, ground_truth_color=:green,
                                 sigma_cutoff::Float64=3.0,
                                 detection_marker=:circle, ground_truth_marker=:cross,
                                 detection_markersize::Float64=8.0, ground_truth_markersize::Float64=10.0,
                                 linewidth::Float64=1.0,
                                 detection_linestyle=:dash, ground_truth_linestyle=:solid)
    # Create base image
    lscene = Spec.Imshow(img; imagekw=(interpolate=interpolate,))

    # Add ground truth blobs (green, solid) - convert to circles first
    gt_circles = Circle.(ground_truth, Ref(sigma_cutoff))
    gt_specs = Spec.Locus.(gt_circles; color=ground_truth_color,
                          linewidth=linewidth, linestyle=ground_truth_linestyle)
    append!(lscene.plots, gt_specs)

    # Add detection blobs (blue, dashed) - convert to circles first
    det_circles = Circle.(detections, Ref(sigma_cutoff))
    det_specs = Spec.Locus.(det_circles; color=detection_color,
                           linewidth=linewidth, linestyle=detection_linestyle)
    append!(lscene.plots, det_specs)

    return MakieSpec.GridLayout([lscene]; rowgaps=Makie.Fixed(0), colgaps=Makie.Fixed(0))
end

# Convert arguments for PlotSpec integration with recipes
# Atomic convert_arguments for blobs only
Makie.used_attributes(::Type{<:Plot}, ::Vector{<:AbstractBlob}) = (:color, :sigma_cutoff, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob}; color=:green, sigma_cutoff=3.0, ...)

Convert blobs to PlotSpec vector for Makie recipe integration.
Allows users to call: plot(blobs; color=:red, sigma_cutoff=2.0) or plot!(blobs; ...)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, blobs::Vector{<:AbstractBlob};
                                color=:green, sigma_cutoff::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0,
                                linewidth::Float64=1.0, linestyle=:solid)
    # Convert blobs to circles and return PlotSpec vector
    circles = Circle.(blobs, Ref(sigma_cutoff))
    return Spec.Locus.(circles; color=color,
                      linewidth=linewidth, linestyle=linestyle)
end

# Atomic convert_arguments for blob detections with dashed outlines
Makie.used_attributes(::Type{<:Plot}, ::Vector{IsoBlobDetection}) = (:color, :sigma_cutoff, :marker, :markersize, :linewidth, :linestyle)

"""
    Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection}; color=:blue, linestyle=:dash, ...)

Convert blob detections to PlotSpec vector with dashed outlines for Makie recipe integration.
Allows users to call: plot(detections; color=:blue, linestyle=:dash) or plot!(detections; ...)
"""
function Makie.convert_arguments(::Type{<:AbstractPlot}, detections::Vector{IsoBlobDetection};
                                color=:blue, sigma_cutoff::Float64=3.0,
                                marker=:cross, markersize::Float64=15.0, linewidth::Float64=1.0,
                                linestyle=:dash)
    # IsoBlobDetection is already an AbstractBlob, convert to circles
    circles = Circle.(detections, Ref(sigma_cutoff))
    return Spec.Locus.(circles; color=color,
                      linewidth=linewidth, linestyle=linestyle)
end

# =============================================================================
# SpecApi Module - Composable Plotting Functions
# =============================================================================

"""
Module for SpecApi-style composable plotting functions.

Import with: `import VisualGeometryCore.Spec`
Or use as: `using VisualGeometryCore; import VisualGeometryCore.Spec as S`

Provides:
- `Spec.Imshow()` - Image display with y-axis flipping
- `Spec.Locus()` - Geometric locus visualization for circles and ellipses (use with blobs: `Locus.(Circle.(blobs, sigma_cutoff))`)
"""
