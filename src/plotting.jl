# Plotting functionality for VisualGeometryCore types
# Moved from BlobBoards.jl to make available across all projects

"""
    PlotBlobs

Makie recipe for plotting blob features with optional scaling circles and polarity filtering.
"""
@recipe(PlotBlobs) do scene
    Attributes(
        scale_factor=3.0,
        marker=:cross,
        color=:green,
        markersize=15.0,
        strokewidth=3.0,
        alpha=1.0,
        polarity=nothing,
        plot_scale=true,
        plot_uncertainty=false
    )
end

function Makie.plot!(plt::PlotBlobs)
    blobs = plt[1][]
    features = if plt.polarity[] == PositiveFeature
        filter(f -> f.polarity == PositiveFeature, blobs)
    elseif plt.polarity[] == NegativeFeature
        filter(f -> f.polarity == NegativeFeature, blobs)
    else
        blobs
    end
    isempty(features) && return plt

    centers = [ustrip.(f.center) for f in features]
    scales = [ustrip(f.σ) for f in features]

    scatter!(plt, centers;
        markersize=plt.markersize[],
        color=plt.color[],
        alpha=plt.alpha[],
        marker=plt.marker[]
    )

    if plt.plot_scale[]
        circles = GeometryBasics.Circle[]
        for (c, s) in zip(centers, scales)
            point = (length(c) >= 2) ? [c[1], c[2]] : c
            radius = s * plt.scale_factor[]
            circle = GeometryBasics.Circle(GeometryBasics.Point2f(point...), radius)
            push!(circles, circle)
        end
        if !isempty(circles)
            poly!(plt, circles;
                color=:transparent,
                strokecolor=plt.color[],
                strokewidth=plt.strokewidth[],
                alpha=plt.alpha[]
            )
        end
    end
    return plt
end

using GLMakie

"""
    plot_detection_results(pattern, detected_blobs, ground_truth_blobs=nothing; kwargs...)

Plot blob detection results overlaid on a pattern image using GLMakie.

# Arguments
- `pattern`: Background image pattern
- `detected_blobs`: Vector of detected blob features
- `ground_truth_blobs=nothing`: Optional ground truth blobs for comparison

# Keyword Arguments
- `title="Detection Results"`: Plot title
- `size=(800, 800)`: Figure size
- `detected_color=:red`: Color for detected blobs
- `ground_truth_color=:green`: Color for ground truth blobs
- `scale_factor=3.0`: Scaling factor for blob circles

Returns a GLMakie Figure object.
"""
function plot_detection_results(pattern, detected_blobs, ground_truth_blobs=nothing;
                               title::String="Detection Results",
                               size::Tuple=(800, 800),
                               detected_color=:red,
                               ground_truth_color=:green,
                               scale_factor::Float64=3.0)
    @static if Base.find_package("GLMakie") !== nothing
        GLMakie.activate!()
        fig = GLMakie.Figure(size=size)
        ax = GLMakie.Axis(fig[1,1], title=title)

        # Display image with proper orientation for blob coordinates
        # Use rotr90 to match the coordinate system expected by blob data
        image!(ax, rotr90(pattern))

        # Set up axis for image coordinates (flip y-axis to match image convention)
        ax.yreversed = true
        ax.aspect = DataAspect()

        if ground_truth_blobs !== nothing
            plotblobs!(ax, ground_truth_blobs;
                color=ground_truth_color,
                scale_factor=scale_factor,
                alpha=0.7
            )
        end
        plotblobs!(ax, detected_blobs;
            color=detected_color,
            scale_factor=scale_factor
        )
        return fig
    else
        error("GLMakie not available. Install GLMakie for interactive visualization.")
    end
end

"""
    plot_pattern(pattern, blobs; kwargs...)

Render a pattern image and overlay blob detections using GLMakie.

# Arguments
- `pattern`: Background image pattern
- `blobs`: Vector of blob features to overlay

# Keyword Arguments
- `title="Pattern"`: Plot title
- `size=(800, 800)`: Figure size
- `color=:green`: Color for blob visualization
- `scale_factor=3.0`: Scaling factor for blob circles

Returns a GLMakie Figure object.
"""
function plot_pattern(pattern, blobs;
                     title::String="Pattern",
                     size::Tuple=(800, 800),
                     color=:green,
                     scale_factor::Float64=3.0)
    @static if Base.find_package("GLMakie") !== nothing
        GLMakie.activate!()
        fig = GLMakie.Figure(size=size)
        ax = GLMakie.Axis(fig[1,1], title=title)

        # Display image with proper orientation for blob coordinates
        # Use rotr90 to match the coordinate system expected by blob data
        image!(ax, rotr90(pattern))

        # Set up axis for image coordinates (flip y-axis to match image convention)
        ax.yreversed = true
        ax.aspect = DataAspect()

        # Plot blobs using the recipe
        plotblobs!(ax, blobs; color=color, scale_factor=scale_factor)
        return fig
    else
        error("GLMakie not available. Install GLMakie for interactive visualization.")
    end
end

"""
    plot_blob_pattern(pattern)

Create a simple image plot of a blob pattern using Makie Spec interface.

# Arguments
- `pattern`: Image pattern to display

Returns a Makie LScene object using the Spec API.
"""
function plot_blob_pattern(pattern)
    imshow = Makie.Spec.Image(rotr90(pattern); interpolate=false)
    scn = Makie.Spec.LScene(
        show_axis=false,
        plots=[imshow];
        # scenekw=(; camera=campixel!)  # TODO: Define campixel! camera
    )
    return scn
end