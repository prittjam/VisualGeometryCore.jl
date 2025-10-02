"""
Example demonstrating VisualGeometryCore's plotting with sequential plotting.

This example shows two ways to plot images and blobs:
1. Simple image display via convert_arguments
2. Sequential plotting: plot image first, then add blobs with plot!
"""

using VisualGeometryCore
using VisualGeometryCore: IsoBlob, pd
using TestImages
using GLMakie
using GeometryBasics: Point2

# Load a test image
println("Loading test image...")
img = testimage("cameraman")
height, width = size(img)
println("Image size: ", (height, width))

# Example 1: Simple image display using convert_arguments
println("\nExample 1: Simple image display - plot(img)")
fig1, _, _ = plot(img; size=(width, height))
display(fig1)

# Example 2: Sequential plotting using SpecApi - plot image, then add blob overlay
println("\nExample 2: Sequential SpecApi - plot image then add blobs")
blobs = [
    IsoBlob(Point2(width/4*pd, height/4*pd), 20.0pd),
    IsoBlob(Point2(3*width/4*pd, height/4*pd), 15.0pd),
    IsoBlob(Point2(width/2*pd, height/2*pd), 25.0pd),
]

using VisualGeometryCore: imshow, plotblobs
import Makie.SpecApi as S
using GLMakie: Fixed

# Build the plot sequentially with SpecApi
lscene = imshow(img)
blob_specs = plotblobs(blobs; color=:cyan, scale_factor=3.0)
append!(lscene.plots, blob_specs)
layout = S.GridLayout([lscene]; rowgaps=Fixed(0), colgaps=Fixed(0))
fig2, _, _ = plot(layout; size=(width, height))
display(fig2)

println("\n✓ Examples completed!")
println("\nUsage patterns:")
println("  plot(img)                                  # Display image")
println("  lscene = imshow(img)                       # Build with SpecApi")
println("  append!(lscene.plots, plotblobs(blobs))    # Add blobs sequentially")
