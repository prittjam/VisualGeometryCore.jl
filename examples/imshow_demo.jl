"""
Example demonstrating VisualGeometryCore's imshow function with SpecApi.

This example shows:
1. Loading an image from TestImages
2. Displaying it with proper y-axis orientation using imshow
3. Adding blob overlays to the image
"""

using VisualGeometryCore
using VisualGeometryCore: imshow, plotblobs, IsoBlob, pd
using TestImages
using GLMakie
using GeometryBasics: Point2
import Makie.SpecApi as S

# Load a test image
println("Loading test image...")
img = testimage("cameraman")
height, width = size(img)
println("Image size: ", (height, width))

# Example 1: Simple imshow
println("\nExample 1: Simple image display")
lscene = imshow(img)
layout = S.GridLayout([lscene]; rowgaps=Fixed(0), colgaps=Fixed(0))
fig1 = Figure(; size=(width, height))
plot!(fig1, layout)
display(fig1)

# Example 2: Image with blob overlays
println("\nExample 2: Image with blob overlays")
blobs = [
    IsoBlob(Point2(width/4*pd, height/4*pd), 20.0pd),
    IsoBlob(Point2(3*width/4*pd, height/4*pd), 15.0pd),
    IsoBlob(Point2(width/2*pd, height/2*pd), 25.0pd),
]

lscene_with_blobs = imshow(img)
blob_specs = plotblobs(blobs; color=:cyan, scale_factor=3.0, linewidth=2.0)
append!(lscene_with_blobs.plots, blob_specs)
layout_blobs = S.GridLayout([lscene_with_blobs]; rowgaps=Fixed(0), colgaps=Fixed(0))
fig2 = Figure(; size=(width, height))
plot!(fig2, layout_blobs)
display(fig2)

println("\n✓ Examples completed!")
println("\nUsage pattern:")
println("  1. lscene = imshow(img)")
println("  2. append!(lscene.plots, plotblobs(blobs))")
println("  3. layout = S.GridLayout([lscene]; rowgaps=Fixed(0), colgaps=Fixed(0))")
println("  4. fig = Figure(; size=(width, height))")
println("  5. plot!(fig, layout)")
