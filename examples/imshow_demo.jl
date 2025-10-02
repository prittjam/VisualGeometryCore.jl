"""
Example demonstrating VisualGeometryCore's imshow function with SpecApi.

This example shows:
1. Loading an image from TestImages
2. Displaying it with proper y-axis orientation using imshow
3. Adding blob overlays to the image
4. Composing multiple plot elements in a Figure using SpecApi
"""

using VisualGeometryCore
using VisualGeometryCore: imshow, plotblobs, IsoBlob, pd
using TestImages
using GLMakie
using GeometryBasics: Point2
import Makie.SpecApi as S


# Example: Manual y-flip using scene transformation (post-processing)
println("Manual Example 1: Y-flip via scene transformation...")
img = testimage("cameraman")
h, w = size(img)
ish = S.Image(img'; interpolate=false)
lscene = S.LScene(show_axis=false, plots=[ish])
layout = S.GridLayout([lscene])
f1, _, _ = plot(layout)

# Post-process: flip y-axis via scene transformation
for item in f1.layout.content
    ls = item.content
    if ls isa Makie.LScene
        ls.scene.transformation.scale[] = (1.0, -1.0, 1.0)
        ls.scene.transformation.translation[] = (0.0, Float64(h), 0.0)
    end
end
display(f1)

# Example: Y-flip using flipped limits (declarative, at spec level)
println("\nManual Example 2: Y-flip via flipped limits...")
using GeometryBasics: Rect2f
ish2 = S.Image(img'; interpolate=false)
# Use flipped y-limits: y goes from height to 0 instead of 0 to height
lscene2 = S.LScene(show_axis=false, plots=[ish2];
                   scenekw=(camera=campixel!, limits=Rect2f(0, w, h, 0)))
layout2 = S.GridLayout([lscene2])
f2, _, _ = plot(layout2)
display(f2)

# Example: Multiple scenes with flipped limits
println("\nManual Example 3: Multiple scenes with flipped limits...")
scenemat = [lscene2 lscene2; lscene2 lscene2]
layout3 = S.GridLayout(scenemat, colgaps=[Fixed(0)], rowgaps=[Fixed(0)])
f3, _, _ = plot(layout3)
display(f3)


# Load a test image
println("\n\nNow testing imshow function...")
println("Loading test image...")
img = testimage("cameraman")
println("Image size: ", size(img))

# Example 1: Simple imshow with SpecApi
println("\nExample 1: Simple image display")
lscene = imshow(img)
# Wrap LScene in GridLayout and plot the spec
layout = S.GridLayout([lscene])
fig1, _, _ = plot(layout)
display(fig1)

# Example 2: imshow with interpolation
println("\nExample 2: Image display with interpolation")
lscene_interp = imshow(img; interpolate=true)
layout_interp = S.GridLayout([lscene_interp])
fig2, _, _ = plot(layout_interp)
display(fig2)

# Example 3: Image with blob overlays
println("\nExample 3: Image with blob overlays")
# Create some example blobs at interesting points
height, width = size(img)
blobs = [
    IsoBlob(Point2(width/4*pd, height/4*pd), 20.0pd),
    IsoBlob(Point2(3*width/4*pd, height/4*pd), 15.0pd),
    IsoBlob(Point2(width/2*pd, height/2*pd), 25.0pd),
    IsoBlob(Point2(width/4*pd, 3*height/4*pd), 18.0pd),
    IsoBlob(Point2(3*width/4*pd, 3*height/4*pd), 22.0pd),
]

# Create image display and add blob overlays
lscene_with_blobs = imshow(img)
blob_specs = plotblobs(blobs; color=:cyan, scale_factor=3.0, linewidth=2.0)
append!(lscene_with_blobs.plots, blob_specs)
# Wrap in GridLayout and plot the spec
layout_blobs = S.GridLayout([lscene_with_blobs])
fig3, _, _ = plot(layout_blobs)
display(fig3)

# Example 4: Multiple blob layers with different colors
println("\nExample 4: Multiple blob layers")
# Create two sets of blobs
large_blobs = [
    IsoBlob(Point2(width/3*pd, height/3*pd), 30.0pd),
    IsoBlob(Point2(2*width/3*pd, 2*height/3*pd), 35.0pd),
]

small_blobs = [
    IsoBlob(Point2(width/6*pd, height/6*pd), 10.0pd),
    IsoBlob(Point2(5*width/6*pd, height/6*pd), 12.0pd),
    IsoBlob(Point2(width/2*pd, 5*height/6*pd), 11.0pd),
]

# Compose them on the image
lscene_multi = imshow(img)
append!(lscene_multi.plots, plotblobs(large_blobs; color=:red, scale_factor=3.0))
append!(lscene_multi.plots, plotblobs(small_blobs; color=:green, scale_factor=3.0))
# Wrap in GridLayout and plot the spec
layout_multi = S.GridLayout([lscene_multi])
fig4, _, _ = plot(layout_multi)
display(fig4)

println("\n✓ All examples completed!")
println("\nKey features demonstrated:")
println("  • imshow provides correct y-axis orientation automatically")
println("  • Returns LScene BlockSpec that can be wrapped in S.GridLayout")
println("  • Use plot(layout) to create Figure from spec")
println("  • Direct access to lscene.plots for adding overlays")
println("  • Multiple plot layers can be added incrementally")

println("\nUsage pattern:")
println("  1. lscene = imshow(img)")
println("  2. append!(lscene.plots, plotblobs(blobs))")
println("  3. layout = S.GridLayout([lscene])")
println("  4. fig, _, _ = plot(layout)")
