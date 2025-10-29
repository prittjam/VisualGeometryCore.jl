# BlobBoards Examples

This directory contains example scripts demonstrating how to use BlobBoards for generating and detecting blob patterns.

## Detection Examples

### Pattern Detection Demo

**File:** `pattern_detection_demo.jl`

Demonstrates blob detection on a pure pattern (no borders):
- Loads test pattern from `test/data/blob_pattern_KQn.png`
- Detects blobs using Hessian-Laplace detector
- Compares detections with ground truth
- Visualizes results in three panels: ground truth, detected, and overlay

**To run:**
```julia
julia> include("examples/pattern_detection_demo.jl")
```

**What you'll see:**
- Three visualization panels showing ground truth, detected blobs, and overlay
- Detection statistics including match rate, position accuracy, and scale accuracy
- Interactive Makie figure for zooming and inspection

---

### Board Detection Demo

**File:** `board_detection_demo.jl`

Demonstrates blob detection on a full board (with borders, rulers, and QR codes):
- Generates a small A5 board for testing
- Detects blobs on the full board image
- Handles coordinate transformations between pattern and board frames
- Computes detection statistics including false positives

**To run:**
```julia
julia> include("examples/board_detection_demo.jl")
```

**What you'll see:**
- Three visualization panels: ground truth, detected, and overlay
- Detection statistics including false positives and missed detections
- Board files saved in a temporary directory for manual inspection

**Note:** The board borders and rulers may introduce false positive detections that are not in the pattern area.

---

## Other Examples

### Scalespace Demo
**File:** `scalespace_demo.jl`
Demonstrates scale-space pyramid construction and visualization.

### Gaussian Filtering Demo
**File:** `gaussian_filtering_demo.jl`
Shows Gaussian filtering at different scales.

### Ellipse Transforms Demo
**File:** `ellipse_transforms_demo.jl`
Demonstrates ellipse transformations and affine covariant regions.

### Imshow Demo
**File:** `imshow_demo.jl`
Shows basic image visualization with the `imshow` recipe.

---

## Key Concepts

### Pixel Conventions
The examples use `pixel_convention` parameter to specify coordinate systems:
- `:makie` - First pixel center at (0.5, 0.5), corner at (0, 0)
- `:colmap` / `:opencv` - First pixel center at (0, 0), corner at (-0.5, -0.5)
- `:matlab` - First pixel center at (1, 1), corner at (0.5, 0.5)

**Important:** Ground truth and detector must use the same convention for accurate comparison!

### Reference Frames (Board Only)
For board images, specify the coordinate origin:
- `:pattern` - Origin at pattern area top-left (default)
- `:board` - Origin at full board top-left (includes borders)

Use `:board` when detecting on the full board image.

### Blob Visualization
The Makie recipes automatically handle visualization:
```julia
# Simple blob plotting
blobs!(ax, blobs; color=:red, linewidth=2)

# With labels
blobs!(ax, blobs; color=:green, label="Ground Truth")
axislegend(ax)
```

---

## Detection Parameters

The Hessian-Laplace detector parameters used in examples:
- `peak_threshold=0.003` - Minimum response strength
- `edge_threshold=10.0` - Edge suppression ratio
- `first_octave=-1` - Start at half resolution
- `octave_resolution=3` - Scales per octave
- `base_scale=2.015874` - Initial scale for pyramid

Adjust these for your specific use case!

---

## Tips

1. **Zoom in** on the Makie figures to inspect individual blobs
2. **Adjust thresholds** if you get too many/few detections
3. **Match conventions** - ensure ground truth and detector use same `pixel_convention`
4. **Check coordinates** - board detection requires `reference_frame=:board`
5. **Save figures** using Makie's save functionality: `save("output.png", fig)`
