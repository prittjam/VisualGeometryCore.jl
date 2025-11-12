# =============================================================================
# Image Processing Utilities
# =============================================================================

"""
    vlfeat_upsample(src::Matrix{Gray{T}}) where T

Upsample image using VLFeat's bilinear interpolation method.

This matches the `copy_and_upsample` function from VLFeat scalespace.c,
which creates a 2x upsampled image using the following pattern for each pixel:
- dst[0] = v00
- dst[1] = 0.5 * (v00 + v10)  # horizontal neighbor
- dst[width*2] = 0.5 * (v00 + v01)  # vertical neighbor
- dst[width*2+1] = 0.25 * (v00 + v01 + v10 + v11)  # diagonal

# Arguments
- `src`: Source grayscale image as Matrix{Gray{T}}

# Returns
- Upsampled image as Matrix{Gray{Float32}} with dimensions 2x the input

# Examples
```julia
using Colors
img = rand(Gray{Float32}, 32, 32)
upsampled = vlfeat_upsample(img)  # Returns 64×64 image
```
"""
function vlfeat_upsample(src::Matrix{Gray{T}}) where T
    height, width = size(src)
    dst = Matrix{Gray{Float32}}(undef, height * 2, width * 2)

    for y in 1:height
        oy = (y < height) ? width : 0
        v10 = src[y, 1]
        v11 = (y < height) ? src[y+1, 1] : src[y, 1]

        for x in 1:width
            ox = (x < width) ? 1 : 0
            v00 = v10
            v01 = v11
            v10 = src[y, x + ox]
            v11 = (y < height) ? src[y+1, x + ox] : src[y, x + ox]

            # Convert to Float32 for arithmetic
            f00, f01, f10, f11 = Float32(v00.val), Float32(v01.val), Float32(v10.val), Float32(v11.val)

            dst_y = 2*y - 1
            dst_x = 2*x - 1
            dst[dst_y, dst_x] = Gray{Float32}(f00)
            dst[dst_y, dst_x+1] = Gray{Float32}(0.5f0 * (f00 + f10))
            dst[dst_y+1, dst_x] = Gray{Float32}(0.5f0 * (f00 + f01))
            dst[dst_y+1, dst_x+1] = Gray{Float32}(0.25f0 * (f00 + f01 + f10 + f11))
        end
    end

    return dst
end

"""
    vlfeat_upsample!(dst::AbstractMatrix{Gray{Float32}}, src::Matrix{Gray{T}}) where T

Mutating version of VLFeat's bilinear upsampling that writes directly into pre-allocated destination.

This matches the `copy_and_upsample` function from VLFeat scalespace.c,
which creates a 2x upsampled image using the following pattern for each pixel:
- dst[0] = v00
- dst[1] = 0.5 * (v00 + v10)  # horizontal neighbor
- dst[width*2] = 0.5 * (v00 + v01)  # vertical neighbor
- dst[width*2+1] = 0.25 * (v00 + v01 + v10 + v11)  # diagonal
"""
function vlfeat_upsample!(dst::AbstractMatrix{Gray{Float32}}, src::Matrix{Gray{T}}) where T
    height, width = size(src)
    @assert size(dst) == (height * 2, width * 2) "Destination must be 2x the source size"

    for y in 1:height
        oy = (y < height) ? width : 0
        v10 = src[y, 1]
        v11 = (y < height) ? src[y+1, 1] : src[y, 1]

        for x in 1:width
            ox = (x < width) ? 1 : 0
            v00 = v10
            v01 = v11
            v10 = src[y, x + ox]
            v11 = (y < height) ? src[y+1, x + ox] : src[y, x + ox]

            # Convert to Float32 for arithmetic
            f00, f01, f10, f11 = Float32(v00.val), Float32(v01.val), Float32(v10.val), Float32(v11.val)

            dst_y = 2*y - 1
            dst_x = 2*x - 1
            dst[dst_y, dst_x] = Gray{Float32}(f00)
            dst[dst_y, dst_x+1] = Gray{Float32}(0.5f0 * (f00 + f10))
            dst[dst_y+1, dst_x] = Gray{Float32}(0.5f0 * (f00 + f01))
            dst[dst_y+1, dst_x+1] = Gray{Float32}(0.25f0 * (f00 + f01 + f10 + f11))
        end
    end

    return dst
end
