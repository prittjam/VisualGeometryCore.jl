"""
VLFeat-compatible response computations.

This module provides response functions that exactly match VLFeat's implementations,
including proper scale normalization.
"""

using ImageFiltering: imfilter, centered
using ImageCore: channelview
using StaticArrays

"""
    vlfeat_hessian_det!(det_out::Matrix{Float32}, gaussian_img::Matrix{Gray{Float32}},
                        sigma::Float64, step::Float64)

Compute VLFeat-compatible Hessian determinant with scale normalization.

This function exactly matches VLFeat's `_vl_det_hessian_response` function from covdet.c.

# Arguments
- `det_out`: Preallocated output matrix for Hessian determinant
- `gaussian_img`: Input Gaussian-smoothed image
- `sigma`: Scale (sigma) of the Gaussian level
- `step`: Sampling step (2^octave, where octave can be negative for upsampled octaves)

# VLFeat Formula
- Lxx = -p21 + 2*p22 - p23  (horizontal second derivative using pixels p21,p22,p23)
- Lyy = -p12 + 2*p22 - p32  (vertical second derivative using pixels p12,p22,p32)
- Lxy = (p11 - p31 - p13 + p33)/4  (cross derivative using corner pixels)
- det(H) = (Lxx * Lyy - Lxy^2) * (sigma/step)^4
- Kernels: Lxx uses [-1,2,-1] horizontally, Lyy uses [-1;2;-1] vertically

The scale normalization factor `(sigma/step)^4` is critical for scale-invariant detection.
"""
function vlfeat_hessian_det!(det_out::Matrix{Float32}, gaussian_img::AbstractMatrix{Gray{Float32}},
                             sigma::Float64, step::Float64)
    # Scale normalization factor
    factor = Float32((sigma / step)^4)

    # Hessian kernels matching VLFeat exactly
    # Lxx: horizontal second derivative: -left + 2*center - right
    Lxx_kernel = centered(SMatrix{1,3}([-1, 2, -1]))
    # Lyy: vertical second derivative: -top + 2*center - bottom
    Lyy_kernel = centered(SMatrix{3,1}([-1, 2, -1]))
    # Lxy: cross derivative (note: VLFeat divides by 4)
    Lxy_kernel = centered(SMatrix{3,3}([0.25 0 -0.25; 0 0 0; -0.25 0 0.25]))

    # Extract numeric data
    img_data = channelview(gaussian_img)

    # Compute Hessian components
    Lxx = imfilter(img_data, Lxx_kernel, "replicate")
    Lyy = imfilter(img_data, Lyy_kernel, "replicate")
    Lxy = imfilter(img_data, Lxy_kernel, "replicate")

    # Compute determinant with scale normalization
    # det(H) = Lxx * Lyy - Lxy^2
    @. det_out = (Lxx * Lyy - Lxy * Lxy) * factor

    return det_out
end

"""
    vlfeat_hessian_det(gaussian_img::Matrix{Gray{Float32}}, sigma::Float64, step::Float64)

Non-mutating version that allocates output.
"""
function vlfeat_hessian_det(gaussian_img::AbstractMatrix{Gray{Float32}}, sigma::Float64, step::Float64)
    det_out = Matrix{Float32}(undef, size(gaussian_img))
    return vlfeat_hessian_det!(det_out, gaussian_img, sigma, step)
end
