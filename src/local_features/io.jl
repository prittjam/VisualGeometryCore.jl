"""
I/O functions for saving and loading scale space responses.
"""

using FileIO
using ImageCore: channelview

"""
    save_responses(ss::AbstractScaleSpace, output_dir::String; prefix::String="level")

Save all levels of a scale space to disk as TIFF images.

# Arguments
- `ss::AbstractScaleSpace`: Scale space or response to save
- `output_dir::String`: Directory to save images to (will be created if needed)
- `prefix::String`: Prefix for output filenames (default: "level")

# Example
```julia
save_responses(scale_space, "output/levels")
save_responses(hessian_resp, "output/hessian", prefix="hessian")
```

Output files are named: `{prefix}_o{octave}_s{subdivision}.tif`
"""
function save_responses(ss::AbstractScaleSpace, output_dir::String; prefix::String="level")
    mkpath(output_dir)
    saved_count = 0
    skipped_count = 0

    for level in ss.levels
        saved, skipped = save_level_data(level.data, level, output_dir, prefix)
        saved_count += saved
        skipped_count += skipped
    end

    println("\n✓ Saved $saved_count images to $output_dir")
    if skipped_count > 0
        println("⚠ Skipped $skipped_count uninitialized levels")
    end
end

"""
    save_level_data(data::Matrix{Gray{Float32}}, level, output_dir::String, prefix::String)

Save a single scale space level to disk.

Returns `(saved_count, skipped_count)` tuple.
"""
function save_level_data(data::Matrix{Gray{Float32}}, level, output_dir::String, prefix::String)
    if any(isnan, channelview(data))
        println("Skipped: o=$(level.octave), s=$(level.subdivision) (contains NaN values - uninitialized)")
        return 0, 1
    end

    min_val, max_val = extrema(channelview(data))
    filename = joinpath(output_dir, "$(prefix)_o$(level.octave)_s$(level.subdivision).tif")
    save(filename, data)

    sz = level_size(level)
    println("Saved: $filename (σ=$(round(level.sigma, digits=3)), size=$(sz.width)×$(sz.height), range=[$min_val, $max_val])")
    return 1, 0
end

"""
    save_level_data(data::SubArray{Gray{Float32},2,Array{Gray{Float32},3}}, level, output_dir::String, prefix::String)

Save a single scale space level (view into octave cube) to disk.

Returns `(saved_count, skipped_count)` tuple.
"""
function save_level_data(data::SubArray{Gray{Float32},2,Array{Gray{Float32},3}}, level, output_dir::String, prefix::String)
    if any(isnan, channelview(data))
        println("Skipped: o=$(level.octave), s=$(level.subdivision) (contains NaN values - uninitialized)")
        return 0, 1
    end

    min_val, max_val = extrema(channelview(data))
    filename = joinpath(output_dir, "$(prefix)_o$(level.octave)_s$(level.subdivision).tif")
    save(filename, data)

    sz = level_size(level)
    println("Saved: $filename (σ=$(round(level.sigma, digits=3)), size=$(sz.width)×$(sz.height), range=[$min_val, $max_val])")
    return 1, 0
end
