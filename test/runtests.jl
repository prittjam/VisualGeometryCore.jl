#!/usr/bin/env julia

using Test

@testset "VisualGeometryCore Tests" begin
    include("test_coordinate_conversion.jl")
    include("test_transforms.jl")
    include("test_conic_roundtrip.jl")
    include("test_integration.jl")
    include("test_python_interface.jl")
    include("test_vlfeat_comparison.jl")
    include("test_scalespace_tiff_comparison.jl")
end