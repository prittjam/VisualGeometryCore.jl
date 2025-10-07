#!/usr/bin/env julia

using Test

@testset "VisualGeometryCore Tests" begin
    include("test_coordinate_conversion.jl")
    include("test_transforms.jl")
    include("test_integration.jl")
end