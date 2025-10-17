# Test ScaleSpace Indexing Optimization
using Test

"""
    generate_arrays_old(octave_range, subdivision_range)

Generate octaves and subdivisions arrays using the current nested loop approach.
"""
function generate_arrays_old(octave_range, subdivision_range)
    octaves = Int[]
    subdivisions = Int[]
    for o in octave_range
        for s in subdivision_range
            push!(octaves, o)
            push!(subdivisions, s)
        end
    end
    return octaves, subdivisions
end

"""
    generate_arrays_new(octave_range, subdivision_range)

Generate octaves and subdivisions arrays using the new repeat approach with inner/outer.
"""
function generate_arrays_new(octave_range, subdivision_range)
    n_subdivisions = length(subdivision_range)
    octaves = repeat(collect(octave_range), inner=n_subdivisions)
    subdivisions = repeat(collect(subdivision_range), outer=length(octave_range))
    return octaves, subdivisions
end

@testset "ScaleSpace Indexing Optimization Tests" begin
    
    @testset "Array Generation Equivalence" begin
        # Test various range combinations
        test_cases = [
            (0:2, 0:3, "Standard case"),
            (-1:1, 0:2, "Negative octaves"),
            (0:0, 0:4, "Single octave"),
            (0:3, 0:0, "Single subdivision"),
            (5:7, 2:5, "Offset ranges"),
            (-2:2, -1:3, "Mixed negative ranges"),
            (10:12, 0:1, "Large octave values"),
            (0:1, 10:12, "Large subdivision values")
        ]
        
        for (octave_range, subdivision_range, description) in test_cases
            @testset "$description" begin
                old_octaves, old_subdivisions = generate_arrays_old(octave_range, subdivision_range)
                new_octaves, new_subdivisions = generate_arrays_new(octave_range, subdivision_range)
                
                @test old_octaves == new_octaves
                @test old_subdivisions == new_subdivisions
                @test length(old_octaves) == length(new_octaves)
                @test length(old_subdivisions) == length(new_subdivisions)
            end
        end
    end
    
    @testset "Octave-Major Ordering Verification" begin
        octave_range = 0:2
        subdivision_range = 0:2
        
        old_octaves, old_subdivisions = generate_arrays_old(octave_range, subdivision_range)
        new_octaves, new_subdivisions = generate_arrays_new(octave_range, subdivision_range)
        
        # Expected ordering: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
        expected_octaves = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        expected_subdivisions = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        
        @test old_octaves == expected_octaves
        @test old_subdivisions == expected_subdivisions
        @test new_octaves == expected_octaves
        @test new_subdivisions == expected_subdivisions
    end
    
    @testset "Edge Cases" begin
        @testset "Empty ranges" begin
            # Empty octave range
            octaves1, subdivisions1 = generate_arrays_old(1:0, 0:2)
            octaves2, subdivisions2 = generate_arrays_new(1:0, 0:2)
            @test octaves1 == octaves2 == Int[]
            @test subdivisions1 == subdivisions2 == Int[]
            
            # Empty subdivision range  
            octaves3, subdivisions3 = generate_arrays_old(0:2, 1:0)
            octaves4, subdivisions4 = generate_arrays_new(0:2, 1:0)
            @test octaves3 == octaves4 == Int[]
            @test subdivisions3 == subdivisions4 == Int[]
        end
        
        @testset "Single element ranges" begin
            octaves1, subdivisions1 = generate_arrays_old(5:5, 3:3)
            octaves2, subdivisions2 = generate_arrays_new(5:5, 3:3)
            @test octaves1 == octaves2 == [5]
            @test subdivisions1 == subdivisions2 == [3]
        end
    end
    
    @testset "Performance Benchmarking" begin
        # Test with larger ranges to see performance difference
        large_octave_range = -5:10
        large_subdivision_range = 0:15
        
        # Warm up both functions
        generate_arrays_old(0:1, 0:1)
        generate_arrays_new(0:1, 0:1)
        
        # Benchmark old approach
        old_time = @elapsed begin
            for _ in 1:1000
                generate_arrays_old(large_octave_range, large_subdivision_range)
            end
        end
        
        # Benchmark new approach
        new_time = @elapsed begin
            for _ in 1:1000
                generate_arrays_new(large_octave_range, large_subdivision_range)
            end
        end
        
        println("Performance comparison:")
        println("  Old approach: $(round(old_time * 1000, digits=3)) ms")
        println("  New approach: $(round(new_time * 1000, digits=3)) ms")
        println("  Speedup: $(round(old_time / new_time, digits=2))x")
        
        # New approach should be at least as fast (allow 20% tolerance for noise)
        @test new_time <= old_time * 1.2
    end
end

println("Array generation comparison tests completed successfully!")