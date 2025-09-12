# Essential types for function arguments
export Size2                    # For specifying pattern dimensions
export IsoBlob                  # Blob objects returned by pattern functions

# Blob board I/O functions  
export load_blob_board          # JSON I/O for blob board calibration patterns

# Essential units for measurements
export pd, mm, inch, dpi, px, pt, μm # Core units: pixels/dots, millimeters, inches, dots-per-inch, pixels, points

"""
    load_blob_board(filepath::AbstractString) -> Vector{IsoBlob}

Load and parse a blob board calibration pattern from a JSON file.

This function uses the existing StructTypes infrastructure to directly
deserialize JSON blob data to IsoBlob objects with proper unit handling.

# Arguments
- `filepath::AbstractString`: Path to the JSON blob board file

# Returns
- `Vector{IsoBlob}`: Vector of detected/defined blobs

# Example
```julia
blobs = load_blob_board("path/to/blob_board.json")
```
"""
function load_blob_board(filepath::AbstractString)
    # Read and parse JSON file
    json_str = read(filepath, String)
    
    # Parse the JSON structure - looking for "blobs" array
    data = JSON3.read(json_str)
    
    # Convert blob board JSON format to IsoBlob objects
    blob_data = data.blobs
    blobs = Vector{IsoBlob}(undef, length(blob_data))
    
    for i in eachindex(blob_data)
        blob = blob_data[i]
        
        # Parse center coordinates with units
        x_data = blob.center[1]
        y_data = blob.center[2]
        x_val = x_data.value * Unitful.uparse(x_data.unit)
        y_val = y_data.value * Unitful.uparse(y_data.unit)
        center = Point2(ustrip(x_val), ustrip(y_val))
        
        # Parse σ with units  
        σ_data = blob.σ
        σ_val = σ_data.value * Unitful.uparse(σ_data.unit)
        σ = ustrip(σ_val)
        
        blobs[i] = IsoBlob(center, σ)
    end
    
    return blobs
end
