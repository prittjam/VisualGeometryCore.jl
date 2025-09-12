# JSON deserialization for blob board calibration patterns

"""
    _UnitfulValue

Internal struct for JSON deserialization of unitful values.
Maps JSON objects like `{"value": 210, "unit": "mm"}` to Julia values.
"""
struct _UnitfulValue
    value::Float64
    unit::String
end
StructTypes.StructType(::Type{_UnitfulValue}) = StructTypes.Struct()

"""
    _JsonBlob

Internal struct for JSON deserialization of blob objects.
Maps JSON blob objects with center coordinates and σ parameters.
"""
struct _JsonBlob
    center::Vector{_UnitfulValue}
    σ::_UnitfulValue
end
StructTypes.StructType(::Type{_JsonBlob}) = StructTypes.Struct()

"""
    _BlobFile

Internal struct for JSON deserialization of blob board files.
Maps the top-level JSON structure containing an array of blobs.
"""
struct _BlobFile
    blobs::Vector{_JsonBlob}
end
StructTypes.StructType(::Type{_BlobFile}) = StructTypes.Struct()

"""
    convert_blob(json_blob::_JsonBlob) -> IsoBlob

Convert a JSON blob representation to an `IsoBlob` object.

This function:
1. Extracts center coordinates with units from the JSON structure
2. Evaluates unit strings to create proper unitful quantities
3. Strips units to create the final unitless `IsoBlob`

# Arguments
- `json_blob::_JsonBlob`: JSON blob object with center and σ fields

# Returns
- `IsoBlob`: Converted blob with Point2 center and Float64 σ

# Example
```julia
# Assuming json_blob contains {"center": [{"value": 10.0, "unit": "mm"}, ...], "σ": {"value": 2.0, "unit": "mm"}}
blob = convert_blob(json_blob)
```
"""
function convert_blob(json_blob::_JsonBlob)
    # Extract center coordinates with units
    x_val = json_blob.center[1].value * eval(Symbol(json_blob.center[1].unit))
    y_val = json_blob.center[2].value * eval(Symbol(json_blob.center[2].unit))
    
    # Convert to Point2 (unitless for IsoBlob) 
    center = Point2(ustrip(x_val), ustrip(y_val))
    
    # Extract σ with units and strip units
    σ_val = json_blob.σ.value * eval(Symbol(json_blob.σ.unit))
    σ = ustrip(σ_val)
    
    return IsoBlob(center, σ)
end

"""
    load_blob_board(filepath::AbstractString) -> Vector{IsoBlob}

Load and parse a blob board calibration pattern from a JSON file.

This function handles the complete process of:
1. Reading the JSON file
2. Parsing the unitful values 
3. Converting to `IsoBlob` objects
4. Returning the vector of blobs

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
    json_data = JSON3.read(read(filepath, String), _BlobFile)
    blobs = [convert_blob(jb) for jb in json_data.blobs]
    return blobs
end
