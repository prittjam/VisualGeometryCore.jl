# ==============================================================================
# Camera Visualization Functions
# ==============================================================================

"""
    frustum!(lscene, camera::Camera{CameraModel{PhysicalIntrinsics,...}}, sensor_bounds; kwargs...)

Add camera frustum visualization for camera with physical intrinsics.
Uses focal length as default depth for natural frustum visualization.

# Arguments
- `lscene`: LScene with 3D camera
- `camera`: Camera object with PhysicalIntrinsics
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
- `depth=nothing`: Frustum depth (mm). If nothing, uses focal length
- `color=:orange`: Frustum color
- `linewidth=2.0`: Width of frustum edge lines

# Examples
```julia
S.frustum!(scene3d, camera, sensor_bounds)  # Uses focal length as depth
S.frustum!(scene3d, camera, sensor_bounds; depth=300.0)  # Custom depth
```
"""
function frustum!(lscene, camera::Camera{<:CameraModel{PhysicalIntrinsics}}, sensor_bounds;
                  depth=nothing,
                  kwargs...)
    # Use focal length as default depth for physical cameras
    focal_depth = camera.model.intrinsics.f
    frustum_depth = something(depth, focal_depth)
    return frustum!(lscene, camera, sensor_bounds, frustum_depth; kwargs...)
end

"""
    frustum!(lscene, camera, sensor_bounds, depth; kwargs...)

Add camera frustum visualization to a 3D LScene with explicit depth.

# Arguments
- `lscene`: LScene with 3D camera
- `camera`: Camera object
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates
- `depth`: Depth of frustum from camera center (mm)

# Keyword Arguments
- `color=:orange`: Frustum color
- `linewidth=2.0`: Width of frustum edge lines
- `show_near_plane=false`: Show near plane rectangle
- `near_depth=10.0`: Near plane depth (if show_near_plane=true)
- `show_up_indicator=true`: Show triangle indicator at top edge pointing in camera's up direction
- `indicator_size=40.0`: Height of the up indicator triangle in pixels (image coordinates)
- `image=nothing`: Optional image to display on the far plane (texture mapped to frustum rectangle)

# Examples
```julia
S.frustum!(scene3d, camera, sensor_bounds, 250.0; color=:cyan)
S.frustum!(scene3d, camera, sensor_bounds, 250.0; image=camera_view)  # Show image on far plane
```
"""
function frustum!(lscene, camera, sensor_bounds, depth;
                  color=:orange,
                  linewidth=2.0,
                  show_near_plane=false,
                  near_depth=10.0,
                  show_up_indicator=true,
                  indicator_size=40.0,
                  image=nothing)

    # Get camera pose and center
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)

    # Unproject sensor corners to 3D in camera space, then transform to world
    corners_2d = coordinates(sensor_bounds)
    corners_cam = unproject.(Ref(camera.model), corners_2d, depth)
    corners_world = Point3f.(cam_pose.(corners_cam))

    # Optional: show image on far plane (before drawing wireframe)
    if !isnothing(image)
        # Create a unit rectangle mesh with UV coordinates using GeometryBasics
        unit_rect_mesh = GeometryBasics.uv_normal_mesh(Rect(0, 0, 1, 1))

        # Transform the unit rectangle to the frustum far plane
        # Extract positions and transform them to world space
        rect_positions = GeometryBasics.coordinates(unit_rect_mesh)

        # Map [0,1]x[0,1] rectangle to frustum far plane corners
        # corners_world: [p1, p2, p3, p4] from coordinates(sensor_bounds)
        transformed_positions = map(rect_positions) do p
            u, v = p[1], p[2]
            # Bilinear interpolation: p = (1-u)(1-v)p1 + u(1-v)p2 + uv*p3 + (1-u)v*p4
            p1, p2, p3, p4 = corners_world[1], corners_world[2], corners_world[3], corners_world[4]
            Point3f((1-u)*(1-v)*p1 + u*(1-v)*p2 + u*v*p3 + (1-u)*v*p4)
        end

        # Create new mesh with transformed positions but keep UV coordinates and faces
        far_mesh = GeometryBasics.Mesh(
            transformed_positions,
            GeometryBasics.faces(unit_rect_mesh);
            uv=GeometryBasics.texturecoordinates(unit_rect_mesh)
        )

        # Display textured mesh
        # Use parent() to unwrap OffsetArrays (safe for regular arrays too)
        # Makie will use the UV coordinates from the mesh to map the image
        push!(lscene.plots, MakieSpec.Mesh(far_mesh;
            color=parent(image),
            shading=Makie.NoShading,
            interpolate=true))
    end

    # Create frustum mesh: pyramid from camera center to far plane
    vertices = [cam_center, corners_world...]
    faces = [
        TriangleFace(1, 2, 3),  # side 1
        TriangleFace(1, 3, 4),  # side 2
        TriangleFace(1, 4, 5),  # side 3
        TriangleFace(1, 5, 2),  # side 4
    ]
    frustum_mesh = GeometryBasics.Mesh(vertices, faces)

    # Draw frustum as wireframe mesh
    push!(lscene.plots, MakieSpec.Wireframe(frustum_mesh;
        color=color, linewidth=linewidth))

    # Optional: near plane as rectangular mesh
    if show_near_plane
        near_cam = unproject.(Ref(camera.model), corners_2d, near_depth)
        near_world = Point3f.(cam_pose.(near_cam))

        # Create near plane rectangle as two triangles
        near_faces = [
            TriangleFace(1, 2, 3),
            TriangleFace(1, 3, 4)
        ]
        near_mesh = GeometryBasics.Mesh(near_world, near_faces)

        push!(lscene.plots, MakieSpec.Wireframe(near_mesh;
            color=color, linewidth=linewidth))
    end

    # Add up indicator: small triangle at midpoint of top edge pointing in -y direction (camera up)
    if show_up_indicator
        # Use center function and adjust y to top edge
        sensor_center = center(sensor_bounds)
        top_y = sensor_bounds.origin[2]  # Top edge (minimum y, since y points down)
        base_half_width = sensor_bounds.widths[1] * 0.02  # 2% of sensor width

        # Construct triangle in 2D image space
        tip_2d = Point2(sensor_center[1], top_y - indicator_size)
        base_left_2d = Point2(sensor_center[1] - base_half_width, top_y)
        base_right_2d = Point2(sensor_center[1] + base_half_width, top_y)
        triangle_2d = GeometryBasics.Triangle(tip_2d, base_left_2d, base_right_2d)

        # Broadcast unproject over triangle vertices (convert to SVector for unproject) and transform to world
        vertices_world = Point3f.(cam_pose.(unproject.(Ref(camera.model), SVector{2}.(triangle_2d), Ref(depth))))

        # Create 3D triangle from unprojected vertices
        indicator_triangle = GeometryBasics.Triangle(vertices_world...)

        push!(lscene.plots, MakieSpec.Mesh(indicator_triangle; color=color))
    end

    return nothing
end

"""
    camera!(lscene, camera, sensor_bounds; axis_length=50.0, frustum_depth=200.0, kwargs...)

Add camera visualization to a 3D LScene, including camera center, coordinate axes, and frustum.

# Arguments
- `lscene`: LScene with 3D camera (cam3d!)
- `camera`: Camera object with model and extrinsics
- `sensor_bounds`: Rect2 representing sensor bounds in image coordinates

# Keyword Arguments
- `axis_length=50.0`: Length of camera coordinate axes (mm)
- `frustum_depth=200.0`: Depth of frustum visualization (mm)
- `show_center=true`: Show camera center marker
- `show_axes=true`: Show camera coordinate frame axes
- `show_frustum=true`: Show camera frustum
- `center_color=:red`: Color for camera center marker
- `center_size=20.0`: Size of camera center marker
- `frustum_color=:orange`: Color for frustum
- `frustum_linewidth=2.0`: Width of frustum edges
- `frustum_image=nothing`: Optional image to display on frustum far plane

# Examples
```julia
import VisualGeometryCore.Spec as S

# Create 3D scene
scene3d = S.LScene3D()

# Add camera visualization
S.camera!(scene3d, camera, sensor_bounds)

# Customize appearance
S.camera!(scene3d, camera, sensor_bounds;
    axis_length=100.0,
    frustum_depth=300.0,
    frustum_color=:blue)
```
"""
function camera!(lscene, camera, sensor_bounds;
                 axis_length=50.0,
                 frustum_depth=200.0,
                 show_center=true,
                 show_axes=true,
                 show_frustum=true,
                 center_color=:red,
                 center_size=20.0,
                 frustum_color=:orange,
                 frustum_linewidth=2.0,
                 frustum_image=nothing)

    # Get camera pose (camera-to-world)
    cam_pose = pose(camera)
    cam_center = Point3f(cam_pose.t)
    R_c2w = cam_pose.R  # Camera-to-world rotation (columns are camera axes in world coords)

    # 1. Camera center
    if show_center
        push!(lscene.plots, MakieSpec.Scatter([cam_center];
            color=center_color,
            markersize=center_size))
    end

    # 2. Camera coordinate axes
    if show_axes
        # Transform scaled canonical basis vectors to world coordinates via pose
        basis = canonical_basis(SVector{3,Float64})
        axis_ends = Point3f.(cam_pose.(basis .* axis_length))

        # X-axis (red), Y-axis (green), Z-axis/optical axis (blue)
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[1]];
            color=:red, linewidth=3))
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[2]];
            color=:green, linewidth=3))
        push!(lscene.plots, MakieSpec.Lines(
            [cam_center, axis_ends[3]];
            color=:blue, linewidth=3))

        # Add markers at endpoints to indicate direction
        push!(lscene.plots, MakieSpec.Scatter(
            collect(axis_ends);
            color=[:red, :green, :blue],
            markersize=8))
    end

    # 3. Camera frustum
    if show_frustum
        frustum!(lscene, camera, sensor_bounds, frustum_depth;
            color=frustum_color,
            linewidth=frustum_linewidth,
            image=frustum_image)
    end

    return nothing
end

"""
    LScene3D(; show_axis=true, width=600, height=600, kwargs...)

Create a 3D LScene with interactive Camera3D controls.

# Keyword Arguments
- `show_axis=true`: Show 3D axis
- `width=600`: Scene width
- `height=600`: Scene height
- `scenekw`: Additional scene keyword arguments

# Returns
- LScene configured for 3D visualization

# Examples
```julia
scene3d = S.LScene3D()

# Add visualizations
S.board!(scene3d, X)
S.camera!(scene3d, camera)

# Include in layout
layout = S.GridLayout([(1,1) => scene3d])
```
"""
function LScene3D(;
                  show_axis=true,
                  width=600,
                  height=600,
                  plots=PlotSpec[],
                  scenekw=(;))

    lscene = MakieSpec.LScene(
        plots=plots,
        show_axis=show_axis,
        width=Makie.Fixed(width),
        height=Makie.Fixed(height),
        scenekw=scenekw
    )

    # Add callback to enable Camera3D controls
    function setup_camera3d(lscene_block)
        cam3d!(lscene_block.scene)
        return
    end

    push!(lscene.then_funcs, setup_camera3d)

    return lscene
end
