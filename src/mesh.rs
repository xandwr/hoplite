//! 3D mesh primitives and spatial transforms for GPU rendering.
//!
//! This module provides the core building blocks for 3D rendering in Hoplite:
//!
//! - [`Vertex3d`] — The vertex format used by all meshes, containing position, normal, and UV data
//! - [`Mesh`] — GPU-resident geometry with vertex and index buffers
//! - [`Transform`] — Position, rotation, and scale for placing meshes in 3D space
//!
//! # Creating Meshes
//!
//! Meshes can be created from raw vertex/index data or using built-in primitives:
//!
//! ```no_run
//! use hoplite::*;
//!
//! fn main() {
//!     run(|ctx| {
//!         // Built-in primitives
//!         let cube = Mesh::cube(&ctx.gpu);
//!         let sphere = Mesh::sphere(&ctx.gpu, 32, 16);
//!         let plane = Mesh::plane(&ctx.gpu, 10.0);
//!
//!         // Custom mesh from vertices
//!         let vertices = vec![
//!             Vertex3d::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0]),
//!             Vertex3d::new([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
//!             Vertex3d::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
//!         ];
//!         let indices = vec![0, 1, 2];
//!         let triangle = Mesh::new(&ctx.gpu, &vertices, &indices);
//!
//!         move |frame| {
//!             // Meshes are rendered via MeshNode in the render graph
//!         }
//!     });
//! }
//! ```
//!
//! # Transforms
//!
//! [`Transform`] uses a builder pattern for ergonomic positioning:
//!
//! ```
//! use hoplite::{Transform, Vec3, Quat};
//!
//! // Builder pattern
//! let transform = Transform::new()
//!     .position(Vec3::new(0.0, 2.0, -5.0))
//!     .rotation(Quat::from_rotation_y(0.5))
//!     .uniform_scale(2.0);
//!
//! // Quick positioning
//! let positioned = Transform::from_position(Vec3::new(1.0, 0.0, 0.0));
//! ```
//!
//! # Vertex Layout
//!
//! The [`Vertex3d`] struct uses the following GPU layout (32 bytes per vertex):
//!
//! | Attribute | Format    | Offset | Shader Location |
//! |-----------|-----------|--------|-----------------|
//! | position  | Float32x3 | 0      | 0               |
//! | normal    | Float32x3 | 12     | 1               |
//! | uv        | Float32x2 | 24     | 2               |
//!
//! This layout is exposed via [`Vertex3d::LAYOUT`] for custom pipeline creation.

use crate::gpu::GpuContext;
use glam::{Mat4, Vec3};

/// A vertex for 3D mesh rendering with position, normal, and texture coordinates.
///
/// This struct is the fundamental building block for all 3D geometry in Hoplite.
/// It uses `#[repr(C)]` to ensure a predictable memory layout for GPU upload,
/// and derives [`bytemuck::Pod`] and [`bytemuck::Zeroable`] for safe casting
/// to byte slices.
///
/// # Memory Layout
///
/// Each vertex occupies 32 bytes:
/// - `position`: 12 bytes (3 × f32) at offset 0
/// - `normal`: 12 bytes (3 × f32) at offset 12
/// - `uv`: 8 bytes (2 × f32) at offset 24
///
/// # Example
///
/// ```
/// use hoplite::Vertex3d;
///
/// // Create a vertex with position, normal pointing up, and UV at origin
/// let vertex = Vertex3d::new(
///     [0.0, 1.0, 0.0],  // position
///     [0.0, 1.0, 0.0],  // normal (pointing up)
///     [0.5, 0.5],       // uv (center of texture)
/// );
/// ```
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3d {
    /// The 3D position of this vertex in model space.
    pub position: [f32; 3],
    /// The surface normal vector (should be normalized for correct lighting).
    pub normal: [f32; 3],
    /// Texture coordinates, typically in the range [0, 1].
    pub uv: [f32; 2],
}

impl Vertex3d {
    /// The wgpu vertex buffer layout descriptor for this vertex type.
    ///
    /// Use this when creating custom render pipelines that need to read
    /// [`Vertex3d`] data. The layout defines:
    /// - **Array stride**: 32 bytes per vertex
    /// - **Step mode**: Per-vertex (not per-instance)
    /// - **Attributes**: position (loc 0), normal (loc 1), uv (loc 2)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    ///     vertex: wgpu::VertexState {
    ///         module: &shader,
    ///         entry_point: Some("vs_main"),
    ///         buffers: &[Vertex3d::LAYOUT],
    ///         ..Default::default()
    ///     },
    ///     // ...
    /// });
    /// ```
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex3d>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            // normal
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            // uv
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x2,
            },
        ],
    };

    /// Creates a new vertex with the given position, normal, and UV coordinates.
    ///
    /// # Arguments
    ///
    /// * `position` - The 3D position in model space
    /// * `normal` - The surface normal vector (should be normalized)
    /// * `uv` - Texture coordinates, typically in [0, 1] range
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::Vertex3d;
    ///
    /// let vertex = Vertex3d::new(
    ///     [1.0, 0.0, 0.0],   // right of origin
    ///     [0.0, 1.0, 0.0],   // normal pointing up
    ///     [1.0, 0.0],        // top-right of texture
    /// );
    /// ```
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }
}

/// GPU-resident 3D mesh geometry with vertex and index buffers.
///
/// A `Mesh` holds the GPU buffers required to render 3D geometry. Once created,
/// the mesh data lives on the GPU and can be rendered efficiently. Meshes are
/// immutable after creation—to render different geometry, create a new mesh.
///
/// # Creating Meshes
///
/// ## Built-in Primitives
///
/// Hoplite provides several common primitives:
///
/// ```no_run
/// use hoplite::*;
///
/// run(|ctx| {
///     let cube = Mesh::cube(&ctx.gpu);           // Unit cube centered at origin
///     let sphere = Mesh::sphere(&ctx.gpu, 32, 16); // Sphere with 32 segments, 16 rings
///     let plane = Mesh::plane(&ctx.gpu, 5.0);    // 5×5 plane on XZ axis
///     move |_| {}
/// });
/// ```
///
/// ## Custom Geometry
///
/// For custom meshes, provide vertex and index data:
///
/// ```no_run
/// use hoplite::*;
///
/// run(|ctx| {
///     // A simple triangle
///     let vertices = vec![
///         Vertex3d::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0]),
///         Vertex3d::new([-1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
///         Vertex3d::new([1.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
///     ];
///     let indices = vec![0, 1, 2];
///     let triangle = Mesh::new(&ctx.gpu, &vertices, &indices);
///     move |_| {}
/// });
/// ```
///
/// # Rendering
///
/// Meshes are rendered through [`MeshNode`](crate::MeshNode) in the render graph.
/// During setup, add meshes to the queue and enable mesh rendering:
///
/// ```no_run
/// use hoplite::*;
///
/// run(|ctx| {
///     ctx.enable_mesh_rendering();
///     let cube_idx = ctx.mesh_cube();  // Returns mesh index
///
///     move |frame| {
///         frame.draw_mesh(cube_idx, Transform::new(), Color::WHITE);
///     }
/// });
/// ```
///
/// # Winding Order
///
/// All built-in primitives use counter-clockwise (CCW) winding order for front faces.
/// Custom meshes should follow this convention for correct backface culling.
#[derive(Debug)]
pub struct Mesh {
    /// The GPU buffer containing vertex data.
    pub(crate) vertex_buffer: wgpu::Buffer,
    /// The GPU buffer containing index data (u32 indices).
    pub(crate) index_buffer: wgpu::Buffer,
    /// The number of indices in the mesh (determines draw call size).
    pub(crate) index_count: u32,
}

impl Mesh {
    /// Creates a mesh from raw vertex and index data.
    ///
    /// This uploads the provided geometry data to GPU buffers. The mesh is
    /// ready to render immediately after creation.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for buffer allocation
    /// * `vertices` - Slice of vertices defining the mesh geometry
    /// * `indices` - Slice of u32 indices defining triangles (3 indices per triangle)
    ///
    /// # Panics
    ///
    /// Does not panic, but an empty mesh will not render anything.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hoplite::*;
    ///
    /// run(|ctx| {
    ///     // Create a quad from two triangles
    ///     let vertices = vec![
    ///         Vertex3d::new([-0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0]),
    ///         Vertex3d::new([ 0.5, -0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]),
    ///         Vertex3d::new([ 0.5,  0.5, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0]),
    ///         Vertex3d::new([-0.5,  0.5, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0]),
    ///     ];
    ///     let indices = vec![0, 1, 2, 2, 3, 0];
    ///     let quad = Mesh::new(&ctx.gpu, &vertices, &indices);
    ///     move |_| {}
    /// });
    /// ```
    pub fn new(gpu: &GpuContext, vertices: &[Vertex3d], indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt;

        let vertex_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Vertex Buffer"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

        let index_buffer = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Mesh Index Buffer"),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }

    /// Creates a unit cube centered at the origin.
    ///
    /// The cube spans from -0.5 to 0.5 on all axes, making it exactly 1 unit
    /// on each side. Each face has its own set of vertices with correct normals
    /// for flat shading and independent UV coordinates.
    ///
    /// # Geometry Details
    ///
    /// - **Dimensions**: 1×1×1 units
    /// - **Center**: Origin (0, 0, 0)
    /// - **Vertices**: 24 (4 per face, for correct normals)
    /// - **Triangles**: 12 (2 per face)
    /// - **UV mapping**: Each face maps the full \[0,1\] texture range
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hoplite::*;
    ///
    /// run(|ctx| {
    ///     ctx.enable_mesh_rendering();
    ///     let cube_idx = ctx.mesh_cube();
    ///
    ///     move |frame| {
    ///         // Render at 2x scale, positioned above origin
    ///         let transform = Transform::new()
    ///             .position(Vec3::new(0.0, 1.0, 0.0))
    ///             .uniform_scale(2.0);
    ///         frame.draw_mesh(cube_idx, transform, Color::WHITE);
    ///     }
    /// });
    /// ```
    pub fn cube(gpu: &GpuContext) -> Self {
        // Each face has its own vertices for correct normals
        #[rustfmt::skip]
        let vertices = vec![
            // Front face (Z+)
            Vertex3d::new([-0.5, -0.5,  0.5], [ 0.0,  0.0,  1.0], [0.0, 0.0]),
            Vertex3d::new([ 0.5, -0.5,  0.5], [ 0.0,  0.0,  1.0], [1.0, 0.0]),
            Vertex3d::new([ 0.5,  0.5,  0.5], [ 0.0,  0.0,  1.0], [1.0, 1.0]),
            Vertex3d::new([-0.5,  0.5,  0.5], [ 0.0,  0.0,  1.0], [0.0, 1.0]),
            // Back face (Z-)
            Vertex3d::new([ 0.5, -0.5, -0.5], [ 0.0,  0.0, -1.0], [0.0, 0.0]),
            Vertex3d::new([-0.5, -0.5, -0.5], [ 0.0,  0.0, -1.0], [1.0, 0.0]),
            Vertex3d::new([-0.5,  0.5, -0.5], [ 0.0,  0.0, -1.0], [1.0, 1.0]),
            Vertex3d::new([ 0.5,  0.5, -0.5], [ 0.0,  0.0, -1.0], [0.0, 1.0]),
            // Top face (Y+)
            Vertex3d::new([-0.5,  0.5,  0.5], [ 0.0,  1.0,  0.0], [0.0, 0.0]),
            Vertex3d::new([ 0.5,  0.5,  0.5], [ 0.0,  1.0,  0.0], [1.0, 0.0]),
            Vertex3d::new([ 0.5,  0.5, -0.5], [ 0.0,  1.0,  0.0], [1.0, 1.0]),
            Vertex3d::new([-0.5,  0.5, -0.5], [ 0.0,  1.0,  0.0], [0.0, 1.0]),
            // Bottom face (Y-)
            Vertex3d::new([-0.5, -0.5, -0.5], [ 0.0, -1.0,  0.0], [0.0, 0.0]),
            Vertex3d::new([ 0.5, -0.5, -0.5], [ 0.0, -1.0,  0.0], [1.0, 0.0]),
            Vertex3d::new([ 0.5, -0.5,  0.5], [ 0.0, -1.0,  0.0], [1.0, 1.0]),
            Vertex3d::new([-0.5, -0.5,  0.5], [ 0.0, -1.0,  0.0], [0.0, 1.0]),
            // Right face (X+)
            Vertex3d::new([ 0.5, -0.5,  0.5], [ 1.0,  0.0,  0.0], [0.0, 0.0]),
            Vertex3d::new([ 0.5, -0.5, -0.5], [ 1.0,  0.0,  0.0], [1.0, 0.0]),
            Vertex3d::new([ 0.5,  0.5, -0.5], [ 1.0,  0.0,  0.0], [1.0, 1.0]),
            Vertex3d::new([ 0.5,  0.5,  0.5], [ 1.0,  0.0,  0.0], [0.0, 1.0]),
            // Left face (X-)
            Vertex3d::new([-0.5, -0.5, -0.5], [-1.0,  0.0,  0.0], [0.0, 0.0]),
            Vertex3d::new([-0.5, -0.5,  0.5], [-1.0,  0.0,  0.0], [1.0, 0.0]),
            Vertex3d::new([-0.5,  0.5,  0.5], [-1.0,  0.0,  0.0], [1.0, 1.0]),
            Vertex3d::new([-0.5,  0.5, -0.5], [-1.0,  0.0,  0.0], [0.0, 1.0]),
        ];

        #[rustfmt::skip]
        let indices: Vec<u32> = vec![
            0,  1,  2,  2,  3,  0,  // front
            4,  5,  6,  6,  7,  4,  // back
            8,  9,  10, 10, 11, 8,  // top
            12, 13, 14, 14, 15, 12, // bottom
            16, 17, 18, 18, 19, 16, // right
            20, 21, 22, 22, 23, 20, // left
        ];

        Self::new(gpu, &vertices, &indices)
    }

    /// Creates a UV sphere centered at the origin with configurable tessellation.
    ///
    /// The sphere has a radius of 0.5 (diameter of 1 unit) and is generated using
    /// latitude/longitude subdivision. Higher segment and ring counts produce
    /// smoother spheres at the cost of more vertices.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for buffer allocation
    /// * `segments` - Number of longitudinal divisions (around the equator)
    /// * `rings` - Number of latitudinal divisions (pole to pole)
    ///
    /// # Geometry Details
    ///
    /// - **Radius**: 0.5 units (diameter 1.0)
    /// - **Center**: Origin (0, 0, 0)
    /// - **Vertices**: `(segments + 1) × (rings + 1)`
    /// - **Triangles**: `segments × rings × 2`
    /// - **UV mapping**: Equirectangular projection (u = longitude, v = latitude)
    ///
    /// # Recommended Values
    ///
    /// | Quality    | Segments | Rings |
    /// |------------|----------|-------|
    /// | Low        | 16       | 8     |
    /// | Medium     | 32       | 16    |
    /// | High       | 64       | 32    |
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hoplite::*;
    ///
    /// run(|ctx| {
    ///     ctx.enable_mesh_rendering();
    ///     // Medium quality sphere
    ///     let sphere_idx = ctx.mesh_sphere(32, 16);
    ///
    ///     move |frame| {
    ///         frame.draw_mesh(sphere_idx, Transform::new(), Color::WHITE);
    ///     }
    /// });
    /// ```
    pub fn sphere(gpu: &GpuContext, segments: u32, rings: u32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for ring in 0..=rings {
            let phi = std::f32::consts::PI * ring as f32 / rings as f32;
            let y = phi.cos();
            let ring_radius = phi.sin();

            for seg in 0..=segments {
                let theta = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
                let x = ring_radius * theta.cos();
                let z = ring_radius * theta.sin();

                let position = [x * 0.5, y * 0.5, z * 0.5];
                let normal = [x, y, z];
                let uv = [seg as f32 / segments as f32, ring as f32 / rings as f32];

                vertices.push(Vertex3d::new(position, normal, uv));
            }
        }

        for ring in 0..rings {
            for seg in 0..segments {
                let current = ring * (segments + 1) + seg;
                let next = current + segments + 1;

                indices.push(current);
                indices.push(next);
                indices.push(current + 1);

                indices.push(current + 1);
                indices.push(next);
                indices.push(next + 1);
            }
        }

        Self::new(gpu, &vertices, &indices)
    }

    /// Creates a flat rectangular plane on the XZ axis (horizontal ground plane).
    ///
    /// The plane is centered at the origin with normals pointing up (+Y). This
    /// is useful for ground planes, floors, or any horizontal surface.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for buffer allocation
    /// * `size` - The width and depth of the plane (it's square)
    ///
    /// # Geometry Details
    ///
    /// - **Dimensions**: `size × size` units on XZ plane
    /// - **Center**: Origin (0, 0, 0)
    /// - **Y position**: 0 (lies on the XZ plane)
    /// - **Normal**: Pointing up (0, 1, 0)
    /// - **Vertices**: 4
    /// - **Triangles**: 2
    /// - **UV mapping**: Full \[0,1\] range across the plane
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hoplite::*;
    ///
    /// run(|ctx| {
    ///     ctx.enable_mesh_rendering();
    ///     // Create a 10×10 ground plane
    ///     let ground_idx = ctx.mesh_plane(10.0);
    ///
    ///     move |frame| {
    ///         // Render as a white floor
    ///         frame.draw_mesh(ground_idx, Transform::new(), Color::WHITE);
    ///     }
    /// });
    /// ```
    pub fn plane(gpu: &GpuContext, size: f32) -> Self {
        let half = size * 0.5;
        let vertices = vec![
            Vertex3d::new([-half, 0.0, -half], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3d::new([half, 0.0, -half], [0.0, 1.0, 0.0], [1.0, 0.0]),
            Vertex3d::new([half, 0.0, half], [0.0, 1.0, 0.0], [1.0, 1.0]),
            Vertex3d::new([-half, 0.0, half], [0.0, 1.0, 0.0], [0.0, 1.0]),
        ];

        let indices = vec![0, 1, 2, 2, 3, 0];

        Self::new(gpu, &vertices, &indices)
    }
}

/// A 3D transformation representing position, rotation, and scale.
///
/// `Transform` is the primary way to position meshes in 3D space. It stores
/// translation, rotation (as a quaternion), and scale separately, then combines
/// them into a 4×4 transformation matrix for rendering.
///
/// # Builder Pattern
///
/// Transform uses a fluent builder pattern for ergonomic construction:
///
/// ```
/// use hoplite::{Transform, Vec3, Quat};
///
/// let transform = Transform::new()
///     .position(Vec3::new(0.0, 5.0, -10.0))
///     .rotation(Quat::from_rotation_y(std::f32::consts::PI / 4.0))
///     .uniform_scale(2.0);
/// ```
///
/// # Transformation Order
///
/// When converted to a matrix via [`Transform::matrix()`], transformations are
/// applied in the standard order: **Scale → Rotate → Translate** (SRT).
/// This means:
/// 1. The mesh is scaled around its local origin
/// 2. Then rotated around its local origin
/// 3. Finally translated to its world position
///
/// # Default Values
///
/// A default transform places the object at the origin with no rotation and
/// unit scale:
/// - `position`: `(0, 0, 0)`
/// - `rotation`: Identity quaternion (no rotation)
/// - `scale`: `(1, 1, 1)`
///
/// # Example
///
/// ```no_run
/// use hoplite::*;
///
/// run(|ctx| {
///     ctx.enable_mesh_rendering();
///     let cube_idx = ctx.mesh_cube();
///     let mut angle = 0.0f32;
///
///     move |frame| {
///         angle += frame.dt;
///
///         // Spinning cube above the origin
///         let transform = Transform::new()
///             .position(Vec3::new(0.0, 2.0, 0.0))
///             .rotation(Quat::from_rotation_y(angle));
///         frame.draw_mesh(cube_idx, transform, Color::WHITE);
///     }
/// });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    /// World-space position (translation).
    pub position: Vec3,
    /// Rotation as a unit quaternion.
    pub rotation: glam::Quat,
    /// Scale factors for each axis.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// Creates a new identity transform (origin, no rotation, unit scale).
    ///
    /// This is equivalent to `Transform::default()`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::Transform;
    ///
    /// let transform = Transform::new();
    /// // Equivalent to Transform::default()
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a transform positioned at the given location.
    ///
    /// This is a convenience constructor for the common case of positioning
    /// an object without rotation or scaling.
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Vec3};
    ///
    /// let transform = Transform::from_position(Vec3::new(1.0, 2.0, 3.0));
    /// assert_eq!(transform.position, Vec3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    /// Sets the position (translation) component.
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Vec3};
    ///
    /// let transform = Transform::new().position(Vec3::new(5.0, 0.0, -3.0));
    /// ```
    pub fn position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    /// Sets the rotation component using a quaternion.
    ///
    /// For common rotation operations, use glam's quaternion constructors:
    /// - `Quat::from_rotation_x(angle)` — Rotate around X axis
    /// - `Quat::from_rotation_y(angle)` — Rotate around Y axis
    /// - `Quat::from_rotation_z(angle)` — Rotate around Z axis
    /// - `Quat::from_axis_angle(axis, angle)` — Rotate around arbitrary axis
    /// - `Quat::from_euler(order, x, y, z)` — From Euler angles
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Quat};
    ///
    /// // Rotate 45 degrees around the Y axis
    /// let transform = Transform::new()
    ///     .rotation(Quat::from_rotation_y(std::f32::consts::PI / 4.0));
    /// ```
    pub fn rotation(mut self, rotation: glam::Quat) -> Self {
        self.rotation = rotation;
        self
    }

    /// Sets non-uniform scale factors for each axis.
    ///
    /// Use this when you need different scale values on different axes.
    /// For uniform scaling, prefer [`Transform::uniform_scale()`].
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Vec3};
    ///
    /// // Stretch 2x on X, 0.5x on Y, 1x on Z
    /// let transform = Transform::new().scale(Vec3::new(2.0, 0.5, 1.0));
    /// ```
    pub fn scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Sets uniform scale on all axes.
    ///
    /// This is the most common scaling operation. The scale value is applied
    /// equally to X, Y, and Z.
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Vec3};
    ///
    /// let transform = Transform::new().uniform_scale(2.0);
    /// assert_eq!(transform.scale, Vec3::new(2.0, 2.0, 2.0));
    /// ```
    pub fn uniform_scale(mut self, scale: f32) -> Self {
        self.scale = Vec3::splat(scale);
        self
    }

    /// Converts this transform to a 4×4 transformation matrix.
    ///
    /// The matrix applies transformations in SRT order (Scale, Rotate, Translate),
    /// which is the standard convention for 3D graphics.
    ///
    /// This is called automatically during mesh rendering, but you can use it
    /// directly if you need the raw matrix for custom shaders or calculations.
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Transform, Vec3, Mat4};
    ///
    /// let transform = Transform::new()
    ///     .position(Vec3::new(1.0, 0.0, 0.0))
    ///     .uniform_scale(2.0);
    ///
    /// let matrix = transform.matrix();
    /// // matrix can be passed to shaders or used for point transformation
    /// ```
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}
