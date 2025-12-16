use crate::gpu::GpuContext;
use glam::{Mat4, Vec3};

/// Vertex for 3D mesh rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex3d {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex3d {
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

    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> Self {
        Self {
            position,
            normal,
            uv,
        }
    }
}

/// A 3D mesh with vertex and index buffers.
#[derive(Debug)]
pub struct Mesh {
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) index_buffer: wgpu::Buffer,
    pub(crate) index_count: u32,
}

impl Mesh {
    /// Create a mesh from raw vertex and index data.
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

    /// Create a unit cube centered at the origin.
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

    /// Create a unit sphere centered at the origin.
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

    /// Create a flat plane on the XZ axis.
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

/// A transform for positioning meshes in 3D space.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: glam::Quat,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            ..Default::default()
        }
    }

    pub fn position(mut self, position: Vec3) -> Self {
        self.position = position;
        self
    }

    pub fn rotation(mut self, rotation: glam::Quat) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn uniform_scale(mut self, scale: f32) -> Self {
        self.scale = Vec3::splat(scale);
        self
    }

    /// Convert to a 4x4 transformation matrix.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}
