//! Fullscreen shader effect passes for screen-space and world-space rendering.
//!
//! This module provides [`EffectPass`], a rendering primitive for fullscreen shader effects.
//! It renders a single triangle that covers the entire screen, allowing custom fragment shaders
//! to process every pixel.
//!
//! # Two Modes of Operation
//!
//! - **Screen-space** ([`EffectPass::new`]): Basic uniforms with resolution and time.
//!   Suitable for 2D effects like color grading, vignettes, and procedural backgrounds.
//!
//! - **World-space** ([`EffectPass::new_world`]): Extended uniforms including camera
//!   position and orientation. Suitable for raymarching, volumetric effects, and any
//!   technique that needs to cast rays into 3D space.
//!
//! # Shader Requirements
//!
//! Your WGSL shader must define vertex and fragment entry points named `vs` and `fs`.
//! The vertex shader should generate fullscreen triangle coordinates:
//!
//! ```wgsl
//! @vertex
//! fn vs(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
//!     let x = f32(i32(vertex_index) - 1);
//!     let y = f32(i32(vertex_index & 1u) * 2 - 1);
//!     return vec4f(x, y, 0.0, 1.0);
//! }
//! ```
//!
//! # Example
//!
//! ```no_run
//! use hoplite::{GpuContext, EffectPass};
//!
//! let shader = r#"
//!     struct Uniforms { resolution: vec2f, time: f32 }
//!     @group(0) @binding(0) var<uniform> u: Uniforms;
//!
//!     @vertex
//!     fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
//!         let x = f32(i32(vi) - 1);
//!         let y = f32(i32(vi & 1u) * 2 - 1);
//!         return vec4f(x, y, 0.0, 1.0);
//!     }
//!
//!     @fragment
//!     fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
//!         let uv = pos.xy / u.resolution;
//!         return vec4f(uv, 0.5 + 0.5 * sin(u.time), 1.0);
//!     }
//! "#;
//!
//! let effect = EffectPass::new(&gpu, shader);
//! effect.render(&gpu, &mut render_pass, time);
//! ```

use crate::camera::Camera;
use crate::gpu::GpuContext;

/// Standard uniforms available to all screen-space effect passes.
///
/// This struct is uploaded to the GPU each frame and bound at `@group(0) @binding(0)`.
/// The layout matches WGSL struct alignment requirements (16-byte aligned).
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct Uniforms {
///     resolution: vec2f,
///     time: f32,
/// }
/// @group(0) @binding(0) var<uniform> u: Uniforms;
/// ```
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniforms {
    /// Render target resolution in pixels `[width, height]`.
    pub resolution: [f32; 2],
    /// Elapsed time in seconds since application start.
    pub time: f32,
    /// Padding for 16-byte alignment.
    pub _padding: f32,
}

/// Extended uniforms for world-space effect passes.
///
/// Includes all [`ScreenUniforms`] fields plus camera position, orientation vectors,
/// field of view, and aspect ratio. This allows shaders to reconstruct world-space
/// rays for raymarching, volumetric effects, and similar techniques.
///
/// # WGSL Declaration
///
/// ```wgsl
/// struct Uniforms {
///     resolution: vec2f,
///     time: f32,
///     fov: f32,
///     camera_pos: vec3f,
///     // Note: padding fields are implicit in WGSL
///     camera_forward: vec3f,
///     camera_right: vec3f,
///     camera_up: vec3f,
///     aspect: f32,
/// }
/// @group(0) @binding(0) var<uniform> u: Uniforms;
/// ```
///
/// # Ray Construction
///
/// To cast a ray from the camera through a pixel:
///
/// ```wgsl
/// let uv = (pos.xy / u.resolution) * 2.0 - 1.0;
/// let ray_dir = normalize(
///     u.camera_forward +
///     uv.x * u.aspect * tan(u.fov * 0.5) * u.camera_right +
///     uv.y * tan(u.fov * 0.5) * u.camera_up
/// );
/// ```
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldUniforms {
    /// Render target resolution in pixels `[width, height]`.
    pub resolution: [f32; 2],
    /// Elapsed time in seconds since application start.
    pub time: f32,
    /// Vertical field of view in radians.
    pub fov: f32,
    /// Camera position in world space.
    pub camera_pos: [f32; 3],
    /// Padding for 16-byte alignment.
    pub _pad1: f32,
    /// Camera forward direction (normalized).
    pub camera_forward: [f32; 3],
    /// Padding for 16-byte alignment.
    pub _pad2: f32,
    /// Camera right direction (normalized).
    pub camera_right: [f32; 3],
    /// Padding for 16-byte alignment.
    pub _pad3: f32,
    /// Camera up direction (normalized, orthogonal to forward).
    pub camera_up: [f32; 3],
    /// Aspect ratio (width / height).
    pub aspect: f32,
}

/// A fullscreen shader effect pass.
///
/// Renders a fullscreen triangle with a custom fragment shader, providing a simple
/// way to implement screen-space or world-space effects. The pass handles uniform
/// buffer management, bind group creation, and pipeline setup.
///
/// # Modes
///
/// - **Screen-space** (via [`new`](Self::new)): Uses `ScreenUniforms` with resolution and time.
/// - **World-space** (via [`new_world`](Self::new_world)): Uses `WorldUniforms` with full camera data.
///
/// # Example
///
/// ```no_run
/// use hoplite::{GpuContext, EffectPass};
///
/// // Screen-space effect
/// let vignette = EffectPass::new(&gpu, include_str!("shaders/vignette.wgsl"));
/// vignette.render(&gpu, &mut render_pass, time);
///
/// // World-space effect (raymarching)
/// let raymarch = EffectPass::new_world(&gpu, include_str!("shaders/raymarch.wgsl"));
/// raymarch.render_with_camera(&gpu, &mut render_pass, time, &camera);
/// ```
pub struct EffectPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uses_camera: bool,
}

impl EffectPass {
    /// Create a new screen-space effect pass from WGSL shader source.
    ///
    /// The shader receives `ScreenUniforms` at `@group(0) @binding(0)` and must
    /// define `vs` and `fs` entry points. Use [`render`](Self::render) to draw.
    ///
    /// See the module-level documentation for shader requirements.
    pub fn new(gpu: &GpuContext, shader_source: &str) -> Self {
        Self::create(gpu, shader_source, false)
    }

    /// Create a new world-space effect pass from WGSL shader source.
    ///
    /// The shader receives `WorldUniforms` at `@group(0) @binding(0)` and must
    /// define `vs` and `fs` entry points. Use [`render_with_camera`](Self::render_with_camera) to draw.
    ///
    /// See `WorldUniforms` for the uniform layout and ray construction example.
    pub fn new_world(gpu: &GpuContext, shader_source: &str) -> Self {
        Self::create(gpu, shader_source, true)
    }

    /// Internal constructor that creates the pipeline and resources.
    fn create(gpu: &GpuContext, shader_source: &str, uses_camera: bool) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Effect Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let buffer_size = if uses_camera {
            std::mem::size_of::<WorldUniforms>()
        } else {
            std::mem::size_of::<ScreenUniforms>()
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Effect Uniforms"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Effect Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Effect Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Effect Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Effect Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group,
            uses_camera,
        }
    }

    /// Render a screen-space effect (no camera).
    ///
    /// Uploads `ScreenUniforms` and draws a fullscreen triangle.
    ///
    /// # Panics
    ///
    /// Panics if this effect was created with [`new_world`](Self::new_world).
    /// Use [`render_with_camera`](Self::render_with_camera) for world-space effects.
    pub fn render(&self, gpu: &GpuContext, render_pass: &mut wgpu::RenderPass, time: f32) {
        assert!(
            !self.uses_camera,
            "This effect requires a camera. Use render_with_camera() instead."
        );

        let uniforms = ScreenUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            time,
            _padding: 0.0,
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Render a world-space effect with camera data.
    ///
    /// Uploads `WorldUniforms` (including camera position and orientation) and draws
    /// a fullscreen triangle. The shader can use this data to cast rays into the scene.
    ///
    /// # Panics
    ///
    /// Panics if this effect was created with [`new`](Self::new).
    /// Use [`render`](Self::render) for screen-space effects.
    pub fn render_with_camera(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        camera: &Camera,
    ) {
        assert!(
            self.uses_camera,
            "This effect doesn't use a camera. Use render() instead."
        );

        let uniforms = WorldUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            time,
            fov: camera.fov,
            camera_pos: camera.position.to_array(),
            _pad1: 0.0,
            camera_forward: camera.forward.to_array(),
            _pad2: 0.0,
            camera_right: camera.right().to_array(),
            _pad3: 0.0,
            camera_up: camera.orthogonal_up().to_array(),
            aspect: gpu.aspect(),
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Returns whether this effect pass uses camera data.
    ///
    /// If `true`, use [`render_with_camera`](Self::render_with_camera).
    /// If `false`, use [`render`](Self::render).
    pub fn uses_camera(&self) -> bool {
        self.uses_camera
    }
}
