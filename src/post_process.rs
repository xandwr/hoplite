//! Post-processing pass infrastructure for fullscreen effects.
//!
//! This module provides two types of post-processing passes:
//!
//! - [`PostProcessPass`]: A simple post-processing pass with basic uniforms (resolution, time)
//! - [`WorldPostProcessPass`]: An extended pass that includes camera information for world-space effects
//!
//! Both passes render a fullscreen triangle and sample from an input texture, making them
//! suitable for effects like bloom, color grading, vignette, fog, and screen-space reflections.
//!
//! # Shader Requirements
//!
//! Your WGSL shader must define a vertex shader entry point `vs` and fragment shader entry point `fs`.
//! The vertex shader should generate fullscreen triangle coordinates. A typical implementation:
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
//! use hoplite::{GpuContext, PostProcessPass};
//!
//! let shader_source = r#"
//!     struct Uniforms {
//!         resolution: vec2f,
//!         time: f32,
//!     }
//!     @group(0) @binding(0) var<uniform> u: Uniforms;
//!     @group(0) @binding(1) var input_texture: texture_2d<f32>;
//!     @group(0) @binding(2) var input_sampler: sampler;
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
//!         return textureSample(input_texture, input_sampler, uv);
//!     }
//! "#;
//!
//! // let pass = PostProcessPass::new(&gpu, shader_source);
//! ```

use crate::camera::Camera;
use crate::gpu::GpuContext;

/// Standard uniforms for post-processing passes.
///
/// This struct is uploaded to the GPU as a uniform buffer and provides
/// basic information needed by most post-processing shaders.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessUniforms {
    /// Screen resolution in pixels as `[width, height]`.
    pub resolution: [f32; 2],
    /// Elapsed time in seconds since the application started.
    pub time: f32,
    /// Padding for 16-byte alignment (required by GPU uniform buffers).
    pub _padding: f32,
}

/// Extended uniforms for world-space post-processing effects.
///
/// This struct extends `PostProcessUniforms` with camera information,
/// enabling effects that need to reconstruct world-space positions or
/// ray directions (e.g., volumetric fog, screen-space reflections, depth-based effects).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldPostProcessUniforms {
    /// Screen resolution in pixels as `[width, height]`.
    pub resolution: [f32; 2],
    /// Elapsed time in seconds since the application started.
    pub time: f32,
    /// Camera field of view in radians.
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

/// A post-processing pass that samples from an input texture.
///
/// This pass provides a simple interface for fullscreen post-processing effects.
/// It automatically manages uniform buffers, bind groups, and pipeline state.
///
/// # Shader Bindings
///
/// Your shader receives the following bindings:
///
/// | Binding | Type | Description |
/// |---------|------|-------------|
/// | 0 | `uniform` | Uniforms with `resolution` and `time` fields |
/// | 1 | `texture_2d<f32>` | Input texture from the previous pass |
/// | 2 | `sampler` | Linear filtering sampler for the input texture |
///
pub struct PostProcessPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PostProcessPass {
    /// Creates a new post-processing pass from WGSL shader source.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context to create resources on
    /// * `shader_source` - WGSL shader source code
    ///
    /// # Shader Requirements
    ///
    /// The shader must define:
    /// - A vertex entry point named `vs`
    /// - A fragment entry point named `fs`
    /// - The following bindings:
    ///
    /// ```wgsl
    /// struct Uniforms {
    ///     resolution: vec2f,
    ///     time: f32,
    /// }
    /// @group(0) @binding(0) var<uniform> u: Uniforms;
    /// @group(0) @binding(1) var input_texture: texture_2d<f32>;
    /// @group(0) @binding(2) var input_sampler: sampler;
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shader source fails to compile.
    pub fn new(gpu: &GpuContext, shader_source: &str) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Uniforms"),
            size: std::mem::size_of::<PostProcessUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Pipeline"),
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
            bind_group_layout,
            sampler,
        }
    }

    /// Creates a bind group for the given input texture.
    ///
    /// This is useful when you need to manage bind groups manually,
    /// for example when caching them across frames for performance.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `input_view` - Texture view to sample from in the shader
    pub fn create_bind_group(
        &self,
        gpu: &GpuContext,
        input_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Renders the post-processing effect to the current render pass.
    ///
    /// This method updates the uniform buffer with the current resolution and time,
    /// creates a bind group for the input texture, and issues a draw call for a
    /// fullscreen triangle (3 vertices).
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context (used for resolution and queue access)
    /// * `render_pass` - The active render pass to draw into
    /// * `time` - Current time in seconds (passed to shader as `u.time`)
    /// * `input_view` - Texture view from the previous pass to sample from
    pub fn render(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        input_view: &wgpu::TextureView,
    ) {
        let uniforms = PostProcessUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            time,
            _padding: 0.0,
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = self.create_bind_group(gpu, input_view);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// A post-processing pass with camera uniforms for world-space effects.
///
/// This pass extends [`PostProcessPass`] with full camera information, enabling
/// effects that need to reconstruct world-space positions or compute view rays.
///
/// # Use Cases
///
/// - Volumetric fog and atmospheric scattering
/// - Screen-space reflections (SSR)
/// - Depth-based effects (DOF, god rays)
/// - Ray marching effects that need camera rays
///
/// # Shader Bindings
///
/// Your shader receives the following bindings:
///
/// | Binding | Type | Description |
/// |---------|------|-------------|
/// | 0 | `uniform` | Uniforms with resolution, time, and camera data |
/// | 1 | `texture_2d<f32>` | Input texture from the previous pass |
/// | 2 | `sampler` | Linear filtering sampler for the input texture |
///
/// # Example: Computing View Rays
///
/// ```wgsl
/// fn get_ray_direction(uv: vec2f) -> vec3f {
///     let ndc = uv * 2.0 - 1.0;
///     let half_height = tan(u.fov * 0.5);
///     let half_width = half_height * u.aspect;
///     return normalize(
///         u.camera_forward +
///         u.camera_right * ndc.x * half_width +
///         u.camera_up * ndc.y * half_height
///     );
/// }
/// ```
pub struct WorldPostProcessPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl WorldPostProcessPass {
    /// Creates a new world-space post-processing pass from WGSL shader source.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context to create resources on
    /// * `shader_source` - WGSL shader source code
    ///
    /// # Shader Requirements
    ///
    /// The shader must define:
    /// - A vertex entry point named `vs`
    /// - A fragment entry point named `fs`
    /// - The following bindings:
    ///
    /// ```wgsl
    /// struct Uniforms {
    ///     resolution: vec2f,
    ///     time: f32,
    ///     fov: f32,
    ///     camera_pos: vec3f,
    ///     _pad1: f32,
    ///     camera_forward: vec3f,
    ///     _pad2: f32,
    ///     camera_right: vec3f,
    ///     _pad3: f32,
    ///     camera_up: vec3f,
    ///     aspect: f32,
    /// }
    /// @group(0) @binding(0) var<uniform> u: Uniforms;
    /// @group(0) @binding(1) var input_texture: texture_2d<f32>;
    /// @group(0) @binding(2) var input_sampler: sampler;
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the shader source fails to compile.
    pub fn new(gpu: &GpuContext, shader_source: &str) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("WorldPostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WorldPostProcess Uniforms"),
            size: std::mem::size_of::<WorldPostProcessUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("WorldPostProcess Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("WorldPostProcess Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Input texture
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("WorldPostProcess Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("WorldPostProcess Pipeline"),
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
            bind_group_layout,
            sampler,
        }
    }

    /// Creates a bind group for the given input texture.
    ///
    /// This is useful when you need to manage bind groups manually,
    /// for example when caching them across frames for performance.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `input_view` - Texture view to sample from in the shader
    pub fn create_bind_group(
        &self,
        gpu: &GpuContext,
        input_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WorldPostProcess Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// Renders the post-processing effect with camera data.
    ///
    /// This method updates the uniform buffer with the current resolution, time,
    /// and camera state, creates a bind group for the input texture, and issues
    /// a draw call for a fullscreen triangle (3 vertices).
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context (used for resolution and queue access)
    /// * `render_pass` - The active render pass to draw into
    /// * `time` - Current time in seconds (passed to shader as `u.time`)
    /// * `camera` - Camera to extract position and orientation from
    /// * `input_view` - Texture view from the previous pass to sample from
    pub fn render(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        camera: &Camera,
        input_view: &wgpu::TextureView,
    ) {
        let uniforms = WorldPostProcessUniforms {
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

        let bind_group = self.create_bind_group(gpu, input_view);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
