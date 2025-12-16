use crate::camera::Camera;
use crate::gpu::GpuContext;

/// Standard uniforms for post-processing passes.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PostProcessUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub _padding: f32,
}

/// Extended uniforms for world-space post-processing (includes camera).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldPostProcessUniforms {
    // Base uniforms
    pub resolution: [f32; 2],
    pub time: f32,
    pub fov: f32,
    // Camera
    pub camera_pos: [f32; 3],
    pub _pad1: f32,
    pub camera_forward: [f32; 3],
    pub _pad2: f32,
    pub camera_right: [f32; 3],
    pub _pad3: f32,
    pub camera_up: [f32; 3],
    pub aspect: f32,
}

/// A post-processing pass that samples from an input texture.
///
/// The shader receives:
/// - `u.resolution`: screen resolution
/// - `u.time`: elapsed time
/// - `input_texture` + `input_sampler`: the previous pass output
pub struct PostProcessPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PostProcessPass {
    /// Create a new post-processing pass from WGSL shader source.
    ///
    /// The shader should define:
    /// ```wgsl
    /// struct Uniforms {
    ///     resolution: vec2f,
    ///     time: f32,
    /// }
    /// @group(0) @binding(0) var<uniform> u: Uniforms;
    /// @group(0) @binding(1) var input_texture: texture_2d<f32>;
    /// @group(0) @binding(2) var input_sampler: sampler;
    /// ```
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

    /// Create a bind group for the given input texture.
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

    /// Render the post-process effect.
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

/// A world-space post-processing pass with camera uniforms.
///
/// The shader receives:
/// - `u.resolution`: screen resolution
/// - `u.time`: elapsed time
/// - `u.fov`: field of view in radians
/// - `u.camera_pos`, `u.camera_forward`, `u.camera_right`, `u.camera_up`: camera vectors
/// - `u.aspect`: aspect ratio
/// - `input_texture` + `input_sampler`: the previous pass output
pub struct WorldPostProcessPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl WorldPostProcessPass {
    /// Create a new world-space post-processing pass from WGSL shader source.
    ///
    /// The shader should define:
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

    /// Create a bind group for the given input texture.
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

    /// Render the post-process effect with camera data.
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
            camera_pos: camera.position,
            _pad1: 0.0,
            camera_forward: camera.forward,
            _pad2: 0.0,
            camera_right: camera.right(),
            _pad3: 0.0,
            camera_up: camera.orthogonal_up(),
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
