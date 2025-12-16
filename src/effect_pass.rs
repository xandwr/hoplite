use crate::camera::Camera;
use crate::gpu::GpuContext;

/// Standard uniforms available to all effect passes (screen-space).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub _padding: f32,
}

/// Extended uniforms for world-space effect passes (includes camera).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WorldUniforms {
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

/// A full-screen shader effect pass.
///
/// Renders a full-screen triangle with a custom fragment shader.
/// Can be used for screen-space effects (post-processing, UI backgrounds)
/// or world-space effects (raymarching, black holes) when `.with_camera()` is called.
pub struct EffectPass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    uses_camera: bool,
}

impl EffectPass {
    /// Create a new screen-space effect pass from WGSL shader source.
    ///
    /// The shader should define:
    /// ```wgsl
    /// struct Uniforms {
    ///     resolution: vec2f,
    ///     time: f32,
    /// }
    /// @group(0) @binding(0) var<uniform> u: Uniforms;
    /// ```
    pub fn new(gpu: &GpuContext, shader_source: &str) -> Self {
        Self::create(gpu, shader_source, false)
    }

    /// Create a new world-space effect pass from WGSL shader source.
    ///
    /// The shader should define:
    /// ```wgsl
    /// struct Uniforms {
    ///     resolution: vec2f,
    ///     time: f32,
    ///     fov: f32,
    ///     camera_pos: vec3f,
    ///     camera_forward: vec3f,
    ///     camera_right: vec3f,
    ///     camera_up: vec3f,
    ///     aspect: f32,
    /// }
    /// @group(0) @binding(0) var<uniform> u: Uniforms;
    /// ```
    pub fn new_world(gpu: &GpuContext, shader_source: &str) -> Self {
        Self::create(gpu, shader_source, true)
    }

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

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Returns whether this effect pass uses camera data.
    pub fn uses_camera(&self) -> bool {
        self.uses_camera
    }
}
