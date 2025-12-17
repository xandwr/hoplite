//! GPU rendering pass for scene transitions.
//!
//! This module provides shaders and pipelines for rendering transition effects
//! between scenes, including fade-to-color and crossfade transitions.

use crate::draw2d::Color;
use crate::gpu::GpuContext;

/// Uniforms for transition rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TransitionUniforms {
    /// Screen resolution in pixels.
    resolution: [f32; 2],
    /// Blend progress (interpretation depends on transition type).
    progress: f32,
    /// Padding for alignment.
    _pad: f32,
    /// Overlay/fade color (RGBA).
    color: [f32; 4],
}

/// GPU resources for rendering scene transitions.
pub struct TransitionPass {
    /// Pipeline for fade-to-color overlay.
    fade_pipeline: wgpu::RenderPipeline,
    /// Pipeline for crossfade blending.
    crossfade_pipeline: wgpu::RenderPipeline,
    /// Uniform buffer.
    uniform_buffer: wgpu::Buffer,
    /// Bind group layout for fade (uniforms + one texture).
    fade_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group layout for crossfade (uniforms + two textures).
    crossfade_bind_group_layout: wgpu::BindGroupLayout,
    /// Texture sampler.
    sampler: wgpu::Sampler,
}

impl TransitionPass {
    /// Create a new transition pass with GPU resources.
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        // Create fade shader
        let fade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fade Transition Shader"),
            source: wgpu::ShaderSource::Wgsl(FADE_SHADER.into()),
        });

        // Create crossfade shader
        let crossfade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Crossfade Transition Shader"),
            source: wgpu::ShaderSource::Wgsl(CROSSFADE_SHADER.into()),
        });

        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transition Uniforms"),
            size: std::mem::size_of::<TransitionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Transition Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Fade bind group layout (uniforms + scene texture + sampler)
        let fade_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Fade Transition Bind Group Layout"),
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
                    // Scene texture
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

        // Crossfade bind group layout (uniforms + old texture + new texture + sampler)
        let crossfade_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Crossfade Transition Bind Group Layout"),
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
                    // Old scene texture
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
                    // New scene texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
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
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create fade pipeline
        let fade_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fade Transition Pipeline Layout"),
            bind_group_layouts: &[&fade_bind_group_layout],
            push_constant_ranges: &[],
        });

        let fade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fade Transition Pipeline"),
            layout: Some(&fade_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &fade_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &fade_shader,
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

        // Create crossfade pipeline
        let crossfade_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Crossfade Transition Pipeline Layout"),
                bind_group_layouts: &[&crossfade_bind_group_layout],
                push_constant_ranges: &[],
            });

        let crossfade_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Crossfade Transition Pipeline"),
            layout: Some(&crossfade_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &crossfade_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &crossfade_shader,
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
            fade_pipeline,
            crossfade_pipeline,
            uniform_buffer,
            fade_bind_group_layout,
            crossfade_bind_group_layout,
            sampler,
        }
    }

    /// Render a fade-to-color transition.
    ///
    /// This blends the scene texture with a solid color based on the progress.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `encoder` - Command encoder to record into
    /// * `target` - Target texture view to render to
    /// * `scene_view` - Scene texture to blend with color
    /// * `color` - The fade color
    /// * `overlay_alpha` - How much of the color to show (0.0 = all scene, 1.0 = all color)
    pub fn render_fade(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene_view: &wgpu::TextureView,
        color: Color,
        overlay_alpha: f32,
    ) {
        let uniforms = TransitionUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            progress: overlay_alpha,
            _pad: 0.0,
            color: [color.r, color.g, color.b, color.a],
        };

        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fade Transition Bind Group"),
            layout: &self.fade_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Fade Transition Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.fade_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Render a crossfade transition between two scenes.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `encoder` - Command encoder to record into
    /// * `target` - Target texture view to render to
    /// * `old_scene_view` - Texture of the outgoing scene
    /// * `new_scene_view` - Texture of the incoming scene
    /// * `blend` - Blend factor (0.0 = all old, 1.0 = all new)
    pub fn render_crossfade(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        old_scene_view: &wgpu::TextureView,
        new_scene_view: &wgpu::TextureView,
        blend: f32,
    ) {
        let uniforms = TransitionUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            progress: blend,
            _pad: 0.0,
            color: [0.0, 0.0, 0.0, 0.0], // Not used for crossfade
        };

        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Crossfade Transition Bind Group"),
            layout: &self.crossfade_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(old_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(new_scene_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Crossfade Transition Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.crossfade_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// Fade transition shader - blends scene with solid color.
const FADE_SHADER: &str = r#"
struct Uniforms {
    resolution: vec2f,
    progress: f32,
    _pad: f32,
    color: vec4f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var scene_texture: texture_2d<f32>;
@group(0) @binding(2) var scene_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    // Fullscreen triangle
    let x = f32(i32(vi) - 1);
    let y = f32(i32(vi & 1u) * 2 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = pos.xy / u.resolution;
    let scene = textureSample(scene_texture, scene_sampler, uv);

    // Blend scene with overlay color based on progress
    // progress = 0: full scene, progress = 1: full overlay color
    return mix(scene, u.color, u.progress);
}
"#;

/// Crossfade transition shader - blends two scenes together.
const CROSSFADE_SHADER: &str = r#"
struct Uniforms {
    resolution: vec2f,
    progress: f32,
    _pad: f32,
    color: vec4f, // Not used, but keeps struct consistent
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var old_texture: texture_2d<f32>;
@group(0) @binding(2) var new_texture: texture_2d<f32>;
@group(0) @binding(3) var tex_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
    // Fullscreen triangle
    let x = f32(i32(vi) - 1);
    let y = f32(i32(vi & 1u) * 2 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = pos.xy / u.resolution;
    let old_scene = textureSample(old_texture, tex_sampler, uv);
    let new_scene = textureSample(new_texture, tex_sampler, uv);

    // Crossfade: blend from old to new based on progress
    return mix(old_scene, new_scene, u.progress);
}
"#;
