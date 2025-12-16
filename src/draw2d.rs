use crate::assets::{Assets, FontId};
use crate::gpu::GpuContext;
use crate::ui::Color;

/// Vertex for 2D sprite/text rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2d {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex2d {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex2d>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // position
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            // uv
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x2,
            },
            // color
            wgpu::VertexAttribute {
                offset: 16,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
}

/// Uniforms for 2D rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Draw2dUniforms {
    resolution: [f32; 2],
    _padding: [f32; 2],
}

const MAX_VERTICES: usize = 16384;

/// Immediate-mode 2D drawing API for sprites and text.
///
/// All draw calls are batched and rendered in a single pass at the end of the frame.
pub struct Draw2d {
    // Pipelines
    colored_pipeline: wgpu::RenderPipeline,
    textured_pipeline: wgpu::RenderPipeline,

    // Shared resources
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,

    // Per-font bind groups (cached)
    font_bind_groups: Vec<Option<wgpu::BindGroup>>,

    // Current frame batches
    colored_vertices: Vec<Vertex2d>,
    text_batches: Vec<(FontId, Vec<Vertex2d>)>,
}

impl Draw2d {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        // Create shaders
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Draw2d Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/draw2d.wgsl").into()),
        });

        // Uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw2d Uniforms"),
            size: std::mem::size_of::<Draw2dUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uniform bind group layout (group 0)
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Draw2d Uniform Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Draw2d Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Texture bind group layout (group 1)
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Draw2d Texture Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Pipeline layouts
        let colored_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Draw2d Colored Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout],
                push_constant_ranges: &[],
            });

        let textured_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Draw2d Textured Pipeline Layout"),
                bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
                push_constant_ranges: &[],
            });

        // Blend state for alpha blending
        let blend_state = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };

        // Colored pipeline (no texture)
        let colored_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Draw2d Colored Pipeline"),
            layout: Some(&colored_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Vertex2d::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_colored"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(blend_state),
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

        // Textured pipeline (for fonts/sprites)
        let textured_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Draw2d Textured Pipeline"),
            layout: Some(&textured_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Vertex2d::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_textured"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(blend_state),
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

        // Vertex buffer
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Draw2d Vertex Buffer"),
            size: (MAX_VERTICES * std::mem::size_of::<Vertex2d>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            colored_pipeline,
            textured_pipeline,
            vertex_buffer,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            font_bind_groups: Vec::new(),
            colored_vertices: Vec::with_capacity(1024),
            text_batches: Vec::new(),
        }
    }

    /// Clear all draw calls for the new frame.
    pub fn clear(&mut self) {
        self.colored_vertices.clear();
        self.text_batches.clear();
    }

    /// Draw a colored rectangle.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        let c = [color.r, color.g, color.b, color.a];
        let uv = [0.0, 0.0]; // Not used for colored quads

        self.colored_vertices.extend_from_slice(&[
            Vertex2d {
                position: [x, y],
                uv,
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv,
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv,
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv,
                color: c,
            },
            Vertex2d {
                position: [x + w, y + h],
                uv,
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv,
                color: c,
            },
        ]);
    }

    /// Draw text at the given position.
    pub fn text(
        &mut self,
        assets: &Assets,
        font_id: FontId,
        x: f32,
        y: f32,
        text: &str,
        color: Color,
    ) {
        let Some(font) = assets.font(font_id) else {
            return;
        };

        let c = [color.r, color.g, color.b, color.a];
        let mut cursor_x = x;
        let baseline_y = y + font.size(); // Offset to baseline

        // Find or create batch for this font
        let batch_idx = self
            .text_batches
            .iter()
            .position(|(id, _)| *id == font_id)
            .unwrap_or_else(|| {
                self.text_batches.push((font_id, Vec::new()));
                self.text_batches.len() - 1
            });

        for ch in text.chars() {
            let Some(glyph) = font.glyph(ch) else {
                cursor_x += font.size() * 0.5; // Fallback advance for missing glyphs
                continue;
            };

            if glyph.width > 0 && glyph.height > 0 {
                let gx = cursor_x + glyph.offset_x;
                // Y offset: fontdue's ymin is distance from baseline to top of glyph
                // We need to go down from baseline, then up by the glyph height
                let gy = baseline_y - glyph.offset_y - glyph.height as f32;

                let gw = glyph.width as f32;
                let gh = glyph.height as f32;

                // UV coordinates from atlas
                let u0 = glyph.uv[0];
                let v0 = glyph.uv[1];
                let u1 = u0 + glyph.uv[2];
                let v1 = v0 + glyph.uv[3];

                self.text_batches[batch_idx].1.extend_from_slice(&[
                    Vertex2d {
                        position: [gx, gy],
                        uv: [u0, v0],
                        color: c,
                    },
                    Vertex2d {
                        position: [gx + gw, gy],
                        uv: [u1, v0],
                        color: c,
                    },
                    Vertex2d {
                        position: [gx, gy + gh],
                        uv: [u0, v1],
                        color: c,
                    },
                    Vertex2d {
                        position: [gx + gw, gy],
                        uv: [u1, v0],
                        color: c,
                    },
                    Vertex2d {
                        position: [gx + gw, gy + gh],
                        uv: [u1, v1],
                        color: c,
                    },
                    Vertex2d {
                        position: [gx, gy + gh],
                        uv: [u0, v1],
                        color: c,
                    },
                ]);
            }

            cursor_x += glyph.advance;
        }
    }

    /// Ensure we have bind groups for all loaded fonts.
    pub(crate) fn update_font_bind_groups(&mut self, gpu: &GpuContext, assets: &Assets) {
        // Grow the bind group cache if needed
        while self.font_bind_groups.len() < assets.fonts.len() {
            self.font_bind_groups.push(None);
        }

        // Create bind groups for any new fonts
        for (i, font) in assets.fonts.iter().enumerate() {
            if self.font_bind_groups[i].is_none() {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Font Bind Group"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&font.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&font.sampler),
                        },
                    ],
                });
                self.font_bind_groups[i] = Some(bind_group);
            }
        }
    }

    /// Render all batched draw calls.
    pub fn render(&self, gpu: &GpuContext, render_pass: &mut wgpu::RenderPass, _assets: &Assets) {
        // Update uniforms
        let uniforms = Draw2dUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            _padding: [0.0, 0.0],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        // Render colored quads
        if !self.colored_vertices.is_empty() {
            gpu.queue.write_buffer(
                &self.vertex_buffer,
                0,
                bytemuck::cast_slice(&self.colored_vertices),
            );

            render_pass.set_pipeline(&self.colored_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(0..self.colored_vertices.len() as u32, 0..1);
        }

        // Render text batches
        let mut offset = self.colored_vertices.len();
        for (font_id, vertices) in &self.text_batches {
            if vertices.is_empty() {
                continue;
            }

            let Some(bind_group) = self
                .font_bind_groups
                .get(font_id.0)
                .and_then(|bg| bg.as_ref())
            else {
                continue;
            };

            gpu.queue.write_buffer(
                &self.vertex_buffer,
                (offset * std::mem::size_of::<Vertex2d>()) as u64,
                bytemuck::cast_slice(vertices),
            );

            render_pass.set_pipeline(&self.textured_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(offset as u32..(offset + vertices.len()) as u32, 0..1);

            offset += vertices.len();
        }
    }
}
