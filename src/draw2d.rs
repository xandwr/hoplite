use crate::assets::{Assets, FontId};
use crate::gpu::GpuContext;
use crate::texture::Sprite;

/// Index into the sprite storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpriteId(pub usize);

/// A rectangle in screen-space pixel coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// RGBA color with premultiplied alpha.
#[derive(Clone, Copy, Debug)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    pub const WHITE: Color = Color::rgba(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Color = Color::rgba(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);

    /// Semi-transparent dark background for debug panels.
    pub const DEBUG_BG: Color = Color::rgba(0.1, 0.1, 0.1, 0.85);
    /// Accent color for borders.
    pub const DEBUG_BORDER: Color = Color::rgba(0.4, 0.4, 0.4, 1.0);
}

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
    sprite_pipeline: wgpu::RenderPipeline,

    // Shared resources
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,

    // Per-font bind groups (cached)
    font_bind_groups: Vec<Option<wgpu::BindGroup>>,

    // Sprite storage and bind groups
    pub(crate) sprites: Vec<Sprite>,
    sprite_bind_groups: Vec<Option<wgpu::BindGroup>>,

    // Current frame batches
    colored_vertices: Vec<Vertex2d>,
    text_batches: Vec<(FontId, Vec<Vertex2d>)>,
    sprite_batches: Vec<(SpriteId, Vec<Vertex2d>)>,
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

        // Textured pipeline (for fonts - uses R8 alpha mask)
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

        // Sprite pipeline (for RGBA sprites)
        let sprite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Draw2d Sprite Pipeline"),
            layout: Some(&textured_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Vertex2d::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sprite"),
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
            sprite_pipeline,
            vertex_buffer,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            font_bind_groups: Vec::new(),
            sprites: Vec::new(),
            sprite_bind_groups: Vec::new(),
            colored_vertices: Vec::with_capacity(1024),
            text_batches: Vec::new(),
            sprite_batches: Vec::new(),
        }
    }

    /// Add a sprite and return its ID.
    pub fn add_sprite(&mut self, sprite: Sprite) -> SpriteId {
        let id = SpriteId(self.sprites.len());
        self.sprites.push(sprite);
        self.sprite_bind_groups.push(None); // Will be created lazily
        id
    }

    /// Get a sprite by ID.
    pub fn get_sprite(&self, id: SpriteId) -> Option<&Sprite> {
        self.sprites.get(id.0)
    }

    /// Clear all draw calls for the new frame.
    pub fn clear(&mut self) {
        self.colored_vertices.clear();
        self.text_batches.clear();
        self.sprite_batches.clear();
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

    /// Draw a sprite at the given position.
    ///
    /// The sprite is drawn at its native size. Use `sprite_scaled` for custom sizing.
    pub fn sprite(&mut self, sprite_id: SpriteId, x: f32, y: f32, tint: Color) {
        let Some(sprite) = self.sprites.get(sprite_id.0) else {
            return;
        };
        let w = sprite.width as f32;
        let h = sprite.height as f32;
        self.sprite_rect(sprite_id, x, y, w, h, tint);
    }

    /// Draw a sprite scaled to fit a rectangle.
    pub fn sprite_scaled(
        &mut self,
        sprite_id: SpriteId,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        tint: Color,
    ) {
        self.sprite_rect(sprite_id, x, y, w, h, tint);
    }

    /// Draw a sprite with a sub-region (for sprite sheets).
    ///
    /// `src_rect` defines the source rectangle in pixel coordinates within the sprite.
    pub fn sprite_region(
        &mut self,
        sprite_id: SpriteId,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        src_x: f32,
        src_y: f32,
        src_w: f32,
        src_h: f32,
        tint: Color,
    ) {
        let Some(sprite) = self.sprites.get(sprite_id.0) else {
            return;
        };

        let tex_w = sprite.width as f32;
        let tex_h = sprite.height as f32;

        // Convert source rect to UV coordinates
        let u0 = src_x / tex_w;
        let v0 = src_y / tex_h;
        let u1 = (src_x + src_w) / tex_w;
        let v1 = (src_y + src_h) / tex_h;

        let c = [tint.r, tint.g, tint.b, tint.a];

        // Find or create batch for this sprite
        let batch_idx = self
            .sprite_batches
            .iter()
            .position(|(id, _)| *id == sprite_id)
            .unwrap_or_else(|| {
                self.sprite_batches.push((sprite_id, Vec::new()));
                self.sprite_batches.len() - 1
            });

        self.sprite_batches[batch_idx].1.extend_from_slice(&[
            Vertex2d {
                position: [x, y],
                uv: [u0, v0],
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv: [u1, v0],
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv: [u0, v1],
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv: [u1, v0],
                color: c,
            },
            Vertex2d {
                position: [x + w, y + h],
                uv: [u1, v1],
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv: [u0, v1],
                color: c,
            },
        ]);
    }

    /// Internal: draw a sprite filling a rectangle with full UV range.
    fn sprite_rect(&mut self, sprite_id: SpriteId, x: f32, y: f32, w: f32, h: f32, tint: Color) {
        let c = [tint.r, tint.g, tint.b, tint.a];

        // Find or create batch for this sprite
        let batch_idx = self
            .sprite_batches
            .iter()
            .position(|(id, _)| *id == sprite_id)
            .unwrap_or_else(|| {
                self.sprite_batches.push((sprite_id, Vec::new()));
                self.sprite_batches.len() - 1
            });

        self.sprite_batches[batch_idx].1.extend_from_slice(&[
            Vertex2d {
                position: [x, y],
                uv: [0.0, 0.0],
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv: [1.0, 0.0],
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv: [0.0, 1.0],
                color: c,
            },
            Vertex2d {
                position: [x + w, y],
                uv: [1.0, 0.0],
                color: c,
            },
            Vertex2d {
                position: [x + w, y + h],
                uv: [1.0, 1.0],
                color: c,
            },
            Vertex2d {
                position: [x, y + h],
                uv: [0.0, 1.0],
                color: c,
            },
        ]);
    }

    /// Draw a bordered panel with optional title bar.
    ///
    /// This is a convenience method for drawing debug overlays and UI panels.
    /// For more control, use the individual `rect()` and `text()` methods.
    pub fn panel(&mut self, x: f32, y: f32, width: f32, height: f32) -> PanelBuilder<'_> {
        PanelBuilder {
            draw2d: self,
            x,
            y,
            width,
            height,
            background: Color::DEBUG_BG,
            border: Some(Color::DEBUG_BORDER),
            title: None,
            title_font: None,
        }
    }

    /// Ensure we have bind groups for all loaded fonts and sprites.
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

        // Create bind groups for any new sprites
        for (i, sprite) in self.sprites.iter().enumerate() {
            if self
                .sprite_bind_groups
                .get(i)
                .map(|bg| bg.is_none())
                .unwrap_or(true)
            {
                let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Sprite Bind Group"),
                    layout: &self.texture_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&sprite.view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sprite.sampler),
                        },
                    ],
                });
                if i >= self.sprite_bind_groups.len() {
                    self.sprite_bind_groups.push(Some(bind_group));
                } else {
                    self.sprite_bind_groups[i] = Some(bind_group);
                }
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

        // Render sprite batches
        for (sprite_id, vertices) in &self.sprite_batches {
            if vertices.is_empty() {
                continue;
            }

            let Some(bind_group) = self
                .sprite_bind_groups
                .get(sprite_id.0)
                .and_then(|bg| bg.as_ref())
            else {
                continue;
            };

            gpu.queue.write_buffer(
                &self.vertex_buffer,
                (offset * std::mem::size_of::<Vertex2d>()) as u64,
                bytemuck::cast_slice(vertices),
            );

            render_pass.set_pipeline(&self.sprite_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.set_bind_group(1, bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.draw(offset as u32..(offset + vertices.len()) as u32, 0..1);

            offset += vertices.len();
        }
    }
}

/// Builder for drawing panels with backgrounds, borders, and titles.
pub struct PanelBuilder<'a> {
    draw2d: &'a mut Draw2d,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    background: Color,
    border: Option<Color>,
    title: Option<String>,
    title_font: Option<FontId>,
}

impl<'a> PanelBuilder<'a> {
    /// Set the background color.
    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Set the border color.
    pub fn border(mut self, color: Color) -> Self {
        self.border = Some(color);
        self
    }

    /// Remove the border.
    pub fn no_border(mut self) -> Self {
        self.border = None;
        self
    }

    /// Add a title bar with the given text and font.
    pub fn title(mut self, text: impl Into<String>, font: FontId) -> Self {
        self.title = Some(text.into());
        self.title_font = Some(font);
        self
    }

    /// Draw the panel. Call this to finalize and render the panel.
    pub fn draw(self, assets: &Assets) {
        let border_width = 1.0;
        let title_height = 22.0;

        // Draw background
        self.draw2d
            .rect(self.x, self.y, self.width, self.height, self.background);

        // Draw border if present
        if let Some(border_color) = self.border {
            // Top
            self.draw2d
                .rect(self.x, self.y, self.width, border_width, border_color);
            // Bottom
            self.draw2d.rect(
                self.x,
                self.y + self.height - border_width,
                self.width,
                border_width,
                border_color,
            );
            // Left
            self.draw2d
                .rect(self.x, self.y, border_width, self.height, border_color);
            // Right
            self.draw2d.rect(
                self.x + self.width - border_width,
                self.y,
                border_width,
                self.height,
                border_color,
            );
        }

        // Draw title bar if present
        if let (Some(title_text), Some(font_id)) = (&self.title, self.title_font) {
            let title_bg = Color::rgba(0.15, 0.15, 0.15, 0.95);
            self.draw2d
                .rect(self.x, self.y, self.width, title_height, title_bg);
            self.draw2d.text(
                assets,
                font_id,
                self.x + 8.0,
                self.y + 4.0,
                title_text,
                Color::WHITE,
            );
        }
    }
}
