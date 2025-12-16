//! Immediate-mode 2D drawing API for sprites, text, and UI elements.
//!
//! This module provides a simple, batched 2D rendering system built on top of wgpu.
//! All draw calls are collected during the frame and rendered in a single pass,
//! minimizing GPU state changes and draw calls.
//!
//! # Architecture
//!
//! The rendering system uses three separate pipelines:
//! - **Colored pipeline**: For solid-color rectangles (no texture sampling)
//! - **Textured pipeline**: For font rendering (R8 alpha mask textures)
//! - **Sprite pipeline**: For RGBA sprite rendering
//!
//! Draw calls are batched by texture to minimize bind group switches. Each frame:
//! 1. Call drawing methods ([`Draw2d::rect`], [`Draw2d::text`], [`Draw2d::sprite`], etc.)
//! 2. Call [`Draw2d::render`] to flush all batched geometry to the GPU
//! 3. Call [`Draw2d::clear`] to reset batches for the next frame
//!
//! # Coordinate System
//!
//! All coordinates are in screen-space pixels with the origin at the top-left corner.
//! X increases rightward, Y increases downward.
//!
//! # Example
//!
//! ```ignore
//! // During frame update
//! draw2d.rect(10.0, 10.0, 100.0, 50.0, Color::rgb(0.2, 0.4, 0.8));
//! draw2d.text(&assets, font_id, 20.0, 20.0, "Hello!", Color::WHITE);
//! draw2d.sprite(sprite_id, 150.0, 10.0, Color::WHITE);
//!
//! // During render pass
//! draw2d.render(&gpu, &mut render_pass, &assets);
//! draw2d.clear();
//! ```

use crate::assets::{Assets, FontId};
use crate::gpu::GpuContext;
use crate::texture::Sprite;

/// Index into the sprite storage.
///
/// Obtained from [`Draw2d::add_sprite`] and used to reference sprites in draw calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SpriteId(pub usize);

/// A rectangle in screen-space pixel coordinates.
///
/// The origin is at the top-left corner, with X increasing rightward
/// and Y increasing downward.
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    /// X coordinate of the top-left corner.
    pub x: f32,
    /// Y coordinate of the top-left corner.
    pub y: f32,
    /// Width of the rectangle in pixels.
    pub width: f32,
    /// Height of the rectangle in pixels.
    pub height: f32,
}

impl Rect {
    /// Creates a new rectangle with the given position and dimensions.
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

/// RGBA color with components in the range `[0.0, 1.0]`.
///
/// Colors are used for tinting sprites, coloring rectangles, and styling text.
/// The alpha component controls transparency (0.0 = fully transparent, 1.0 = fully opaque).
///
/// # Predefined Colors
///
/// Several commonly-used colors are provided as constants:
/// - [`Color::WHITE`], [`Color::BLACK`], [`Color::TRANSPARENT`]
/// - [`Color::DEBUG_BG`], [`Color::DEBUG_BORDER`] for debug UI styling
#[derive(Clone, Copy, Debug)]
pub struct Color {
    /// Red component (0.0 to 1.0).
    pub r: f32,
    /// Green component (0.0 to 1.0).
    pub g: f32,
    /// Blue component (0.0 to 1.0).
    pub b: f32,
    /// Alpha component (0.0 = transparent, 1.0 = opaque).
    pub a: f32,
}

impl Color {
    /// Creates a color from RGBA components.
    ///
    /// All components should be in the range `[0.0, 1.0]`.
    pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Creates an opaque color from RGB components.
    ///
    /// Equivalent to `Color::rgba(r, g, b, 1.0)`.
    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self { r, g, b, a: 1.0 }
    }

    /// Fully opaque white.
    pub const WHITE: Color = Color::rgba(1.0, 1.0, 1.0, 1.0);
    /// Fully opaque black.
    pub const BLACK: Color = Color::rgba(0.0, 0.0, 0.0, 1.0);
    /// Fully transparent (invisible).
    pub const TRANSPARENT: Color = Color::rgba(0.0, 0.0, 0.0, 0.0);

    /// Semi-transparent dark background for debug panels.
    pub const DEBUG_BG: Color = Color::rgba(0.1, 0.1, 0.1, 0.85);
    /// Gray accent color for panel borders.
    pub const DEBUG_BORDER: Color = Color::rgba(0.4, 0.4, 0.4, 1.0);
}

/// Vertex format for 2D sprite and text rendering.
///
/// Each vertex contains:
/// - **Position**: Screen-space coordinates in pixels
/// - **UV**: Texture coordinates (0.0 to 1.0)
/// - **Color**: RGBA tint color
///
/// This struct is `#[repr(C)]` and implements [`bytemuck::Pod`] for direct
/// GPU buffer uploads.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex2d {
    /// Screen-space position in pixels `[x, y]`.
    pub position: [f32; 2],
    /// Texture coordinates `[u, v]` in range `[0.0, 1.0]`.
    pub uv: [f32; 2],
    /// RGBA color for tinting `[r, g, b, a]`.
    pub color: [f32; 4],
}

impl Vertex2d {
    /// Vertex buffer layout descriptor for wgpu pipeline creation.
    ///
    /// Defines the memory layout:
    /// - Location 0: `position` as `Float32x2` (offset 0)
    /// - Location 1: `uv` as `Float32x2` (offset 8)
    /// - Location 2: `color` as `Float32x4` (offset 16)
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

/// Uniform buffer data for 2D rendering.
///
/// Contains the screen resolution for converting pixel coordinates to
/// normalized device coordinates in the vertex shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Draw2dUniforms {
    /// Screen resolution `[width, height]` in pixels.
    resolution: [f32; 2],
    /// Padding to align to 16 bytes (required by wgpu uniform buffers).
    _padding: [f32; 2],
}

/// Maximum number of vertices that can be batched per frame.
///
/// With 6 vertices per quad, this allows approximately 2,730 quads per frame.
const MAX_VERTICES: usize = 16384;

/// Immediate-mode 2D drawing API for sprites, text, and shapes.
///
/// `Draw2d` provides a simple interface for rendering 2D graphics on top of
/// your 3D scene or as a standalone 2D application. All draw calls are batched
/// and rendered efficiently in a single pass.
///
/// # Usage Pattern
///
/// ```ignore
/// // 1. Issue draw calls during your update/draw phase
/// draw2d.rect(x, y, width, height, Color::WHITE);
/// draw2d.text(&assets, font_id, x, y, "Hello", Color::BLACK);
/// draw2d.sprite(sprite_id, x, y, Color::WHITE);
///
/// // 2. Render everything in your render pass
/// draw2d.render(&gpu, &mut render_pass, &assets);
///
/// // 3. Clear batches for the next frame
/// draw2d.clear();
/// ```
///
/// # Batching Strategy
///
/// Draw calls are grouped by their texture requirements:
/// - Colored rectangles are batched together (no texture)
/// - Text is batched per-font (each font has its own atlas texture)
/// - Sprites are batched per-sprite (each sprite is a separate texture)
///
/// This minimizes GPU state changes while maintaining draw order within each batch type.
/// Note that colored geometry is always drawn first, followed by text, then sprites.
pub struct Draw2d {
    // Pipelines for different rendering modes
    /// Pipeline for solid-color rectangles (no texture sampling).
    colored_pipeline: wgpu::RenderPipeline,
    /// Pipeline for font rendering (R8 alpha mask textures).
    textured_pipeline: wgpu::RenderPipeline,
    /// Pipeline for RGBA sprite rendering.
    sprite_pipeline: wgpu::RenderPipeline,

    // Shared GPU resources
    /// Dynamic vertex buffer for all 2D geometry.
    vertex_buffer: wgpu::Buffer,
    /// Uniform buffer containing screen resolution.
    uniform_buffer: wgpu::Buffer,
    /// Bind group for uniforms (group 0).
    uniform_bind_group: wgpu::BindGroup,
    /// Layout for texture bind groups (group 1).
    texture_bind_group_layout: wgpu::BindGroupLayout,

    // Per-font bind groups (cached, indexed by FontId)
    font_bind_groups: Vec<Option<wgpu::BindGroup>>,

    // Sprite storage and bind groups
    /// All registered sprites.
    pub(crate) sprites: Vec<Sprite>,
    /// Cached bind groups for sprites (indexed by SpriteId).
    sprite_bind_groups: Vec<Option<wgpu::BindGroup>>,

    // Current frame vertex batches
    /// Vertices for solid-color rectangles.
    colored_vertices: Vec<Vertex2d>,
    /// Vertices for text, grouped by font.
    text_batches: Vec<(FontId, Vec<Vertex2d>)>,
    /// Vertices for sprites, grouped by sprite texture.
    sprite_batches: Vec<(SpriteId, Vec<Vertex2d>)>,
}

impl Draw2d {
    /// Creates a new 2D drawing context.
    ///
    /// Initializes all GPU resources including:
    /// - Render pipelines for colored, textured, and sprite rendering
    /// - Vertex and uniform buffers
    /// - Bind group layouts
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context containing the device and surface configuration
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

    /// Registers a sprite texture and returns its ID for later use.
    ///
    /// The sprite's GPU bind group will be created lazily on first render.
    ///
    /// # Arguments
    ///
    /// * `sprite` - The sprite texture to register
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] that can be used with [`Draw2d::sprite`] and related methods.
    pub fn add_sprite(&mut self, sprite: Sprite) -> SpriteId {
        let id = SpriteId(self.sprites.len());
        self.sprites.push(sprite);
        self.sprite_bind_groups.push(None); // Will be created lazily
        id
    }

    /// Returns a reference to the sprite with the given ID, if it exists.
    pub fn get_sprite(&self, id: SpriteId) -> Option<&Sprite> {
        self.sprites.get(id.0)
    }

    /// Clears all batched draw calls for the new frame.
    ///
    /// Call this at the end of each frame after [`Draw2d::render`] to prepare
    /// for the next frame's draw calls.
    pub fn clear(&mut self) {
        self.colored_vertices.clear();
        self.text_batches.clear();
        self.sprite_batches.clear();
    }

    /// Draws a solid-color rectangle.
    ///
    /// # Arguments
    ///
    /// * `x`, `y` - Top-left corner position in pixels
    /// * `w`, `h` - Width and height in pixels
    /// * `color` - Fill color
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

    /// Draws text at the given position.
    ///
    /// Text is rendered using the specified font from the asset system.
    /// The position specifies the top-left corner of the text bounding box.
    ///
    /// # Arguments
    ///
    /// * `assets` - Asset manager containing loaded fonts
    /// * `font_id` - ID of the font to use (from [`Assets::load_font`])
    /// * `x`, `y` - Top-left corner position in pixels
    /// * `text` - The string to render
    /// * `color` - Text color
    ///
    /// # Notes
    ///
    /// - Missing glyphs are skipped with a fallback advance
    /// - Each font is batched separately for efficient rendering
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

    /// Draws a sprite at its native size.
    ///
    /// The sprite is drawn at its original pixel dimensions. Use [`Draw2d::sprite_scaled`]
    /// for custom sizing or [`Draw2d::sprite_region`] for sprite sheet sub-regions.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID of the sprite (from [`Draw2d::add_sprite`])
    /// * `x`, `y` - Top-left corner position in pixels
    /// * `tint` - Color multiplier (use [`Color::WHITE`] for no tinting)
    pub fn sprite(&mut self, sprite_id: SpriteId, x: f32, y: f32, tint: Color) {
        let Some(sprite) = self.sprites.get(sprite_id.0) else {
            return;
        };
        let w = sprite.width as f32;
        let h = sprite.height as f32;
        self.sprite_rect(sprite_id, x, y, w, h, tint);
    }

    /// Draws a sprite scaled to fit a rectangle.
    ///
    /// The sprite texture is stretched to fill the specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID of the sprite (from [`Draw2d::add_sprite`])
    /// * `x`, `y` - Top-left corner position in pixels
    /// * `w`, `h` - Destination width and height in pixels
    /// * `tint` - Color multiplier (use [`Color::WHITE`] for no tinting)
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

    /// Draws a sub-region of a sprite (for sprite sheets).
    ///
    /// This is useful for sprite sheets where multiple frames or tiles are
    /// packed into a single texture.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID of the sprite (from [`Draw2d::add_sprite`])
    /// * `x`, `y` - Destination top-left corner in pixels
    /// * `w`, `h` - Destination width and height in pixels
    /// * `src_x`, `src_y` - Source region top-left corner in pixels (within the sprite)
    /// * `src_w`, `src_h` - Source region dimensions in pixels
    /// * `tint` - Color multiplier (use [`Color::WHITE`] for no tinting)
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

    /// Draws a sprite filling a rectangle with full UV range (0,0 to 1,1).
    ///
    /// Internal helper used by [`Draw2d::sprite`] and [`Draw2d::sprite_scaled`].
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

    /// Creates a panel builder for drawing bordered UI panels.
    ///
    /// Returns a [`PanelBuilder`] that allows customizing the panel's appearance
    /// before drawing. This is a convenience method for creating debug overlays
    /// and simple UI elements.
    ///
    /// # Arguments
    ///
    /// * `x`, `y` - Top-left corner position in pixels
    /// * `width`, `height` - Panel dimensions in pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// draw2d.panel(10.0, 10.0, 200.0, 100.0)
    ///     .background(Color::DEBUG_BG)
    ///     .border(Color::DEBUG_BORDER)
    ///     .title("Debug Panel", font_id)
    ///     .draw(&assets);
    /// ```
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

    /// Creates GPU bind groups for newly loaded fonts and sprites.
    ///
    /// This method lazily creates bind groups for any fonts or sprites that
    /// don't yet have them. Call this once per frame before rendering to ensure
    /// all textures are ready for use.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `assets` - Asset manager containing loaded fonts
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

    /// Renders all batched draw calls to the given render pass.
    ///
    /// This method flushes all accumulated geometry from the current frame:
    /// 1. Colored rectangles (using the colored pipeline)
    /// 2. Text batches (using the textured pipeline, one draw per font)
    /// 3. Sprite batches (using the sprite pipeline, one draw per sprite texture)
    ///
    /// Call [`Draw2d::clear`] after this to prepare for the next frame.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for buffer uploads
    /// * `render_pass` - The active render pass to draw into
    /// * `_assets` - Asset manager (currently unused but kept for API consistency)
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

/// Builder for drawing panels with backgrounds, borders, and optional titles.
///
/// Created via [`Draw2d::panel`]. Use the builder methods to customize the
/// panel's appearance, then call [`PanelBuilder::draw`] to render it.
///
/// # Default Appearance
///
/// - Background: [`Color::DEBUG_BG`] (semi-transparent dark)
/// - Border: [`Color::DEBUG_BORDER`] (gray, 1px)
/// - No title bar
///
/// # Example
///
/// ```ignore
/// draw2d.panel(10.0, 10.0, 200.0, 150.0)
///     .background(Color::rgba(0.0, 0.0, 0.2, 0.9))
///     .border(Color::WHITE)
///     .title("Settings", font_id)
///     .draw(&assets);
/// ```
pub struct PanelBuilder<'a> {
    /// Reference to the Draw2d instance for issuing draw calls.
    draw2d: &'a mut Draw2d,
    /// X coordinate of the panel's top-left corner.
    x: f32,
    /// Y coordinate of the panel's top-left corner.
    y: f32,
    /// Width of the panel in pixels.
    width: f32,
    /// Height of the panel in pixels.
    height: f32,
    /// Background fill color.
    background: Color,
    /// Border color, or `None` for no border.
    border: Option<Color>,
    /// Optional title text.
    title: Option<String>,
    /// Font for the title (required if `title` is set).
    title_font: Option<FontId>,
}

impl<'a> PanelBuilder<'a> {
    /// Sets the background fill color.
    ///
    /// Default: [`Color::DEBUG_BG`]
    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    /// Sets the border color.
    ///
    /// The border is drawn as a 1-pixel outline around the panel.
    ///
    /// Default: [`Color::DEBUG_BORDER`]
    pub fn border(mut self, color: Color) -> Self {
        self.border = Some(color);
        self
    }

    /// Removes the border from the panel.
    pub fn no_border(mut self) -> Self {
        self.border = None;
        self
    }

    /// Adds a title bar with the given text and font.
    ///
    /// The title bar is rendered as a darker strip at the top of the panel
    /// with the text left-aligned and white-colored.
    ///
    /// # Arguments
    ///
    /// * `text` - The title text to display
    /// * `font` - Font ID to use for rendering the title
    pub fn title(mut self, text: impl Into<String>, font: FontId) -> Self {
        self.title = Some(text.into());
        self.title_font = Some(font);
        self
    }

    /// Finalizes and draws the panel.
    ///
    /// This consumes the builder and issues draw calls for:
    /// 1. The background rectangle
    /// 2. The border (if enabled)
    /// 3. The title bar (if set)
    ///
    /// # Arguments
    ///
    /// * `assets` - Asset manager for font rendering (required if title is set)
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
