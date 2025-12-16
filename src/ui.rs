use crate::gpu::GpuContext;

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

/// A GUI component that can be rendered.
pub enum GuiComponent {
    /// A floating pane with optional title.
    FloatingPane {
        rect: Rect,
        title: Option<String>,
        background: Color,
        border: Option<Color>,
    },
    /// A text label (placeholder - text rendering not yet implemented).
    Label {
        rect: Rect,
        text: String,
        color: Color,
    },
    /// A simple colored rectangle.
    Quad { rect: Rect, color: Color },
}

impl GuiComponent {
    /// Create a floating pane at the given position.
    pub fn floating_pane(x: f32, y: f32, width: f32, height: f32) -> FloatingPaneBuilder {
        FloatingPaneBuilder {
            rect: Rect::new(x, y, width, height),
            title: None,
            background: Color::DEBUG_BG,
            border: Some(Color::DEBUG_BORDER),
        }
    }

    /// Create a label at the given position.
    pub fn label(x: f32, y: f32, text: impl Into<String>) -> Self {
        GuiComponent::Label {
            rect: Rect::new(x, y, 0.0, 0.0),
            text: text.into(),
            color: Color::WHITE,
        }
    }

    /// Create a simple quad.
    pub fn quad(rect: Rect, color: Color) -> Self {
        GuiComponent::Quad { rect, color }
    }
}

/// Builder for FloatingPane.
pub struct FloatingPaneBuilder {
    rect: Rect,
    title: Option<String>,
    background: Color,
    border: Option<Color>,
}

impl FloatingPaneBuilder {
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn background(mut self, color: Color) -> Self {
        self.background = color;
        self
    }

    pub fn border(mut self, color: Color) -> Self {
        self.border = Some(color);
        self
    }

    pub fn no_border(mut self) -> Self {
        self.border = None;
        self
    }

    pub fn build(self) -> GuiComponent {
        GuiComponent::FloatingPane {
            rect: self.rect,
            title: self.title,
            background: self.background,
            border: self.border,
        }
    }
}

/// Vertex for UI quads.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UiVertex {
    position: [f32; 2],
    color: [f32; 4],
}

impl UiVertex {
    const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<UiVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
}

/// Uniforms for the UI shader.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UiUniforms {
    resolution: [f32; 2],
    _padding: [f32; 2],
}

const MAX_VERTICES: usize = 4096;

/// Immediate-mode UI rendering pass.
///
/// Renders 2D GUI elements on top of the scene, unaffected by post-processing.
/// Uses alpha blending for transparency.
pub struct UiPass {
    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertices: Vec<UiVertex>,
}

impl UiPass {
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("UI Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/ui.wgsl").into()),
        });

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Vertex Buffer"),
            size: (MAX_VERTICES * std::mem::size_of::<UiVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("UI Uniforms"),
            size: std::mem::size_of::<UiUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("UI Bind Group Layout"),
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("UI Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("UI Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("UI Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[UiVertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    // Alpha blending for transparency
                    blend: Some(wgpu::BlendState {
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
                    }),
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
            vertex_buffer,
            uniform_buffer,
            bind_group,
            vertices: Vec::with_capacity(MAX_VERTICES),
        }
    }

    /// Clear all queued UI elements.
    pub fn clear(&mut self) {
        self.vertices.clear();
    }

    /// Add a GUI component to be rendered.
    pub fn add(&mut self, component: &GuiComponent) {
        match component {
            GuiComponent::FloatingPane {
                rect,
                title,
                background,
                border,
            } => {
                // Draw background
                self.push_quad(rect, background);

                // Draw border if present
                if let Some(border_color) = border {
                    let border_width = 1.0;
                    // Top
                    self.push_quad(
                        &Rect::new(rect.x, rect.y, rect.width, border_width),
                        border_color,
                    );
                    // Bottom
                    self.push_quad(
                        &Rect::new(
                            rect.x,
                            rect.y + rect.height - border_width,
                            rect.width,
                            border_width,
                        ),
                        border_color,
                    );
                    // Left
                    self.push_quad(
                        &Rect::new(rect.x, rect.y, border_width, rect.height),
                        border_color,
                    );
                    // Right
                    self.push_quad(
                        &Rect::new(
                            rect.x + rect.width - border_width,
                            rect.y,
                            border_width,
                            rect.height,
                        ),
                        border_color,
                    );
                }

                // Draw title bar if present
                if let Some(_title) = title {
                    let title_height = 24.0;
                    let title_bg = Color::rgba(0.15, 0.15, 0.15, 0.95);
                    self.push_quad(
                        &Rect::new(rect.x, rect.y, rect.width, title_height),
                        &title_bg,
                    );
                    // Title text would go here once text rendering is implemented
                    // For now, just show the title bar
                }
            }
            GuiComponent::Label { rect, color, .. } => {
                // Text rendering not implemented yet - just show a placeholder
                let placeholder = Rect::new(rect.x, rect.y, 8.0, 12.0);
                self.push_quad(&placeholder, color);
            }
            GuiComponent::Quad { rect, color } => {
                self.push_quad(rect, color);
            }
        }
    }

    fn push_quad(&mut self, rect: &Rect, color: &Color) {
        let x0 = rect.x;
        let y0 = rect.y;
        let x1 = rect.x + rect.width;
        let y1 = rect.y + rect.height;
        let c = [color.r, color.g, color.b, color.a];

        // Two triangles for a quad
        self.vertices.push(UiVertex {
            position: [x0, y0],
            color: c,
        });
        self.vertices.push(UiVertex {
            position: [x1, y0],
            color: c,
        });
        self.vertices.push(UiVertex {
            position: [x0, y1],
            color: c,
        });

        self.vertices.push(UiVertex {
            position: [x1, y0],
            color: c,
        });
        self.vertices.push(UiVertex {
            position: [x1, y1],
            color: c,
        });
        self.vertices.push(UiVertex {
            position: [x0, y1],
            color: c,
        });
    }

    /// Render all queued UI elements.
    pub fn render(&self, gpu: &GpuContext, render_pass: &mut wgpu::RenderPass) {
        if self.vertices.is_empty() {
            return;
        }

        let uniforms = UiUniforms {
            resolution: [gpu.width() as f32, gpu.height() as f32],
            _padding: [0.0, 0.0],
        };
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

        gpu.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.vertices));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.draw(0..self.vertices.len() as u32, 0..1);
    }
}
