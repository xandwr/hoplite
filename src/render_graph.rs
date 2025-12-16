use crate::camera::Camera;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;

/// A render target that can be written to by passes.
pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
}

impl RenderTarget {
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: gpu.width(),
                height: gpu.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view }
    }

    pub fn resize(&mut self, gpu: &GpuContext, label: &str) {
        *self = Self::new(gpu, label);
    }
}

/// Context passed to each render node during execution.
pub struct RenderContext<'a> {
    pub gpu: &'a GpuContext,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub time: f32,
    pub camera: &'a Camera,
}

/// A node in the render graph that can produce output.
pub trait RenderNode {
    /// Execute this node, rendering to the provided target view.
    fn execute(&self, ctx: &mut RenderContext, target: &wgpu::TextureView);
}

/// An effect pass wrapped as a render node.
pub struct EffectNode {
    pub effect: EffectPass,
    pub clear_color: Option<wgpu::Color>,
}

impl EffectNode {
    pub fn new(effect: EffectPass) -> Self {
        Self {
            effect,
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    pub fn with_clear(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }

    pub fn no_clear(mut self) -> Self {
        self.clear_color = None;
        self
    }
}

impl RenderNode for EffectNode {
    fn execute(&self, ctx: &mut RenderContext, target: &wgpu::TextureView) {
        let load_op = match self.clear_color {
            Some(color) => wgpu::LoadOp::Clear(color),
            None => wgpu::LoadOp::Load,
        };

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        if self.effect.uses_camera() {
            self.effect
                .render_with_camera(ctx.gpu, &mut render_pass, ctx.time, ctx.camera);
        } else {
            self.effect.render(ctx.gpu, &mut render_pass, ctx.time);
        }
    }
}

/// A composable render graph that chains render passes together.
///
/// # Example
/// ```ignore
/// let graph = RenderGraph::new(&gpu)
///     .node(EffectNode::new(scene_effect))
///     .node(EffectNode::new(bloom_effect).no_clear())
///     .build();
///
/// // In render loop:
/// graph.execute(&gpu, time, &camera);
/// ```
pub struct RenderGraphBuilder {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraphBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a render node to the graph.
    pub fn node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Build the render graph.
    pub fn build(self, _gpu: &GpuContext) -> RenderGraph {
        RenderGraph { nodes: self.nodes }
    }
}

pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl Default for RenderGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// Create a new render graph builder.
    pub fn builder() -> RenderGraphBuilder {
        RenderGraphBuilder::new()
    }

    /// Execute the render graph, presenting to screen.
    ///
    /// All nodes render directly to the screen in sequence.
    /// For multi-pass effects with intermediate textures, nodes can
    /// manage their own render targets internally.
    pub fn execute(&self, gpu: &GpuContext, time: f32, camera: &Camera) {
        let output = gpu.surface.get_current_texture().unwrap();
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        {
            let mut ctx = RenderContext {
                gpu,
                encoder: &mut encoder,
                time,
                camera,
            };

            for node in &self.nodes {
                node.execute(&mut ctx, &view);
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
