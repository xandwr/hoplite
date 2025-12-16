use crate::camera::Camera;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::{HotEffectPass, HotPostProcessPass, HotWorldPostProcessPass};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};

/// A render target that can be written to by passes.
pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    width: u32,
    height: u32,
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
        Self {
            texture,
            view,
            width: gpu.width(),
            height: gpu.height(),
        }
    }

    /// Check if target needs resize and recreate if so.
    pub fn ensure_size(&mut self, gpu: &GpuContext, label: &str) {
        if self.width != gpu.width() || self.height != gpu.height() {
            *self = Self::new(gpu, label);
        }
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
    /// `input` is the previous pass output (None for the first pass).
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    );

    /// Check for hot-reload changes. Called once per frame before execute.
    /// Default implementation does nothing (for non-hot-reloadable nodes).
    fn check_hot_reload(&mut self, _gpu: &GpuContext) {}
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
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        _input: Option<&wgpu::TextureView>,
    ) {
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

/// A post-processing pass that reads from the previous pass output.
pub struct PostProcessNode {
    pub pass: PostProcessPass,
}

impl PostProcessNode {
    pub fn new(pass: PostProcessPass) -> Self {
        Self { pass }
    }
}

impl RenderNode for PostProcessNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let input_view = input.expect("PostProcessNode requires an input from a previous pass");

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.time, input_view);
    }
}

/// A world-space post-processing pass with camera uniforms.
pub struct WorldPostProcessNode {
    pub pass: WorldPostProcessPass,
}

impl WorldPostProcessNode {
    pub fn new(pass: WorldPostProcessPass) -> Self {
        Self { pass }
    }
}

impl RenderNode for WorldPostProcessNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let input_view =
            input.expect("WorldPostProcessNode requires an input from a previous pass");

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.time, ctx.camera, input_view);
    }
}

// ============================================================================
// Hot-Reload Render Nodes
// ============================================================================

/// A hot-reloadable effect pass wrapped as a render node.
pub struct HotEffectNode {
    pub effect: HotEffectPass,
    pub clear_color: Option<wgpu::Color>,
}

impl HotEffectNode {
    pub fn new(effect: HotEffectPass) -> Self {
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

    /// Check for shader changes and recompile if needed.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        self.effect.check_reload(gpu);
    }
}

impl RenderNode for HotEffectNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        _input: Option<&wgpu::TextureView>,
    ) {
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

    fn check_hot_reload(&mut self, gpu: &GpuContext) {
        self.effect.check_reload(gpu);
    }
}

/// A hot-reloadable post-processing pass node.
pub struct HotPostProcessNode {
    pub pass: HotPostProcessPass,
}

impl HotPostProcessNode {
    pub fn new(pass: HotPostProcessPass) -> Self {
        Self { pass }
    }

    /// Check for shader changes and recompile if needed.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        self.pass.check_reload(gpu);
    }
}

impl RenderNode for HotPostProcessNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let input_view = input.expect("HotPostProcessNode requires an input from a previous pass");

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.time, input_view);
    }

    fn check_hot_reload(&mut self, gpu: &GpuContext) {
        self.pass.check_reload(gpu);
    }
}

/// A hot-reloadable world-space post-processing pass node.
pub struct HotWorldPostProcessNode {
    pub pass: HotWorldPostProcessPass,
}

impl HotWorldPostProcessNode {
    pub fn new(pass: HotWorldPostProcessPass) -> Self {
        Self { pass }
    }

    /// Check for shader changes and recompile if needed.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        self.pass.check_reload(gpu);
    }
}

impl RenderNode for HotWorldPostProcessNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let input_view =
            input.expect("HotWorldPostProcessNode requires an input from a previous pass");

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.time, ctx.camera, input_view);
    }

    fn check_hot_reload(&mut self, gpu: &GpuContext) {
        self.pass.check_reload(gpu);
    }
}

/// A composable render graph that chains render passes together.
///
/// # Example
/// ```ignore
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(scene_effect))
///     .node(PostProcessNode::new(lensing_pass))
///     .build(&gpu);
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
    pub fn build(self, gpu: &GpuContext) -> RenderGraph {
        // Create ping-pong buffers for multi-pass rendering
        let target_a = RenderTarget::new(gpu, "RenderGraph Target A");
        let target_b = RenderTarget::new(gpu, "RenderGraph Target B");

        RenderGraph {
            nodes: self.nodes,
            target_a,
            target_b,
        }
    }
}

pub struct RenderGraph {
    nodes: Vec<Box<dyn RenderNode>>,
    target_a: RenderTarget,
    target_b: RenderTarget,
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

    /// Add a node to an existing render graph, returning a new graph.
    pub fn with_node<N: RenderNode + 'static>(mut self, node: N, gpu: &GpuContext) -> Self {
        self.nodes.push(Box::new(node));
        // Ensure we have render targets
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");
        self
    }

    /// Execute the render graph, presenting to screen.
    pub fn execute(&mut self, gpu: &GpuContext, time: f32, camera: &Camera) {
        self.execute_with_ui(gpu, time, camera, |_, _| {});
    }

    /// Check all nodes for hot-reload changes.
    /// Called automatically by execute_with_ui, but can be called manually if needed.
    pub fn check_hot_reload(&mut self, gpu: &GpuContext) {
        for node in &mut self.nodes {
            node.check_hot_reload(gpu);
        }
    }

    /// Execute the render graph with an optional UI pass rendered on top.
    ///
    /// The UI closure receives the GPU context and render pass, allowing
    /// UI elements to be composited on top of the scene after all
    /// post-processing effects have been applied.
    ///
    /// # Example
    /// ```ignore
    /// let mut ui = UiPass::new(&gpu);
    ///
    /// // In render loop:
    /// ui.clear();
    /// ui.add(&GuiComponent::floating_pane(10.0, 10.0, 200.0, 100.0).build());
    /// graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
    ///     ui.render(gpu, pass);
    /// });
    /// ```
    pub fn execute_with_ui<F>(&mut self, gpu: &GpuContext, time: f32, camera: &Camera, ui_fn: F)
    where
        F: FnOnce(&GpuContext, &mut wgpu::RenderPass),
    {
        // Check for hot-reload changes before rendering
        self.check_hot_reload(gpu);

        // Ensure render targets are the right size
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");

        let output = gpu.surface.get_current_texture().unwrap();
        let screen_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        let node_count = self.nodes.len();

        {
            let mut ctx = RenderContext {
                gpu,
                encoder: &mut encoder,
                time,
                camera,
            };

            // For single node, render directly to screen
            if node_count == 1 {
                self.nodes[0].execute(&mut ctx, &screen_view, None);
            } else {
                // Multi-pass: ping-pong between targets, final pass goes to screen
                let mut current_input: Option<&wgpu::TextureView> = None;

                for (i, node) in self.nodes.iter().enumerate() {
                    let is_last = i == node_count - 1;

                    let target = if is_last {
                        &screen_view
                    } else if i % 2 == 0 {
                        &self.target_a.view
                    } else {
                        &self.target_b.view
                    };

                    node.execute(&mut ctx, target, current_input);

                    // Set up input for next pass
                    if !is_last {
                        current_input = Some(if i % 2 == 0 {
                            &self.target_a.view
                        } else {
                            &self.target_b.view
                        });
                    }
                }
            }
        }

        // Render UI on top (if any)
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &screen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            ui_fn(gpu, &mut ui_pass);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
