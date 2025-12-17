//! Post-processing render nodes for screen-space and world-aware effects.

use crate::gpu::GpuContext;
use crate::hot_shader::{HotPostProcessPass, HotWorldPostProcessPass};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::render_graph::{RenderContext, RenderNode};

/// Render node for screen-space post-processing effects.
///
/// `PostProcessNode` wraps a [`PostProcessPass`] that samples the previous pass's
/// output and applies screen-space transformations. Common uses include:
///
/// - Blur effects (Gaussian, box, motion blur)
/// - Bloom and glow
/// - Color grading and tonemapping
/// - Chromatic aberration
/// - Vignette effects
///
/// # Requirements
///
/// This node **requires** a previous pass in the render graph. It will panic
/// if used as the first node, since there's no input texture to sample.
///
/// # Example
///
/// ```ignore
/// let bloom = PostProcessPass::new(&gpu, include_str!("shaders/bloom.wgsl"));
/// let tonemap = PostProcessPass::new(&gpu, include_str!("shaders/tonemap.wgsl"));
///
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(scene))         // First: render scene
///     .node(PostProcessNode::new(bloom))    // Then: apply bloom
///     .node(PostProcessNode::new(tonemap))  // Finally: tonemap
///     .build(&gpu);
/// ```
pub struct PostProcessNode {
    /// The underlying post-process pass containing the shader pipeline.
    pub pass: PostProcessPass,
}

impl PostProcessNode {
    /// Creates a new post-process node.
    ///
    /// # Arguments
    ///
    /// * `pass` - The post-process pass to wrap
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

/// Render node for world-aware post-processing effects.
///
/// `WorldPostProcessNode` wraps a [`WorldPostProcessPass`] that has access to
/// camera uniforms (view matrix, projection matrix, camera position) in addition
/// to the previous pass's output. This enables effects that need world-space
/// information:
///
/// - Volumetric fog and atmospheric scattering
/// - Screen-space raymarching
/// - Depth-based effects (DOF, SSAO)
/// - World-space lens flares
/// - God rays / light shafts
///
/// # Requirements
///
/// This node **requires** a previous pass in the render graph. It will panic
/// if used as the first node, since there's no input texture to sample.
///
/// # Camera Access
///
/// The shader receives camera uniforms automatically, allowing reconstruction
/// of world-space positions from screen coordinates.
///
/// # Example
///
/// ```ignore
/// let fog = WorldPostProcessPass::new(&gpu, include_str!("shaders/volumetric_fog.wgsl"));
///
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(scene))
///     .node(WorldPostProcessNode::new(fog))  // Can access camera for fog calculations
///     .build(&gpu);
/// ```
pub struct WorldPostProcessNode {
    /// The underlying world post-process pass containing the shader pipeline.
    pub pass: WorldPostProcessPass,
}

impl WorldPostProcessNode {
    /// Creates a new world post-process node.
    ///
    /// # Arguments
    ///
    /// * `pass` - The world post-process pass to wrap
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
// Hot-Reload Post-Process Nodes
// ============================================================================

/// Hot-reloadable render node for screen-space post-processing.
///
/// `HotPostProcessNode` is the hot-reloadable variant of [`PostProcessNode`].
/// It wraps a [`HotPostProcessPass`] that monitors shader files and recompiles
/// automatically when changes are detected.
///
/// # Requirements
///
/// Like [`PostProcessNode`], this node requires a previous pass in the graph.
///
/// # Example
///
/// ```ignore
/// let blur = HotPostProcessPass::new(&gpu, "shaders/blur.wgsl")?;
///
/// let mut graph = RenderGraph::builder()
///     .node(HotEffectNode::new(scene))
///     .node(HotPostProcessNode::new(blur))
///     .build(&gpu);
/// ```
pub struct HotPostProcessNode {
    /// The hot-reloadable post-process pass that watches for shader changes.
    pub pass: HotPostProcessPass,
}

impl HotPostProcessNode {
    /// Creates a new hot-reloadable post-process node.
    ///
    /// # Arguments
    ///
    /// * `pass` - The hot-reloadable post-process pass to wrap
    pub fn new(pass: HotPostProcessPass) -> Self {
        Self { pass }
    }

    /// Manually triggers a hot-reload check.
    ///
    /// This is called automatically by the render graph, but can be invoked
    /// manually if needed outside the normal render loop.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for recompiling shaders if changes are detected
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

/// Hot-reloadable render node for world-aware post-processing.
///
/// `HotWorldPostProcessNode` is the hot-reloadable variant of [`WorldPostProcessNode`].
/// It wraps a [`HotWorldPostProcessPass`] that monitors shader files and recompiles
/// automatically, while still providing access to camera uniforms.
///
/// # Requirements
///
/// Like [`WorldPostProcessNode`], this node requires a previous pass in the graph.
///
/// # Example
///
/// ```ignore
/// let fog = HotWorldPostProcessPass::new(&gpu, "shaders/fog.wgsl")?;
///
/// let mut graph = RenderGraph::builder()
///     .node(HotEffectNode::new(scene))
///     .node(HotWorldPostProcessNode::new(fog))
///     .build(&gpu);
/// ```
pub struct HotWorldPostProcessNode {
    /// The hot-reloadable world post-process pass that watches for shader changes.
    pub pass: HotWorldPostProcessPass,
}

impl HotWorldPostProcessNode {
    /// Creates a new hot-reloadable world post-process node.
    ///
    /// # Arguments
    ///
    /// * `pass` - The hot-reloadable world post-process pass to wrap
    pub fn new(pass: HotWorldPostProcessPass) -> Self {
        Self { pass }
    }

    /// Manually triggers a hot-reload check.
    ///
    /// This is called automatically by the render graph, but can be invoked
    /// manually if needed outside the normal render loop.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for recompiling shaders if changes are detected
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
