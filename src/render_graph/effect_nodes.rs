//! Effect render nodes for full-screen shader effects.

use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::HotEffectPass;
use crate::render_graph::{RenderContext, RenderNode};

/// Render node for full-screen shader effects.
///
/// `EffectNode` wraps an [`EffectPass`] for use in a render graph. Effect passes
/// are typically used as the first node in a graph to render procedural backgrounds,
/// raymarched scenes, or other full-screen shader effects.
///
/// Unlike post-process nodes, effect nodes do not require input from a previous pass
/// and can render independently.
///
/// # Clear Behavior
///
/// By default, the target is cleared to black before rendering. Use [`with_clear`](Self::with_clear)
/// to specify a different clear color, or [`no_clear`](Self::no_clear) to load the existing
/// target contents (useful for layering multiple effects).
///
/// # Example
///
/// ```ignore
/// let scene = EffectPass::new(&gpu, include_str!("shaders/scene.wgsl"));
/// let node = EffectNode::new(scene)
///     .with_clear(wgpu::Color::BLUE);
///
/// let graph = RenderGraph::builder()
///     .node(node)
///     .build(&gpu);
/// ```
pub struct EffectNode {
    /// The underlying effect pass containing the shader pipeline.
    pub effect: EffectPass,
    /// Clear color for the render target. `None` means load existing contents.
    pub clear_color: Option<wgpu::Color>,
}

impl EffectNode {
    /// Creates a new effect node with default black clear color.
    ///
    /// # Arguments
    ///
    /// * `effect` - The effect pass to wrap
    pub fn new(effect: EffectPass) -> Self {
        Self {
            effect,
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    /// Sets a custom clear color for the render target.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to clear the target to before rendering
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    pub fn with_clear(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }

    /// Disables clearing, preserving existing target contents.
    ///
    /// Useful when layering multiple effect passes on top of each other
    /// or when the effect shader explicitly handles all pixels.
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
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

// ============================================================================
// Hot-Reload Effect Node
// ============================================================================

/// Hot-reloadable render node for full-screen shader effects.
///
/// `HotEffectNode` is the hot-reloadable variant of [`EffectNode`]. It wraps a
/// [`HotEffectPass`] that monitors shader files for changes and automatically
/// recompiles the pipeline when modifications are detected.
///
/// # Hot-Reload Behavior
///
/// - Shader files are checked for modifications once per frame
/// - If changes are detected, the shader is recompiled asynchronously
/// - Compilation errors are logged but don't crash the application
/// - The previous working shader continues rendering until the new one compiles
///
/// # Example
///
/// ```ignore
/// // Load shader from file (not embedded) for hot-reloading
/// let scene = HotEffectPass::new(&gpu, "shaders/scene.wgsl")?;
/// let node = HotEffectNode::new(scene);
///
/// let mut graph = RenderGraph::builder()
///     .node(node)
///     .build(&gpu);
///
/// // Edit shaders/scene.wgsl while running - changes apply automatically!
/// ```
pub struct HotEffectNode {
    /// The hot-reloadable effect pass that watches for shader changes.
    pub effect: HotEffectPass,
    /// Clear color for the render target. `None` means load existing contents.
    pub clear_color: Option<wgpu::Color>,
}

impl HotEffectNode {
    /// Creates a new hot-reloadable effect node with default black clear color.
    ///
    /// # Arguments
    ///
    /// * `effect` - The hot-reloadable effect pass to wrap
    pub fn new(effect: HotEffectPass) -> Self {
        Self {
            effect,
            clear_color: Some(wgpu::Color::BLACK),
        }
    }

    /// Sets a custom clear color for the render target.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to clear the target to before rendering
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    pub fn with_clear(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }

    /// Disables clearing, preserving existing target contents.
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    pub fn no_clear(mut self) -> Self {
        self.clear_color = None;
        self
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
