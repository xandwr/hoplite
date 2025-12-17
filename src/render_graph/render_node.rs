//! The core render node trait for the render graph.

use crate::gpu::GpuContext;
use crate::render_graph::RenderContext;

/// Trait for render graph nodes that can execute rendering operations.
///
/// Implement this trait to create custom render passes that integrate with the
/// render graph system. Each node receives the previous pass's output (if any)
/// and writes to a target texture view.
///
/// # Execution Flow
///
/// 1. `check_hot_reload()` is called once per frame for all nodes
/// 2. `execute()` is called in sequence, with ping-pong buffer management
/// 3. The final node renders directly to the screen
///
/// # Implementing Custom Nodes
///
/// ```ignore
/// struct MyCustomNode {
///     pipeline: wgpu::RenderPipeline,
///     bind_group: wgpu::BindGroup,
/// }
///
/// impl RenderNode for MyCustomNode {
///     fn execute(
///         &self,
///         ctx: &mut RenderContext,
///         target: &wgpu::TextureView,
///         input: Option<&wgpu::TextureView>,
///     ) {
///         let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
///             color_attachments: &[Some(wgpu::RenderPassColorAttachment {
///                 view: target,
///                 // ... configure load/store ops
///             })],
///             // ...
///         });
///         pass.set_pipeline(&self.pipeline);
///         pass.set_bind_group(0, &self.bind_group, &[]);
///         pass.draw(0..3, 0..1);  // Full-screen triangle
///     }
/// }
/// ```
pub trait RenderNode {
    /// Executes this node's rendering operations.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Render context with GPU access, encoder, time, and camera
    /// * `target` - Texture view to render into (either intermediate buffer or screen)
    /// * `input` - Previous pass output, or `None` for the first node in the graph
    ///
    /// # Panics
    ///
    /// Post-processing nodes typically panic if `input` is `None`, as they require
    /// a source texture to sample from.
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    );

    /// Called once per frame before `execute()` to check for hot-reload changes.
    ///
    /// Override this method for nodes that support hot-reloading (e.g., shader
    /// file watching). The default implementation does nothing.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for recompiling shaders if changes are detected
    fn check_hot_reload(&mut self, _gpu: &GpuContext) {}
}
