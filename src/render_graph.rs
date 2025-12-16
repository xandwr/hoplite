//! Composable render graph system for multi-pass rendering pipelines.
//!
//! This module provides a flexible, node-based render graph architecture that allows
//! chaining multiple render passes together with automatic ping-pong buffer management.
//! Each pass can read from the previous pass's output and write to its own render target,
//! enabling complex post-processing chains and multi-stage rendering effects.
//!
//! # Architecture
//!
//! The render graph uses a linear pipeline model:
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │  EffectNode │───▶│ PostProcess │───▶│ PostProcess │───▶│   Screen    │
//! │  (Scene)    │    │   Node 1    │    │   Node 2    │    │  (Final)    │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//!       │                  │                  │
//!       ▼                  ▼                  ▼
//!   Target A ◀────────▶ Target B        (ping-pong)
//! ```
//!
//! # Node Types
//!
//! - [`EffectNode`] / [`HotEffectNode`]: Full-screen shader effects (backgrounds, procedural scenes)
//! - [`PostProcessNode`] / [`HotPostProcessNode`]: Screen-space post-processing (blur, bloom, color grading)
//! - [`WorldPostProcessNode`] / [`HotWorldPostProcessNode`]: Post-processing with camera/world data (raymarching, fog)
//! - [`MeshNode`]: 3D mesh rendering with depth testing
//!
//! Hot-reload variants automatically watch shader files and recompile on changes.
//!
//! # Example
//!
//! ```ignore
//! use hoplite::{RenderGraph, EffectNode, PostProcessNode};
//!
//! // Build a render graph with a scene and post-processing
//! let mut graph = RenderGraph::builder()
//!     .node(EffectNode::new(scene_effect))           // Render procedural scene
//!     .node(PostProcessNode::new(bloom_pass))        // Apply bloom
//!     .node(PostProcessNode::new(tonemap_pass))      // Tonemap to screen
//!     .build(&gpu);
//!
//! // In render loop:
//! loop {
//!     graph.execute(&gpu, time, &camera);
//! }
//! ```
//!
//! # With UI Overlay
//!
//! ```ignore
//! graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
//!     ui.render(gpu, pass);  // UI composited on top of final output
//! });
//! ```

use crate::camera::Camera;
use crate::draw2d::Color;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::{HotEffectPass, HotPostProcessPass, HotWorldPostProcessPass};
use crate::mesh::{Mesh, Transform};
use crate::mesh_pass::{DrawCall, MeshPass};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::texture::Texture;
use std::cell::RefCell;
use std::rc::Rc;

/// An off-screen render target used for intermediate pass results.
///
/// Render targets are GPU textures that can be both rendered to (as a color attachment)
/// and sampled from (as a texture binding). This dual capability enables ping-pong
/// rendering where one pass writes to target A while reading from target B, then
/// the next pass reverses the roles.
///
/// The render graph automatically manages two render targets internally and handles
/// resizing when the window dimensions change.
///
/// # Fields
///
/// * `texture` - The underlying wgpu texture resource
/// * `view` - A texture view for binding as either render target or sampler input
/// * `width` - Current width in pixels (tracks GPU surface size)
/// * `height` - Current height in pixels (tracks GPU surface size)
pub struct RenderTarget {
    /// The underlying GPU texture that stores pixel data.
    pub texture: wgpu::Texture,
    /// A view into the texture, used for render pass attachments and shader sampling.
    pub view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RenderTarget {
    /// Creates a new render target matching the current GPU surface dimensions.
    ///
    /// The texture is created with:
    /// - Same format as the surface (typically `Bgra8UnormSrgb`)
    /// - `RENDER_ATTACHMENT` usage for writing via render passes
    /// - `TEXTURE_BINDING` usage for sampling in subsequent passes
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context providing device and surface configuration
    /// * `label` - Debug label for the texture (visible in GPU debuggers like RenderDoc)
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

    /// Checks if the target dimensions match the GPU surface and recreates if needed.
    ///
    /// This should be called at the start of each frame to handle window resizes.
    /// If the dimensions differ, a new texture is allocated and the old one is dropped.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context to check dimensions against
    /// * `label` - Debug label for the recreated texture
    pub fn ensure_size(&mut self, gpu: &GpuContext, label: &str) {
        if self.width != gpu.width() || self.height != gpu.height() {
            *self = Self::new(gpu, label);
        }
    }
}

/// Execution context passed to each render node during graph traversal.
///
/// This struct bundles all the resources a render node needs to execute its
/// rendering operations. It is created fresh for each frame and passed through
/// the entire node chain.
///
/// # Lifetime
///
/// The `'a` lifetime ties all references to the frame's scope, ensuring nodes
/// cannot hold onto resources beyond the current frame.
///
/// # Fields
///
/// * `gpu` - Access to device, queue, and surface configuration
/// * `encoder` - Command encoder for recording GPU commands
/// * `time` - Elapsed time in seconds (for animations and effects)
/// * `camera` - Current camera state for view/projection matrices
pub struct RenderContext<'a> {
    /// GPU context providing access to device, queue, and configuration.
    pub gpu: &'a GpuContext,
    /// Command encoder for recording render pass commands.
    /// Nodes append their commands to this encoder.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Elapsed time in seconds since application start.
    /// Used for animating shaders and time-based effects.
    pub time: f32,
    /// Current camera providing view and projection matrices.
    /// Available for nodes that need world-space or screen-space transformations.
    pub camera: &'a Camera,
}

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
// Hot-Reload Render Nodes
// ============================================================================
//
// These variants of the render nodes support hot-reloading, allowing shader
// modifications to be applied without restarting the application. They watch
// the shader source files and automatically recompile when changes are detected.
//
// Hot-reload nodes are ideal for development workflows where rapid iteration
// on shaders is desired. For production builds, consider using the non-hot
// variants to avoid file system overhead.

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

// ============================================================================
// Mesh Render Node
// ============================================================================
//
// The mesh rendering system provides 3D geometry rendering with depth testing,
// texturing, and per-instance coloring. It uses a deferred queuing model where
// draw calls are accumulated during the frame and rendered in a single batch.

/// A queued mesh draw call stored in the shared mesh queue.
///
/// This struct represents a single mesh instance to be rendered, with its
/// transform, color tint, and optional texture. Draw calls are accumulated
/// in a [`MeshQueue`] during the frame and processed by [`MeshNode`].
///
/// # Fields
///
/// * `mesh_index` - Index into [`MeshQueue::meshes`]
/// * `transform` - World-space transformation (position, rotation, scale)
/// * `color` - RGBA color tint applied to the mesh
/// * `texture_index` - Optional index into [`MeshQueue::textures`]
pub struct QueuedMesh {
    /// Index of the mesh in the queue's mesh array.
    pub mesh_index: usize,
    /// World-space transformation for this instance.
    pub transform: Transform,
    /// Color tint multiplied with vertex colors and textures.
    pub color: Color,
    /// Optional texture index. `None` uses vertex colors only.
    pub texture_index: Option<usize>,
}

/// Shared storage for meshes, textures, and the per-frame draw queue.
///
/// `MeshQueue` provides a central repository for 3D assets and a queue for
/// draw calls. It is typically wrapped in `Rc<RefCell<>>` and shared between
/// the render graph and application code.
///
/// # Usage Pattern
///
/// ```ignore
/// // Setup: create queue and register meshes
/// let queue = Rc::new(RefCell::new(MeshQueue::new()));
/// let cube_idx = queue.borrow_mut().add_mesh(cube_mesh);
/// let tex_idx = queue.borrow_mut().add_texture(wood_texture);
///
/// // Each frame: queue draw calls
/// queue.borrow_mut().draw(cube_idx, transform, Color::WHITE);
/// queue.borrow_mut().draw_textured(cube_idx, transform2, Color::WHITE, tex_idx);
///
/// // Render graph processes the queue automatically
/// graph.execute(&gpu, time, &camera);
///
/// // After frame: clear for next frame
/// queue.borrow_mut().clear_queue();
/// ```
///
/// # Thread Safety
///
/// This struct is not thread-safe. Use `Rc<RefCell<>>` for single-threaded
/// applications or `Arc<Mutex<>>` for multi-threaded scenarios.
pub struct MeshQueue {
    /// Registered meshes, indexed by the values returned from [`add_mesh`](Self::add_mesh).
    pub meshes: Vec<Mesh>,
    /// Registered textures, indexed by the values returned from [`add_texture`](Self::add_texture).
    pub textures: Vec<Texture>,
    /// Per-frame draw queue, cleared at the end of each frame.
    pub draw_queue: Vec<QueuedMesh>,
}

impl MeshQueue {
    /// Creates a new empty mesh queue.
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            textures: Vec::new(),
            draw_queue: Vec::new(),
        }
    }

    /// Registers a mesh and returns its index for later use.
    ///
    /// Meshes are stored permanently until the queue is dropped. Use the
    /// returned index with [`draw`](Self::draw) or [`draw_textured`](Self::draw_textured).
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh geometry to register
    ///
    /// # Returns
    ///
    /// An index that can be used to reference this mesh in draw calls.
    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        idx
    }

    /// Registers a texture and returns its index for later use.
    ///
    /// Textures are stored permanently until the queue is dropped. Use the
    /// returned index with [`draw_textured`](Self::draw_textured).
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to register
    ///
    /// # Returns
    ///
    /// An index that can be used to reference this texture in draw calls.
    pub fn add_texture(&mut self, texture: Texture) -> usize {
        let idx = self.textures.len();
        self.textures.push(texture);
        idx
    }

    /// Queues a mesh for rendering this frame without a texture.
    ///
    /// The mesh will be rendered using vertex colors multiplied by the
    /// provided color tint.
    ///
    /// # Arguments
    ///
    /// * `mesh_index` - Index from [`add_mesh`](Self::add_mesh)
    /// * `transform` - World-space transformation
    /// * `color` - Color tint (use `Color::WHITE` for no tinting)
    pub fn draw(&mut self, mesh_index: usize, transform: Transform, color: Color) {
        self.draw_queue.push(QueuedMesh {
            mesh_index,
            transform,
            color,
            texture_index: None,
        });
    }

    /// Queues a textured mesh for rendering this frame.
    ///
    /// The mesh will be rendered with the specified texture, with colors
    /// multiplied by the color tint.
    ///
    /// # Arguments
    ///
    /// * `mesh_index` - Index from [`add_mesh`](Self::add_mesh)
    /// * `transform` - World-space transformation
    /// * `color` - Color tint (use `Color::WHITE` for no tinting)
    /// * `texture_index` - Index from [`add_texture`](Self::add_texture)
    pub fn draw_textured(
        &mut self,
        mesh_index: usize,
        transform: Transform,
        color: Color,
        texture_index: usize,
    ) {
        self.draw_queue.push(QueuedMesh {
            mesh_index,
            transform,
            color,
            texture_index: Some(texture_index),
        });
    }

    /// Clears the draw queue for the next frame.
    ///
    /// Call this at the end of each frame after the render graph has executed.
    /// Registered meshes and textures are preserved.
    pub fn clear_queue(&mut self) {
        self.draw_queue.clear();
    }
}

impl Default for MeshQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Render node for 3D mesh rendering with depth testing.
///
/// `MeshNode` renders all meshes queued in its associated [`MeshQueue`].
/// It supports:
///
/// - Depth testing for correct occlusion
/// - Per-instance transforms and color tints
/// - Optional texturing
/// - Compositing over previous pass output (background blitting)
///
/// # Integration with Render Graph
///
/// When used after other nodes (e.g., an effect pass for a background),
/// `MeshNode` first blits the input texture, then renders meshes on top
/// with depth testing. This allows 3D objects to be composited over
/// procedural backgrounds.
///
/// # Clear Behavior
///
/// By default, `MeshNode` does **not** clear the target, allowing it to
/// render on top of the previous pass. Use [`with_clear`](Self::with_clear)
/// if you want to clear to a solid color first.
///
/// # Example
///
/// ```ignore
/// let queue = Rc::new(RefCell::new(MeshQueue::new()));
/// let cube_idx = queue.borrow_mut().add_mesh(Mesh::cube(&gpu));
///
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(sky_effect))                    // Background
///     .node(MeshNode::new(&gpu, Rc::clone(&queue)))         // 3D meshes on top
///     .node(PostProcessNode::new(tonemap))                  // Post-process
///     .build(&gpu);
///
/// // In render loop:
/// queue.borrow_mut().draw(cube_idx, Transform::default(), Color::WHITE);
/// graph.execute(&gpu, time, &camera);
/// queue.borrow_mut().clear_queue();
/// ```
pub struct MeshNode {
    /// The mesh rendering pass with pipeline and depth buffer.
    pub pass: MeshPass,
    /// Shared queue containing meshes, textures, and draw calls.
    pub queue: Rc<RefCell<MeshQueue>>,
    /// Optional clear color. `None` preserves previous pass output.
    pub clear_color: Option<wgpu::Color>,
}

impl MeshNode {
    /// Creates a new mesh render node.
    ///
    /// The node is configured to render on top of the previous pass by default
    /// (no clearing). The depth buffer is created at the current GPU surface size.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for creating the mesh pass and depth buffer
    /// * `queue` - Shared mesh queue (typically `Rc<RefCell<MeshQueue>>`)
    pub fn new(gpu: &GpuContext, queue: Rc<RefCell<MeshQueue>>) -> Self {
        Self {
            pass: MeshPass::new(gpu),
            queue,
            clear_color: None, // Don't clear by default - render on top of previous pass
        }
    }

    /// Sets a clear color, causing the target to be cleared before rendering.
    ///
    /// Use this when `MeshNode` is the first node in the graph or when you
    /// don't want to preserve the previous pass output.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to clear to before rendering meshes
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    pub fn with_clear(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }
}

impl RenderNode for MeshNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let queue = self.queue.borrow();

        // Build draw calls from the queue
        let draw_calls: Vec<DrawCall> = queue
            .draw_queue
            .iter()
            .filter_map(|q| {
                queue.meshes.get(q.mesh_index).map(|mesh| DrawCall {
                    mesh,
                    transform: q.transform,
                    color: q.color,
                    texture: q.texture_index.and_then(|idx| queue.textures.get(idx)),
                })
            })
            .collect();

        // If there's an input texture, we need to blit it first as the background
        if let Some(input_view) = input {
            // First pass: blit the input texture to the target (no depth)
            let mut blit_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Blit Pass"),
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
            self.pass.blit(ctx.gpu, &mut blit_pass, input_view);
        }

        // If no meshes to draw, we're done (background is already blitted)
        if draw_calls.is_empty() {
            // If there was no input either, we need to at least clear the target
            if input.is_none() {
                let _clear_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mesh Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                self.clear_color.unwrap_or(wgpu::Color::BLACK),
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            return;
        }

        // Second pass: render meshes on top with depth testing
        // Use Load since we already blitted (or Clear if no input and clear_color is set)
        let load_op = if input.is_some() {
            wgpu::LoadOp::Load
        } else {
            match self.clear_color {
                Some(color) => wgpu::LoadOp::Clear(color),
                None => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            }
        };

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mesh Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.pass.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.camera, ctx.time, &draw_calls);
    }

    fn check_hot_reload(&mut self, gpu: &GpuContext) {
        self.pass.ensure_depth_size(gpu);
    }
}

/// Builder for constructing render graphs with a fluent API.
///
/// `RenderGraphBuilder` provides a chainable interface for assembling render
/// pipelines. Nodes are executed in the order they are added.
///
/// # Example
///
/// ```ignore
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(scene_effect))      // First: render scene
///     .node(PostProcessNode::new(bloom))        // Then: apply bloom
///     .node(PostProcessNode::new(tonemap))      // Finally: tonemap
///     .build(&gpu);
/// ```
///
/// # Node Ordering
///
/// Nodes execute in insertion order. The first node receives no input
/// (`input` is `None`), while subsequent nodes receive the previous
/// node's output. The final node renders directly to the screen.
pub struct RenderGraphBuilder {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraphBuilder {
    /// Creates a new empty render graph builder.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Adds a render node to the graph.
    ///
    /// Nodes are executed in the order they are added. Any type implementing
    /// [`RenderNode`] can be added.
    ///
    /// # Arguments
    ///
    /// * `node` - The render node to add
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    ///
    /// # Type Parameters
    ///
    /// * `N` - Any type implementing `RenderNode + 'static`
    pub fn node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Builds the render graph, allocating ping-pong buffers.
    ///
    /// This method finalizes the graph and creates the intermediate render
    /// targets needed for multi-pass rendering. Two render targets are
    /// allocated at the current GPU surface size.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for creating render targets
    ///
    /// # Returns
    ///
    /// A ready-to-use [`RenderGraph`].
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

/// A composable render graph that executes a chain of render passes.
///
/// `RenderGraph` manages a sequence of render nodes and the intermediate
/// buffers needed for multi-pass rendering. It handles:
///
/// - Ping-pong buffer management for pass chaining
/// - Automatic render target resizing on window resize
/// - Hot-reload checking for all nodes
/// - Final presentation to the screen
/// - Optional UI overlay compositing
///
/// # Buffer Management
///
/// For multi-pass rendering, the graph uses two intermediate render targets
/// (ping-pong buffers). Each pass alternates between reading from one buffer
/// and writing to the other, with the final pass writing directly to the screen.
///
/// ```text
/// Pass 0: None → Target A
/// Pass 1: Target A → Target B
/// Pass 2: Target B → Target A
/// Pass 3: Target A → Screen
/// ```
///
/// For single-node graphs, no intermediate buffers are used.
///
/// # Example
///
/// ```ignore
/// // Create a render graph
/// let mut graph = RenderGraph::builder()
///     .node(EffectNode::new(scene))
///     .node(PostProcessNode::new(bloom))
///     .build(&gpu);
///
/// // Simple rendering
/// graph.execute(&gpu, time, &camera);
///
/// // Or with UI overlay
/// graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
///     ui.render(gpu, pass);
/// });
/// ```
pub struct RenderGraph {
    /// The sequence of render nodes to execute.
    nodes: Vec<Box<dyn RenderNode>>,
    /// First ping-pong buffer for intermediate results.
    target_a: RenderTarget,
    /// Second ping-pong buffer for intermediate results.
    target_b: RenderTarget,
}

impl Default for RenderGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RenderGraph {
    /// Creates a new render graph builder.
    ///
    /// This is the recommended way to construct a render graph. Use the builder's
    /// fluent API to add nodes, then call [`build`](RenderGraphBuilder::build).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let graph = RenderGraph::builder()
    ///     .node(EffectNode::new(scene))
    ///     .node(PostProcessNode::new(bloom))
    ///     .build(&gpu);
    /// ```
    pub fn builder() -> RenderGraphBuilder {
        RenderGraphBuilder::new()
    }

    /// Adds a node to an existing render graph.
    ///
    /// This method allows dynamically extending a render graph after construction.
    /// The render targets are resized if needed to match the current GPU surface.
    ///
    /// # Arguments
    ///
    /// * `node` - The render node to add
    /// * `gpu` - GPU context for potential target resizing
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Add a new post-process effect at runtime
    /// graph = graph.with_node(PostProcessNode::new(new_effect), &gpu);
    /// ```
    pub fn with_node<N: RenderNode + 'static>(mut self, node: N, gpu: &GpuContext) -> Self {
        self.nodes.push(Box::new(node));
        // Ensure we have render targets
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");
        self
    }

    /// Executes the render graph and presents to the screen.
    ///
    /// This is the main method called each frame. It:
    /// 1. Checks all nodes for hot-reload changes
    /// 2. Ensures render targets match the current window size
    /// 3. Executes each node in sequence with ping-pong buffering
    /// 4. Presents the final result to the screen
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `time` - Elapsed time in seconds (passed to shaders)
    /// * `camera` - Current camera state
    ///
    /// # Panics
    ///
    /// Panics if the surface texture cannot be acquired.
    pub fn execute(&mut self, gpu: &GpuContext, time: f32, camera: &Camera) {
        self.execute_with_ui(gpu, time, camera, |_, _| {});
    }

    /// Checks all nodes for hot-reload changes.
    ///
    /// This is called automatically by [`execute`](Self::execute) and
    /// [`execute_with_ui`](Self::execute_with_ui), but can be invoked manually
    /// if you need to trigger hot-reload checks outside the normal render loop.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for shader recompilation
    pub fn check_hot_reload(&mut self, gpu: &GpuContext) {
        for node in &mut self.nodes {
            node.check_hot_reload(gpu);
        }
    }

    /// Executes the render graph with a UI overlay pass.
    ///
    /// Similar to [`execute`](Self::execute), but allows rendering UI elements
    /// on top of the final output. The UI closure is called after all render
    /// nodes have executed, with the render pass targeting the screen.
    ///
    /// The UI pass uses `LoadOp::Load` to preserve the rendered scene, so UI
    /// elements are composited on top.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `time` - Elapsed time in seconds
    /// * `camera` - Current camera state
    /// * `ui_fn` - Closure that receives `(&GpuContext, &mut wgpu::RenderPass)` for UI rendering
    ///
    /// # Example
    ///
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
    ///
    /// # Panics
    ///
    /// Panics if the surface texture cannot be acquired.
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
