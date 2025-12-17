//! Scene-specific setup context for configuring per-scene render pipelines.

use crate::draw2d::Color;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::{HotEffectPass, HotPostProcessPass, HotWorldPostProcessPass};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::render_graph::{
    EffectNode, HotEffectNode, HotPostProcessNode, HotWorldPostProcessNode, MeshNode, MeshQueue,
    PostProcessNode, RenderGraph, WorldPostProcessNode,
};
use std::cell::RefCell;
use std::rc::Rc;

/// Context for configuring a scene's render pipeline during setup.
///
/// This is similar to [`SetupContext`](crate::SetupContext) but only provides
/// methods for configuring the render graph. Asset loading (meshes, textures,
/// sprites) is done through the main `SetupContext` since assets are shared
/// across all scenes.
///
/// # Example
///
/// ```ignore
/// ctx.scene("underwater", |scene| {
///     scene.background_color(Color::rgb(0.0, 0.1, 0.3));
///     scene.enable_mesh_rendering();
///     scene.hot_post_process("shaders/underwater_distortion.wgsl");
///     scene.hot_post_process("shaders/caustics.wgsl");
///
///     move |frame| {
///         // Scene frame logic
///     }
/// });
/// ```
pub struct SceneSetupContext<'a> {
    /// GPU context for creating render resources.
    pub gpu: &'a GpuContext,
    /// The render graph being built for this scene.
    graph_builder: &'a mut Option<RenderGraph>,
    /// Shared mesh queue (for creating MeshNode).
    mesh_queue: &'a Rc<RefCell<MeshQueue>>,
}

impl<'a> SceneSetupContext<'a> {
    /// Create a new scene setup context.
    pub(crate) fn new(
        gpu: &'a GpuContext,
        graph_builder: &'a mut Option<RenderGraph>,
        mesh_queue: &'a Rc<RefCell<MeshQueue>>,
    ) -> Self {
        Self {
            gpu,
            graph_builder,
            mesh_queue,
        }
    }

    // ========================================================================
    // Background Color
    // ========================================================================

    /// Set a solid background color for this scene.
    ///
    /// This is the simplest way to set a background - no shader required.
    /// Use this when you just want a solid color behind your content.
    ///
    /// # Arguments
    ///
    /// * `color` - Background color
    ///
    /// # Example
    ///
    /// ```ignore
    /// scene.background_color(Color::rgb(0.1, 0.1, 0.15));
    /// scene.enable_mesh_rendering();
    /// ```
    pub fn background_color(&mut self, color: Color) -> &mut Self {
        let shader = format!(
            r#"
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Uniforms {{
    resolution: vec2f,
    time: f32,
}}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {{
    var pos = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    return vec4f(pos[vertex_index], 0.0, 1.0);
}}

@fragment
fn fs() -> @location(0) vec4f {{
    return vec4f({:.6}, {:.6}, {:.6}, {:.6});
}}
"#,
            color.r, color.g, color.b, color.a
        );
        let effect = EffectPass::new(self.gpu, &shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    // ========================================================================
    // Shader Effect Methods (Embedded)
    // ========================================================================

    /// Add a fullscreen screen-space shader effect.
    ///
    /// Screen-space effects do not receive camera uniforms - they operate purely
    /// in normalized device coordinates. Use [`Self::effect_world`] if your shader
    /// needs camera information.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    pub fn effect(&mut self, shader: &str) -> &mut Self {
        let effect = EffectPass::new(self.gpu, shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    /// Add a fullscreen world-space shader effect.
    ///
    /// World-space effects receive camera uniforms (view matrix, projection matrix,
    /// camera position, etc.) allowing them to perform 3D calculations like ray
    /// marching or world-space lighting.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    pub fn effect_world(&mut self, shader: &str) -> &mut Self {
        let effect = EffectPass::new_world(self.gpu, shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    /// Add a screen-space post-processing effect.
    ///
    /// Post-processing effects read from the previous render pass output and write
    /// to a new buffer. They're ideal for effects like color grading, vignette,
    /// or simple blur that don't need 3D information.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    pub fn post_process(&mut self, shader: &str) -> &mut Self {
        let pass = PostProcessPass::new(self.gpu, shader);
        self.add_node(PostProcessNode::new(pass));
        self
    }

    /// Add a world-space post-processing effect.
    ///
    /// Similar to [`Self::post_process`], but also receives camera uniforms.
    /// Useful for effects that need depth-based calculations, world-space
    /// fog, or other 3D-aware post-processing.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    pub fn post_process_world(&mut self, shader: &str) -> &mut Self {
        let pass = WorldPostProcessPass::new(self.gpu, shader);
        self.add_node(WorldPostProcessNode::new(pass));
        self
    }

    // ========================================================================
    // Shader Effect Methods (Hot-Reloadable)
    // ========================================================================

    /// Add a hot-reloadable fullscreen screen-space shader effect.
    ///
    /// The shader is loaded from a file path on disk and will automatically
    /// reload whenever the file is modified.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    pub fn hot_effect(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable fullscreen world-space shader effect.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    pub fn hot_effect_world(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new_world(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable screen-space post-processing effect.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    pub fn hot_post_process(&mut self, path: &str) -> &mut Self {
        match HotPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable world-space post-processing effect.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    pub fn hot_post_process_world(&mut self, path: &str) -> &mut Self {
        match HotWorldPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotWorldPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    // ========================================================================
    // 3D Mesh Rendering
    // ========================================================================

    /// Enable 3D mesh rendering in this scene's pipeline.
    ///
    /// This adds a mesh rendering pass to the render graph. The position in the
    /// pipeline determines what effects are applied to meshes:
    ///
    /// - Call **before** post-processing to apply effects to meshes
    /// - Call **after** effects to render meshes on top of shader backgrounds
    ///
    /// # Example
    ///
    /// ```ignore
    /// scene.hot_effect_world("shaders/background.wgsl")  // Rendered first
    ///      .enable_mesh_rendering()                        // Meshes on top
    ///      .hot_post_process("shaders/bloom.wgsl");        // Bloom applied to everything
    /// ```
    pub fn enable_mesh_rendering(&mut self) -> &mut Self {
        let mesh_node = MeshNode::new(self.gpu, Rc::clone(self.mesh_queue));
        self.add_node(mesh_node);
        self
    }

    /// Internal helper to add a render node to the graph.
    fn add_node<N: crate::render_graph::RenderNode + 'static>(&mut self, node: N) {
        if self.graph_builder.is_none() {
            *self.graph_builder = Some(RenderGraph::builder().node(node).build(self.gpu));
        } else {
            let old = self.graph_builder.take().unwrap();
            *self.graph_builder = Some(old.with_node(node, self.gpu));
        }
    }
}
