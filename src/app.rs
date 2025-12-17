//! Application framework and main entry point for Hoplite.
//!
//! This module provides the core application lifecycle management for Hoplite applications,
//! including window creation, event handling, and the render loop. It exposes a simple,
//! closure-based API that handles all the complexity of GPU initialization, resource
//! management, and frame timing.
//!
//! # Architecture
//!
//! The application framework is built around two phases:
//!
//! 1. **Setup Phase**: Called once at startup via [`SetupContext`]. Use this to:
//!    - Load fonts, textures, and sprites
//!    - Configure the render pipeline (effects, post-processing)
//!    - Create 3D meshes
//!    - Set up any initial state your application needs
//!
//! 2. **Frame Phase**: Called every frame via [`Frame`]. Use this to:
//!    - Read input state
//!    - Update game/application logic
//!    - Issue draw commands (text, rectangles, sprites, meshes)
//!
//! # Quick Start
//!
//! The simplest Hoplite application:
//!
//! ```ignore
//! use hoplite::app::run;
//!
//! fn main() {
//!     run(|ctx| {
//!         // Setup: load a font for text rendering
//!         ctx.default_font(16.0);
//!
//!         // Return the frame closure
//!         move |frame| {
//!             frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
//!         }
//!     });
//! }
//! ```
//!
//! # Render Pipeline
//!
//! Hoplite uses a node-based render graph. The order you add effects matters:
//!
//! ```ignore
//! run(|ctx| {
//!     ctx.default_font(16.0);
//!
//!     // 1. Background effect (rendered first, clears screen)
//!     ctx.effect_world(include_str!("background.wgsl"));
//!
//!     // 2. Enable 3D mesh rendering
//!     ctx.enable_mesh_rendering();
//!
//!     // 3. Post-processing (applied after meshes)
//!     ctx.post_process_world(include_str!("bloom.wgsl"));
//!
//!     // 4. 2D UI is always rendered last, on top of everything
//!
//!     move |frame| { /* ... */ }
//! });
//! ```
//!
//! # Hot Reloading
//!
//! For rapid shader development, use the `hot_*` variants which reload from disk:
//!
//! ```ignore
//! ctx.hot_effect("shaders/background.wgsl");      // Reloads on file change
//! ctx.hot_post_process("shaders/bloom.wgsl");     // Reloads on file change
//! ```

use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::assets::{Assets, FontId};
use crate::camera::Camera;
use crate::draw2d::SpriteId;
use crate::draw2d::{Color, Draw2d};
use crate::ecs::{MeshId, TextureId};
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::{HotEffectPass, HotPostProcessPass, HotWorldPostProcessPass};
use crate::input::Input;
use crate::mesh::{Mesh, Transform};
use crate::picking::{self, PickResult, Ray, RayHit};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::render_graph::{
    EffectNode, HotEffectNode, HotPostProcessNode, HotWorldPostProcessNode, MeshNode, MeshQueue,
    PostProcessNode, RenderGraph, WorldPostProcessNode,
};
use crate::texture::{Sprite, Texture};
use std::cell::RefCell;
use std::rc::Rc;

/// Context provided during the setup phase of a Hoplite application.
///
/// `SetupContext` is passed to your setup closure and provides methods for:
/// - Loading and configuring fonts, textures, and sprites
/// - Building the render pipeline (effects, post-processing)
/// - Creating 3D meshes and enabling mesh rendering
///
/// The setup phase runs once before the first frame. Any resources loaded here
/// remain available throughout the application's lifetime.
///
/// # Example
///
/// ```ignore
/// run(|ctx| {
///     // Load the default font at 16pt
///     ctx.default_font(16.0);
///
///     // Add a fullscreen shader effect
///     ctx.effect_world(include_str!("background.wgsl"));
///
///     // Load a sprite for use in the frame loop
///     let player_sprite = ctx.sprite_from_file("assets/player.png").unwrap();
///
///     // Create a cube mesh
///     ctx.enable_mesh_rendering();
///     let cube = ctx.mesh_cube();
///
///     move |frame| {
///         frame.sprite(player_sprite, 100.0, 100.0);
///         // ...
///     }
/// });
/// ```
///
/// # Render Pipeline Order
///
/// The order in which you call effect/post-process methods determines the render order:
/// 1. Effects are rendered first (each can read the previous output)
/// 2. Mesh rendering (if enabled) happens at its position in the chain
/// 3. Post-processing effects are applied in order
/// 4. 2D UI (text, sprites, rectangles) is always rendered last
pub struct SetupContext<'a> {
    /// GPU context for creating GPU resources directly.
    pub gpu: &'a GpuContext,
    /// Asset manager for loading fonts and other managed resources.
    pub assets: &'a mut Assets,
    /// 2D drawing context for registering sprites.
    pub draw: &'a mut Draw2d,
    /// ECS world for spawning entities during setup.
    pub world: &'a mut hecs::World,
    /// Storage for the default font ID (set via [`Self::default_font`]).
    default_font: &'a mut Option<FontId>,
    /// The render graph being built (lazily initialized on first effect/pass).
    graph_builder: &'a mut Option<RenderGraph>,
    /// Shared mesh queue for 3D rendering.
    mesh_queue: &'a Rc<RefCell<MeshQueue>>,
}

impl<'a> SetupContext<'a> {
    /// Load and register the default font at the specified size.
    ///
    /// This font will be used by [`Frame::text`] and [`Frame::text_color`] for
    /// convenient text rendering without needing to pass a font ID each time.
    ///
    /// # Arguments
    ///
    /// * `size` - Font size in pixels (e.g., 16.0 for standard UI text)
    ///
    /// # Returns
    ///
    /// The [`FontId`] for the loaded font, which can also be used with the
    /// lower-level `Draw2d::text` method if needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.default_font(16.0);
    /// // Later in frame:
    /// frame.text(10.0, 10.0, "Hello!");
    /// ```
    pub fn default_font(&mut self, size: f32) -> FontId {
        let font = self.assets.default_font(self.gpu, size);
        *self.default_font = Some(font);
        font
    }

    // ========================================================================
    // Background Color
    // ========================================================================

    /// Set a solid background color for the application.
    ///
    /// This is the simplest way to set a background - no shader required.
    /// Use this when you just want a solid color behind your content.
    ///
    /// For dynamic backgrounds, use [`effect`](Self::effect) or
    /// [`hot_effect`](Self::hot_effect) instead.
    ///
    /// # Arguments
    ///
    /// * `color` - Background color (use [`Color`](crate::Color) for convenience)
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.background_color(Color::rgb(0.1, 0.1, 0.15)); // Dark blue-gray
    /// ctx.enable_mesh_rendering();
    ///
    /// move |frame| {
    ///     frame.mesh(cube).at(0.0, 0.0, -5.0).draw();
    /// }
    /// ```
    pub fn background_color(&mut self, color: Color) -> &mut Self {
        // Create a minimal shader that just outputs the color
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

    /// Add a fullscreen screen-space shader effect to the render pipeline.
    ///
    /// Screen-space effects do not receive camera uniforms - they operate purely
    /// in normalized device coordinates. Use [`Self::effect_world`] if your shader
    /// needs camera information.
    ///
    /// Effects are rendered in the order they're added. The first effect clears
    /// the screen; subsequent effects can read the previous effect's output.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.effect(include_str!("shaders/gradient.wgsl"));
    /// ```
    pub fn effect(&mut self, shader: &str) -> &mut Self {
        let effect = EffectPass::new(self.gpu, shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    /// Add a fullscreen world-space shader effect to the render pipeline.
    ///
    /// World-space effects receive camera uniforms (view matrix, projection matrix,
    /// camera position, etc.) allowing them to perform 3D calculations like ray
    /// marching or world-space lighting.
    ///
    /// # Arguments
    ///
    /// * `shader` - WGSL shader source code (typically via `include_str!`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Ray marching shader that uses camera position and direction
    /// ctx.effect_world(include_str!("shaders/raymarching.wgsl"));
    /// ```
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.post_process(include_str!("shaders/vignette.wgsl"));
    /// ```
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
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Gravitational lensing effect that needs world-space ray directions
    /// ctx.post_process_world(include_str!("shaders/lensing.wgsl"));
    /// ```
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
    /// reload whenever the file is modified. This is invaluable during shader
    /// development - save your file and see changes instantly without restarting.
    ///
    /// If the shader fails to load or compile, an error is printed to stderr
    /// and the effect is skipped (the application continues running).
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.hot_effect("shaders/gradient.wgsl");
    /// // Now edit gradient.wgsl and save - changes appear immediately!
    /// ```
    pub fn hot_effect(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable fullscreen world-space shader effect.
    ///
    /// Like [`Self::hot_effect`], but the shader receives camera uniforms for
    /// 3D calculations. See [`Self::effect_world`] for details on world-space effects.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.hot_effect_world("shaders/raymarching.wgsl");
    /// ```
    pub fn hot_effect_world(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new_world(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable screen-space post-processing effect.
    ///
    /// Like [`Self::hot_effect`], but configured as a post-processing pass that
    /// reads from the previous render output. See [`Self::post_process`] for details.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.hot_post_process("shaders/bloom.wgsl");
    /// ```
    pub fn hot_post_process(&mut self, path: &str) -> &mut Self {
        match HotPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable world-space post-processing effect.
    ///
    /// Like [`Self::hot_post_process`], but also receives camera uniforms.
    /// See [`Self::post_process_world`] for details.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the WGSL shader file on disk
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.hot_post_process_world("shaders/lensing.wgsl");
    /// ```
    pub fn hot_post_process_world(&mut self, path: &str) -> &mut Self {
        match HotWorldPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotWorldPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Internal helper to add a render node to the graph.
    ///
    /// Lazily initializes the render graph on first use, then appends
    /// subsequent nodes to the existing graph.
    fn add_node<N: crate::render_graph::RenderNode + 'static>(&mut self, node: N) {
        if self.graph_builder.is_none() {
            *self.graph_builder = Some(RenderGraph::builder().node(node).build(self.gpu));
        } else {
            // Rebuild with new node - could optimize later with incremental updates
            let old = self.graph_builder.take().unwrap();
            *self.graph_builder = Some(old.with_node(node, self.gpu));
        }
    }

    // ========================================================================
    // 3D Mesh Methods
    // ========================================================================

    /// Enable 3D mesh rendering in the pipeline.
    ///
    /// This adds a mesh rendering pass to the render graph. The position in the
    /// pipeline determines what effects are applied to meshes:
    ///
    /// - Call **before** post-processing to apply effects to meshes
    /// - Call **after** effects to render meshes on top of shader backgrounds
    ///
    /// # Returns
    ///
    /// `&mut Self` for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// ctx.effect_world(include_str!("background.wgsl"))  // Rendered first
    ///    .enable_mesh_rendering()                         // Meshes on top
    ///    .post_process(include_str!("bloom.wgsl"));       // Bloom applied to everything
    /// ```
    pub fn enable_mesh_rendering(&mut self) -> &mut Self {
        let mesh_node = MeshNode::new(self.gpu, Rc::clone(self.mesh_queue));
        self.add_node(mesh_node);
        self
    }

    /// Create a unit cube mesh (1x1x1, centered at origin).
    ///
    /// # Returns
    ///
    /// A type-safe [`MeshId`] for use with [`Frame::mesh`] or [`Frame::draw_mesh`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cube = ctx.mesh_cube();
    /// // In frame (builder style):
    /// frame.mesh(cube).at(0.0, 0.0, -5.0).color(Color::RED).draw();
    /// // Or classic style:
    /// frame.draw_mesh(cube, Transform::translation(0.0, 0.0, -5.0), Color::RED);
    /// ```
    pub fn mesh_cube(&mut self) -> MeshId {
        let mesh = Mesh::cube(self.gpu);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Create a UV sphere mesh with the specified tessellation.
    ///
    /// Higher segment/ring counts produce smoother spheres at the cost of
    /// more vertices. A sphere with 16 segments and 12 rings is usually
    /// sufficient for most purposes.
    ///
    /// # Arguments
    ///
    /// * `segments` - Number of horizontal divisions (longitude lines)
    /// * `rings` - Number of vertical divisions (latitude lines)
    ///
    /// # Returns
    ///
    /// A type-safe [`MeshId`] for use with [`Frame::mesh`] or [`Frame::draw_mesh`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sphere = ctx.mesh_sphere(32, 24);  // Smooth sphere
    /// let low_poly = ctx.mesh_sphere(8, 6);  // Chunky sphere
    /// ```
    pub fn mesh_sphere(&mut self, segments: u32, rings: u32) -> MeshId {
        let mesh = Mesh::sphere(self.gpu, segments, rings);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Create a flat horizontal plane mesh.
    ///
    /// The plane is centered at the origin, lying flat on the XZ plane
    /// (normal pointing up in +Y).
    ///
    /// # Arguments
    ///
    /// * `size` - Total width/depth of the plane (extends ±size/2 from center)
    ///
    /// # Returns
    ///
    /// A type-safe [`MeshId`] for use with [`Frame::mesh`] or [`Frame::draw_mesh`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ground = ctx.mesh_plane(100.0);  // 100x100 ground plane
    /// ```
    pub fn mesh_plane(&mut self, size: f32) -> MeshId {
        let mesh = Mesh::plane(self.gpu, size);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Add a custom mesh to the mesh queue.
    ///
    /// Use this to add meshes created manually or loaded from external sources.
    ///
    /// # Arguments
    ///
    /// * `mesh` - A [`Mesh`] instance with vertex and index data
    ///
    /// # Returns
    ///
    /// A type-safe [`MeshId`] for use with [`Frame::mesh`] or [`Frame::draw_mesh`].
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    // ========================================================================
    // 3D Texture Methods
    // ========================================================================

    /// Add a texture to the texture pool for 3D mesh rendering.
    ///
    /// Textures are applied to meshes using [`Frame::mesh`] builder or [`Frame::draw_mesh_textured`].
    ///
    /// # Arguments
    ///
    /// * `texture` - A [`Texture`] instance
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`] for use with mesh rendering.
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        self.mesh_queue.borrow_mut().add_texture(texture)
    }

    /// Load a texture from a file path.
    ///
    /// Supports common image formats (PNG, JPEG, etc.) via the `image` crate.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`] on success, or an [`image::ImageError`] on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let brick_tex = ctx.texture_from_file("assets/brick.png")?;
    /// // In frame (builder style):
    /// frame.mesh(cube).texture(brick_tex).draw();
    /// // Or classic style:
    /// frame.draw_mesh_textured(cube, transform, Color::WHITE, brick_tex);
    /// ```
    pub fn texture_from_file(&mut self, path: &str) -> Result<TextureId, image::ImageError> {
        let texture = Texture::from_file(self.gpu, path)?;
        Ok(self.add_texture(texture))
    }

    /// Load a texture from embedded bytes.
    ///
    /// Useful for bundling textures directly in the executable via `include_bytes!`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw image file bytes (PNG, JPEG, etc.)
    /// * `label` - Debug label for the texture (shown in GPU debugging tools)
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`] on success, or an [`image::ImageError`] on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tex = ctx.texture_from_bytes(
    ///     include_bytes!("../assets/brick.png"),
    ///     "brick texture"
    /// )?;
    /// ```
    pub fn texture_from_bytes(
        &mut self,
        bytes: &[u8],
        label: &str,
    ) -> Result<TextureId, image::ImageError> {
        let texture = Texture::from_bytes(self.gpu, bytes, label)?;
        Ok(self.add_texture(texture))
    }

    /// Create a procedural blocky noise texture.
    ///
    /// Generates a random pattern of earthy colors (browns, grays) suitable
    /// for dirt/stone surfaces. Uses nearest-neighbor filtering for a
    /// classic blocky aesthetic.
    ///
    /// # Arguments
    ///
    /// * `size` - Texture dimensions (creates a size×size texture)
    /// * `seed` - Random seed for reproducible generation
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`].
    pub fn texture_blocky_noise(&mut self, size: u32, seed: u32) -> TextureId {
        let texture = Texture::blocky_noise(self.gpu, size, seed);
        self.add_texture(texture)
    }

    /// Create a procedural blocky grass texture.
    ///
    /// Generates a green grass pattern with color variation.
    ///
    /// # Arguments
    ///
    /// * `size` - Texture dimensions (creates a size×size texture)
    /// * `seed` - Random seed for reproducible generation
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`].
    pub fn texture_blocky_grass(&mut self, size: u32, seed: u32) -> TextureId {
        let texture = Texture::blocky_grass(self.gpu, size, seed);
        self.add_texture(texture)
    }

    /// Create a procedural blocky stone texture.
    ///
    /// Generates a gray stone pattern with cracks and variation.
    ///
    /// # Arguments
    ///
    /// * `size` - Texture dimensions (creates a size×size texture)
    /// * `seed` - Random seed for reproducible generation
    ///
    /// # Returns
    ///
    /// A type-safe [`TextureId`].
    pub fn texture_blocky_stone(&mut self, size: u32, seed: u32) -> TextureId {
        let texture = Texture::blocky_stone(self.gpu, size, seed);
        self.add_texture(texture)
    }

    // ========================================================================
    // 2D Sprite Methods
    // ========================================================================

    /// Add a pre-created sprite to the 2D layer.
    ///
    /// Sprites are rendered on top of all 3D content and effects as part
    /// of the UI layer.
    ///
    /// # Arguments
    ///
    /// * `sprite` - A [`Sprite`] instance
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] for use with [`Frame::sprite`] and related methods.
    pub fn add_sprite(&mut self, sprite: Sprite) -> SpriteId {
        self.draw.add_sprite(sprite)
    }

    /// Load a 2D sprite from a file path with linear (smooth) filtering.
    ///
    /// Linear filtering smoothly interpolates between pixels when the sprite
    /// is scaled. Use [`Self::sprite_from_file_nearest`] for pixel art.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] on success, or an [`image::ImageError`] on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let player = ctx.sprite_from_file("assets/player.png")?;
    /// // In frame:
    /// frame.sprite(player, 100.0, 200.0);
    /// ```
    pub fn sprite_from_file(&mut self, path: &str) -> Result<SpriteId, image::ImageError> {
        let sprite = Sprite::from_file(self.gpu, path)?;
        Ok(self.add_sprite(sprite))
    }

    /// Load a 2D sprite from a file with nearest-neighbor (pixelated) filtering.
    ///
    /// Nearest-neighbor filtering preserves sharp pixel edges when scaling,
    /// making it ideal for pixel art sprites.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the image file
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] on success, or an [`image::ImageError`] on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let character = ctx.sprite_from_file_nearest("assets/character_16x16.png")?;
    /// // Scale up without blur:
    /// frame.sprite_scaled(character, x, y, 64.0, 64.0);
    /// ```
    pub fn sprite_from_file_nearest(&mut self, path: &str) -> Result<SpriteId, image::ImageError> {
        let sprite = Sprite::from_file_nearest(self.gpu, path)?;
        Ok(self.add_sprite(sprite))
    }

    /// Load a 2D sprite from embedded bytes with linear filtering.
    ///
    /// Useful for bundling sprites directly in the executable via `include_bytes!`.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw image file bytes (PNG, JPEG, etc.)
    /// * `label` - Debug label for GPU debugging tools
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] on success, or an [`image::ImageError`] on failure.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let icon = ctx.sprite_from_bytes(
    ///     include_bytes!("../assets/icon.png"),
    ///     "app icon"
    /// )?;
    /// ```
    pub fn sprite_from_bytes(
        &mut self,
        bytes: &[u8],
        label: &str,
    ) -> Result<SpriteId, image::ImageError> {
        let sprite = Sprite::from_bytes(self.gpu, bytes, label)?;
        Ok(self.add_sprite(sprite))
    }

    /// Load a 2D sprite from embedded bytes with nearest-neighbor filtering.
    ///
    /// Combines embedded loading with pixel-art-friendly filtering.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Raw image file bytes (PNG, JPEG, etc.)
    /// * `label` - Debug label for GPU debugging tools
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] on success, or an [`image::ImageError`] on failure.
    pub fn sprite_from_bytes_nearest(
        &mut self,
        bytes: &[u8],
        label: &str,
    ) -> Result<SpriteId, image::ImageError> {
        let sprite = Sprite::from_bytes_nearest(self.gpu, bytes, label)?;
        Ok(self.add_sprite(sprite))
    }

    /// Create a sprite from a procedural blocky noise texture.
    ///
    /// Useful for testing and demos. Generates a blocky, earthy pattern.
    ///
    /// # Arguments
    ///
    /// * `size` - Sprite dimensions (creates a size×size sprite)
    /// * `seed` - Random seed for reproducible generation
    ///
    /// # Returns
    ///
    /// A [`SpriteId`] for the generated sprite.
    pub fn sprite_blocky_noise(&mut self, size: u32, seed: u32) -> SpriteId {
        let data = generate_blocky_noise_data(size, seed);
        let sprite = Sprite::from_rgba_nearest(self.gpu, &data, size, size, "Blocky Noise Sprite");
        self.add_sprite(sprite)
    }
}

/// Generate RGBA pixel data for a blocky noise texture.
///
/// This is an internal helper function that generates procedural texture data
/// using a simple hash-based approach. The result has an earthy, blocky
/// aesthetic.
///
/// # Arguments
///
/// * `size` - Width and height of the texture in pixels
/// * `seed` - Random seed for reproducible generation
///
/// # Returns
///
/// A `Vec<u8>` containing RGBA pixel data (4 bytes per pixel, row-major order).
fn generate_blocky_noise_data(size: u32, seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (size * size * 4) as usize];

    // Earthy color palette for the blocky aesthetic
    let colors: &[[u8; 3]] = &[
        [139, 90, 43],   // Brown (dirt)
        [128, 128, 128], // Gray (stone)
        [85, 85, 85],    // Dark gray
        [160, 120, 60],  // Light brown
        [100, 70, 40],   // Dark brown
        [90, 90, 90],    // Medium gray
        [120, 100, 70],  // Tan
        [70, 60, 50],    // Very dark brown
    ];

    /// Simple hash function for pseudo-random per-pixel values.
    fn hash(x: u32, y: u32, seed: u32) -> u32 {
        let mut h = seed;
        h = h.wrapping_add(x.wrapping_mul(374761393));
        h = h.wrapping_add(y.wrapping_mul(668265263));
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177);
        h ^= h >> 16;
        h
    }

    for y in 0..size {
        for x in 0..size {
            let idx = ((y * size + x) * 4) as usize;
            let h = hash(x, y, seed);
            let color_idx = (h % colors.len() as u32) as usize;
            let base = colors[color_idx];
            // Add slight per-pixel variation for more natural look
            let variation = ((hash(x + 1000, y + 1000, seed) % 30) as i32) - 15;

            data[idx] = (base[0] as i32 + variation).clamp(0, 255) as u8;
            data[idx + 1] = (base[1] as i32 + variation).clamp(0, 255) as u8;
            data[idx + 2] = (base[2] as i32 + variation).clamp(0, 255) as u8;
            data[idx + 3] = 255; // Fully opaque
        }
    }

    data
}

/// Context provided each frame for rendering and game logic.
///
/// `Frame` is passed to your frame closure every frame and provides:
/// - **Timing info**: elapsed time, delta time, FPS
/// - **Input state**: keyboard, mouse (via the [`Input`] field)
/// - **Camera control**: position, rotation, zoom
/// - **Drawing methods**: text, rectangles, sprites, meshes
///
/// For simple applications, the convenience methods (`text`, `rect`, `sprite`, etc.)
/// are usually sufficient. For advanced use cases, the underlying `gpu`, `assets`,
/// and `draw` fields provide direct access to lower-level APIs.
///
/// # Example
///
/// ```ignore
/// move |frame| {
///     // Update camera based on input
///     if frame.input.key_held(KeyCode::KeyW) {
///         frame.camera.position.z -= 5.0 * frame.dt;
///     }
///
///     // Draw UI
///     frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
///     frame.text(10.0, 30.0, &format!("Time: {:.1}s", frame.time));
///
///     // Draw 3D meshes
///     let transform = Transform::translation(0.0, 0.0, -5.0)
///         .rotate_y(frame.time);
///     frame.draw_mesh(cube_mesh, transform, Color::RED);
/// }
/// ```
pub struct Frame<'a> {
    /// GPU context for advanced rendering operations.
    ///
    /// Provides direct access to wgpu device, queue, and surface for custom
    /// rendering code that goes beyond the built-in helpers.
    pub gpu: &'a GpuContext,

    /// Asset manager containing loaded fonts.
    ///
    /// Use with `Draw2d::text` for custom font rendering. The convenience
    /// methods `Frame::text` and `Frame::text_color` use the default font
    /// set during setup.
    pub assets: &'a Assets,

    /// Low-level 2D drawing API.
    ///
    /// Provides direct control over 2D rendering including custom vertex
    /// batching and advanced text layout. The `Frame::text`, `Frame::rect`,
    /// and `Frame::sprite` methods delegate to this internally.
    pub draw: &'a mut Draw2d,

    /// Camera for 3D rendering.
    ///
    /// Modify position, rotation, field of view, etc. to control the viewpoint.
    /// Changes take effect immediately for subsequent draw calls.
    pub camera: &'a mut Camera,

    /// Input state for this frame.
    ///
    /// Query keyboard and mouse state. Input is captured once per frame,
    /// so the same query returns the same result throughout a frame.
    pub input: &'a Input,

    /// ECS world for entity management.
    ///
    /// Query and mutate entities each frame. Spawn new entities, update
    /// components, or despawn entities based on game logic.
    pub world: &'a mut hecs::World,

    /// Total elapsed time since application start, in seconds.
    ///
    /// Useful for animations and time-based effects. Continuously increases;
    /// never resets during the application lifetime.
    pub time: f32,

    /// Time elapsed since the previous frame, in seconds.
    ///
    /// Use this for frame-rate-independent movement and physics:
    /// `position += velocity * frame.dt`
    ///
    /// Typically around 0.016 (60 FPS) or 0.008 (120 FPS).
    pub dt: f32,

    /// Default font set during setup (if any).
    default_font: Option<FontId>,

    /// Shared mesh queue for 3D draw calls.
    mesh_queue: Rc<RefCell<MeshQueue>>,

    /// Window handle for cursor control.
    window: &'a Window,
}

impl Frame<'_> {
    // ========================================================================
    // Timing & Screen Info
    // ========================================================================

    /// Calculate the current frames per second based on delta time.
    ///
    /// This is a simple reciprocal of `dt`. For smoothed FPS display,
    /// consider averaging over multiple frames in your application code.
    ///
    /// # Returns
    ///
    /// FPS as a float. Returns 0.0 if `dt` is zero (shouldn't happen in practice).
    ///
    /// # Example
    ///
    /// ```ignore
    /// frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
    /// ```
    pub fn fps(&self) -> f32 {
        if self.dt > 0.0 { 1.0 / self.dt } else { 0.0 }
    }

    /// Get the current window/screen width in pixels.
    ///
    /// Useful for positioning UI elements relative to screen edges or for
    /// calculating aspect ratios.
    pub fn width(&self) -> u32 {
        self.gpu.width()
    }

    /// Get the current window/screen height in pixels.
    ///
    /// Useful for positioning UI elements relative to screen edges or for
    /// calculating aspect ratios.
    pub fn height(&self) -> u32 {
        self.gpu.height()
    }

    // ========================================================================
    // Cursor Control
    // ========================================================================

    /// Capture and hide the mouse cursor for FPS-style controls.
    ///
    /// When captured, the cursor is hidden and confined to the window.
    /// Mouse delta will continue to report movement. Call [`release_cursor`](Self::release_cursor)
    /// to restore normal cursor behavior.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Capture cursor on click
    /// if frame.input.mouse_pressed(MouseButton::Left) {
    ///     frame.capture_cursor();
    /// }
    /// // Release on Escape
    /// if frame.input.key_pressed(KeyCode::Escape) {
    ///     frame.release_cursor();
    /// }
    /// ```
    pub fn capture_cursor(&self) {
        use winit::window::CursorGrabMode;
        // Try confined mode first (keeps cursor in window), fall back to locked (hides cursor)
        let _ = self
            .window
            .set_cursor_grab(CursorGrabMode::Confined)
            .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Locked));
        self.window.set_cursor_visible(false);
    }

    /// Release the mouse cursor, restoring normal behavior.
    ///
    /// This undoes [`capture_cursor`](Self::capture_cursor), showing the cursor
    /// and allowing it to leave the window.
    pub fn release_cursor(&self) {
        use winit::window::CursorGrabMode;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
    }

    // ========================================================================
    // Text Rendering
    // ========================================================================

    /// Draw white text at the given screen position using the default font.
    ///
    /// Coordinates are in screen pixels with (0, 0) at top-left.
    ///
    /// # Panics
    ///
    /// Panics if no default font was set during setup via [`SetupContext::default_font`].
    ///
    /// # Arguments
    ///
    /// * `x` - X position in screen pixels
    /// * `y` - Y position in screen pixels
    /// * `text` - The string to render
    ///
    /// # Example
    ///
    /// ```ignore
    /// frame.text(10.0, 10.0, "Hello, Hoplite!");
    /// frame.text(10.0, 30.0, &format!("Score: {}", score));
    /// ```
    pub fn text(&mut self, x: f32, y: f32, text: &str) {
        self.text_color(x, y, text, Color::WHITE)
    }

    /// Draw colored text at the given screen position using the default font.
    ///
    /// # Panics
    ///
    /// Panics if no default font was set during setup.
    ///
    /// # Arguments
    ///
    /// * `x` - X position in screen pixels
    /// * `y` - Y position in screen pixels
    /// * `text` - The string to render
    /// * `color` - Text color
    ///
    /// # Example
    ///
    /// ```ignore
    /// frame.text_color(10.0, 50.0, "WARNING!", Color::RED);
    /// frame.text_color(10.0, 70.0, "Health OK", Color::GREEN);
    /// ```
    pub fn text_color(&mut self, x: f32, y: f32, text: &str, color: Color) {
        let font = self
            .default_font
            .expect("No default font set. Call ctx.default_font() in setup.");
        self.draw.text(self.assets, font, x, y, text, color);
    }

    // ========================================================================
    // Rectangle & Panel Rendering
    // ========================================================================

    /// Draw a solid colored rectangle.
    ///
    /// # Arguments
    ///
    /// * `x` - X position of top-left corner in screen pixels
    /// * `y` - Y position of top-left corner in screen pixels
    /// * `w` - Width in pixels
    /// * `h` - Height in pixels
    /// * `color` - Fill color
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw a red health bar background
    /// frame.rect(10.0, 100.0, 200.0, 20.0, Color::rgb(0.2, 0.0, 0.0));
    /// // Draw the health fill
    /// frame.rect(10.0, 100.0, health_pct * 200.0, 20.0, Color::RED);
    /// ```
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.draw.rect(x, y, w, h, color);
    }

    /// Draw a UI panel with a styled background and border.
    ///
    /// Panels provide a consistent look for UI containers. For panels with
    /// titles, use [`Self::panel_titled`].
    ///
    /// # Arguments
    ///
    /// * `x` - X position of top-left corner
    /// * `y` - Y position of top-left corner
    /// * `w` - Panel width
    /// * `h` - Panel height
    ///
    /// # Returns
    ///
    /// The Y coordinate where content should begin (same as `y` for titleless panels).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let content_y = frame.panel(10.0, 10.0, 200.0, 150.0);
    /// frame.text(20.0, content_y + 10.0, "Panel content here");
    /// ```
    pub fn panel(&mut self, x: f32, y: f32, w: f32, h: f32) -> f32 {
        self.draw.panel(x, y, w, h).draw(self.assets);
        y
    }

    /// Draw a UI panel with a title bar.
    ///
    /// The title bar uses the default font and has a distinct background color.
    ///
    /// # Panics
    ///
    /// Panics if no default font was set during setup.
    ///
    /// # Arguments
    ///
    /// * `x` - X position of top-left corner
    /// * `y` - Y position of top-left corner
    /// * `w` - Panel width
    /// * `h` - Total panel height (including title bar)
    /// * `title` - Title text to display
    ///
    /// # Returns
    ///
    /// The Y coordinate where content should begin (below the title bar, ~22px offset).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let content_y = frame.panel_titled(10.0, 10.0, 250.0, 200.0, "Inventory");
    /// frame.text(20.0, content_y + 10.0, "Sword x1");
    /// frame.text(20.0, content_y + 30.0, "Potion x3");
    /// ```
    pub fn panel_titled(&mut self, x: f32, y: f32, w: f32, h: f32, title: &str) -> f32 {
        let font = self
            .default_font
            .expect("Panel with title requires default font. Call ctx.default_font() in setup.");
        self.draw
            .panel(x, y, w, h)
            .title(title, font)
            .draw(self.assets);
        y + 22.0 // Title bar height
    }

    // ========================================================================
    // Camera Control
    // ========================================================================

    /// Set the camera for this frame.
    ///
    /// This is a cleaner alternative to `*frame.camera = camera`.
    ///
    /// # Arguments
    ///
    /// * `camera` - The camera configuration to use
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut orbit = OrbitCamera::new().distance(10.0);
    /// move |frame| {
    ///     orbit.update(frame.input, frame.dt);
    ///     frame.set_camera(orbit.camera());
    /// }
    /// ```
    pub fn set_camera(&mut self, camera: Camera) {
        *self.camera = camera;
    }

    // ========================================================================
    // 3D Mesh Rendering
    // ========================================================================

    /// Start building a mesh draw call with a fluent API.
    ///
    /// This is the preferred way to draw meshes - it's more readable and
    /// discoverable than the classic `draw_mesh` methods.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh to draw (from `ctx.mesh_cube()`, etc.)
    ///
    /// # Returns
    ///
    /// A [`MeshBuilder`] for configuring and drawing the mesh.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Simple: draw at position with color
    /// frame.mesh(cube).at(0.0, 2.0, -5.0).color(Color::RED).draw();
    ///
    /// // With texture
    /// frame.mesh(cube).at(0.0, 0.0, -5.0).texture(brick_tex).draw();
    ///
    /// // Full control
    /// frame.mesh(cube)
    ///     .transform(Transform::new()
    ///         .position(Vec3::Y * 2.0)
    ///         .rotation(Quat::from_rotation_y(frame.time)))
    ///     .color(Color::rgb(0.8, 0.2, 0.2))
    ///     .texture(wood_tex)
    ///     .draw();
    /// ```
    pub fn mesh(&mut self, mesh: MeshId) -> MeshBuilder<'_> {
        MeshBuilder {
            queue: &self.mesh_queue,
            mesh,
            transform: Transform::default(),
            color: Color::WHITE,
            texture: None,
        }
    }

    /// Draw a 3D mesh with the given transform and color.
    ///
    /// For a more fluent API, consider using [`Self::mesh`] instead.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle returned by `SetupContext::mesh_*` or `add_mesh`
    /// * `transform` - Position, rotation, and scale of the mesh
    /// * `color` - Solid color for the mesh
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Spinning red cube
    /// let transform = Transform::translation(0.0, 0.0, -5.0)
    ///     .rotate_y(frame.time * 0.5);
    /// frame.draw_mesh(cube, transform, Color::RED);
    /// ```
    pub fn draw_mesh(&mut self, mesh: MeshId, transform: Transform, color: Color) {
        self.mesh_queue.borrow_mut().draw(mesh, transform, color);
    }

    /// Draw a 3D mesh with default white color.
    ///
    /// Convenience method equivalent to `draw_mesh(mesh, transform, Color::WHITE)`.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle returned by `SetupContext::mesh_*` or `add_mesh`
    /// * `transform` - Position, rotation, and scale of the mesh
    pub fn draw_mesh_white(&mut self, mesh: MeshId, transform: Transform) {
        self.draw_mesh(mesh, transform, Color::WHITE);
    }

    /// Draw a textured 3D mesh with a color tint.
    ///
    /// The texture is sampled and multiplied by the color. Use `Color::WHITE`
    /// for no tinting (show texture as-is).
    ///
    /// For a more fluent API, consider using [`Self::mesh`] instead:
    /// `frame.mesh(cube).texture(tex).draw()`
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle returned by `SetupContext::mesh_*` or `add_mesh`
    /// * `transform` - Position, rotation, and scale of the mesh
    /// * `color` - Tint color (multiplied with texture)
    /// * `texture` - Handle returned by `SetupContext::texture_*` or `add_texture`
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Textured cube with red tint
    /// frame.draw_mesh_textured(cube, transform, Color::RED, brick_texture);
    ///
    /// // Textured cube with no tint
    /// frame.draw_mesh_textured(cube, transform, Color::WHITE, brick_texture);
    /// ```
    pub fn draw_mesh_textured(
        &mut self,
        mesh: MeshId,
        transform: Transform,
        color: Color,
        texture: TextureId,
    ) {
        self.mesh_queue
            .borrow_mut()
            .draw_textured(mesh, transform, color, texture);
    }

    /// Draw a textured 3D mesh with no color tint.
    ///
    /// Convenience method equivalent to `draw_mesh_textured(mesh, transform, Color::WHITE, texture)`.
    ///
    /// For a more fluent API, consider using [`Self::mesh`] instead:
    /// `frame.mesh(cube).texture(tex).draw()`
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle returned by `SetupContext::mesh_*` or `add_mesh`
    /// * `transform` - Position, rotation, and scale of the mesh
    /// * `texture` - Handle returned by `SetupContext::texture_*` or `add_texture`
    pub fn draw_mesh_textured_white(
        &mut self,
        mesh: MeshId,
        transform: Transform,
        texture: TextureId,
    ) {
        self.draw_mesh_textured(mesh, transform, Color::WHITE, texture);
    }

    // ========================================================================
    // 2D Sprite Rendering
    // ========================================================================

    /// Draw a 2D sprite at its native size.
    ///
    /// The sprite is rendered at its original pixel dimensions. Use
    /// [`Self::sprite_scaled`] to draw at a different size.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID returned by `SetupContext::sprite_*` or `add_sprite`
    /// * `x` - X position of top-left corner in screen pixels
    /// * `y` - Y position of top-left corner in screen pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw player sprite at its natural size
    /// frame.sprite(player_sprite, player_x, player_y);
    /// ```
    pub fn sprite(&mut self, sprite_id: SpriteId, x: f32, y: f32) {
        self.draw.sprite(sprite_id, x, y, Color::WHITE);
    }

    /// Draw a 2D sprite with a color tint.
    ///
    /// The tint color is multiplied with the sprite's pixels. Use this for
    /// damage flashes, team colors, or fading effects.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID returned by `SetupContext::sprite_*` or `add_sprite`
    /// * `x` - X position of top-left corner in screen pixels
    /// * `y` - Y position of top-left corner in screen pixels
    /// * `tint` - Color to multiply with sprite pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Flash red when damaged
    /// let tint = if is_damaged { Color::RED } else { Color::WHITE };
    /// frame.sprite_tinted(player_sprite, x, y, tint);
    ///
    /// // Fade out (50% opacity)
    /// frame.sprite_tinted(sprite, x, y, Color::rgba(1.0, 1.0, 1.0, 0.5));
    /// ```
    pub fn sprite_tinted(&mut self, sprite_id: SpriteId, x: f32, y: f32, tint: Color) {
        self.draw.sprite(sprite_id, x, y, tint);
    }

    /// Draw a 2D sprite scaled to fit a rectangle.
    ///
    /// The sprite is stretched or shrunk to exactly fill the specified dimensions.
    /// For pixel art, consider using nearest-neighbor filtered sprites
    /// (loaded via `sprite_from_file_nearest`) to preserve crisp edges.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID returned by `SetupContext::sprite_*` or `add_sprite`
    /// * `x` - X position of top-left corner
    /// * `y` - Y position of top-left corner
    /// * `w` - Desired width in pixels
    /// * `h` - Desired height in pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Scale 16x16 pixel art to 64x64
    /// frame.sprite_scaled(character, x, y, 64.0, 64.0);
    /// ```
    pub fn sprite_scaled(&mut self, sprite_id: SpriteId, x: f32, y: f32, w: f32, h: f32) {
        self.draw.sprite_scaled(sprite_id, x, y, w, h, Color::WHITE);
    }

    /// Draw a 2D sprite scaled with a color tint.
    ///
    /// Combines scaling and tinting in a single call.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID returned by `SetupContext::sprite_*` or `add_sprite`
    /// * `x` - X position of top-left corner
    /// * `y` - Y position of top-left corner
    /// * `w` - Desired width in pixels
    /// * `h` - Desired height in pixels
    /// * `tint` - Color to multiply with sprite pixels
    pub fn sprite_scaled_tinted(
        &mut self,
        sprite_id: SpriteId,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        tint: Color,
    ) {
        self.draw.sprite_scaled(sprite_id, x, y, w, h, tint);
    }

    /// Draw a sub-region of a sprite (for sprite sheets/atlases).
    ///
    /// This allows drawing individual frames from a sprite sheet by specifying
    /// a source rectangle within the sprite texture.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID of the sprite sheet
    /// * `x`, `y` - Destination position on screen
    /// * `w`, `h` - Destination size on screen
    /// * `src_x`, `src_y` - Source rectangle position within the sprite (in pixels)
    /// * `src_w`, `src_h` - Source rectangle size (in pixels)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Draw frame 3 from a 16x16 sprite sheet (4 columns)
    /// let frame_idx = 3;
    /// let src_x = (frame_idx % 4) as f32 * 16.0;
    /// let src_y = (frame_idx / 4) as f32 * 16.0;
    /// frame.sprite_region(sheet, x, y, 32.0, 32.0, src_x, src_y, 16.0, 16.0);
    /// ```
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
    ) {
        self.draw.sprite_region(
            sprite_id,
            x,
            y,
            w,
            h,
            src_x,
            src_y,
            src_w,
            src_h,
            Color::WHITE,
        );
    }

    /// Draw a sub-region of a sprite with a color tint.
    ///
    /// Combines sprite sheet support with color tinting.
    ///
    /// # Arguments
    ///
    /// * `sprite_id` - ID of the sprite sheet
    /// * `x`, `y` - Destination position on screen
    /// * `w`, `h` - Destination size on screen
    /// * `src_x`, `src_y` - Source rectangle position within the sprite
    /// * `src_w`, `src_h` - Source rectangle size
    /// * `tint` - Color to multiply with sprite pixels
    pub fn sprite_region_tinted(
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
        self.draw
            .sprite_region(sprite_id, x, y, w, h, src_x, src_y, src_w, src_h, tint);
    }

    // ========================================================================
    // ECS Rendering
    // ========================================================================

    /// Render all entities with [`Transform`](crate::Transform) and [`RenderMesh`](crate::RenderMesh) components.
    ///
    /// This queries the ECS world for entities with both components and queues
    /// them for rendering. Call this during your frame update to render all
    /// ECS-managed entities.
    ///
    /// Requires [`SetupContext::enable_mesh_rendering`] to be called during setup.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Setup:
    /// ctx.enable_mesh_rendering();
    /// let cube = ctx.mesh_cube();
    ///
    /// ctx.world.spawn((
    ///     Transform::new().position(Vec3::new(0.0, 0.0, -5.0)),
    ///     RenderMesh::new(cube, Color::RED),
    /// ));
    ///
    /// // Frame loop:
    /// move |frame| {
    ///     frame.render_world();
    /// }
    /// ```
    pub fn render_world(&mut self) {
        use crate::ecs::RenderMesh;
        use crate::mesh::Transform;

        for (_, (transform, render_mesh)) in self.world.query::<(&Transform, &RenderMesh)>().iter()
        {
            if let Some(texture) = render_mesh.texture {
                self.mesh_queue.borrow_mut().draw_textured(
                    render_mesh.mesh,
                    *transform,
                    render_mesh.color,
                    texture,
                );
            } else {
                self.mesh_queue
                    .borrow_mut()
                    .draw(render_mesh.mesh, *transform, render_mesh.color);
            }
        }
    }

    // ========================================================================
    // 3D Picking / Raycasting
    // ========================================================================

    /// Create a ray from the current mouse position for picking.
    ///
    /// This creates a ray that starts at the camera and passes through the
    /// mouse cursor position in 3D space. Use this for custom raycasting
    /// against game objects.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ray = frame.mouse_ray();
    /// // Test against custom geometry, planes, etc.
    /// if let Some(t) = ray.intersect_aabb(min, max) {
    ///     let hit_point = ray.point_at(t);
    /// }
    /// ```
    pub fn mouse_ray(&self) -> Ray {
        let mouse = self.input.mouse_position();
        let aspect = self.gpu.width() as f32 / self.gpu.height() as f32;

        Ray::from_screen(
            mouse.x,
            mouse.y,
            self.gpu.width() as f32,
            self.gpu.height() as f32,
            self.camera.view_matrix(),
            self.camera
                .projection_matrix(aspect, self.camera.near, self.camera.far),
        )
    }

    /// Cast a ray from the mouse and find the closest entity with a collider.
    ///
    /// This is the primary method for implementing mouse picking in your game.
    /// It tests the ray against all entities that have both a `Transform` and
    /// `Collider` component.
    ///
    /// # Returns
    ///
    /// The closest hit entity, or `None` if nothing was hit.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check what the mouse is hovering over
    /// if let Some(hit) = frame.pick_collider() {
    ///     frame.text(10.0, 10.0, &format!("Entity: {:?}", hit.entity));
    ///
    ///     // Handle clicks
    ///     if frame.input.mouse_pressed(MouseButton::Left) {
    ///         // Select or interact with the entity
    ///     }
    /// }
    /// ```
    pub fn pick_collider(&self) -> PickResult {
        let ray = self.mouse_ray();
        picking::raycast(self.world, &ray)
    }

    /// Cast a ray from the mouse and find all entities with colliders.
    ///
    /// Like [`pick_collider`](Self::pick_collider), but returns all hits
    /// sorted by distance (closest first). Useful for selection through
    /// transparent objects or implementing "pick all in line" mechanics.
    ///
    /// # Returns
    ///
    /// A vector of all hits, sorted by distance (closest first).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let hits = frame.pick_collider_all();
    /// for hit in hits {
    ///     println!("Hit {:?} at distance {}", hit.entity, hit.distance);
    /// }
    /// ```
    pub fn pick_collider_all(&self) -> Vec<RayHit> {
        let ray = self.mouse_ray();
        picking::raycast_all(self.world, &ray)
    }

    /// Cast a custom ray and find the closest entity with a collider.
    ///
    /// Use this when you need to cast rays from positions other than the mouse,
    /// such as for AI line-of-sight or projectile trajectories.
    ///
    /// # Arguments
    ///
    /// * `ray` - The ray to cast
    ///
    /// # Returns
    ///
    /// The closest hit entity, or `None` if nothing was hit.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Check if enemy can see player
    /// let ray = Ray::new(enemy_pos, (player_pos - enemy_pos).normalize());
    /// if let Some(hit) = frame.raycast(&ray) {
    ///     // Something is in the way
    /// }
    /// ```
    pub fn raycast(&self, ray: &Ray) -> PickResult {
        picking::raycast(self.world, ray)
    }

    /// Cast a custom ray and find all entities with colliders.
    ///
    /// # Arguments
    ///
    /// * `ray` - The ray to cast
    ///
    /// # Returns
    ///
    /// A vector of all hits, sorted by distance (closest first).
    pub fn raycast_all(&self, ray: &Ray) -> Vec<RayHit> {
        picking::raycast_all(self.world, ray)
    }
}

/// Builder for configuring and drawing a 3D mesh.
///
/// Created by [`Frame::mesh`]. Use the builder methods to configure the mesh's
/// transform, color, and texture, then call [`draw`](Self::draw) to queue it for rendering.
///
/// # Example
///
/// ```ignore
/// // Simple usage
/// frame.mesh(cube).at(0.0, 2.0, -5.0).color(Color::RED).draw();
///
/// // With texture
/// frame.mesh(cube).at(0.0, 0.0, -5.0).texture(brick_tex).draw();
///
/// // With full transform control
/// frame.mesh(cube)
///     .transform(Transform::new()
///         .position(Vec3::new(0.0, 1.0, -5.0))
///         .rotation(Quat::from_rotation_y(time)))
///     .color(Color::rgb(0.8, 0.2, 0.2))
///     .draw();
/// ```
pub struct MeshBuilder<'a> {
    queue: &'a Rc<RefCell<MeshQueue>>,
    mesh: MeshId,
    transform: Transform,
    color: Color,
    texture: Option<TextureId>,
}

impl MeshBuilder<'_> {
    /// Set the mesh position in world space.
    ///
    /// This is a convenience method that sets only the translation component.
    /// For rotation or scale, use [`transform`](Self::transform).
    ///
    /// # Arguments
    ///
    /// * `x`, `y`, `z` - World position coordinates
    pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.transform = Transform::from_position(glam::Vec3::new(x, y, z));
        self
    }

    /// Set the mesh position from a Vec3.
    ///
    /// This is a convenience method that sets only the translation component.
    ///
    /// # Arguments
    ///
    /// * `pos` - World position as a Vec3
    pub fn position(mut self, pos: glam::Vec3) -> Self {
        self.transform = Transform::from_position(pos);
        self
    }

    /// Set the full transform (position, rotation, scale).
    ///
    /// Use this when you need rotation or non-uniform scaling.
    ///
    /// # Arguments
    ///
    /// * `transform` - Full transform configuration
    pub fn transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the mesh color or tint.
    ///
    /// When used without a texture, this is the solid color of the mesh.
    /// When used with a texture, this is multiplied with the texture colors.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to apply
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Apply a texture to the mesh.
    ///
    /// The texture will be sampled using the mesh's UV coordinates and
    /// multiplied by the color (default white = no tint).
    ///
    /// # Arguments
    ///
    /// * `texture` - Texture handle from `ctx.texture_*` methods
    pub fn texture(mut self, texture: TextureId) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Queue the mesh for rendering.
    ///
    /// This must be called to actually draw the mesh. The builder pattern
    /// allows you to configure all options, then draw with a single call.
    pub fn draw(self) {
        let mut queue = self.queue.borrow_mut();
        if let Some(texture) = self.texture {
            queue.draw_textured(self.mesh, self.transform, self.color, texture);
        } else {
            queue.draw(self.mesh, self.transform, self.color);
        }
    }
}

/// Configuration options for creating a Hoplite application window.
///
/// Use this struct with [`run_with_config`] to customize the window title
/// and initial dimensions. For default settings (800x600 window titled "Hoplite"),
/// use [`run`] instead.
///
/// # Example
///
/// ```ignore
/// use hoplite::app::{run_with_config, AppConfig};
///
/// run_with_config(
///     AppConfig::new()
///         .title("My Game")
///         .size(1280, 720),
///     |ctx| {
///         // setup...
///         move |frame| {
///             // frame loop...
///         }
///     }
/// );
/// ```
pub struct AppConfig {
    /// Window title displayed in the title bar.
    pub title: String,
    /// Initial window width in pixels.
    pub width: u32,
    /// Initial window height in pixels.
    pub height: u32,
}

impl Default for AppConfig {
    /// Create default configuration: 800x600 window titled "Hoplite".
    fn default() -> Self {
        Self {
            title: "Hoplite".to_string(),
            width: 800,
            height: 600,
        }
    }
}

impl AppConfig {
    /// Create a new configuration with default values.
    ///
    /// Equivalent to [`AppConfig::default()`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the window title.
    ///
    /// # Arguments
    ///
    /// * `title` - Any type that can be converted to `String`
    ///
    /// # Example
    ///
    /// ```ignore
    /// AppConfig::new().title("Space Invaders")
    /// ```
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Set the initial window dimensions.
    ///
    /// The window may be resized by the user after creation. Use `Frame::width()`
    /// and `Frame::height()` to get the current dimensions during rendering.
    ///
    /// # Arguments
    ///
    /// * `width` - Window width in pixels
    /// * `height` - Window height in pixels
    ///
    /// # Example
    ///
    /// ```ignore
    /// AppConfig::new().size(1920, 1080)  // Full HD
    /// ```
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
}

/// Run a Hoplite application with default configuration.
///
/// This is the main entry point for most Hoplite applications. It creates a
/// window (800x600, titled "Hoplite"), initializes the GPU, and runs your
/// setup and frame closures.
///
/// For custom window configuration, use [`run_with_config`] instead.
///
/// # Type Parameters
///
/// * `S` - Setup closure type: `FnOnce(&mut SetupContext) -> F`
/// * `F` - Frame closure type: `FnMut(&mut Frame)`
///
/// # Arguments
///
/// * `setup` - A closure that receives a [`SetupContext`] for initialization
///   and returns a frame closure. The frame closure is called every frame.
///
/// # Panics
///
/// Panics if:
/// - The GPU cannot be initialized (no compatible adapter found)
/// - The window cannot be created
/// - Event loop creation fails
///
/// # Example
///
/// ```ignore
/// use hoplite::app::run;
///
/// fn main() {
///     run(|ctx| {
///         // Setup phase: load resources, configure pipeline
///         ctx.default_font(16.0);
///         ctx.effect_world(include_str!("background.wgsl"));
///
///         // Return the frame closure (called every frame)
///         move |frame| {
///             frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
///         }
///     });
/// }
/// ```
pub fn run<S, F>(setup: S)
where
    S: FnOnce(&mut SetupContext) -> F + 'static,
    F: FnMut(&mut Frame) + 'static,
{
    run_with_config(AppConfig::default(), setup);
}

/// Run a Hoplite application with custom window configuration.
///
/// Like [`run`], but allows specifying window title and dimensions.
///
/// # Type Parameters
///
/// * `S` - Setup closure type: `FnOnce(&mut SetupContext) -> F`
/// * `F` - Frame closure type: `FnMut(&mut Frame)`
///
/// # Arguments
///
/// * `config` - Window configuration (title, size)
/// * `setup` - A closure that receives a [`SetupContext`] for initialization
///   and returns a frame closure
///
/// # Panics
///
/// Same conditions as [`run`].
///
/// # Example
///
/// ```ignore
/// use hoplite::app::{run_with_config, AppConfig};
///
/// fn main() {
///     run_with_config(
///         AppConfig::new()
///             .title("Black Hole Simulator")
///             .size(1280, 720),
///         |ctx| {
///             ctx.default_font(16.0);
///             ctx.effect_world(include_str!("scene.wgsl"))
///                .post_process_world(include_str!("lensing.wgsl"));
///
///             move |frame| {
///                 frame.text(10.0, 10.0, "Hello!");
///             }
///         }
///     );
/// }
/// ```
pub fn run_with_config<S, F>(config: AppConfig, setup: S)
where
    S: FnOnce(&mut SetupContext) -> F + 'static,
    F: FnMut(&mut Frame) + 'static,
{
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = HopliteApp::Pending {
        config,
        setup: Some(Box::new(move |gpu, assets, draw, mesh_queue, world| {
            let mut default_font = None;
            let mut graph_builder = None;

            let mut ctx = SetupContext {
                gpu,
                assets,
                draw,
                world,
                default_font: &mut default_font,
                graph_builder: &mut graph_builder,
                mesh_queue,
            };

            let frame_fn = setup(&mut ctx);

            (
                Box::new(frame_fn) as Box<dyn FnMut(&mut Frame)>,
                default_font,
                graph_builder,
            )
        })),
    };

    event_loop.run_app(&mut app).unwrap();
}

/// Type alias for the internal setup function.
///
/// This boxed closure is created from the user's setup closure and handles
/// the actual initialization when the window becomes available. It receives
/// the GPU context and mutable references to assets and 2D drawing context,
/// and returns the frame closure along with optional default font and render graph.
type SetupFn = Box<
    dyn FnOnce(
        &GpuContext,
        &mut Assets,
        &mut Draw2d,
        &Rc<RefCell<MeshQueue>>,
        &mut hecs::World,
    ) -> (
        Box<dyn FnMut(&mut Frame)>,
        Option<FontId>,
        Option<RenderGraph>,
    ),
>;

/// Internal application state machine.
///
/// The Hoplite application lifecycle has two states:
///
/// - **Pending**: Initial state before the window is created. Holds the
///   configuration and setup closure.
/// - **Running**: Active state after initialization. Holds all runtime
///   resources and the frame closure.
///
/// The transition from `Pending` to `Running` happens in the [`ApplicationHandler::resumed`]
/// callback when the window system is ready.
enum HopliteApp {
    /// Application is waiting for window creation.
    Pending {
        /// Window configuration (title, size).
        config: AppConfig,
        /// User's setup closure (consumed during initialization).
        setup: Option<SetupFn>,
    },
    /// Application is running the main loop.
    Running {
        /// Native window handle (Arc for sharing with wgpu surface).
        window: Arc<Window>,
        /// GPU context containing device, queue, and surface.
        gpu: GpuContext,
        /// Loaded fonts and other managed assets.
        assets: Assets,
        /// 2D rendering state (batched draw calls).
        draw_2d: Draw2d,
        /// 3D camera state.
        camera: Camera,
        /// Input state (keyboard, mouse).
        input: Input,
        /// ECS world for entity management.
        world: hecs::World,
        /// User's frame closure (called every frame).
        frame_fn: Box<dyn FnMut(&mut Frame)>,
        /// Default font ID if one was set during setup.
        default_font: Option<FontId>,
        /// Optional render graph for shader effects and 3D rendering.
        render_graph: Option<RenderGraph>,
        /// Shared queue of mesh draw calls for the current frame.
        mesh_queue: Rc<RefCell<MeshQueue>>,
        /// Time when the application started (for `Frame::time`).
        start_time: Instant,
        /// Time of the last frame (for `Frame::dt` calculation).
        last_frame: Instant,
    },
}

/// Implementation of winit's [`ApplicationHandler`] trait for the Hoplite app.
///
/// This handles the window lifecycle events and implements the main render loop.
impl ApplicationHandler for HopliteApp {
    /// Called when the application is resumed (window becomes available).
    ///
    /// On first call, this transitions from `Pending` to `Running` by:
    /// 1. Creating the window with the configured attributes
    /// 2. Initializing the GPU context
    /// 3. Creating asset and 2D drawing systems
    /// 4. Running the user's setup closure
    /// 5. Transitioning to the `Running` state
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let HopliteApp::Pending { config, setup } = self {
            let window_attrs = WindowAttributes::default()
                .with_title(&config.title)
                .with_inner_size(winit::dpi::LogicalSize::new(config.width, config.height));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            let gpu = GpuContext::new(window.clone());
            let mut assets = Assets::new();
            let mut draw_2d = Draw2d::new(&gpu);

            // Create shared mesh queue for 3D rendering
            let mesh_queue = Rc::new(RefCell::new(MeshQueue::new()));

            // Create ECS world
            let mut world = hecs::World::new();

            // Run user's setup closure to get the frame function
            let setup_fn = setup.take().unwrap();
            let (frame_fn, default_font, render_graph) =
                setup_fn(&gpu, &mut assets, &mut draw_2d, &mesh_queue, &mut world);

            *self = HopliteApp::Running {
                window,
                gpu,
                assets,
                draw_2d,
                camera: Camera::new(),
                input: Input::new(),
                world,
                frame_fn,
                default_font,
                render_graph,
                mesh_queue,
                start_time: Instant::now(),
                last_frame: Instant::now(),
            };
        }
    }

    /// Handle window events (input, resize, close, redraw).
    ///
    /// The main render loop happens in `RedrawRequested`:
    /// 1. Calculate frame timing (dt, elapsed time)
    /// 2. Clear 2D and mesh state from previous frame
    /// 3. Create a [`Frame`] context and call the user's frame closure
    /// 4. Execute the render graph (or fallback to 2D-only rendering)
    /// 5. Request the next frame
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let HopliteApp::Running {
            window,
            gpu,
            assets,
            draw_2d,
            camera,
            input,
            world,
            frame_fn,
            default_font,
            render_graph,
            mesh_queue,
            start_time,
            last_frame,
        } = self
        else {
            return;
        };

        input.handle_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let time = start_time.elapsed().as_secs_f32();
                let dt = now.duration_since(*last_frame).as_secs_f32();
                *last_frame = now;

                // Clear draw_2d for new frame and update font bind groups
                draw_2d.clear();
                draw_2d.update_font_bind_groups(gpu, assets);

                // Clear mesh queue for new frame
                mesh_queue.borrow_mut().clear_queue();

                // Create frame context
                let mut frame = Frame {
                    gpu,
                    assets,
                    draw: draw_2d,
                    camera,
                    input,
                    world,
                    time,
                    dt,
                    default_font: *default_font,
                    mesh_queue: Rc::clone(mesh_queue),
                    window,
                };

                // Run user's frame function
                frame_fn(&mut frame);

                // Execute render graph if present, otherwise just render UI
                if let Some(graph) = render_graph {
                    graph.execute_with_ui(gpu, time, camera, |gpu, pass| {
                        draw_2d.render(gpu, pass, assets);
                    });
                } else {
                    // No render graph - just render 2D content to screen
                    render_2d_only(gpu, draw_2d, assets);
                }

                input.begin_frame();
                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Fallback renderer for applications without a render graph.
///
/// When no shader effects or 3D rendering are configured, this function
/// provides a simple path to render 2D content directly to the screen.
/// It clears the screen to black and renders all 2D draw calls (text,
/// rectangles, sprites).
///
/// This is used internally when the user doesn't call any effect/post-process
/// methods during setup.
///
/// # Arguments
///
/// * `gpu` - GPU context for accessing device, queue, and surface
/// * `draw_2d` - 2D drawing context with batched draw calls
/// * `assets` - Asset manager (needed for font textures)
fn render_2d_only(gpu: &GpuContext, draw_2d: &Draw2d, assets: &Assets) {
    // Get the next frame's texture to render to
    let output = gpu.surface.get_current_texture().unwrap();
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Create command encoder for this frame
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("2D Only Encoder"),
        });

    // Begin render pass: clear to black, then render 2D content
    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("2D Only Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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

        // Render all batched 2D draw calls
        draw_2d.render(gpu, &mut render_pass, assets);
    }

    // Submit commands and present the frame
    gpu.queue.submit(std::iter::once(encoder.finish()));
    output.present();
}
