//! Scene projection for rendering one scene onto a mesh surface.
//!
//! This module enables portal-style rendering where one scene is rendered to a texture
//! and displayed on a mesh in another scene. Common use cases:
//! - Windows looking out from cockpits
//! - Portals between areas
//! - Security camera feeds
//! - Mirrors and reflective surfaces
//!
//! # Example
//!
//! ```ignore
//! use hoplite::*;
//!
//! run_with_scenes(|ctx| {
//!     let window_mesh = ctx.mesh_quad();
//!     
//!     // Scene to project (exterior view)
//!     ctx.scene("exterior", |scene| {
//!         scene.enable_mesh_rendering();
//!         // ... scene setup
//!     });
//!     
//!     // Scene with projection (interior with window)
//!     ctx.scene("interior", |scene| {
//!         scene.enable_mesh_rendering();
//!         
//!         let mut projection = SceneProjection::new(
//!             "exterior",
//!             Camera::new().at(Vec3::new(0.0, 5.0, 10.0)),
//!             512, 512,
//!             ctx.gpu()
//!         );
//!         
//!         move |frame| {
//!             // Projection texture is automatically updated by scene manager
//!             
//!             // Draw window with projected scene
//!             frame.mesh(window_mesh)
//!                 .project_scene(&projection)
//!                 .draw();
//!         }
//!     });
//! });
//! ```

use crate::camera::Camera;
use crate::gpu::GpuContext;
use crate::texture::Texture;

/// Projects a scene onto a mesh surface for portal/window rendering.
///
/// A `SceneProjection` renders a source scene to an offscreen texture that can be
/// applied to meshes in other scenes. The projection has its own camera, allowing
/// independent viewpoint control (e.g., a security camera with a fixed view, or a
/// portal that transforms the player's view).
///
/// # Note on Texture Registration
///
/// The projection's texture must be registered with the mesh queue using
/// `add_projection_texture` before it can be used with meshes.
///
/// # Fields
///
/// * `source_scene` - Name of the scene to render
/// * `camera` - Camera used to render the source scene
/// * `width`, `height` - Texture resolution
pub struct SceneProjection {
    /// Name of the source scene to render.
    pub source_scene: String,

    /// Camera used to render the source scene.
    pub camera: Camera,

    /// Render target texture for the projected scene.
    pub(crate) texture: wgpu::Texture,

    /// Texture view for rendering.
    pub(crate) texture_view: wgpu::TextureView,

    /// Width of the projection texture in pixels.
    pub width: u32,

    /// Height of the projection texture in pixels.
    pub height: u32,
}

impl SceneProjection {
    /// Create a new scene projection with custom resolution.
    ///
    /// # Arguments
    ///
    /// * `source_scene` - Name of the scene to render (must be registered)
    /// * `camera` - Camera viewpoint for rendering the source scene
    /// * `width` - Texture width in pixels (e.g., 512, 1024)
    /// * `height` - Texture height in pixels
    /// * `gpu` - GPU context for creating texture resources
    ///
    /// # Example
    ///
    /// ```ignore
    /// let projection = SceneProjection::new(
    ///     "exterior",
    ///     Camera::new().at(Vec3::new(0.0, 2.0, 5.0)).looking_at(Vec3::ZERO),
    ///     512, 512,
    ///     &gpu
    /// );
    /// ```
    pub fn new(
        source_scene: impl Into<String>,
        camera: Camera,
        width: u32,
        height: u32,
        gpu: &GpuContext,
    ) -> Self {
        // Create render target texture
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Projection Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            source_scene: source_scene.into(),
            camera,
            texture,
            texture_view,
            width,
            height,
        }
    }

    /// Update the projection camera.
    ///
    /// Use this to dynamically adjust the projection viewpoint (e.g., for portal-style
    /// view transformations based on player position).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Update camera to face the player
    /// let look_dir = (player_pos - portal_pos).normalize();
    /// projection.set_camera(Camera::new().at(portal_pos).looking_in(look_dir));
    /// ```
    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    /// Get the texture view for rendering.
    ///
    /// This is used by the scene manager to render the source scene.
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.texture_view
    }

    /// Convert this projection to a Texture for registration with the mesh queue.
    ///
    /// This creates a Texture wrapper using the projection's texture view and sampler.
    /// Register this texture with `ctx.add_texture()` to get a TextureId for rendering.
    ///
    /// **Important:** Continue to render to the projection using `scene_manager.render_scene_to_target()`
    /// to update the texture contents. Both the Texture and SceneProjection reference the same
    /// underlying GPU texture.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In scene setup
    /// let projection = SceneProjection::new("exterior", camera, 512, 512, ctx.gpu());
    /// let proj_texture = projection.as_texture(ctx.gpu());
    /// let texture_id = ctx.add_texture(proj_texture);
    ///
    /// // In frame closure
    /// scene_manager.render_scene_to_target(
    ///     &projection.source_scene,
    ///     projection.texture_view(),
    ///     &projection.camera,
    ///     frame.gpu,
    ///     frame.time
    /// );
    /// frame.mesh(window_mesh).texture(texture_id).draw();
    /// ```
    pub fn as_texture(&self, gpu: &GpuContext) -> Texture {
        // Use the projection's existing texture and create compatible view/sampler
        let view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Scene Projection Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create a dummy placeholder texture handle (only view and sampler are used for rendering)
        let placeholder = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Projection Placeholder"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        Texture::from_wgpu_resources(placeholder, view, sampler, self.width, self.height)
    }
}
