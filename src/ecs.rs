//! ECS components for entity-based rendering.
//!
//! This module provides components for rendering entities using the `hecs` ECS library.
//! Entities with [`Transform`](crate::Transform) and [`RenderMesh`] components are
//! automatically rendered when [`Frame::render_world`](crate::Frame::render_world) is called.
//!
//! # Example
//!
//! ```ignore
//! use hoplite::*;
//!
//! run(|ctx| {
//!     ctx.enable_mesh_rendering();
//!     let cube = ctx.mesh_cube();
//!
//!     // Spawn an entity with mesh components
//!     ctx.world.spawn((
//!         Transform::new().position(Vec3::new(0.0, 0.0, -5.0)),
//!         RenderMesh::new(cube, Color::RED),
//!     ));
//!
//!     move |frame| {
//!         frame.render_world();
//!     }
//! });
//! ```

use crate::draw2d::Color;

/// Type-safe handle to a mesh stored in the MeshQueue.
///
/// Obtained from mesh creation methods like [`SetupContext::mesh_cube`](crate::SetupContext::mesh_cube).
/// This newtype wrapper prevents accidentally passing texture indices where mesh indices are expected.
///
/// # Example
///
/// ```ignore
/// let cube: MeshId = ctx.mesh_cube();
/// frame.mesh(cube).draw();  // Type-safe: can't pass a TextureId here
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshId(pub(crate) usize);

/// Type-safe handle to a texture stored in the MeshQueue.
///
/// Obtained from texture creation methods like [`SetupContext::texture_from_file`](crate::SetupContext::texture_from_file).
/// This newtype wrapper prevents accidentally passing mesh indices where texture indices are expected.
///
/// # Example
///
/// ```ignore
/// let tex: TextureId = ctx.texture_from_file("brick.png")?;
/// frame.mesh(cube).texture(tex).draw();  // Type-safe: can't pass a MeshId here
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureId(pub(crate) usize);

// Keep MeshHandle and TextureHandle as aliases for backwards compatibility in ECS contexts
/// Alias for [`MeshId`] - used in ECS components.
pub type MeshHandle = MeshId;

/// Alias for [`TextureId`] - used in ECS components.
pub type TextureHandle = TextureId;

/// Component for rendering a mesh on an entity.
///
/// Attach this component along with a [`Transform`](crate::Transform) to make an entity
/// renderable. Call [`Frame::render_world`](crate::Frame::render_world) to render all
/// entities with these components.
///
/// # Example
///
/// ```ignore
/// // Untextured colored mesh
/// world.spawn((
///     Transform::new().position(Vec3::Y * 2.0),
///     RenderMesh::new(MeshHandle(cube), Color::RED),
/// ));
///
/// // Textured mesh
/// world.spawn((
///     Transform::new(),
///     RenderMesh::with_texture(MeshHandle(cube), Color::WHITE, TextureHandle(tex)),
/// ));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct RenderMesh {
    /// Handle to the mesh geometry.
    pub mesh: MeshHandle,
    /// Color tint applied to the mesh.
    pub color: Color,
    /// Optional texture. If `None`, uses vertex colors only.
    pub texture: Option<TextureHandle>,
}

impl RenderMesh {
    /// Create a new render mesh component with a color tint.
    pub fn new(mesh: MeshHandle, color: Color) -> Self {
        Self {
            mesh,
            color,
            texture: None,
        }
    }

    /// Create a new render mesh component with a texture.
    pub fn with_texture(mesh: MeshHandle, color: Color, texture: TextureHandle) -> Self {
        Self {
            mesh,
            color,
            texture: Some(texture),
        }
    }
}
