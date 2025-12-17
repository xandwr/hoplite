//! # Hoplite
//!
//! **A creative coding framework for Rust that gets out of your way.**
//!
//! Write shaders, render 3D scenes, and build visualizations with a single closure.
//! No boilerplate, no ceremony—just code for the screen.
//!
//! ## Quick Start
//!
//! ```no_run
//! use hoplite::*;
//!
//! fn main() {
//!     run(|ctx| {
//!         ctx.default_font(16.0);
//!         ctx.background_color(Color::rgb(0.1, 0.1, 0.15));
//!         ctx.enable_mesh_rendering();
//!
//!         let cube = ctx.mesh_cube();
//!
//!         move |frame| {
//!             // Fluent mesh builder API
//!             frame.mesh(cube)
//!                 .at(0.0, 0.0, -5.0)
//!                 .color(Color::RED)
//!                 .draw();
//!
//!             frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
//!         }
//!     });
//! }
//! ```
//!
//! ## Philosophy
//!
//! - **One closure, one call** — Setup and frame logic live in closures. No traits to implement.
//! - **Hot reload everything** — Edit WGSL shaders and watch them update instantly.
//! - **Escape hatches everywhere** — Start simple, access the full wgpu API when needed.
//! - **Type-safe handles** — `MeshId` and `TextureId` prevent mix-ups at compile time.
//!
//! See the [repository](https://github.com/xandwr/hoplite) for full documentation and examples.

mod app;
mod assets;
mod camera;
mod draw2d;
mod ecs;
mod effect_pass;
mod freelook_camera;
mod geometry;
mod gpu;
mod hot_shader;
mod input;
mod mesh;
mod mesh_pass;
mod orbit_camera;
mod picking;
mod post_process;
mod render_graph;
pub mod scene;
mod scene_projection;
mod texture;

pub use app::{
    AppConfig, Frame, MeshBuilder, MeshLoader, SceneSetupContext, SetupContext, run,
    run_with_config, run_with_scenes, run_with_scenes_config,
};
pub use assets::{Assets, FontAtlas, FontId};
pub use camera::Camera;
pub use draw2d::{Color, Draw2d, PanelBuilder, Rect, SpriteId};
pub use effect_pass::EffectPass;
pub use freelook_camera::{FreelookCamera, FreelookMode, SeatedConfig};
pub use geometry::{GeometryError, GeometryLoader, PendingGeometry, RawGeometry};
pub use gpu::GpuContext;
pub use hot_shader::{HotEffectPass, HotPostProcessPass, HotShader, HotWorldPostProcessPass};
pub use input::Input;
pub use mesh::{Mesh, Transform, Vertex3d};
pub use mesh_pass::MeshPass;
pub use orbit_camera::{OrbitCamera, OrbitMode};
pub use post_process::{PostProcessPass, WorldPostProcessPass};
pub use render_graph::{
    EffectNode, HotEffectNode, HotPostProcessNode, HotWorldPostProcessNode, MeshNode, MeshQueue,
    PostProcessNode, RenderContext, RenderGraph, RenderGraphBuilder, RenderNode, RenderTarget,
    WorldPostProcessNode,
};
pub use scene_projection::SceneProjection;
pub use texture::{Sprite, Texture};

// Re-export glam math types for convenience
pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

// Re-export commonly used winit types for convenience
pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode;

// ECS support and type-safe handles
pub use ecs::{MeshHandle, MeshId, RenderMesh, TextureHandle, TextureId};
pub use hecs::{Entity, World};

// 3D picking and collision
pub use picking::{Collider, PickResult, Ray, RayHit};
