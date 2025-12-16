mod app;
mod assets;
mod camera;
mod draw2d;
mod effect_pass;
mod gpu;
mod hot_shader;
mod input;
mod mesh;
mod mesh_pass;
mod orbit_camera;
mod post_process;
mod render_graph;
mod texture;

pub use app::{AppConfig, Frame, SetupContext, run, run_with_config};
pub use assets::{Assets, FontAtlas, FontId};
pub use camera::Camera;
pub use draw2d::{Color, Draw2d, PanelBuilder, Rect};
pub use effect_pass::EffectPass;
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
pub use texture::Texture;

// Re-export glam math types for convenience
pub use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

// Re-export commonly used winit types for convenience
pub use winit::event::MouseButton;
pub use winit::keyboard::KeyCode;
