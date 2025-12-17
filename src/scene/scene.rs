//! Scene definition and identifier types.

use crate::Frame;
use crate::camera::Camera;
use crate::render_graph::{MeshQueue, RenderGraph};
use std::cell::RefCell;
use std::rc::Rc;

/// Unique identifier for a scene.
///
/// Scene IDs are strings that uniquely identify scenes within a [`SceneManager`](super::SceneManager).
/// They're used when switching between scenes via [`Frame::switch_to`] and [`Frame::switch_to_with`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SceneId(pub(crate) String);

impl SceneId {
    /// Create a new scene ID from a string.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Get the scene ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SceneId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for SceneId {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for SceneId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// A scene with its own camera, render graph, and frame logic.
///
/// Scenes are created via [`SetupContext::scene`] and managed by the [`SceneManager`](super::SceneManager).
/// Each scene maintains independent state that persists across scene switches (unless reset in `on_enter`).
pub struct Scene {
    /// Unique identifier for this scene.
    pub id: SceneId,

    /// Scene's camera (independent per scene).
    pub camera: Camera,

    /// Scene's render graph (can have different post-processing).
    pub render_graph: Option<RenderGraph>,

    /// Reference to the shared mesh queue (shared storage, per-scene draw queue).
    pub mesh_queue: Rc<RefCell<MeshQueue>>,

    /// The scene's frame update closure.
    pub(crate) frame_fn: Box<dyn FnMut(&mut Frame)>,

    /// Optional callback when entering this scene.
    pub(crate) on_enter: Option<Box<dyn FnMut()>>,

    /// Optional callback when exiting this scene.
    pub(crate) on_exit: Option<Box<dyn FnMut()>>,
}

impl Scene {
    /// Create a new scene with the given ID.
    pub(crate) fn new(
        id: SceneId,
        render_graph: Option<RenderGraph>,
        mesh_queue: Rc<RefCell<MeshQueue>>,
        frame_fn: Box<dyn FnMut(&mut Frame)>,
    ) -> Self {
        Self {
            id,
            camera: Camera::new(),
            render_graph,
            mesh_queue,
            frame_fn,
            on_enter: None,
            on_exit: None,
        }
    }

    /// Call the scene's enter callback if set.
    pub(crate) fn enter(&mut self) {
        if let Some(ref mut callback) = self.on_enter {
            callback();
        }
    }

    /// Call the scene's exit callback if set.
    pub(crate) fn exit(&mut self) {
        if let Some(ref mut callback) = self.on_exit {
            callback();
        }
    }
}

/// Builder for configuring scene lifecycle hooks.
///
/// Returned by [`SetupContext::scene`] to allow chaining `on_enter` and `on_exit` hooks.
///
/// # Example
///
/// ```ignore
/// ctx.scene("game", |scene| {
///     scene.enable_mesh_rendering();
///     let mut score = 0;
///
///     move |frame| {
///         frame.text(10.0, 10.0, &format!("Score: {}", score));
///     }
/// })
/// .on_enter(|| println!("Starting game!"))
/// .on_exit(|| println!("Leaving game..."));
/// ```
pub struct SceneBuilder<'a> {
    pub(crate) scene_id: SceneId,
    pub(crate) manager: &'a mut super::SceneManager,
}

impl<'a> SceneBuilder<'a> {
    /// Set a callback to run when entering this scene.
    ///
    /// Called after scene becomes active, before the first frame update.
    pub fn on_enter<F: FnMut() + 'static>(self, callback: F) -> Self {
        if let Some(scene) = self.manager.scenes.get_mut(&self.scene_id.0) {
            scene.on_enter = Some(Box::new(callback));
        }
        self
    }

    /// Set a callback to run when exiting this scene.
    ///
    /// Called before transition begins, while scene is still active.
    pub fn on_exit<F: FnMut() + 'static>(self, callback: F) -> Self {
        if let Some(scene) = self.manager.scenes.get_mut(&self.scene_id.0) {
            scene.on_exit = Some(Box::new(callback));
        }
        self
    }
}
