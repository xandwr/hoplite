//! Scene management for Hoplite.
//!
//! This module provides a scene-based architecture for organizing independent
//! rendering contexts, each with its own camera, render graph, and state.
//!
//! # Overview
//!
//! Scenes are self-contained units of game content. Each scene can have:
//! - Its own camera (position, orientation, FOV)
//! - Its own render pipeline (different post-processing effects)
//! - Its own frame logic (update closure)
//! - Optional lifecycle hooks (`on_enter`, `on_exit`)
//!
//! # Example
//!
//! ```ignore
//! use hoplite::*;
//!
//! fn main() {
//!     run_with_scenes(|ctx| {
//!         ctx.default_font(16.0);
//!         let cube = ctx.mesh_cube();
//!
//!         // External ship view
//!         ctx.scene("external", |scene| {
//!             scene.hot_effect_world("shaders/space.wgsl");
//!             scene.enable_mesh_rendering();
//!
//!             let mut orbit = OrbitCamera::new();
//!
//!             move |frame| {
//!                 orbit.update(frame.input, frame.dt);
//!                 frame.set_camera(orbit.camera());
//!                 frame.mesh(cube).draw();
//!
//!                 if frame.input.key_pressed(KeyCode::Enter) {
//!                     frame.switch_to("cockpit");
//!                 }
//!             }
//!         });
//!
//!         // Cockpit view
//!         ctx.scene("cockpit", |scene| {
//!             scene.background_color(Color::BLACK);
//!             scene.enable_mesh_rendering();
//!
//!             let mut freelook = FreelookCamera::new();
//!
//!             move |frame| {
//!                 freelook.update(frame.input, frame.dt);
//!                 frame.set_camera(freelook.camera());
//!
//!                 if frame.input.key_pressed(KeyCode::Escape) {
//!                     frame.switch_to_with("external", Transition::fade_to_black(0.5));
//!                 }
//!             }
//!         });
//!
//!         ctx.start_scene("external");
//!     });
//! }
//! ```

mod manager;
pub mod scene;
mod setup;
mod transition;
mod transition_pass;

pub use manager::SceneManager;
pub use scene::{Scene, SceneBuilder, SceneId};
pub use setup::SceneSetupContext;
pub use transition::{ActiveTransition, Easing, Transition, TransitionKind};
pub use transition_pass::TransitionPass;
