//! Scene manager for storing and switching between scenes.

use super::scene::{Scene, SceneBuilder, SceneId};
use super::transition::{ActiveTransition, Transition, TransitionPhase};
use super::transition_pass::TransitionPass;
use crate::Assets;
use crate::camera::Camera;
use crate::draw2d::Draw2d;
use crate::gpu::GpuContext;
use crate::render_graph::{MeshQueue, RenderTarget};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Manages multiple scenes and transitions between them.
///
/// The scene manager is responsible for:
/// - Storing registered scenes
/// - Tracking the active scene
/// - Handling scene transitions with various effects
/// - Coordinating rendering during transitions
///
/// # Example
///
/// Scene management is typically done through the `run_with_scenes` API,
/// but the manager can also be used directly for advanced use cases.
pub struct SceneManager {
    /// All registered scenes by name.
    pub(crate) scenes: HashMap<String, Scene>,

    /// Name of the currently active scene.
    active_scene: Option<String>,

    /// Active transition state (if any).
    transition: Option<ActiveTransition>,

    /// GPU pass for rendering transitions.
    transition_pass: Option<TransitionPass>,

    /// Render target for capturing scene output during crossfade.
    crossfade_capture: Option<RenderTarget>,

    /// Second render target for the other scene during crossfade.
    crossfade_capture_2: Option<RenderTarget>,

    /// Queued scene switch (processed at start of next frame).
    pending_switch: Option<(String, Transition)>,
}

impl SceneManager {
    /// Create a new empty scene manager.
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            active_scene: None,
            transition: None,
            transition_pass: None,
            crossfade_capture: None,
            crossfade_capture_2: None,
            pending_switch: None,
        }
    }

    /// Initialize GPU resources for transitions.
    ///
    /// This must be called after GPU context is available.
    pub fn init_gpu_resources(&mut self, gpu: &GpuContext) {
        self.transition_pass = Some(TransitionPass::new(gpu));
        self.crossfade_capture = Some(RenderTarget::new(gpu, "Scene Crossfade Capture 1"));
        self.crossfade_capture_2 = Some(RenderTarget::new(gpu, "Scene Crossfade Capture 2"));
    }

    /// Register a scene with the manager.
    pub(crate) fn register(&mut self, scene: Scene) {
        let name = scene.id.0.clone();
        self.scenes.insert(name, scene);
    }

    /// Set the initial active scene.
    ///
    /// This should be called during setup to specify which scene to start with.
    pub fn set_active(&mut self, scene_name: impl Into<String>) {
        let name = scene_name.into();
        if self.scenes.contains_key(&name) {
            // Call on_enter for the initial scene
            if let Some(scene) = self.scenes.get_mut(&name) {
                scene.enter();
            }
            self.active_scene = Some(name);
        } else {
            eprintln!("[scene] Warning: Scene '{}' not found", name);
        }
    }

    /// Get the name of the currently active scene.
    pub fn active_scene(&self) -> Option<&str> {
        self.active_scene.as_deref()
    }

    /// Get a mutable reference to the active scene.
    pub fn active_scene_mut(&mut self) -> Option<&mut Scene> {
        self.active_scene
            .as_ref()
            .and_then(|name| self.scenes.get_mut(name))
    }

    /// Get the camera from the active scene.
    pub fn active_camera(&self) -> Option<&Camera> {
        self.active_scene
            .as_ref()
            .and_then(|name| self.scenes.get(name))
            .map(|s| &s.camera)
    }

    /// Get a mutable reference to the active scene's camera.
    pub fn active_camera_mut(&mut self) -> Option<&mut Camera> {
        self.active_scene
            .as_ref()
            .and_then(|name| self.scenes.get_mut(name))
            .map(|s| &mut s.camera)
    }

    /// Request a scene switch (instant).
    pub fn switch_to(&mut self, scene_name: impl Into<String>) {
        self.switch_to_with(scene_name, Transition::instant());
    }

    /// Request a scene switch with a transition effect.
    ///
    /// The switch is queued and processed at the start of the next frame.
    pub fn switch_to_with(&mut self, scene_name: impl Into<String>, transition: Transition) {
        let name = scene_name.into();
        if !self.scenes.contains_key(&name) {
            eprintln!("[scene] Warning: Scene '{}' not found", name);
            return;
        }

        // Don't switch if we're already on this scene and there's no transition
        if self.active_scene.as_ref() == Some(&name) && self.transition.is_none() {
            return;
        }

        self.pending_switch = Some((name, transition));
    }

    /// Check if a transition is currently in progress.
    pub fn is_transitioning(&self) -> bool {
        self.transition.is_some()
    }

    /// Process pending scene switches and update transition state.
    ///
    /// Returns true if the active scene changed.
    pub fn update(&mut self, time: f32) -> bool {
        let mut scene_changed = false;

        // Process pending switch
        if let Some((target_name, transition)) = self.pending_switch.take() {
            let source_name = self.active_scene.clone().unwrap_or_default();

            // Call on_exit for current scene (before transition starts)
            if let Some(current) = self.active_scene_mut() {
                current.exit();
            }

            // Start the transition
            self.transition = Some(ActiveTransition::new(
                transition,
                source_name,
                target_name,
                time,
            ));
        }

        // Update active transition
        if let Some(ref mut active) = self.transition {
            let completed = active.update(time);

            // Check if we hit midpoint (time to swap scenes)
            if active.is_midpoint() && !active.is_crossfade() {
                // Swap to the new scene
                let target = active.target_scene.clone();
                self.active_scene = Some(target.clone());

                // Call on_enter for new scene
                if let Some(scene) = self.scenes.get_mut(&target) {
                    scene.enter();
                }

                scene_changed = true;
            }

            // For crossfade, swap immediately (both scenes render during transition)
            if active.is_crossfade() && self.active_scene.as_ref() != Some(&active.target_scene) {
                let target = active.target_scene.clone();
                let _source = active.source_scene.clone();

                // For crossfade, we keep both scenes and the old one renders to capture buffer
                self.active_scene = Some(target.clone());

                // Call on_enter for new scene
                if let Some(scene) = self.scenes.get_mut(&target) {
                    scene.enter();
                }

                scene_changed = true;
            }

            // Clean up completed transition
            if completed {
                self.transition = None;
            }
        }

        scene_changed
    }

    /// Execute the active scene's frame logic.
    pub fn run_frame(
        &mut self,
        gpu: &GpuContext,
        assets: &Assets,
        draw: &mut Draw2d,
        input: &crate::Input,
        world: &mut hecs::World,
        time: f32,
        dt: f32,
        mesh_queue: &Rc<RefCell<MeshQueue>>,
        window: &winit::window::Window,
        default_font: Option<crate::assets::FontId>,
    ) {
        if let Some(scene_name) = self.active_scene.clone() {
            if let Some(scene) = self.scenes.get_mut(&scene_name) {
                // Create frame context with scene's camera
                let mut frame = crate::Frame {
                    gpu,
                    assets,
                    draw,
                    camera: &mut scene.camera,
                    input,
                    world,
                    time,
                    dt,
                    default_font,
                    mesh_queue: Rc::clone(mesh_queue),
                    window,
                    scene_switch: None,
                };

                // Run scene's frame function
                (scene.frame_fn)(&mut frame);

                // Check if scene requested a switch
                if let Some((target, transition)) = frame.scene_switch.take() {
                    self.switch_to_with(target, transition);
                }
            }
        }
    }

    /// Render the scene(s) with transition effects if active.
    pub fn render(&mut self, gpu: &GpuContext, time: f32, draw_2d: &Draw2d, assets: &Assets) {
        // Ensure render targets are properly sized
        if let Some(ref mut target) = self.crossfade_capture {
            target.ensure_size(gpu, "Crossfade Capture 1");
        }
        if let Some(ref mut target) = self.crossfade_capture_2 {
            target.ensure_size(gpu, "Crossfade Capture 2");
        }

        // Get the output texture
        let output = match gpu.surface.get_current_texture() {
            Ok(output) => output,
            Err(e) => {
                eprintln!("[scene] Failed to get surface texture: {}", e);
                return;
            }
        };
        let screen_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Check if we're in a transition - extract data before mutable borrows
        let transition_info = self.transition.as_ref().map(|t| {
            (
                t.is_crossfade(),
                t.fade_color(),
                t.get_fade_alpha(),
                t.phase.clone(),
                t.source_scene.clone(),
                t.target_scene.clone(),
            )
        });

        match transition_info {
            Some((true, _, _, phase, source, target)) => {
                // Crossfade transition
                self.render_crossfade(
                    gpu,
                    time,
                    &screen_view,
                    draw_2d,
                    assets,
                    phase,
                    source,
                    target,
                );
            }
            Some((false, Some(color), (_, overlay_alpha), phase, source, target)) => {
                // Fade-to-color transition
                self.render_fade(
                    gpu,
                    time,
                    &screen_view,
                    draw_2d,
                    assets,
                    phase,
                    source,
                    target,
                    color,
                    overlay_alpha,
                );
            }
            _ => {
                // No transition or instant - render normally
                self.render_normal(gpu, time, &screen_view, draw_2d, assets);
            }
        }

        output.present();
    }

    /// Render without any transition effects.
    fn render_normal(
        &mut self,
        gpu: &GpuContext,
        time: f32,
        screen_view: &wgpu::TextureView,
        draw_2d: &Draw2d,
        assets: &Assets,
    ) {
        if let Some(scene_name) = &self.active_scene {
            if let Some(scene) = self.scenes.get_mut(scene_name) {
                if let Some(ref mut graph) = scene.render_graph {
                    graph.execute_with_ui(gpu, time, &scene.camera, |gpu, pass| {
                        draw_2d.render(gpu, pass, assets);
                    });
                } else {
                    // No render graph - just render 2D
                    render_2d_only(gpu, screen_view, draw_2d, assets);
                }
            }
        }
    }

    /// Render with fade-to-color transition.
    fn render_fade(
        &mut self,
        gpu: &GpuContext,
        time: f32,
        screen_view: &wgpu::TextureView,
        draw_2d: &Draw2d,
        assets: &Assets,
        phase: TransitionPhase,
        source_scene: String,
        target_scene: String,
        fade_color: crate::draw2d::Color,
        overlay_alpha: f32,
    ) {
        // Determine which scene to render based on transition phase
        let scene_name = match phase {
            TransitionPhase::FadingOut | TransitionPhase::Midpoint => source_scene,
            TransitionPhase::FadingIn => target_scene,
            TransitionPhase::Crossfading => target_scene, // Shouldn't happen
        };

        // Get render target for intermediate rendering
        let capture = self.crossfade_capture.as_ref().unwrap();

        // Render scene to capture target
        if let Some(scene) = self.scenes.get_mut(&scene_name) {
            if let Some(ref mut graph) = scene.render_graph {
                graph.execute_to_target(gpu, time, &scene.camera, &capture.view);
            } else {
                // Clear to black if no render graph
                let mut encoder =
                    gpu.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Fade Scene Clear"),
                        });
                {
                    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Clear Pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &capture.view,
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
                }
                gpu.queue.submit(std::iter::once(encoder.finish()));
            }
        }

        // Apply fade overlay and render to screen
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Fade Transition Encoder"),
            });

        if let Some(ref pass) = self.transition_pass {
            pass.render_fade(
                gpu,
                &mut encoder,
                screen_view,
                &capture.view,
                fade_color,
                overlay_alpha,
            );
        }

        // Render UI on top
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fade UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen_view,
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
            draw_2d.render(gpu, &mut ui_pass, assets);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render with crossfade transition.
    fn render_crossfade(
        &mut self,
        gpu: &GpuContext,
        time: f32,
        screen_view: &wgpu::TextureView,
        draw_2d: &Draw2d,
        assets: &Assets,
        _phase: TransitionPhase,
        source_scene: String,
        target_scene: String,
    ) {
        // Get the blend value from the transition
        let blend = self
            .transition
            .as_ref()
            .map(|t| t.get_crossfade_blend())
            .unwrap_or(0.0);

        let capture_1 = self.crossfade_capture.as_ref().unwrap();
        let capture_2 = self.crossfade_capture_2.as_ref().unwrap();

        // Render old scene to capture_1
        if let Some(scene) = self.scenes.get_mut(&source_scene) {
            if let Some(ref mut graph) = scene.render_graph {
                graph.execute_to_target(gpu, time, &scene.camera, &capture_1.view);
            }
        }

        // Render new scene to capture_2
        if let Some(scene) = self.scenes.get_mut(&target_scene) {
            if let Some(ref mut graph) = scene.render_graph {
                graph.execute_to_target(gpu, time, &scene.camera, &capture_2.view);
            }
        }

        // Blend and render to screen
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Crossfade Transition Encoder"),
            });

        if let Some(ref pass) = self.transition_pass {
            pass.render_crossfade(
                gpu,
                &mut encoder,
                screen_view,
                &capture_1.view,
                &capture_2.view,
                blend,
            );
        }

        // Render UI on top
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Crossfade UI Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: screen_view,
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
            draw_2d.render(gpu, &mut ui_pass, assets);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Create a scene builder for configuring lifecycle hooks.
    pub(crate) fn scene_builder(&mut self, scene_id: SceneId) -> SceneBuilder<'_> {
        SceneBuilder {
            scene_id,
            manager: self,
        }
    }
}

impl Default for SceneManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Fallback renderer for scenes without a render graph.
fn render_2d_only(gpu: &GpuContext, target: &wgpu::TextureView, draw_2d: &Draw2d, assets: &Assets) {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("2D Only Encoder"),
        });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("2D Only Pass"),
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

        draw_2d.render(gpu, &mut render_pass, assets);
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
}
