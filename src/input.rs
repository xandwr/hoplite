//! Input handling for keyboard and mouse events.
//!
//! This module provides the [`Input`] struct, which tracks the state of keyboard keys
//! and mouse buttons across frames. It distinguishes between three states for each input:
//!
//! - **Down**: The input is currently held (persists across frames)
//! - **Pressed**: The input was just pressed this frame (single-frame event)
//! - **Released**: The input was just released this frame (single-frame event)
//!
//! # Frame Lifecycle
//!
//! The input system follows a specific frame lifecycle:
//!
//! 1. Call [`Input::begin_frame`] at the start of each frame to clear per-frame state
//! 2. Process window events via [`Input::handle_event`] during the event loop
//! 3. Query input state using the various accessor methods during update/render
//!
//! # Example
//!
//! ```ignore
//! let mut input = Input::new();
//!
//! // In your event loop
//! input.begin_frame();
//! for event in events {
//!     input.handle_event(&event);
//! }
//!
//! // In your update logic
//! if input.key_pressed(KeyCode::Space) {
//!     // Jump (only triggers once per press)
//! }
//! if input.key_down(KeyCode::KeyW) {
//!     // Move forward (triggers every frame while held)
//! }
//! ```

use std::collections::HashSet;

use glam::Vec2;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Tracks input state for keyboard and mouse across frames.
///
/// This struct maintains three categories of state for both keyboard keys and mouse buttons:
/// - **Down state**: Which inputs are currently held down (persists until released)
/// - **Pressed state**: Which inputs were pressed this frame (cleared each frame)
/// - **Released state**: Which inputs were released this frame (cleared each frame)
///
/// Additionally tracks mouse position, movement delta, and scroll wheel input.
///
/// # Thread Safety
///
/// This struct is not thread-safe and should be accessed from a single thread,
/// typically the main thread that owns the window.
pub struct Input {
    /// Keys currently held down.
    keys_down: HashSet<KeyCode>,
    /// Keys pressed this frame (cleared at the start of each frame).
    keys_pressed: HashSet<KeyCode>,
    /// Keys released this frame (cleared at the start of each frame).
    keys_released: HashSet<KeyCode>,
    /// Mouse buttons currently held down.
    mouse_buttons_down: HashSet<MouseButton>,
    /// Mouse buttons pressed this frame (cleared at the start of each frame).
    mouse_buttons_pressed: HashSet<MouseButton>,
    /// Mouse buttons released this frame (cleared at the start of each frame).
    mouse_buttons_released: HashSet<MouseButton>,
    /// Current mouse position in window coordinates (pixels from top-left).
    mouse_position: Vec2,
    /// Mouse movement delta accumulated this frame.
    mouse_delta: Vec2,
    /// Scroll wheel delta accumulated this frame, normalized to "lines".
    scroll_delta: Vec2,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            keys_down: HashSet::new(),
            keys_pressed: HashSet::new(),
            keys_released: HashSet::new(),
            mouse_buttons_down: HashSet::new(),
            mouse_buttons_pressed: HashSet::new(),
            mouse_buttons_released: HashSet::new(),
            mouse_position: Vec2::ZERO,
            mouse_delta: Vec2::ZERO,
            scroll_delta: Vec2::ZERO,
        }
    }
}

impl Input {
    /// Creates a new input tracker with all state cleared.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resets per-frame input state.
    ///
    /// This must be called at the start of each frame, before processing any window events.
    /// It clears the "pressed" and "released" states for keys and mouse buttons,
    /// as well as the mouse movement and scroll deltas.
    ///
    /// The "down" states are preserved, as they represent inputs that are still held.
    pub fn begin_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse_buttons_pressed.clear();
        self.mouse_buttons_released.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// Processes a window event and updates input state accordingly.
    ///
    /// This method handles the following event types:
    /// - [`WindowEvent::KeyboardInput`]: Updates key down/pressed/released state
    /// - [`WindowEvent::MouseInput`]: Updates mouse button down/pressed/released state
    /// - [`WindowEvent::CursorMoved`]: Updates mouse position and accumulates movement delta
    /// - [`WindowEvent::MouseWheel`]: Accumulates scroll delta (normalized to lines)
    ///
    /// Other event types are ignored.
    ///
    /// # Key Press Detection
    ///
    /// Keys are only marked as "pressed" on the first event when transitioning from
    /// released to pressed. Held keys that generate repeat events will not trigger
    /// additional "pressed" states.
    pub fn handle_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if !self.keys_down.contains(&key) {
                                self.keys_pressed.insert(key);
                            }
                            self.keys_down.insert(key);
                        }
                        ElementState::Released => {
                            self.keys_down.remove(&key);
                            self.keys_released.insert(key);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => match state {
                ElementState::Pressed => {
                    if !self.mouse_buttons_down.contains(button) {
                        self.mouse_buttons_pressed.insert(*button);
                    }
                    self.mouse_buttons_down.insert(*button);
                }
                ElementState::Released => {
                    self.mouse_buttons_down.remove(button);
                    self.mouse_buttons_released.insert(*button);
                }
            },
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                self.mouse_delta += new_pos - self.mouse_position;
                self.mouse_position = new_pos;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    winit::event::MouseScrollDelta::LineDelta(x, y) => Vec2::new(*x, *y),
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        Vec2::new(pos.x as f32, pos.y as f32) / 120.0
                    }
                };
                self.scroll_delta += d;
            }
            _ => {}
        }
    }

    /// Returns `true` if the key is currently held down.
    ///
    /// This returns `true` for every frame that the key remains pressed,
    /// making it suitable for continuous actions like movement.
    #[inline]
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    /// Returns `true` if the key was pressed this frame.
    ///
    /// This only returns `true` for the single frame when the key transitions
    /// from released to pressed, making it suitable for discrete actions like jumping.
    #[inline]
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Returns `true` if the key was released this frame.
    ///
    /// This only returns `true` for the single frame when the key transitions
    /// from pressed to released.
    #[inline]
    pub fn key_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }

    /// Returns `true` if the mouse button is currently held down.
    ///
    /// This returns `true` for every frame that the button remains pressed.
    #[inline]
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&button)
    }

    /// Returns `true` if the mouse button was pressed this frame.
    ///
    /// This only returns `true` for the single frame when the button transitions
    /// from released to pressed.
    #[inline]
    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.contains(&button)
    }

    /// Returns `true` if the mouse button was released this frame.
    ///
    /// This only returns `true` for the single frame when the button transitions
    /// from pressed to released.
    #[inline]
    pub fn mouse_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons_released.contains(&button)
    }

    /// Returns the current mouse position in window coordinates.
    ///
    /// The position is measured in pixels from the top-left corner of the window's
    /// client area. The value is updated whenever a [`WindowEvent::CursorMoved`] event
    /// is processed.
    #[inline]
    pub fn mouse_position(&self) -> Vec2 {
        self.mouse_position
    }

    /// Returns the accumulated mouse movement delta for this frame.
    ///
    /// The delta represents the total mouse movement since the last call to
    /// [`begin_frame`](Self::begin_frame). Positive X is rightward, positive Y is downward.
    #[inline]
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// Returns the accumulated scroll wheel delta for this frame.
    ///
    /// The delta is normalized to "lines" (typically one notch of a scroll wheel).
    /// For pixel-based scroll events (e.g., touchpad), the value is divided by 120
    /// to approximate line-based scrolling.
    ///
    /// - `x`: Horizontal scroll (positive = right)
    /// - `y`: Vertical scroll (positive = down, though this varies by platform)
    #[inline]
    pub fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }

    /// Handles raw mouse motion from device events.
    ///
    /// This is called for `DeviceEvent::MouseMotion` events, which provide
    /// raw mouse movement independent of cursor position. This is essential
    /// for FPS-style camera controls when the cursor is locked (grabbed),
    /// as `CursorMoved` events stop reporting movement when the cursor
    /// hits window boundaries.
    ///
    /// # Arguments
    ///
    /// * `dx` - Horizontal movement (positive = right)
    /// * `dy` - Vertical movement (positive = down)
    pub fn handle_raw_mouse_motion(&mut self, dx: f32, dy: f32) {
        self.mouse_delta += Vec2::new(dx, dy);
    }
}
