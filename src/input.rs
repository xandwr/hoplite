use std::collections::HashSet;

use glam::Vec2;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};

/// Tracks input state for keyboard and mouse.
pub struct Input {
    keys_down: HashSet<KeyCode>,
    keys_pressed: HashSet<KeyCode>,
    keys_released: HashSet<KeyCode>,
    mouse_buttons_down: HashSet<MouseButton>,
    mouse_buttons_pressed: HashSet<MouseButton>,
    mouse_buttons_released: HashSet<MouseButton>,
    mouse_position: Vec2,
    mouse_delta: Vec2,
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Call at the start of each frame to reset per-frame state.
    pub fn begin_frame(&mut self) {
        self.keys_pressed.clear();
        self.keys_released.clear();
        self.mouse_buttons_pressed.clear();
        self.mouse_buttons_released.clear();
        self.mouse_delta = Vec2::ZERO;
        self.scroll_delta = Vec2::ZERO;
    }

    /// Process a window event and update input state.
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

    /// Returns true if the key is currently held down.
    pub fn key_down(&self, key: KeyCode) -> bool {
        self.keys_down.contains(&key)
    }

    /// Returns true if the key was pressed this frame.
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Returns true if the key was released this frame.
    pub fn key_released(&self, key: KeyCode) -> bool {
        self.keys_released.contains(&key)
    }

    /// Returns true if the mouse button is currently held down.
    pub fn mouse_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons_down.contains(&button)
    }

    /// Returns true if the mouse button was pressed this frame.
    pub fn mouse_pressed(&self, button: MouseButton) -> bool {
        self.mouse_buttons_pressed.contains(&button)
    }

    /// Returns true if the mouse button was released this frame.
    pub fn mouse_released(&self, button: MouseButton) -> bool {
        self.mouse_buttons_released.contains(&button)
    }

    /// Current mouse position in window coordinates.
    pub fn mouse_position(&self) -> Vec2 {
        self.mouse_position
    }

    /// Mouse movement delta this frame.
    pub fn mouse_delta(&self) -> Vec2 {
        self.mouse_delta
    }

    /// Scroll wheel delta this frame (in "lines").
    pub fn scroll_delta(&self) -> Vec2 {
        self.scroll_delta
    }
}
