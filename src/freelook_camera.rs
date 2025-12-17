//! A first-person freelook camera controller with seated and unseated modes.
//!
//! This module provides [`FreelookCamera`], a camera controller designed for
//! first-person perspectives. It supports two modes:
//!
//! - **Unseated**: Full freedom of movement and looking via keyboard (WASD) and mouse
//! - **Seated**: Position is locked to a fixed point, with optional view angle constraints
//!
//! # Example
//!
//! ```ignore
//! use hoplite::{FreelookCamera, FreelookMode, Vec3};
//!
//! // Unseated mode - full movement
//! let mut camera = FreelookCamera::new()
//!     .position([0.0, 1.8, 0.0])
//!     .mode(FreelookMode::Unseated);
//!
//! // Seated mode - locked to chair, constrained view
//! let mut seated_camera = FreelookCamera::new()
//!     .mode(FreelookMode::seated([5.0, 1.0, 3.0])
//!         .yaw_range(-45.0_f32.to_radians(), 45.0_f32.to_radians())
//!         .pitch_range(-30.0_f32.to_radians(), 30.0_f32.to_radians())
//!         .facing(Vec3::new(0.0, 0.0, -1.0)));
//!
//! // In frame loop:
//! camera.update(frame.input, frame.dt);
//! *frame.camera = camera.camera();
//! ```

use glam::Vec3;
use winit::keyboard::KeyCode;

use crate::camera::Camera;
use crate::input::Input;

/// Configuration for seated mode with view constraints.
#[derive(Clone, Debug)]
pub struct SeatedConfig {
    /// Fixed world position for the camera when seated.
    pub position: Vec3,
    /// Base yaw (horizontal angle) the seat faces, in radians.
    pub base_yaw: f32,
    /// Base pitch (vertical angle) the seat faces, in radians.
    pub base_pitch: f32,
    /// Minimum yaw offset from base (negative = left), in radians.
    pub min_yaw_offset: f32,
    /// Maximum yaw offset from base (positive = right), in radians.
    pub max_yaw_offset: f32,
    /// Minimum pitch offset from base (negative = down), in radians.
    pub min_pitch_offset: f32,
    /// Maximum pitch offset from base (positive = up), in radians.
    pub max_pitch_offset: f32,
}

impl SeatedConfig {
    /// Create a new seated configuration at the given position.
    ///
    /// By default, looks forward (-Z) with full 360° yaw and ±89° pitch.
    pub fn new(position: impl Into<Vec3>) -> Self {
        Self {
            position: position.into(),
            base_yaw: 0.0,
            base_pitch: 0.0,
            min_yaw_offset: -std::f32::consts::PI,
            max_yaw_offset: std::f32::consts::PI,
            min_pitch_offset: -std::f32::consts::FRAC_PI_2 + 0.01,
            max_pitch_offset: std::f32::consts::FRAC_PI_2 - 0.01,
        }
    }

    /// Set the yaw range as offsets from the base direction (in radians).
    ///
    /// - `min`: Maximum turn to the left (negative value)
    /// - `max`: Maximum turn to the right (positive value)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allow looking 45 degrees left or right
    /// config.yaw_range(-45.0_f32.to_radians(), 45.0_f32.to_radians())
    /// ```
    pub fn yaw_range(mut self, min: f32, max: f32) -> Self {
        self.min_yaw_offset = min;
        self.max_yaw_offset = max;
        self
    }

    /// Set the pitch range as offsets from the base direction (in radians).
    ///
    /// - `min`: Maximum look down (negative value)
    /// - `max`: Maximum look up (positive value)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Allow looking 30 degrees up or down
    /// config.pitch_range(-30.0_f32.to_radians(), 30.0_f32.to_radians())
    /// ```
    pub fn pitch_range(mut self, min: f32, max: f32) -> Self {
        self.min_pitch_offset = min;
        self.max_pitch_offset = max;
        self
    }

    /// Set the base direction the seat faces using a direction vector.
    ///
    /// The camera's yaw/pitch constraints will be relative to this direction.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Seat faces the positive X direction
    /// config.facing(Vec3::X)
    /// ```
    pub fn facing(mut self, direction: impl Into<Vec3>) -> Self {
        let dir = direction.into().normalize_or_zero();
        // Match the convention used in forward_direction(): yaw=0 means looking toward -Z
        self.base_yaw = dir.x.atan2(-dir.z);
        self.base_pitch = dir.y.asin();
        self
    }

    /// Set the base direction using yaw and pitch angles directly (in radians).
    pub fn facing_angles(mut self, yaw: f32, pitch: f32) -> Self {
        self.base_yaw = yaw;
        self.base_pitch = pitch;
        self
    }
}

/// Controls how the freelook camera operates.
#[derive(Clone, Debug)]
pub enum FreelookMode {
    /// Full movement and looking enabled.
    /// Camera position can be moved with WASD/Space/Shift.
    Unseated,
    /// Position is locked; only looking is allowed (within optional constraints).
    Seated(SeatedConfig),
}

impl Default for FreelookMode {
    fn default() -> Self {
        Self::Unseated
    }
}

impl FreelookMode {
    /// Create a seated mode configuration at the given world position.
    ///
    /// Returns a [`SeatedConfig`] that can be further customized with view constraints.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mode = FreelookMode::seated([5.0, 1.0, 3.0])
    ///     .yaw_range(-45.0_f32.to_radians(), 45.0_f32.to_radians())
    ///     .facing(Vec3::NEG_Z);
    /// ```
    pub fn seated(position: impl Into<Vec3>) -> SeatedConfig {
        SeatedConfig::new(position)
    }
}

/// A first-person camera controller with seated and unseated modes.
///
/// # Modes
///
/// - **Unseated**: Full WASD movement + mouse look. Standard FPS controls.
/// - **Seated**: Locked to a world position, optionally constrained view angles.
///   Perfect for sitting in chairs, vehicles, or cutscenes.
///
/// # Controls (Unseated Mode)
///
/// - **W/S**: Move forward/backward
/// - **A/D**: Strafe left/right
/// - **Space**: Move up
/// - **Left Shift**: Move down
/// - **Mouse**: Look around
///
/// # Example
///
/// ```ignore
/// let mut camera = FreelookCamera::new()
///     .position([0.0, 1.8, 0.0])
///     .speed(5.0);
///
/// // In frame loop:
/// camera.update(frame.input, frame.dt);
/// *frame.camera = camera.camera();
///
/// // Sit the player down
/// camera.seat(FreelookMode::seated([10.0, 1.0, 5.0])
///     .yaw_range(-60.0_f32.to_radians(), 60.0_f32.to_radians())
///     .facing(Vec3::NEG_X));
///
/// // Stand back up
/// camera.unseat();
/// ```
#[derive(Clone, Debug)]
pub struct FreelookCamera {
    /// Current camera position (only used in Unseated mode).
    pub position: Vec3,
    /// Horizontal angle in radians (yaw). 0 = looking toward -Z.
    pub yaw: f32,
    /// Vertical angle in radians (pitch). 0 = horizontal, positive = up.
    pub pitch: f32,
    /// Field of view in radians.
    pub fov: f32,
    /// Current mode (Unseated or Seated).
    pub mode: FreelookMode,
    /// Mouse sensitivity for looking.
    pub sensitivity: f32,
    /// Movement speed in units per second (Unseated mode only).
    pub speed: f32,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl Default for FreelookCamera {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            fov: std::f32::consts::FRAC_PI_2,
            mode: FreelookMode::Unseated,
            sensitivity: 0.003,
            speed: 5.0,
            near: 0.1,
            far: 1000.0,
        }
    }
}

impl FreelookCamera {
    /// Create a new freelook camera with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the camera position.
    ///
    /// In Unseated mode, this is the starting position.
    /// In Seated mode, the position is overridden by the seat configuration.
    pub fn position(mut self, position: impl Into<Vec3>) -> Self {
        self.position = position.into();
        self
    }

    /// Set the control mode.
    pub fn mode(mut self, mode: FreelookMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the field of view in degrees.
    pub fn fov(mut self, fov_degrees: f32) -> Self {
        self.fov = fov_degrees.to_radians();
        self
    }

    /// Set the initial yaw (horizontal angle) in radians.
    pub fn yaw(mut self, yaw: f32) -> Self {
        self.yaw = yaw;
        self
    }

    /// Set the initial pitch (vertical angle) in radians.
    pub fn pitch(mut self, pitch: f32) -> Self {
        self.pitch = pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self
    }

    /// Set the initial look direction using a direction vector.
    pub fn looking_toward(mut self, direction: impl Into<Vec3>) -> Self {
        let dir = direction.into().normalize_or_zero();
        self.yaw = dir.x.atan2(-dir.z);
        self.pitch = dir.y.asin().clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self
    }

    /// Set mouse sensitivity.
    pub fn sensitivity(mut self, sensitivity: f32) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set movement speed (Unseated mode only).
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    /// Set near and far clipping planes.
    pub fn clip_planes(mut self, near: f32, far: f32) -> Self {
        self.near = near;
        self.far = far;
        self
    }

    /// Transition to seated mode with the given configuration.
    ///
    /// The camera will be locked to the seat position with optional view constraints.
    ///
    /// # Example
    ///
    /// ```ignore
    /// camera.seat(FreelookMode::seated([5.0, 1.0, 3.0])
    ///     .yaw_range(-45.0_f32.to_radians(), 45.0_f32.to_radians())
    ///     .facing(Vec3::NEG_Z));
    /// ```
    pub fn seat(&mut self, config: SeatedConfig) {
        // Snap to seat's base orientation
        self.yaw = config.base_yaw;
        self.pitch = config.base_pitch;
        self.mode = FreelookMode::Seated(config);
    }

    /// Transition to unseated mode.
    ///
    /// If currently seated, the camera's position will be set to the seat position
    /// so the player doesn't teleport.
    pub fn unseat(&mut self) {
        if let FreelookMode::Seated(config) = &self.mode {
            self.position = config.position;
        }
        self.mode = FreelookMode::Unseated;
    }

    /// Returns true if currently in seated mode.
    pub fn is_seated(&self) -> bool {
        matches!(self.mode, FreelookMode::Seated(_))
    }

    /// Returns true if currently in unseated mode.
    pub fn is_unseated(&self) -> bool {
        matches!(self.mode, FreelookMode::Unseated)
    }

    /// Get the current effective position (accounting for seated mode).
    pub fn effective_position(&self) -> Vec3 {
        match &self.mode {
            FreelookMode::Unseated => self.position,
            FreelookMode::Seated(config) => config.position,
        }
    }

    /// Compute the forward direction vector from current yaw and pitch.
    fn forward_direction(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            -self.yaw.cos() * self.pitch.cos(),
        )
        .normalize_or_zero()
    }

    /// Compute the right direction vector (for strafing).
    fn right_direction(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, self.yaw.sin()).normalize_or_zero()
    }

    /// Update the camera based on input and delta time.
    pub fn update(&mut self, input: &Input, dt: f32) {
        // Mouse look (works in both modes)
        let delta = input.mouse_delta();
        self.yaw += delta.x * self.sensitivity;
        self.pitch -= delta.y * self.sensitivity;

        // Apply constraints based on mode
        match &self.mode {
            FreelookMode::Unseated => {
                // Clamp pitch to avoid gimbal lock (no yaw constraint)
                self.pitch = self.pitch.clamp(
                    -std::f32::consts::FRAC_PI_2 + 0.01,
                    std::f32::consts::FRAC_PI_2 - 0.01,
                );

                // WASD movement
                let forward = self.forward_direction();
                let right = self.right_direction();

                let mut velocity = Vec3::ZERO;

                if input.key_down(KeyCode::KeyW) {
                    velocity += forward;
                }
                if input.key_down(KeyCode::KeyS) {
                    velocity -= forward;
                }
                if input.key_down(KeyCode::KeyA) {
                    velocity -= right;
                }
                if input.key_down(KeyCode::KeyD) {
                    velocity += right;
                }
                if input.key_down(KeyCode::Space) {
                    velocity += Vec3::Y;
                }
                if input.key_down(KeyCode::ShiftLeft) {
                    velocity -= Vec3::Y;
                }

                if velocity.length_squared() > 0.0 {
                    self.position += velocity.normalize() * self.speed * dt;
                }
            }
            FreelookMode::Seated(config) => {
                // Constrain yaw relative to base
                let yaw_offset = self.yaw - config.base_yaw;
                let clamped_yaw_offset =
                    yaw_offset.clamp(config.min_yaw_offset, config.max_yaw_offset);
                self.yaw = config.base_yaw + clamped_yaw_offset;

                // Constrain pitch relative to base
                let pitch_offset = self.pitch - config.base_pitch;
                let clamped_pitch_offset =
                    pitch_offset.clamp(config.min_pitch_offset, config.max_pitch_offset);
                self.pitch = config.base_pitch + clamped_pitch_offset;
            }
        }
    }

    /// Get the current camera state.
    pub fn camera(&self) -> Camera {
        let position = self.effective_position();
        let forward = self.forward_direction();

        Camera {
            position,
            forward,
            up: Vec3::Y,
            fov: self.fov,
            near: self.near,
            far: self.far,
        }
    }
}
