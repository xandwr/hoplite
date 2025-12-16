use crate::camera::Camera;
use crate::input::Input;
use winit::event::MouseButton;

/// Controls how the orbit camera moves.
#[derive(Clone, Copy, Debug)]
pub enum OrbitMode {
    /// User controls camera with mouse drag and scroll wheel.
    Interactive,
    /// Camera auto-rotates around the target, ignoring input.
    AutoRotate {
        /// Rotation speed in radians per second (positive = counterclockwise from above).
        speed: f32,
    },
}

impl Default for OrbitMode {
    fn default() -> Self {
        Self::Interactive
    }
}

/// A camera controller that orbits around a target point.
///
/// # Example
/// ```ignore
/// let mut orbit = OrbitCamera::new()
///     .target(0.0, 0.0, 0.0)
///     .distance(5.0)
///     .mode(OrbitMode::Interactive);
///
/// // In frame loop:
/// orbit.update(frame.input, frame.dt);
/// *frame.camera = orbit.camera();
/// ```
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    /// Point the camera orbits around.
    pub target: [f32; 3],
    /// Distance from target.
    pub distance: f32,
    /// Horizontal angle in radians (yaw).
    pub azimuth: f32,
    /// Vertical angle in radians (pitch), clamped to avoid gimbal lock.
    pub elevation: f32,
    /// Field of view in radians.
    pub fov: f32,
    /// Control mode.
    pub mode: OrbitMode,
    /// Mouse sensitivity for interactive mode.
    pub sensitivity: f32,
    /// Scroll zoom sensitivity.
    pub zoom_sensitivity: f32,
    /// Minimum distance from target.
    pub min_distance: f32,
    /// Maximum distance from target.
    pub max_distance: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: [0.0, 0.0, 0.0],
            distance: 5.0,
            azimuth: 0.0,
            elevation: 0.3,
            fov: std::f32::consts::FRAC_PI_2,
            mode: OrbitMode::Interactive,
            sensitivity: 0.005,
            zoom_sensitivity: 0.5,
            min_distance: 0.5,
            max_distance: 100.0,
        }
    }
}

impl OrbitCamera {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target point to orbit around.
    pub fn target(mut self, x: f32, y: f32, z: f32) -> Self {
        self.target = [x, y, z];
        self
    }

    /// Set the distance from target.
    pub fn distance(mut self, distance: f32) -> Self {
        self.distance = distance.clamp(self.min_distance, self.max_distance);
        self
    }

    /// Set the control mode.
    pub fn mode(mut self, mode: OrbitMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the field of view in degrees.
    pub fn fov(mut self, fov_degrees: f32) -> Self {
        self.fov = fov_degrees.to_radians();
        self
    }

    /// Set the initial azimuth (horizontal angle) in radians.
    pub fn azimuth(mut self, azimuth: f32) -> Self {
        self.azimuth = azimuth;
        self
    }

    /// Set the initial elevation (vertical angle) in radians.
    pub fn elevation(mut self, elevation: f32) -> Self {
        self.elevation = elevation.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
        self
    }

    /// Set mouse sensitivity for interactive mode.
    pub fn sensitivity(mut self, sensitivity: f32) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set scroll zoom sensitivity.
    pub fn zoom_sensitivity(mut self, sensitivity: f32) -> Self {
        self.zoom_sensitivity = sensitivity;
        self
    }

    /// Set distance limits.
    pub fn distance_limits(mut self, min: f32, max: f32) -> Self {
        self.min_distance = min;
        self.max_distance = max;
        self.distance = self.distance.clamp(min, max);
        self
    }

    /// Update the camera based on input and delta time.
    pub fn update(&mut self, input: &Input, dt: f32) {
        match self.mode {
            OrbitMode::Interactive => {
                // Rotate when left mouse button is held
                if input.mouse_down(MouseButton::Left) {
                    let delta = input.mouse_delta();
                    self.azimuth -= delta[0] * self.sensitivity;
                    self.elevation += delta[1] * self.sensitivity;

                    // Clamp elevation to avoid gimbal lock
                    self.elevation = self.elevation.clamp(
                        -std::f32::consts::FRAC_PI_2 + 0.01,
                        std::f32::consts::FRAC_PI_2 - 0.01,
                    );
                }

                // Zoom with scroll wheel
                let scroll = input.scroll_delta();
                if scroll[1].abs() > 0.0 {
                    self.distance -= scroll[1] * self.zoom_sensitivity;
                    self.distance = self.distance.clamp(self.min_distance, self.max_distance);
                }
            }
            OrbitMode::AutoRotate { speed } => {
                self.azimuth += speed * dt;
            }
        }
    }

    /// Get the current camera state.
    pub fn camera(&self) -> Camera {
        // Spherical to Cartesian conversion
        let x = self.distance * self.elevation.cos() * self.azimuth.sin();
        let y = self.distance * self.elevation.sin();
        let z = self.distance * self.elevation.cos() * self.azimuth.cos();

        Camera {
            position: [self.target[0] + x, self.target[1] + y, self.target[2] + z],
            forward: normalize([
                self.target[0] - (self.target[0] + x),
                self.target[1] - (self.target[1] + y),
                self.target[2] - (self.target[2] + z),
            ]),
            up: [0.0, 1.0, 0.0],
            fov: self.fov,
        }
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, -1.0]
    }
}
