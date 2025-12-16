use glam::Vec3;

/// A simple camera for 3D scenes.
///
/// Provides position, orientation, and field of view.
/// Used by `EffectPass` when `.with_camera()` is called.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub up: Vec3,
    pub fov: f32, // radians
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_2, // 90 degrees
        }
    }
}

impl Camera {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn at(mut self, position: impl Into<Vec3>) -> Self {
        self.position = position.into();
        self
    }

    pub fn looking_at(mut self, target: impl Into<Vec3>) -> Self {
        let target = target.into();
        self.forward = (target - self.position).normalize_or_zero();
        self
    }

    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov = fov_degrees.to_radians();
        self
    }

    /// Compute the right vector from forward and up.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize_or_zero()
    }

    /// Recompute up to be orthogonal to forward and right.
    pub fn orthogonal_up(&self) -> Vec3 {
        self.right().cross(self.forward).normalize_or_zero()
    }
}
