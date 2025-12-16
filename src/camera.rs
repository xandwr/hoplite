use glam::{Mat4, Vec3};

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
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            forward: Vec3::NEG_Z,
            up: Vec3::Y,
            fov: std::f32::consts::FRAC_PI_2, // 90 degrees
            near: 0.1,
            far: 1000.0,
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

    /// Compute the view matrix for this camera.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, self.up)
    }

    /// Compute the projection matrix for this camera.
    pub fn projection_matrix(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, near, far)
    }
}
