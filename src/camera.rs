/// A simple camera for 3D scenes.
///
/// Provides position, orientation, and field of view.
/// Used by `EffectPass` when `.with_camera()` is called.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32, // radians
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 5.0],
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov: std::f32::consts::FRAC_PI_2, // 90 degrees
        }
    }
}

impl Camera {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn at(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }

    pub fn looking_at(mut self, target_x: f32, target_y: f32, target_z: f32) -> Self {
        let forward = normalize([
            target_x - self.position[0],
            target_y - self.position[1],
            target_z - self.position[2],
        ]);
        self.forward = forward;
        self
    }

    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov = fov_degrees.to_radians();
        self
    }

    /// Compute the right vector from forward and up.
    pub fn right(&self) -> [f32; 3] {
        normalize(cross(self.forward, self.up))
    }

    /// Recompute up to be orthogonal to forward and right.
    pub fn orthogonal_up(&self) -> [f32; 3] {
        let right = self.right();
        normalize(cross(right, self.forward))
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 0.0]
    }
}
