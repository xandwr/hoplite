//! Camera representation for 3D rendering.
//!
//! This module provides [`Camera`], a simple camera struct that stores position,
//! orientation, and projection parameters. It's used by world-space effect passes
//! and mesh rendering to provide view and projection matrices.
//!
//! # Builder Pattern
//!
//! Camera uses a builder pattern for easy construction:
//!
//! ```
//! use hoplite::Camera;
//!
//! let camera = Camera::new()
//!     .at([0.0, 2.0, 5.0])
//!     .looking_at([0.0, 0.0, 0.0])
//!     .with_fov(60.0);
//! ```
//!
//! # Coordinate System
//!
//! The camera uses a right-handed coordinate system:
//! - +X points right
//! - +Y points up
//! - -Z points into the screen (forward direction)

use glam::{Mat4, Vec3};

/// A 3D camera with position, orientation, and projection parameters.
///
/// The camera stores its position, forward direction, up vector, field of view,
/// and near/far clip planes. It provides methods to compute view and projection
/// matrices for rendering.
///
/// # Example
///
/// ```
/// use hoplite::Camera;
///
/// // Create a camera looking at the origin from above
/// let camera = Camera::new()
///     .at([5.0, 5.0, 5.0])
///     .looking_at([0.0, 0.0, 0.0])
///     .with_fov(45.0);
///
/// // Get matrices for rendering
/// let view = camera.view_matrix();
/// let proj = camera.projection_matrix(16.0 / 9.0, 0.1, 1000.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// Camera position in world space.
    pub position: Vec3,
    /// Normalized direction the camera is facing.
    pub forward: Vec3,
    /// World up vector (used to compute the camera's right and orthogonal up).
    pub up: Vec3,
    /// Vertical field of view in radians.
    pub fov: f32,
    /// Near clipping plane distance.
    pub near: f32,
    /// Far clipping plane distance.
    pub far: f32,
}

impl Default for Camera {
    /// Creates a default camera at (0, 0, 5) looking toward -Z with 90° FOV.
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
    /// Create a new camera with default settings.
    ///
    /// Equivalent to [`Camera::default()`]. Use builder methods like
    /// [`at`](Self::at), [`looking_at`](Self::looking_at), and
    /// [`with_fov`](Self::with_fov) to configure.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the camera position.
    ///
    /// Accepts anything that converts to `Vec3`, including `[f32; 3]` arrays.
    ///
    /// # Example
    ///
    /// ```
    /// # use hoplite::Camera;
    /// let camera = Camera::new().at([0.0, 5.0, 10.0]);
    /// ```
    pub fn at(mut self, position: impl Into<Vec3>) -> Self {
        self.position = position.into();
        self
    }

    /// Point the camera at a target position.
    ///
    /// Computes the forward direction from the current position to the target.
    /// Call this after [`at`](Self::at) to ensure correct orientation.
    ///
    /// # Example
    ///
    /// ```
    /// # use hoplite::Camera;
    /// let camera = Camera::new()
    ///     .at([0.0, 5.0, 10.0])
    ///     .looking_at([0.0, 0.0, 0.0]);
    /// ```
    pub fn looking_at(mut self, target: impl Into<Vec3>) -> Self {
        let target = target.into();
        self.forward = (target - self.position).normalize_or_zero();
        self
    }

    /// Set the vertical field of view in degrees.
    ///
    /// Internally converted to radians. Common values: 45° (telephoto),
    /// 60° (standard), 90° (wide angle).
    ///
    /// # Example
    ///
    /// ```
    /// # use hoplite::Camera;
    /// let camera = Camera::new().with_fov(60.0);
    /// ```
    pub fn with_fov(mut self, fov_degrees: f32) -> Self {
        self.fov = fov_degrees.to_radians();
        self
    }

    /// Compute the right vector from forward and up.
    ///
    /// Returns a normalized vector pointing to the camera's right.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up).normalize_or_zero()
    }

    /// Compute an up vector orthogonal to the forward direction.
    ///
    /// Unlike [`up`](Self::up), this returns a vector that is guaranteed
    /// to be perpendicular to [`forward`](Self::forward), which is needed
    /// for correct view matrix computation.
    pub fn orthogonal_up(&self) -> Vec3 {
        self.right().cross(self.forward).normalize_or_zero()
    }

    /// Compute the view matrix for this camera.
    ///
    /// The view matrix transforms world coordinates to camera (view) space.
    /// Uses right-handed coordinates.
    pub fn view_matrix(&self) -> Mat4 {
        Mat4::look_at_rh(self.position, self.position + self.forward, self.up)
    }

    /// Compute a perspective projection matrix.
    ///
    /// Uses the camera's FOV with the provided aspect ratio and clip planes.
    /// Uses right-handed coordinates with depth range [0, 1].
    ///
    /// # Parameters
    ///
    /// - `aspect`: Width divided by height (e.g., 16.0/9.0)
    /// - `near`: Near clipping plane distance
    /// - `far`: Far clipping plane distance
    pub fn projection_matrix(&self, aspect: f32, near: f32, far: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov, aspect, near, far)
    }
}
