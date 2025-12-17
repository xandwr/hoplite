//! 3D picking and collision detection for mouse interaction with world objects.
//!
//! This module provides tools for detecting what objects the mouse is pointing at
//! in 3D space. It includes:
//!
//! - [`Ray`] — A 3D ray with origin and direction for raycasting
//! - [`Collider`] — Collision shapes (box, sphere) that can be attached to entities
//! - [`RayHit`] — Information about a ray-collider intersection
//!
//! # Example
//!
//! ```ignore
//! use hoplite::*;
//!
//! run(|ctx| {
//!     ctx.enable_mesh_rendering();
//!     let cube = ctx.mesh_cube();
//!
//!     // Spawn a clickable entity
//!     ctx.world.spawn((
//!         Transform::new().position(Vec3::new(0.0, 0.0, -5.0)),
//!         RenderMesh::new(cube, Color::RED),
//!         Collider::box_collider(Vec3::ONE),  // 1x1x1 box collider
//!     ));
//!
//!     move |frame| {
//!         // Check what the mouse is pointing at
//!         if let Some(hit) = frame.pick_collider() {
//!             frame.text(10.0, 10.0, &format!("Hovering entity: {:?}", hit.entity));
//!
//!             if frame.input.mouse_pressed(MouseButton::Left) {
//!                 // Do something with the clicked entity
//!             }
//!         }
//!
//!         frame.render_world();
//!     }
//! });
//! ```

use glam::{Mat4, Vec3};

/// A ray in 3D space, used for raycasting and picking.
///
/// A ray has an origin point and a normalized direction. It represents
/// an infinite line starting at the origin and extending in the direction.
///
/// # Example
///
/// ```
/// use hoplite::{Ray, Vec3};
///
/// // Create a ray from the camera
/// let ray = Ray::new(
///     Vec3::new(0.0, 1.0, 5.0),   // origin (camera position)
///     Vec3::new(0.0, 0.0, -1.0),  // direction (forward)
/// );
///
/// // Get a point along the ray
/// let point_at_10_units = ray.point_at(10.0);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    /// The starting point of the ray.
    pub origin: Vec3,
    /// The normalized direction of the ray.
    pub direction: Vec3,
}

impl Ray {
    /// Create a new ray with the given origin and direction.
    ///
    /// The direction will be normalized automatically.
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize_or_zero(),
        }
    }

    /// Create a ray from screen coordinates using camera matrices.
    ///
    /// This is the primary way to create a picking ray from mouse position.
    ///
    /// # Arguments
    ///
    /// * `screen_x` - X position in screen pixels
    /// * `screen_y` - Y position in screen pixels
    /// * `screen_width` - Total screen width in pixels
    /// * `screen_height` - Total screen height in pixels
    /// * `view_matrix` - Camera view matrix
    /// * `projection_matrix` - Camera projection matrix
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ray = Ray::from_screen(
    ///     mouse_x, mouse_y,
    ///     frame.width() as f32, frame.height() as f32,
    ///     frame.camera.view_matrix(),
    ///     frame.camera.projection_matrix(aspect, 0.1, 1000.0),
    /// );
    /// ```
    pub fn from_screen(
        screen_x: f32,
        screen_y: f32,
        screen_width: f32,
        screen_height: f32,
        view_matrix: Mat4,
        projection_matrix: Mat4,
    ) -> Self {
        // Convert screen coordinates to normalized device coordinates (-1 to 1)
        let ndc_x = (2.0 * screen_x / screen_width) - 1.0;
        let ndc_y = 1.0 - (2.0 * screen_y / screen_height); // Y is flipped

        // Create clip-space coordinates for near and far planes
        let near_clip = glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let far_clip = glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

        // Inverse view-projection to get world coordinates
        let inv_view_proj = (projection_matrix * view_matrix).inverse();

        let near_world = inv_view_proj * near_clip;
        let far_world = inv_view_proj * far_clip;

        // Perspective divide
        let near_point = near_world.truncate() / near_world.w;
        let far_point = far_world.truncate() / far_world.w;

        let direction = (far_point - near_point).normalize_or_zero();

        Self {
            origin: near_point,
            direction,
        }
    }

    /// Get a point along the ray at the given distance from the origin.
    #[inline]
    pub fn point_at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Test intersection with an axis-aligned bounding box (AABB).
    ///
    /// Returns the distance along the ray to the intersection point, or `None`
    /// if the ray doesn't intersect the box.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum corner of the AABB
    /// * `max` - Maximum corner of the AABB
    pub fn intersect_aabb(&self, min: Vec3, max: Vec3) -> Option<f32> {
        let mut t_min = f32::NEG_INFINITY;
        let mut t_max = f32::INFINITY;

        for i in 0..3 {
            let origin = self.origin[i];
            let dir = self.direction[i];
            let box_min = min[i];
            let box_max = max[i];

            if dir.abs() < f32::EPSILON {
                // Ray is parallel to this axis
                if origin < box_min || origin > box_max {
                    return None;
                }
            } else {
                let inv_dir = 1.0 / dir;
                let mut t1 = (box_min - origin) * inv_dir;
                let mut t2 = (box_max - origin) * inv_dir;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                t_min = t_min.max(t1);
                t_max = t_max.min(t2);

                if t_min > t_max {
                    return None;
                }
            }
        }

        // Return the nearest positive intersection
        if t_min > 0.0 {
            Some(t_min)
        } else if t_max > 0.0 {
            Some(t_max)
        } else {
            None
        }
    }

    /// Test intersection with a sphere.
    ///
    /// Returns the distance along the ray to the intersection point, or `None`
    /// if the ray doesn't intersect the sphere.
    ///
    /// # Arguments
    ///
    /// * `center` - Center of the sphere
    /// * `radius` - Radius of the sphere
    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<f32> {
        let oc = self.origin - center;
        let a = self.direction.dot(self.direction);
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_disc = discriminant.sqrt();
        let t1 = (-b - sqrt_disc) / (2.0 * a);
        let t2 = (-b + sqrt_disc) / (2.0 * a);

        // Return the nearest positive intersection
        if t1 > 0.0 {
            Some(t1)
        } else if t2 > 0.0 {
            Some(t2)
        } else {
            None
        }
    }
}

/// A collision shape for picking and hit detection.
///
/// Colliders are simple geometric shapes used for raycasting. They're much
/// faster to test than full mesh geometry and sufficient for most picking needs.
///
/// # Supported Shapes
///
/// - **Box**: Axis-aligned bounding box with half-extents
/// - **Sphere**: Simple sphere with radius
///
/// # Example
///
/// ```
/// use hoplite::{Collider, Vec3};
///
/// // Box collider for a 2x1x2 object (half-extents are 1, 0.5, 1)
/// let box_collider = Collider::box_collider(Vec3::new(2.0, 1.0, 2.0));
///
/// // Sphere collider with radius 1.5
/// let sphere_collider = Collider::sphere(1.5);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum Collider {
    /// Axis-aligned bounding box defined by half-extents.
    /// A box with half_extents (1, 1, 1) spans from (-1, -1, -1) to (1, 1, 1).
    Box {
        /// Half the size of the box on each axis.
        half_extents: Vec3,
    },
    /// Sphere defined by radius.
    Sphere {
        /// Radius of the sphere.
        radius: f32,
    },
}

impl Collider {
    /// Create a box collider from full dimensions.
    ///
    /// The collider will be centered at the entity's position.
    ///
    /// # Arguments
    ///
    /// * `size` - Full dimensions (width, height, depth) of the box
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::{Collider, Vec3};
    ///
    /// // 1x1x1 unit cube (matches Mesh::cube() dimensions)
    /// let collider = Collider::box_collider(Vec3::ONE);
    /// ```
    pub fn box_collider(size: Vec3) -> Self {
        Self::Box {
            half_extents: size * 0.5,
        }
    }

    /// Create a box collider from half-extents.
    ///
    /// Half-extents define the distance from center to each face.
    ///
    /// # Arguments
    ///
    /// * `half_extents` - Half the size on each axis
    pub fn box_half_extents(half_extents: Vec3) -> Self {
        Self::Box { half_extents }
    }

    /// Create a sphere collider.
    ///
    /// # Arguments
    ///
    /// * `radius` - Radius of the sphere
    ///
    /// # Example
    ///
    /// ```
    /// use hoplite::Collider;
    ///
    /// // Sphere with radius 0.5 (matches Mesh::sphere() default radius)
    /// let collider = Collider::sphere(0.5);
    /// ```
    pub fn sphere(radius: f32) -> Self {
        Self::Sphere { radius }
    }

    /// Create a unit box collider (1x1x1).
    ///
    /// Matches the dimensions of `Mesh::cube()`.
    pub fn unit_box() -> Self {
        Self::box_collider(Vec3::ONE)
    }

    /// Create a unit sphere collider (radius 0.5).
    ///
    /// Matches the dimensions of `Mesh::sphere()`.
    pub fn unit_sphere() -> Self {
        Self::Sphere { radius: 0.5 }
    }

    /// Test if a ray intersects this collider at the given transform.
    ///
    /// Returns the distance along the ray to the hit point, or `None` if no hit.
    ///
    /// # Arguments
    ///
    /// * `ray` - The ray to test
    /// * `position` - World position of the collider
    /// * `scale` - Scale of the collider (from Transform)
    pub fn intersect(&self, ray: &Ray, position: Vec3, scale: Vec3) -> Option<f32> {
        match self {
            Collider::Box { half_extents } => {
                let scaled_half = *half_extents * scale;
                let min = position - scaled_half;
                let max = position + scaled_half;
                ray.intersect_aabb(min, max)
            }
            Collider::Sphere { radius } => {
                // Use the average scale for sphere radius
                let avg_scale = (scale.x + scale.y + scale.z) / 3.0;
                ray.intersect_sphere(position, radius * avg_scale)
            }
        }
    }
}

impl Default for Collider {
    fn default() -> Self {
        Self::unit_box()
    }
}

/// Information about a ray-collider intersection.
///
/// Returned by picking methods when a ray hits a collider.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// The entity that was hit.
    pub entity: hecs::Entity,
    /// Distance from ray origin to the hit point.
    pub distance: f32,
    /// World-space position of the hit point.
    pub point: Vec3,
}

/// Result of a raycast against all colliders in the world.
///
/// Contains the closest hit, if any.
pub type PickResult = Option<RayHit>;

/// Cast a ray against all entities with colliders and return hits.
///
/// This is the core picking function. It tests the ray against all entities
/// that have both a `Transform` and `Collider` component.
///
/// # Arguments
///
/// * `world` - The ECS world to query
/// * `ray` - The ray to cast
///
/// # Returns
///
/// A vector of all hits, sorted by distance (closest first).
pub fn raycast_all(world: &hecs::World, ray: &Ray) -> Vec<RayHit> {
    use crate::mesh::Transform;

    let mut hits = Vec::new();

    for (entity, (transform, collider)) in world.query::<(&Transform, &Collider)>().iter() {
        if let Some(distance) = collider.intersect(ray, transform.position, transform.scale) {
            hits.push(RayHit {
                entity,
                distance,
                point: ray.point_at(distance),
            });
        }
    }

    // Sort by distance (closest first)
    hits.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    hits
}

/// Cast a ray and return the closest hit.
///
/// # Arguments
///
/// * `world` - The ECS world to query
/// * `ray` - The ray to cast
///
/// # Returns
///
/// The closest hit, or `None` if nothing was hit.
pub fn raycast(world: &hecs::World, ray: &Ray) -> PickResult {
    raycast_all(world, ray).into_iter().next()
}
