//! Fluent geometry loading for 3D models.
//!
//! This module provides a fun, ergonomic way to load 3D geometry from various file formats.
//! Currently supports STL files, with an extensible architecture for adding more formats.
//!
//! # Quick Start
//!
//! ```no_run
//! use hoplite::*;
//!
//! run(|ctx| {
//!     ctx.enable_mesh_rendering();
//!
//!     // Load an STL file - it's this easy!
//!     let model = ctx.load("my_model.stl").unwrap();
//!
//!     // Or with more control:
//!     let model = ctx.load("my_model.stl")
//!         .centered()      // Center at origin
//!         .normalized()    // Scale to fit in unit cube
//!         .unwrap();
//!
//!     move |frame| {
//!         frame.draw_mesh(model, Transform::new(), Color::WHITE);
//!     }
//! });
//! ```
//!
//! # Supported Formats
//!
//! | Format | Extensions | Notes |
//! |--------|------------|-------|
//! | STL    | `.stl`     | Binary and ASCII, no UV coordinates |
//!
//! # The GeometryLoader
//!
//! For advanced use cases, you can use [`GeometryLoader`] directly:
//!
//! ```ignore
//! use hoplite::*;
//!
//! run(|ctx| {
//!     ctx.enable_mesh_rendering();
//!
//!     // Load from bytes (useful for embedded assets)
//!     let stl_bytes = include_bytes!("../assets/model.stl");
//!     let model = GeometryLoader::from_stl_bytes(&ctx.gpu, stl_bytes)
//!         .centered()
//!         .build()
//!         .unwrap();
//!
//!     move |_| {}
//! });
//! ```

use crate::gpu::GpuContext;
use crate::mesh::{Mesh, Vertex3d};
use glam::{Quat, Vec3};
use std::path::Path;

/// Errors that can occur when loading geometry.
#[derive(Debug)]
pub enum GeometryError {
    /// File could not be read.
    Io(std::io::Error),
    /// File format could not be determined from extension.
    UnknownFormat(String),
    /// The geometry data was invalid or corrupt.
    ParseError(String),
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryError::Io(e) => write!(f, "IO error: {}", e),
            GeometryError::UnknownFormat(ext) => {
                write!(f, "Unknown geometry format: '{}'", ext)
            }
            GeometryError::ParseError(msg) => write!(f, "Parse error: {}", msg),
        }
    }
}

impl std::error::Error for GeometryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GeometryError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for GeometryError {
    fn from(e: std::io::Error) -> Self {
        GeometryError::Io(e)
    }
}

/// Raw geometry data before GPU upload.
///
/// This intermediate representation allows geometry transformations
/// (centering, scaling, etc.) before creating the final GPU mesh.
#[derive(Clone, Debug)]
pub struct RawGeometry {
    /// Vertex positions, normals, and UVs.
    pub vertices: Vec<Vertex3d>,
    /// Triangle indices.
    pub indices: Vec<u32>,
}

impl RawGeometry {
    /// Creates raw geometry from vertices and indices.
    pub fn new(vertices: Vec<Vertex3d>, indices: Vec<u32>) -> Self {
        Self { vertices, indices }
    }

    /// Computes the axis-aligned bounding box.
    ///
    /// Returns `(min, max)` corners of the bounding box.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);

        for v in &self.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }

        (min, max)
    }

    /// Returns the center point of the geometry.
    pub fn center(&self) -> Vec3 {
        let (min, max) = self.bounds();
        (min + max) * 0.5
    }

    /// Returns the size of the bounding box.
    pub fn size(&self) -> Vec3 {
        let (min, max) = self.bounds();
        max - min
    }

    /// Translates all vertices by the given offset.
    pub fn translate(&mut self, offset: Vec3) {
        for v in &mut self.vertices {
            v.position[0] += offset.x;
            v.position[1] += offset.y;
            v.position[2] += offset.z;
        }
    }

    /// Scales all vertices uniformly around the origin.
    pub fn scale(&mut self, factor: f32) {
        for v in &mut self.vertices {
            v.position[0] *= factor;
            v.position[1] *= factor;
            v.position[2] *= factor;
        }
    }

    /// Scales vertices non-uniformly.
    pub fn scale_xyz(&mut self, factors: Vec3) {
        for v in &mut self.vertices {
            v.position[0] *= factors.x;
            v.position[1] *= factors.y;
            v.position[2] *= factors.z;
        }
    }

    /// Rotates all vertices and normals by the given quaternion.
    pub fn rotate(&mut self, rotation: Quat) {
        for v in &mut self.vertices {
            // Rotate position
            let pos = Vec3::from(v.position);
            let rotated_pos = rotation * pos;
            v.position = rotated_pos.into();

            // Rotate normal
            let normal = Vec3::from(v.normal);
            let rotated_normal = rotation * normal;
            v.normal = rotated_normal.into();
        }
    }

    /// Centers the geometry at the origin.
    pub fn recenter(&mut self) {
        let center = self.center();
        self.translate(-center);
    }

    /// Scales the geometry to fit within a unit cube (-0.5 to 0.5).
    pub fn normalize(&mut self) {
        let size = self.size();
        let max_dim = size.x.max(size.y).max(size.z);
        if max_dim > 0.0 {
            self.scale(1.0 / max_dim);
        }
    }

    /// Recalculates vertex normals from face geometry.
    ///
    /// This computes smooth normals by averaging the face normals
    /// of all triangles that share each vertex.
    pub fn recalculate_normals(&mut self) {
        // Reset all normals to zero
        for v in &mut self.vertices {
            v.normal = [0.0, 0.0, 0.0];
        }

        // Accumulate face normals for each vertex
        for tri in self.indices.chunks(3) {
            if tri.len() < 3 {
                continue;
            }
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;

            let p0 = Vec3::from(self.vertices[i0].position);
            let p1 = Vec3::from(self.vertices[i1].position);
            let p2 = Vec3::from(self.vertices[i2].position);

            let edge1 = p1 - p0;
            let edge2 = p2 - p0;
            let face_normal = edge1.cross(edge2);

            // Add to each vertex (weighted by face area, which is |cross product|)
            for &i in &[i0, i1, i2] {
                self.vertices[i].normal[0] += face_normal.x;
                self.vertices[i].normal[1] += face_normal.y;
                self.vertices[i].normal[2] += face_normal.z;
            }
        }

        // Normalize all vertex normals
        for v in &mut self.vertices {
            let n = Vec3::from(v.normal);
            let normalized = n.normalize_or_zero();
            v.normal = normalized.into();
        }
    }

    /// Uploads this geometry to the GPU as a [`Mesh`].
    pub fn upload(&self, gpu: &GpuContext) -> Mesh {
        Mesh::new(gpu, &self.vertices, &self.indices)
    }
}

/// A fluent builder for loading and transforming geometry.
///
/// Use this to load 3D models with optional transformations applied
/// before GPU upload. The builder pattern makes it easy to chain
/// operations in a readable way.
///
/// # Example
///
/// ```no_run
/// use hoplite::*;
///
/// run(|ctx| {
///     ctx.enable_mesh_rendering();
///
///     let mesh = GeometryLoader::from_stl(&ctx.gpu, "model.stl")
///         .centered()           // Move to origin
///         .normalized()         // Fit in unit cube
///         .smooth_normals()     // Recalculate smooth normals
///         .build()
///         .unwrap();
///
///     move |_| {}
/// });
/// ```
pub struct GeometryLoader<'a> {
    gpu: &'a GpuContext,
    pending: PendingGeometry,
}

/// Geometry loading state that doesn't require a GPU reference.
///
/// This allows the loader to be stored separately from the GPU context,
/// enabling more flexible ownership patterns in the API.
#[derive(Clone)]
pub struct PendingGeometry {
    result: Result<RawGeometry, String>,
    center: bool,
    normalize: bool,
    smooth_normals: bool,
    scale_factor: Option<f32>,
    translation: Option<Vec3>,
    rotation: Option<Quat>,
}

impl PendingGeometry {
    /// Load geometry from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let result = Self::load_file(path).map_err(|e| e.to_string());

        Self {
            result,
            center: false,
            normalize: false,
            smooth_normals: false,
            scale_factor: None,
            translation: None,
            rotation: None,
        }
    }

    /// Load STL geometry from a file path.
    pub fn from_stl(path: impl AsRef<Path>) -> Self {
        let result = Self::load_stl_file(path.as_ref()).map_err(|e| e.to_string());

        Self {
            result,
            center: false,
            normalize: false,
            smooth_normals: false,
            scale_factor: None,
            translation: None,
            rotation: None,
        }
    }

    /// Load STL geometry from bytes.
    pub fn from_stl_bytes(bytes: &[u8]) -> Self {
        let result = Self::parse_stl_bytes(bytes).map_err(|e| e.to_string());

        Self {
            result,
            center: false,
            normalize: false,
            smooth_normals: false,
            scale_factor: None,
            translation: None,
            rotation: None,
        }
    }

    /// Create from existing raw geometry.
    pub fn from_raw(geometry: RawGeometry) -> Self {
        Self {
            result: Ok(geometry),
            center: false,
            normalize: false,
            smooth_normals: false,
            scale_factor: None,
            translation: None,
            rotation: None,
        }
    }

    /// Centers the geometry at the origin.
    pub fn centered(mut self) -> Self {
        self.center = true;
        self
    }

    /// Scales the geometry to fit within a unit cube.
    pub fn normalized(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Recalculates smooth vertex normals.
    pub fn smooth_normals(mut self) -> Self {
        self.smooth_normals = true;
        self
    }

    /// Applies a uniform scale factor.
    pub fn scaled(mut self, factor: f32) -> Self {
        self.scale_factor = Some(factor);
        self
    }

    /// Translates the geometry by the given offset.
    pub fn translated(mut self, offset: Vec3) -> Self {
        self.translation = Some(offset);
        self
    }

    /// Reorients the geometry from Z-up to Y-up.
    ///
    /// Many 3D modeling tools export with Z as the up axis, while
    /// game engines typically use Y-up. This applies a -90 degree
    /// rotation around the X axis to convert between them.
    pub fn upright(mut self) -> Self {
        // -90 degrees around X converts Z-up to Y-up
        self.rotation = Some(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2));
        self
    }

    /// Rotates the geometry by a custom quaternion.
    pub fn rotated_by(mut self, rotation: Quat) -> Self {
        self.rotation = Some(rotation);
        self
    }

    /// Finalize and upload to GPU.
    pub fn upload(self, gpu: &GpuContext) -> Result<Mesh, GeometryError> {
        let mut geometry = self.result.map_err(|s| GeometryError::ParseError(s))?;

        // Apply transformations in order
        if self.center {
            geometry.recenter();
        }

        if let Some(rotation) = self.rotation {
            geometry.rotate(rotation);
        }

        if self.normalize {
            geometry.normalize();
        }

        if let Some(scale) = self.scale_factor {
            geometry.scale(scale);
        }

        if self.smooth_normals {
            geometry.recalculate_normals();
        }

        if let Some(offset) = self.translation {
            geometry.translate(offset);
        }

        Ok(geometry.upload(gpu))
    }

    // Internal: Load file with format detection
    fn load_file(path: &Path) -> Result<RawGeometry, GeometryError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();

        match ext.as_str() {
            "stl" => Self::load_stl_file(path),
            _ => Err(GeometryError::UnknownFormat(ext)),
        }
    }

    // Internal: Load STL file
    fn load_stl_file(path: &Path) -> Result<RawGeometry, GeometryError> {
        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::new(file);
        Self::parse_stl(&mut reader)
    }

    // Internal: Parse STL from reader
    fn parse_stl<R: std::io::Read + std::io::Seek>(
        reader: &mut R,
    ) -> Result<RawGeometry, GeometryError> {
        let stl = stl_io::read_stl(reader)
            .map_err(|e| GeometryError::ParseError(format!("STL parse error: {}", e)))?;

        let mut vertices = Vec::with_capacity(stl.faces.len() * 3);
        let mut indices = Vec::with_capacity(stl.faces.len() * 3);

        // stl_io returns an IndexedMesh with a vertex list and indexed triangles
        for (i, face) in stl.faces.iter().enumerate() {
            let normal: [f32; 3] = face.normal.into();

            // Look up the actual vertex positions from the vertex list
            for &vertex_idx in &face.vertices {
                let vertex = &stl.vertices[vertex_idx];
                let position: [f32; 3] = (*vertex).into();
                vertices.push(Vertex3d::new(
                    position,
                    normal,
                    [0.0, 0.0], // STL has no UVs
                ));
            }

            let base = (i * 3) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }

        Ok(RawGeometry::new(vertices, indices))
    }

    // Internal: Parse STL from bytes
    fn parse_stl_bytes(bytes: &[u8]) -> Result<RawGeometry, GeometryError> {
        let mut cursor = std::io::Cursor::new(bytes);
        Self::parse_stl(&mut cursor)
    }
}

impl<'a> GeometryLoader<'a> {
    /// Loads geometry from a file, detecting format from extension.
    ///
    /// Currently supports:
    /// - `.stl` - STL files (binary and ASCII)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use hoplite::*;
    ///
    /// run(|ctx| {
    ///     let mesh = GeometryLoader::from_file(&ctx.gpu, "model.stl")
    ///         .build()
    ///         .unwrap();
    ///     move |_| {}
    /// });
    /// ```
    pub fn from_file(gpu: &'a GpuContext, path: impl AsRef<Path>) -> Self {
        Self {
            gpu,
            pending: PendingGeometry::from_file(path),
        }
    }

    /// Loads an STL file specifically.
    ///
    /// Use this when you know the file is STL, or when the file
    /// doesn't have a standard extension.
    pub fn from_stl(gpu: &'a GpuContext, path: impl AsRef<Path>) -> Self {
        Self {
            gpu,
            pending: PendingGeometry::from_stl(path),
        }
    }

    /// Loads STL geometry from raw bytes.
    ///
    /// Perfect for embedded assets using `include_bytes!`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use hoplite::*;
    ///
    /// const MODEL: &[u8] = include_bytes!("../assets/model.stl");
    ///
    /// run(|ctx| {
    ///     let mesh = GeometryLoader::from_stl_bytes(&ctx.gpu, MODEL)
    ///         .centered()
    ///         .build()
    ///         .unwrap();
    ///     move |_| {}
    /// });
    /// ```
    pub fn from_stl_bytes(gpu: &'a GpuContext, bytes: &[u8]) -> Self {
        Self {
            gpu,
            pending: PendingGeometry::from_stl_bytes(bytes),
        }
    }

    /// Creates a loader from existing raw geometry.
    ///
    /// Useful when you have procedurally generated geometry
    /// and want to apply the same transformations.
    pub fn from_raw(gpu: &'a GpuContext, geometry: RawGeometry) -> Self {
        Self {
            gpu,
            pending: PendingGeometry::from_raw(geometry),
        }
    }

    /// Centers the geometry at the origin.
    ///
    /// Moves the geometry so its bounding box center is at (0, 0, 0).
    /// This is applied before scaling.
    pub fn centered(mut self) -> Self {
        self.pending = self.pending.centered();
        self
    }

    /// Scales the geometry to fit within a unit cube.
    ///
    /// The geometry will fit within -0.5 to 0.5 on all axes,
    /// preserving aspect ratio.
    pub fn normalized(mut self) -> Self {
        self.pending = self.pending.normalized();
        self
    }

    /// Recalculates smooth vertex normals.
    ///
    /// Computes normals by averaging face normals at each vertex.
    /// Useful for STL files which only have face normals.
    pub fn smooth_normals(mut self) -> Self {
        self.pending = self.pending.smooth_normals();
        self
    }

    /// Applies a uniform scale factor.
    ///
    /// This is applied after centering and normalization.
    pub fn scaled(mut self, factor: f32) -> Self {
        self.pending = self.pending.scaled(factor);
        self
    }

    /// Translates the geometry by the given offset.
    ///
    /// This is applied last, after all other transformations.
    pub fn translated(mut self, offset: Vec3) -> Self {
        self.pending = self.pending.translated(offset);
        self
    }

    /// Reorients the geometry from Z-up to Y-up.
    ///
    /// Many 3D modeling tools (Blender, etc.) export with Z as the up axis,
    /// while game engines typically use Y-up. This applies a -90 degree
    /// rotation around the X axis to convert between them.
    ///
    /// This is applied after centering but before normalization/scaling.
    pub fn upright(mut self) -> Self {
        self.pending = self.pending.upright();
        self
    }

    /// Rotates the geometry by a custom quaternion.
    ///
    /// This is applied after centering but before normalization/scaling.
    pub fn rotated_by(mut self, rotation: Quat) -> Self {
        self.pending = self.pending.rotated_by(rotation);
        self
    }

    /// Builds the final mesh, uploading to the GPU.
    ///
    /// Applies all requested transformations in order:
    /// 1. Center (if requested)
    /// 2. Rotate (if requested)
    /// 3. Normalize (if requested)
    /// 4. Scale (if specified)
    /// 5. Smooth normals (if requested)
    /// 6. Translate (if specified)
    ///
    /// Returns the GPU-ready [`Mesh`] or an error.
    pub fn build(self) -> Result<Mesh, GeometryError> {
        self.pending.upload(self.gpu)
    }

    /// Builds the mesh, panicking on error.
    ///
    /// Equivalent to `.build().unwrap()` but with a nicer panic message.
    ///
    /// # Panics
    ///
    /// Panics if the geometry could not be loaded or processed.
    pub fn unwrap(self) -> Mesh {
        self.build().expect("Failed to load geometry")
    }

    /// Builds the mesh, panicking with a custom message on error.
    ///
    /// # Panics
    ///
    /// Panics with the provided message if loading fails.
    pub fn expect(self, msg: &str) -> Mesh {
        self.build().expect(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_geometry_bounds() {
        let vertices = vec![
            Vertex3d::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3d::new([1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3d::new([-1.0, -1.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
        ];
        let indices = vec![0, 1, 2];
        let geom = RawGeometry::new(vertices, indices);

        let (min, max) = geom.bounds();
        assert_eq!(min, Vec3::new(-1.0, -1.0, -1.0));
        assert_eq!(max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn raw_geometry_center() {
        let vertices = vec![
            Vertex3d::new([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3d::new([2.0, 4.0, 6.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
        ];
        let indices = vec![0, 1, 0];
        let geom = RawGeometry::new(vertices, indices);

        assert_eq!(geom.center(), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn raw_geometry_recenter() {
        let vertices = vec![
            Vertex3d::new([2.0, 2.0, 2.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
            Vertex3d::new([4.0, 4.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0]),
        ];
        let indices = vec![0, 1, 0];
        let mut geom = RawGeometry::new(vertices, indices);

        geom.recenter();

        let center = geom.center();
        assert!((center.x).abs() < 0.001);
        assert!((center.y).abs() < 0.001);
        assert!((center.z).abs() < 0.001);
    }
}
