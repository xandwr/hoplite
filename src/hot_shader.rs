//! Hot-reloadable shader infrastructure for rapid iteration.
//!
//! This module provides hot-reloading wrappers around the various shader pass types,
//! enabling live shader editing without restarting the application. When a shader file
//! is modified on disk, the corresponding pass automatically detects the change and
//! recompiles. If compilation fails, the previous working shader is retained.
//!
//! # Available Hot-Reloadable Passes
//!
//! - [`HotShader`]: Low-level shader source watcher that tracks file modifications
//! - [`HotEffectPass`]: Hot-reloadable fullscreen effect (wraps [`EffectPass`])
//! - [`HotPostProcessPass`]: Hot-reloadable post-processing (wraps [`PostProcessPass`])
//! - [`HotWorldPostProcessPass`]: Hot-reloadable world-space post-processing (wraps [`WorldPostProcessPass`])
//!
//! # Usage Pattern
//!
//! All hot-reloadable passes follow the same pattern:
//!
//! 1. Create the pass with a path to a shader file
//! 2. Call `check_reload()` once per frame (typically at the start)
//! 3. Render as usual
//!
//! # Example
//!
//! ```no_run
//! use hoplite::{GpuContext, HotEffectPass};
//!
//! // Create a hot-reloadable effect pass
//! let mut effect = HotEffectPass::new(&gpu, "shaders/my_effect.wgsl")?;
//!
//! // In your render loop:
//! loop {
//!     // Check for shader changes at the start of each frame
//!     effect.check_reload(&gpu);
//!
//!     // Render if valid (gracefully handles compilation failures)
//!     if effect.is_valid() {
//!         effect.render(&gpu, &mut render_pass, time);
//!     }
//! }
//! # Ok::<(), std::io::Error>(())
//! ```
//!
//! # Error Handling
//!
//! Hot-reload compilation errors are logged to stderr with the `[hot-reload]` prefix.
//! The passes use `catch_unwind` to handle potential panics from invalid shaders,
//! ensuring the application remains stable during development.
//!
//! [`EffectPass`]: crate::effect_pass::EffectPass
//! [`PostProcessPass`]: crate::post_process::PostProcessPass
//! [`WorldPostProcessPass`]: crate::post_process::WorldPostProcessPass

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::camera::Camera;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::post_process::{PostProcessPass, WorldPostProcessPass};

/// A shader source that can be hot-reloaded from disk.
///
/// This is a low-level primitive that tracks a shader file's modification time
/// and reloads the source when changes are detected. Higher-level types like
/// [`HotEffectPass`] build on this to provide automatic recompilation.
///
/// # Example
///
/// ```no_run
/// use hoplite::HotShader;
///
/// let mut shader = HotShader::new("shaders/effect.wgsl")?;
///
/// // In your update loop:
/// if shader.check_reload() {
///     println!("Shader changed! New source: {}", shader.source());
///     // Recompile your pipeline here...
/// }
/// # Ok::<(), std::io::Error>(())
/// ```
pub struct HotShader {
    path: PathBuf,
    last_modified: SystemTime,
    source: String,
}

impl HotShader {
    /// Load a shader from the given file path.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or its metadata cannot be accessed.
    pub fn new(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let source = fs::read_to_string(&path)?;
        let last_modified = fs::metadata(&path)?.modified()?;

        Ok(Self {
            path,
            last_modified,
            source,
        })
    }

    /// Check if the shader file has been modified and reload if so.
    ///
    /// Compares the file's current modification time against the cached timestamp.
    /// If newer, reads the file contents and updates the internal source.
    ///
    /// # Returns
    ///
    /// `true` if the shader was reloaded, `false` otherwise (including on errors).
    pub fn check_reload(&mut self) -> bool {
        let Ok(metadata) = fs::metadata(&self.path) else {
            return false;
        };

        let Ok(modified) = metadata.modified() else {
            return false;
        };

        if modified > self.last_modified {
            if let Ok(source) = fs::read_to_string(&self.path) {
                self.source = source;
                self.last_modified = modified;
                return true;
            }
        }

        false
    }

    /// Get the current shader source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the shader file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// A hot-reloadable fullscreen effect pass.
///
/// Watches a shader file and automatically recompiles when changes are detected.
/// Falls back to the last working shader if compilation fails. Wraps [`EffectPass`]
/// with hot-reload capability.
///
/// Supports both screen-space effects (created with [`new`](Self::new)) and world-space
/// effects (created with [`new_world`](Self::new_world)) that have access to camera data.
///
/// # Example
///
/// ```no_run
/// use hoplite::{GpuContext, HotEffectPass};
///
/// // Screen-space effect (no camera)
/// let mut effect = HotEffectPass::new(&gpu, "shaders/vignette.wgsl")?;
///
/// // World-space effect (with camera)
/// let mut raymarcher = HotEffectPass::new_world(&gpu, "shaders/raymarch.wgsl")?;
///
/// // Render loop
/// effect.check_reload(&gpu);
/// effect.render(&gpu, &mut render_pass, time);
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// [`EffectPass`]: crate::effect_pass::EffectPass
pub struct HotEffectPass {
    shader: HotShader,
    pass: Option<EffectPass>,
    uses_camera: bool,
}

impl HotEffectPass {
    /// Create a new screen-space hot-reloadable effect pass.
    ///
    /// The shader will have access to resolution and time uniforms but not camera data.
    /// Use [`new_world`](Self::new_world) for effects that need camera information.
    ///
    /// # Errors
    ///
    /// Returns an error if the shader file cannot be read.
    pub fn new(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source(), false);

        Ok(Self {
            shader,
            pass,
            uses_camera: false,
        })
    }

    /// Create a new world-space hot-reloadable effect pass.
    ///
    /// The shader will have access to extended uniforms including camera position,
    /// orientation, and field of view. Suitable for raymarching, volumetric effects,
    /// and other techniques that need to reconstruct world-space rays.
    ///
    /// # Errors
    ///
    /// Returns an error if the shader file cannot be read.
    pub fn new_world(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source(), true);

        Ok(Self {
            shader,
            pass,
            uses_camera: true,
        })
    }

    /// Attempt to compile the shader, catching panics for safety.
    ///
    /// Returns `None` if compilation fails or panics.
    fn try_compile(gpu: &GpuContext, source: &str, uses_camera: bool) -> Option<EffectPass> {
        // wgpu shader compilation can panic on invalid shaders in some cases,
        // but typically returns errors through validation. We use catch_unwind
        // for extra safety during hot-reload.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if uses_camera {
                EffectPass::new_world(gpu, source)
            } else {
                EffectPass::new(gpu, source)
            }
        }));

        match result {
            Ok(pass) => Some(pass),
            Err(_) => {
                eprintln!("[hot-reload] Shader compilation panicked, keeping previous version");
                None
            }
        }
    }

    /// Check for shader changes and recompile if needed.
    ///
    /// Call this once per frame, typically at the start of your render loop.
    /// If the shader file has been modified, attempts to recompile. On failure,
    /// retains the previous working shader and logs an error to stderr.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        if self.shader.check_reload() {
            eprintln!("[hot-reload] Reloading shader: {:?}", self.shader.path());

            if let Some(new_pass) = Self::try_compile(gpu, self.shader.source(), self.uses_camera) {
                self.pass = Some(new_pass);
                eprintln!("[hot-reload] Shader compiled successfully");
            } else {
                eprintln!("[hot-reload] Shader compilation failed, keeping previous version");
            }
        }
    }

    /// Render a screen-space effect (no camera).
    ///
    /// Does nothing if no valid shader is loaded. Use [`is_valid`](Self::is_valid)
    /// to check if rendering will occur.
    pub fn render(&self, gpu: &GpuContext, render_pass: &mut wgpu::RenderPass, time: f32) {
        if let Some(ref pass) = self.pass {
            pass.render(gpu, render_pass, time);
        }
    }

    /// Render a world-space effect with camera data.
    ///
    /// Use this for effects created with [`new_world`](Self::new_world).
    /// Does nothing if no valid shader is loaded.
    pub fn render_with_camera(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        camera: &Camera,
    ) {
        if let Some(ref pass) = self.pass {
            pass.render_with_camera(gpu, render_pass, time, camera);
        }
    }

    /// Returns whether this effect pass uses camera data.
    pub fn uses_camera(&self) -> bool {
        self.uses_camera
    }

    /// Returns whether a valid shader is currently loaded.
    pub fn is_valid(&self) -> bool {
        self.pass.is_some()
    }
}

/// A hot-reloadable post-processing pass.
///
/// Watches a shader file and automatically recompiles when changes are detected.
/// Wraps [`PostProcessPass`] with hot-reload capability.
///
/// Post-processing passes sample from an input texture, making them suitable for
/// effects like bloom, color grading, and vignette.
///
/// # Example
///
/// ```no_run
/// use hoplite::{GpuContext, HotPostProcessPass};
///
/// let mut bloom = HotPostProcessPass::new(&gpu, "shaders/bloom.wgsl")?;
///
/// // In render loop:
/// bloom.check_reload(&gpu);
/// bloom.render(&gpu, &mut render_pass, time, &scene_texture_view);
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// [`PostProcessPass`]: crate::post_process::PostProcessPass
pub struct HotPostProcessPass {
    shader: HotShader,
    pass: Option<PostProcessPass>,
}

impl HotPostProcessPass {
    /// Create a new hot-reloadable post-processing pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the shader file cannot be read.
    pub fn new(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source());

        Ok(Self { shader, pass })
    }

    /// Attempt to compile the shader, catching panics for safety.
    fn try_compile(gpu: &GpuContext, source: &str) -> Option<PostProcessPass> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            PostProcessPass::new(gpu, source)
        }));

        match result {
            Ok(pass) => Some(pass),
            Err(_) => {
                eprintln!("[hot-reload] Shader compilation panicked, keeping previous version");
                None
            }
        }
    }

    /// Check for shader changes and recompile if needed.
    ///
    /// Call this once per frame. On compilation failure, retains the previous shader.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        if self.shader.check_reload() {
            eprintln!("[hot-reload] Reloading shader: {:?}", self.shader.path());

            if let Some(new_pass) = Self::try_compile(gpu, self.shader.source()) {
                self.pass = Some(new_pass);
                eprintln!("[hot-reload] Shader compiled successfully");
            } else {
                eprintln!("[hot-reload] Shader compilation failed, keeping previous version");
            }
        }
    }

    /// Render the post-process effect.
    ///
    /// Samples from `input_view` and renders to the current render pass target.
    /// Does nothing if no valid shader is loaded.
    pub fn render(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        input_view: &wgpu::TextureView,
    ) {
        if let Some(ref pass) = self.pass {
            pass.render(gpu, render_pass, time, input_view);
        }
    }

    /// Returns whether a valid shader is currently loaded.
    pub fn is_valid(&self) -> bool {
        self.pass.is_some()
    }
}

/// A hot-reloadable world-space post-processing pass.
///
/// Watches a shader file and automatically recompiles when changes are detected.
/// Wraps [`WorldPostProcessPass`] with hot-reload capability.
///
/// Unlike [`HotPostProcessPass`], this pass provides camera information to the shader,
/// enabling world-space effects like fog, atmospheric scattering, and screen-space reflections
/// that need to reconstruct 3D positions from the depth buffer.
///
/// # Example
///
/// ```no_run
/// use hoplite::{GpuContext, HotWorldPostProcessPass};
///
/// let mut fog = HotWorldPostProcessPass::new(&gpu, "shaders/fog.wgsl")?;
///
/// // In render loop:
/// fog.check_reload(&gpu);
/// fog.render(&gpu, &mut render_pass, time, &camera, &scene_texture_view);
/// # Ok::<(), std::io::Error>(())
/// ```
///
/// [`WorldPostProcessPass`]: crate::post_process::WorldPostProcessPass
pub struct HotWorldPostProcessPass {
    shader: HotShader,
    pass: Option<WorldPostProcessPass>,
}

impl HotWorldPostProcessPass {
    /// Create a new hot-reloadable world-space post-processing pass.
    ///
    /// # Errors
    ///
    /// Returns an error if the shader file cannot be read.
    pub fn new(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source());

        Ok(Self { shader, pass })
    }

    /// Attempt to compile the shader, catching panics for safety.
    fn try_compile(gpu: &GpuContext, source: &str) -> Option<WorldPostProcessPass> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            WorldPostProcessPass::new(gpu, source)
        }));

        match result {
            Ok(pass) => Some(pass),
            Err(_) => {
                eprintln!("[hot-reload] Shader compilation panicked, keeping previous version");
                None
            }
        }
    }

    /// Check for shader changes and recompile if needed.
    ///
    /// Call this once per frame. On compilation failure, retains the previous shader.
    pub fn check_reload(&mut self, gpu: &GpuContext) {
        if self.shader.check_reload() {
            eprintln!("[hot-reload] Reloading shader: {:?}", self.shader.path());

            if let Some(new_pass) = Self::try_compile(gpu, self.shader.source()) {
                self.pass = Some(new_pass);
                eprintln!("[hot-reload] Shader compiled successfully");
            } else {
                eprintln!("[hot-reload] Shader compilation failed, keeping previous version");
            }
        }
    }

    /// Render the post-process effect with camera data.
    ///
    /// Samples from `input_view` and renders to the current render pass target.
    /// Camera data is uploaded to the shader for world-space calculations.
    /// Does nothing if no valid shader is loaded.
    pub fn render(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        time: f32,
        camera: &Camera,
        input_view: &wgpu::TextureView,
    ) {
        if let Some(ref pass) = self.pass {
            pass.render(gpu, render_pass, time, camera, input_view);
        }
    }

    /// Returns whether a valid shader is currently loaded.
    pub fn is_valid(&self) -> bool {
        self.pass.is_some()
    }
}
