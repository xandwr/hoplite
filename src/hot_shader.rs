use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::camera::Camera;
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::post_process::{PostProcessPass, WorldPostProcessPass};

/// A shader source that can be hot-reloaded from disk.
pub struct HotShader {
    path: PathBuf,
    last_modified: SystemTime,
    source: String,
}

impl HotShader {
    /// Load a shader from the given file path.
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
    /// Returns `true` if the shader was reloaded.
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
/// Falls back to the last working shader if compilation fails.
pub struct HotEffectPass {
    shader: HotShader,
    pass: Option<EffectPass>,
    uses_camera: bool,
}

impl HotEffectPass {
    /// Create a new screen-space hot-reloadable effect pass.
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
    pub fn new_world(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source(), true);

        Ok(Self {
            shader,
            pass,
            uses_camera: true,
        })
    }

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
    /// Call this once per frame.
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
    pub fn render(&self, gpu: &GpuContext, render_pass: &mut wgpu::RenderPass, time: f32) {
        if let Some(ref pass) = self.pass {
            pass.render(gpu, render_pass, time);
        }
    }

    /// Render a world-space effect with camera data.
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
pub struct HotPostProcessPass {
    shader: HotShader,
    pass: Option<PostProcessPass>,
}

impl HotPostProcessPass {
    /// Create a new hot-reloadable post-processing pass.
    pub fn new(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source());

        Ok(Self { shader, pass })
    }

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
pub struct HotWorldPostProcessPass {
    shader: HotShader,
    pass: Option<WorldPostProcessPass>,
}

impl HotWorldPostProcessPass {
    /// Create a new hot-reloadable world-space post-processing pass.
    pub fn new(gpu: &GpuContext, path: impl AsRef<Path>) -> std::io::Result<Self> {
        let shader = HotShader::new(path)?;
        let pass = Self::try_compile(gpu, shader.source());

        Ok(Self { shader, pass })
    }

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
