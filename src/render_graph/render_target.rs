//! Render targets and execution context for the render graph.

use crate::camera::Camera;
use crate::gpu::GpuContext;

/// An off-screen render target used for intermediate pass results.
///
/// Render targets are GPU textures that can be both rendered to (as a color attachment)
/// and sampled from (as a texture binding). This dual capability enables ping-pong
/// rendering where one pass writes to target A while reading from target B, then
/// the next pass reverses the roles.
///
/// The render graph automatically manages two render targets internally and handles
/// resizing when the window dimensions change.
///
/// # Fields
///
/// * `texture` - The underlying wgpu texture resource
/// * `view` - A texture view for binding as either render target or sampler input
/// * `width` - Current width in pixels (tracks GPU surface size)
/// * `height` - Current height in pixels (tracks GPU surface size)
pub struct RenderTarget {
    /// The underlying GPU texture that stores pixel data.
    pub texture: wgpu::Texture,
    /// A view into the texture, used for render pass attachments and shader sampling.
    pub view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl RenderTarget {
    /// Creates a new render target matching the current GPU surface dimensions.
    ///
    /// The texture is created with:
    /// - Same format as the surface (typically `Bgra8UnormSrgb`)
    /// - `RENDER_ATTACHMENT` usage for writing via render passes
    /// - `TEXTURE_BINDING` usage for sampling in subsequent passes
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context providing device and surface configuration
    /// * `label` - Debug label for the texture (visible in GPU debuggers like RenderDoc)
    pub fn new(gpu: &GpuContext, label: &str) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: gpu.width(),
                height: gpu.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width: gpu.width(),
            height: gpu.height(),
        }
    }

    /// Checks if the target dimensions match the GPU surface and recreates if needed.
    ///
    /// This should be called at the start of each frame to handle window resizes.
    /// If the dimensions differ, a new texture is allocated and the old one is dropped.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context to check dimensions against
    /// * `label` - Debug label for the recreated texture
    pub fn ensure_size(&mut self, gpu: &GpuContext, label: &str) {
        if self.width != gpu.width() || self.height != gpu.height() {
            *self = Self::new(gpu, label);
        }
    }
}

/// Execution context passed to each render node during graph traversal.
///
/// This struct bundles all the resources a render node needs to execute its
/// rendering operations. It is created fresh for each frame and passed through
/// the entire node chain.
///
/// # Lifetime
///
/// The `'a` lifetime ties all references to the frame's scope, ensuring nodes
/// cannot hold onto resources beyond the current frame.
///
/// # Fields
///
/// * `gpu` - Access to device, queue, and surface configuration
/// * `encoder` - Command encoder for recording GPU commands
/// * `time` - Elapsed time in seconds (for animations and effects)
/// * `camera` - Current camera state for view/projection matrices
pub struct RenderContext<'a> {
    /// GPU context providing access to device, queue, and configuration.
    pub gpu: &'a GpuContext,
    /// Command encoder for recording render pass commands.
    /// Nodes append their commands to this encoder.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// Elapsed time in seconds since application start.
    /// Used for animating shaders and time-based effects.
    pub time: f32,
    /// Current camera providing view and projection matrices.
    /// Available for nodes that need world-space or screen-space transformations.
    pub camera: &'a Camera,
}
