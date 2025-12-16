//! Core GPU context and device management.
//!
//! This module provides [`GpuContext`], the central struct that holds all wgpu resources
//! needed for rendering. It manages the device, queue, surface, and surface configuration,
//! and is passed to all rendering passes in the engine.
//!
//! # Initialization
//!
//! A `GpuContext` is created from a winit [`Window`] and handles all the wgpu boilerplate:
//! instance creation, adapter selection, device/queue creation, and surface configuration.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use winit::window::Window;
//! use hoplite::GpuContext;
//!
//! // Create GPU context from a window
//! let gpu = GpuContext::new(window);
//!
//! // Access device for creating resources
//! let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
//!     label: Some("My Buffer"),
//!     size: 1024,
//!     usage: wgpu::BufferUsages::UNIFORM,
//!     mapped_at_creation: false,
//! });
//!
//! // Submit work via the queue
//! gpu.queue.write_buffer(&buffer, 0, &[0u8; 1024]);
//! ```
//!
//! [`Window`]: winit::window::Window

use std::sync::Arc;
use winit::window::Window;

/// Core GPU context holding wgpu resources.
///
/// This struct owns all the fundamental wgpu objects needed for rendering:
/// the surface for presenting to the window, the device for creating GPU resources,
/// the queue for submitting commands, and the surface configuration.
///
/// All fields are public to allow direct access to wgpu APIs when needed.
/// The context is typically created once at startup and passed by reference
/// to all rendering passes.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
/// use hoplite::GpuContext;
///
/// let gpu = GpuContext::new(window);
///
/// // Handle window resize
/// gpu.resize(new_width, new_height);
///
/// // Get current dimensions
/// println!("{}x{} (aspect: {})", gpu.width(), gpu.height(), gpu.aspect());
/// ```
pub struct GpuContext {
    /// The surface for presenting rendered frames to the window.
    pub surface: wgpu::Surface<'static>,
    /// The logical GPU device for creating resources and pipelines.
    pub device: wgpu::Device,
    /// The command queue for submitting work to the GPU.
    pub queue: wgpu::Queue,
    /// Current surface configuration (format, size, present mode).
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// Create a new GPU context from a winit window.
    ///
    /// This performs all wgpu initialization:
    /// 1. Creates a wgpu instance with primary backends (Vulkan, Metal, DX12)
    /// 2. Creates a surface for the window
    /// 3. Requests a suitable GPU adapter
    /// 4. Creates the logical device and command queue
    /// 5. Configures the surface with an sRGB format and Fifo present mode
    ///
    /// # Panics
    ///
    /// Panics if no suitable GPU adapter is found or device creation fails.
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Hoplite Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: Default::default(),
            trace: Default::default(),
            experimental_features: Default::default(),
        }))
        .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    /// Resize the surface to new dimensions.
    ///
    /// Call this when the window is resized. Ignores zero-sized dimensions
    /// to avoid wgpu validation errors (which can occur during window minimize).
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    /// Returns the current surface width in pixels.
    pub fn width(&self) -> u32 {
        self.config.width
    }

    /// Returns the current surface height in pixels.
    pub fn height(&self) -> u32 {
        self.config.height
    }

    /// Returns the current aspect ratio (width / height).
    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height as f32
    }
}
