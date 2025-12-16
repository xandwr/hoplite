use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::assets::{Assets, FontId};
use crate::camera::Camera;
use crate::draw2d::{Color, Draw2d};
use crate::effect_pass::EffectPass;
use crate::gpu::GpuContext;
use crate::hot_shader::{HotEffectPass, HotPostProcessPass, HotWorldPostProcessPass};
use crate::input::Input;
use crate::mesh::{Mesh, Transform};
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::render_graph::{
    EffectNode, HotEffectNode, HotPostProcessNode, HotWorldPostProcessNode, MeshNode, MeshQueue,
    PostProcessNode, RenderGraph, WorldPostProcessNode,
};
use crate::texture::Texture;
use std::cell::RefCell;
use std::rc::Rc;

/// Context provided during app setup.
pub struct SetupContext<'a> {
    pub gpu: &'a GpuContext,
    pub assets: &'a mut Assets,
    default_font: &'a mut Option<FontId>,
    graph_builder: &'a mut Option<RenderGraph>,
    mesh_queue: &'a Rc<RefCell<MeshQueue>>,
}

impl<'a> SetupContext<'a> {
    /// Load the default font at the specified size.
    /// This font will be used by `Frame::text()` for convenience.
    pub fn default_font(&mut self, size: f32) -> FontId {
        let font = self.assets.default_font(size);
        *self.default_font = Some(font);
        font
    }

    /// Add a fullscreen shader effect to the render pipeline.
    ///
    /// Effects are rendered in the order they're added. The first effect
    /// clears the screen, subsequent effects chain together.
    pub fn effect(&mut self, shader: &str) -> &mut Self {
        let effect = EffectPass::new(self.gpu, shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    /// Add a world-space shader effect (receives camera uniforms).
    pub fn effect_world(&mut self, shader: &str) -> &mut Self {
        let effect = EffectPass::new_world(self.gpu, shader);
        self.add_node(EffectNode::new(effect));
        self
    }

    /// Add a screen-space post-processing effect.
    ///
    /// Post-process effects read from the previous pass output.
    pub fn post_process(&mut self, shader: &str) -> &mut Self {
        let pass = PostProcessPass::new(self.gpu, shader);
        self.add_node(PostProcessNode::new(pass));
        self
    }

    /// Add a world-space post-processing effect (receives camera uniforms).
    pub fn post_process_world(&mut self, shader: &str) -> &mut Self {
        let pass = WorldPostProcessPass::new(self.gpu, shader);
        self.add_node(WorldPostProcessNode::new(pass));
        self
    }

    // ========================================================================
    // Hot-reloadable shader methods
    // ========================================================================

    /// Add a hot-reloadable fullscreen shader effect (screen-space).
    ///
    /// The shader is loaded from the given file path and will automatically
    /// reload when the file changes on disk.
    pub fn hot_effect(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable world-space shader effect (receives camera uniforms).
    ///
    /// The shader is loaded from the given file path and will automatically
    /// reload when the file changes on disk.
    pub fn hot_effect_world(&mut self, path: &str) -> &mut Self {
        match HotEffectPass::new_world(self.gpu, path) {
            Ok(effect) => self.add_node(HotEffectNode::new(effect)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable screen-space post-processing effect.
    ///
    /// The shader is loaded from the given file path and will automatically
    /// reload when the file changes on disk.
    pub fn hot_post_process(&mut self, path: &str) -> &mut Self {
        match HotPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    /// Add a hot-reloadable world-space post-processing effect.
    ///
    /// The shader is loaded from the given file path and will automatically
    /// reload when the file changes on disk.
    pub fn hot_post_process_world(&mut self, path: &str) -> &mut Self {
        match HotWorldPostProcessPass::new(self.gpu, path) {
            Ok(pass) => self.add_node(HotWorldPostProcessNode::new(pass)),
            Err(e) => eprintln!("[hot-reload] Failed to load shader '{}': {}", path, e),
        }
        self
    }

    fn add_node<N: crate::render_graph::RenderNode + 'static>(&mut self, node: N) {
        if self.graph_builder.is_none() {
            *self.graph_builder = Some(RenderGraph::builder().node(node).build(self.gpu));
        } else {
            // For now, we rebuild - could optimize later
            let old = self.graph_builder.take().unwrap();
            *self.graph_builder = Some(old.with_node(node, self.gpu));
        }
    }

    // ========================================================================
    // Mesh methods
    // ========================================================================

    /// Enable 3D mesh rendering in the pipeline.
    ///
    /// This adds a mesh rendering pass to the render graph. Call this before
    /// adding post-processing effects if you want meshes to be affected by them.
    ///
    /// Returns self for chaining.
    pub fn enable_mesh_rendering(&mut self) -> &mut Self {
        let mesh_node = MeshNode::new(self.gpu, Rc::clone(self.mesh_queue));
        self.add_node(mesh_node);
        self
    }

    /// Create a unit cube mesh and return its index.
    pub fn mesh_cube(&mut self) -> usize {
        let mesh = Mesh::cube(self.gpu);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Create a sphere mesh and return its index.
    pub fn mesh_sphere(&mut self, segments: u32, rings: u32) -> usize {
        let mesh = Mesh::sphere(self.gpu, segments, rings);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Create a flat plane mesh and return its index.
    pub fn mesh_plane(&mut self, size: f32) -> usize {
        let mesh = Mesh::plane(self.gpu, size);
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    /// Add a custom mesh and return its index.
    pub fn add_mesh(&mut self, mesh: Mesh) -> usize {
        self.mesh_queue.borrow_mut().add_mesh(mesh)
    }

    // ========================================================================
    // Texture methods
    // ========================================================================

    /// Add a texture and return its index.
    pub fn add_texture(&mut self, texture: Texture) -> usize {
        self.mesh_queue.borrow_mut().add_texture(texture)
    }

    /// Load a texture from a file path and return its index.
    pub fn texture_from_file(&mut self, path: &str) -> Result<usize, image::ImageError> {
        let texture = Texture::from_file(self.gpu, path)?;
        Ok(self.add_texture(texture))
    }

    /// Load a texture from embedded bytes and return its index.
    pub fn texture_from_bytes(
        &mut self,
        bytes: &[u8],
        label: &str,
    ) -> Result<usize, image::ImageError> {
        let texture = Texture::from_bytes(self.gpu, bytes, label)?;
        Ok(self.add_texture(texture))
    }

    /// Create a procedural Minecraft-style noise texture and return its index.
    pub fn texture_minecraft_noise(&mut self, size: u32, seed: u32) -> usize {
        let texture = Texture::minecraft_noise(self.gpu, size, seed);
        self.add_texture(texture)
    }

    /// Create a procedural Minecraft-style grass texture and return its index.
    pub fn texture_minecraft_grass(&mut self, size: u32, seed: u32) -> usize {
        let texture = Texture::minecraft_grass(self.gpu, size, seed);
        self.add_texture(texture)
    }

    /// Create a procedural Minecraft-style cobblestone texture and return its index.
    pub fn texture_minecraft_cobblestone(&mut self, size: u32, seed: u32) -> usize {
        let texture = Texture::minecraft_cobblestone(self.gpu, size, seed);
        self.add_texture(texture)
    }
}

/// Context provided each frame for rendering.
///
/// Frame provides a simplified API for common operations. For advanced use,
/// the underlying `gpu`, `assets`, and `draw` fields are still accessible.
pub struct Frame<'a> {
    /// GPU context for advanced rendering.
    pub gpu: &'a GpuContext,
    /// Asset manager for loading fonts, textures, etc.
    pub assets: &'a Assets,
    /// Low-level 2D drawing API.
    pub draw: &'a mut Draw2d,
    /// Current camera state.
    pub camera: &'a mut Camera,
    /// Input state for this frame.
    pub input: &'a Input,
    /// Total elapsed time in seconds.
    pub time: f32,
    /// Delta time since last frame in seconds.
    pub dt: f32,
    /// Default font (if set during setup).
    default_font: Option<FontId>,
    /// Mesh queue for 3D rendering.
    mesh_queue: Rc<RefCell<MeshQueue>>,
}

impl Frame<'_> {
    /// Current frames per second.
    pub fn fps(&self) -> f32 {
        if self.dt > 0.0 { 1.0 / self.dt } else { 0.0 }
    }

    /// Screen width in pixels.
    pub fn width(&self) -> u32 {
        self.gpu.width()
    }

    /// Screen height in pixels.
    pub fn height(&self) -> u32 {
        self.gpu.height()
    }

    /// Draw text at the given position using the default font.
    ///
    /// Panics if no default font was set during setup.
    pub fn text(&mut self, x: f32, y: f32, text: &str) {
        self.text_color(x, y, text, Color::WHITE)
    }

    /// Draw text with a custom color using the default font.
    pub fn text_color(&mut self, x: f32, y: f32, text: &str, color: Color) {
        let font = self
            .default_font
            .expect("No default font set. Call ctx.default_font() in setup.");
        self.draw.text(self.assets, font, x, y, text, color);
    }

    /// Draw a colored rectangle.
    pub fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.draw.rect(x, y, w, h, color);
    }

    /// Draw a panel with background, border, and optional title.
    ///
    /// Returns the content area y-offset (below title bar if present).
    pub fn panel(&mut self, x: f32, y: f32, w: f32, h: f32) -> f32 {
        self.draw.panel(x, y, w, h).draw(self.assets);
        y
    }

    /// Draw a panel with a title bar.
    ///
    /// Returns the content area y-offset (below the title bar).
    pub fn panel_titled(&mut self, x: f32, y: f32, w: f32, h: f32, title: &str) -> f32 {
        let font = self
            .default_font
            .expect("Panel with title requires default font. Call ctx.default_font() in setup.");
        self.draw
            .panel(x, y, w, h)
            .title(title, font)
            .draw(self.assets);
        y + 22.0 // Title bar height
    }

    /// Draw a 3D mesh at the given transform with the specified color.
    ///
    /// Requires `ctx.enable_mesh_rendering()` to be called during setup.
    pub fn draw_mesh(&mut self, mesh_index: usize, transform: Transform, color: Color) {
        self.mesh_queue
            .borrow_mut()
            .draw(mesh_index, transform, color);
    }

    /// Draw a 3D mesh with default white color.
    pub fn draw_mesh_white(&mut self, mesh_index: usize, transform: Transform) {
        self.draw_mesh(mesh_index, transform, Color::WHITE);
    }

    /// Draw a textured 3D mesh at the given transform with the specified color tint.
    ///
    /// The color acts as a tint/multiplier for the texture color.
    /// Use `Color::WHITE` for no tinting.
    pub fn draw_mesh_textured(
        &mut self,
        mesh_index: usize,
        transform: Transform,
        color: Color,
        texture_index: usize,
    ) {
        self.mesh_queue
            .borrow_mut()
            .draw_textured(mesh_index, transform, color, texture_index);
    }

    /// Draw a textured 3D mesh with default white color (no tint).
    pub fn draw_mesh_textured_white(
        &mut self,
        mesh_index: usize,
        transform: Transform,
        texture_index: usize,
    ) {
        self.draw_mesh_textured(mesh_index, transform, Color::WHITE, texture_index);
    }
}

/// Configuration for the app window.
pub struct AppConfig {
    pub title: String,
    pub width: u32,
    pub height: u32,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            title: "Hoplite".to_string(),
            width: 800,
            height: 600,
        }
    }
}

impl AppConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }
}

/// Run a Hoplite application with setup and frame closures.
///
/// # Example
/// ```ignore
/// hoplite::run(|ctx| {
///     ctx.default_font(16.0);
///     ctx.effect_world(include_str!("shader.wgsl"));
///
///     move |frame| {
///         frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
///     }
/// });
/// ```
pub fn run<S, F>(setup: S)
where
    S: FnOnce(&mut SetupContext) -> F + 'static,
    F: FnMut(&mut Frame) + 'static,
{
    run_with_config(AppConfig::default(), setup);
}

/// Run a Hoplite application with custom configuration.
///
/// # Example
/// ```ignore
/// hoplite::run_with_config(
///     AppConfig::new().title("Black Hole").size(1280, 720),
///     |ctx| {
///         ctx.default_font(16.0);
///         ctx.effect_world(include_str!("scene.wgsl"))
///            .post_process_world(include_str!("lensing.wgsl"));
///
///         move |frame| {
///             frame.text(10.0, 10.0, "Hello!");
///         }
///     }
/// );
/// ```
pub fn run_with_config<S, F>(config: AppConfig, setup: S)
where
    S: FnOnce(&mut SetupContext) -> F + 'static,
    F: FnMut(&mut Frame) + 'static,
{
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = HopliteApp::Pending {
        config,
        setup: Some(Box::new(move |gpu, assets, mesh_queue| {
            let mut default_font = None;
            let mut graph_builder = None;

            let mut ctx = SetupContext {
                gpu,
                assets,
                default_font: &mut default_font,
                graph_builder: &mut graph_builder,
                mesh_queue,
            };

            let frame_fn = setup(&mut ctx);

            (
                Box::new(frame_fn) as Box<dyn FnMut(&mut Frame)>,
                default_font,
                graph_builder,
            )
        })),
    };

    event_loop.run_app(&mut app).unwrap();
}

type SetupFn = Box<
    dyn FnOnce(
        &GpuContext,
        &mut Assets,
        &Rc<RefCell<MeshQueue>>,
    ) -> (
        Box<dyn FnMut(&mut Frame)>,
        Option<FontId>,
        Option<RenderGraph>,
    ),
>;

enum HopliteApp {
    Pending {
        config: AppConfig,
        setup: Option<SetupFn>,
    },
    Running {
        window: Arc<Window>,
        gpu: GpuContext,
        assets: Assets,
        draw_2d: Draw2d,
        camera: Camera,
        input: Input,
        frame_fn: Box<dyn FnMut(&mut Frame)>,
        default_font: Option<FontId>,
        render_graph: Option<RenderGraph>,
        mesh_queue: Rc<RefCell<MeshQueue>>,
        start_time: Instant,
        last_frame: Instant,
    },
}

impl ApplicationHandler for HopliteApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if let HopliteApp::Pending { config, setup } = self {
            let window_attrs = WindowAttributes::default()
                .with_title(&config.title)
                .with_inner_size(winit::dpi::LogicalSize::new(config.width, config.height));

            let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
            let gpu = GpuContext::new(window.clone());
            let mut assets = Assets::new(&gpu);
            let draw_2d = Draw2d::new(&gpu);

            // Create shared mesh queue
            let mesh_queue = Rc::new(RefCell::new(MeshQueue::new()));

            let setup_fn = setup.take().unwrap();
            let (frame_fn, default_font, render_graph) = setup_fn(&gpu, &mut assets, &mesh_queue);

            *self = HopliteApp::Running {
                window,
                gpu,
                assets,
                draw_2d,
                camera: Camera::new(),
                input: Input::new(),
                frame_fn,
                default_font,
                render_graph,
                mesh_queue,
                start_time: Instant::now(),
                last_frame: Instant::now(),
            };
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let HopliteApp::Running {
            window,
            gpu,
            assets,
            draw_2d,
            camera,
            input,
            frame_fn,
            default_font,
            render_graph,
            mesh_queue,
            start_time,
            last_frame,
        } = self
        else {
            return;
        };

        input.handle_event(&event);

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let time = start_time.elapsed().as_secs_f32();
                let dt = now.duration_since(*last_frame).as_secs_f32();
                *last_frame = now;

                // Clear draw_2d for new frame and update font bind groups
                draw_2d.clear();
                draw_2d.update_font_bind_groups(gpu, assets);

                // Clear mesh queue for new frame
                mesh_queue.borrow_mut().clear_queue();

                // Create frame context
                let mut frame = Frame {
                    gpu,
                    assets,
                    draw: draw_2d,
                    camera,
                    input,
                    time,
                    dt,
                    default_font: *default_font,
                    mesh_queue: Rc::clone(mesh_queue),
                };

                // Run user's frame function
                frame_fn(&mut frame);

                // Execute render graph if present, otherwise just render UI
                if let Some(graph) = render_graph {
                    graph.execute_with_ui(gpu, time, camera, |gpu, pass| {
                        draw_2d.render(gpu, pass, assets);
                    });
                } else {
                    // No render graph - just render 2D content to screen
                    render_2d_only(gpu, draw_2d, assets);
                }

                input.begin_frame();
                window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Render only 2D content when no render graph is configured.
fn render_2d_only(gpu: &GpuContext, draw_2d: &Draw2d, assets: &Assets) {
    let output = gpu.surface.get_current_texture().unwrap();
    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("2D Only Encoder"),
        });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("2D Only Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        draw_2d.render(gpu, &mut render_pass, assets);
    }

    gpu.queue.submit(std::iter::once(encoder.finish()));
    output.present();
}
