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
use crate::input::Input;
use crate::post_process::{PostProcessPass, WorldPostProcessPass};
use crate::render_graph::{EffectNode, PostProcessNode, RenderGraph, WorldPostProcessNode};

/// Context provided during app setup.
pub struct SetupContext<'a> {
    pub gpu: &'a GpuContext,
    pub assets: &'a mut Assets,
    default_font: &'a mut Option<FontId>,
    graph_builder: &'a mut Option<RenderGraph>,
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

    fn add_node<N: crate::render_graph::RenderNode + 'static>(&mut self, node: N) {
        if self.graph_builder.is_none() {
            *self.graph_builder = Some(RenderGraph::builder().node(node).build(self.gpu));
        } else {
            // For now, we rebuild - could optimize later
            let old = self.graph_builder.take().unwrap();
            *self.graph_builder = Some(old.with_node(node, self.gpu));
        }
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
        setup: Some(Box::new(move |gpu, assets| {
            let mut default_font = None;
            let mut graph_builder = None;

            let mut ctx = SetupContext {
                gpu,
                assets,
                default_font: &mut default_font,
                graph_builder: &mut graph_builder,
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

            let setup_fn = setup.take().unwrap();
            let (frame_fn, default_font, render_graph) = setup_fn(&gpu, &mut assets);

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
