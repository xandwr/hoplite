use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::assets::Assets;
use crate::camera::Camera;
use crate::draw2d::Draw2d;
use crate::gpu::GpuContext;
use crate::input::Input;

/// Context provided during app setup.
pub struct SetupContext<'a> {
    pub gpu: &'a GpuContext,
    pub assets: &'a mut Assets,
}

/// Context provided each frame for rendering.
pub struct Frame<'a> {
    /// GPU context for rendering.
    pub gpu: &'a GpuContext,
    /// Asset manager for loading fonts, textures, etc.
    pub assets: &'a Assets,
    /// 2D drawing API for sprites and text.
    pub draw_2d: &'a mut Draw2d,
    /// Current camera state.
    pub camera: &'a mut Camera,
    /// Input state for this frame.
    pub input: &'a Input,
    /// Total elapsed time in seconds.
    pub time: f32,
    /// Delta time since last frame in seconds.
    pub dt: f32,
}

impl Frame<'_> {
    /// Current frames per second.
    pub fn fps(&self) -> f32 {
        if self.dt > 0.0 { 1.0 / self.dt } else { 0.0 }
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
///     let effect = EffectPass::new_world(ctx.gpu, include_str!("shader.wgsl"));
///     let graph = RenderGraph::builder()
///         .node(EffectNode::new(effect))
///         .build(ctx.gpu);
///
///     move |frame| {
///         frame.camera.position = [frame.time.cos() * 3.0, 1.0, frame.time.sin() * 3.0];
///         graph.execute(frame.gpu, frame.time, frame.camera);
///     }
/// });
/// ```
pub fn run<S, F>(setup: S)
where
    S: FnOnce(SetupContext) -> F + 'static,
    F: FnMut(Frame) + 'static,
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
///         // setup...
///         move |frame| {
///             // render...
///         }
///     }
/// );
/// ```
pub fn run_with_config<S, F>(config: AppConfig, setup: S)
where
    S: FnOnce(SetupContext) -> F + 'static,
    F: FnMut(Frame) + 'static,
{
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = HopliteApp::Pending {
        config,
        setup: Some(Box::new(move |gpu, assets| {
            Box::new(setup(SetupContext { gpu, assets }))
        })),
    };

    event_loop.run_app(&mut app).unwrap();
}

type SetupFn = Box<dyn FnOnce(&GpuContext, &mut Assets) -> Box<dyn FnMut(Frame)>>;

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
        frame_fn: Box<dyn FnMut(Frame)>,
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
            let frame_fn = setup_fn(&gpu, &mut assets);

            *self = HopliteApp::Running {
                window,
                gpu,
                assets,
                draw_2d,
                camera: Camera::new(),
                input: Input::new(),
                frame_fn,
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

                let frame = Frame {
                    gpu,
                    assets,
                    draw_2d,
                    camera,
                    input,
                    time,
                    dt,
                };

                frame_fn(frame);
                input.begin_frame();
                window.request_redraw();
            }
            _ => {}
        }
    }
}
