use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use hoplite::{Camera, EffectNode, EffectPass, GpuContext, RenderGraph};

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    graph: Option<RenderGraph>,
    camera: Camera,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            gpu: None,
            graph: None,
            camera: Camera::new()
                .at(0.0, 0.0, 5.0)
                .looking_at(0.0, 0.0, 0.0)
                .with_fov(90.0),
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("Black Hole"))
                .unwrap(),
        );

        let gpu = GpuContext::new(window.clone());

        let effect = EffectPass::new_world(&gpu, include_str!("shaders/black_hole.wgsl"));

        let graph = RenderGraph::builder()
            .node(EffectNode::new(effect).with_clear(wgpu::Color::BLACK))
            .build(&gpu);

        self.gpu = Some(gpu);
        self.graph = Some(graph);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Some(gpu) = &mut self.gpu {
                    gpu.resize(size.width, size.height);
                }
            }
            WindowEvent::RedrawRequested => {
                if let (Some(gpu), Some(graph)) = (&self.gpu, &self.graph) {
                    let time = self.start_time.elapsed().as_secs_f32();

                    // Slow orbit around the black hole
                    let orbit_speed = 0.15;
                    let orbit_radius = 6.0;
                    self.camera.position = [
                        orbit_radius * (time * orbit_speed).cos(),
                        2.0 * (time * orbit_speed * 0.5).sin(),
                        orbit_radius * (time * orbit_speed).sin(),
                    ];
                    self.camera = self.camera.looking_at(0.0, 0.0, 0.0);

                    graph.execute(gpu, time, &self.camera);
                }

                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
