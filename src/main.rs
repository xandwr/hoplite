use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use hoplite::{Camera, EffectPass, GpuContext, Vec3};

struct App {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext>,
    effect: Option<EffectPass>,
    camera: Camera,
    start_time: Instant,
}

impl Default for App {
    fn default() -> Self {
        Self {
            window: None,
            gpu: None,
            effect: None,
            camera: Camera::new()
                .at(Vec3::new(0.0, 0.0, 3.0))
                .looking_at(Vec3::ZERO)
                .with_fov(90.0),
            start_time: Instant::now(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let gpu = GpuContext::new(window.clone());

        // World-space effect with camera
        let effect = EffectPass::new_world(&gpu, include_str!("shaders/sphere.wgsl"));

        self.gpu = Some(gpu);
        self.effect = Some(effect);
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
                if let (Some(gpu), Some(effect)) = (&self.gpu, &self.effect) {
                    let time = self.start_time.elapsed().as_secs_f32();

                    // Orbit camera around the sphere
                    self.camera.position = Vec3::new(3.0 * time.cos(), 1.0, 3.0 * time.sin());
                    self.camera = self.camera.looking_at(Vec3::ZERO);

                    let output = gpu.surface.get_current_texture().unwrap();
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());

                    let mut encoder = gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                    {
                        let mut render_pass =
                            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                                label: None,
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

                        effect.render_with_camera(gpu, &mut render_pass, time, &self.camera);
                    }

                    gpu.queue.submit(std::iter::once(encoder.finish()));
                    output.present();
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
