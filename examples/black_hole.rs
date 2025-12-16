use hoplite::{AppConfig, Color, OrbitCamera, OrbitMode, Vec3, run_with_config};

fn main() {
    run_with_config(AppConfig::new().title("Black Hole"), |ctx| {
        // Setup: configure render pipeline and load assets
        ctx.default_font(16.0);

        // Hot-reloadable shaders - edit the files and see changes live!
        ctx.hot_effect_world("examples/shaders/black_hole.wgsl")
            .hot_post_process_world("examples/shaders/gravitational_lensing.wgsl");

        // Camera: auto-rotate or interactive orbit
        let mut orbit = OrbitCamera::new()
            .target(Vec3::ZERO)
            .distance(32.0)
            .elevation(0.3)
            .fov(80.0)
            .mode(OrbitMode::Interactive);

        // Frame loop
        move |frame| {
            orbit.update(frame.input, frame.dt);
            *frame.camera = orbit.camera();

            // Draw debug overlay
            let y = frame.panel_titled(10.0, 10.0, 180.0, 80.0, "Debug Overlay");
            frame.text(18.0, y + 8.0, &format!("FPS: {:.1}", frame.fps()));
            frame.text_color(
                18.0,
                y + 28.0,
                &format!("Time: {:.1}s", frame.time),
                Color::rgba(0.7, 0.7, 0.7, 1.0),
            );
        }
    });
}
