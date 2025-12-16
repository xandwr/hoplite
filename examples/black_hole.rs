use hoplite::{AppConfig, Color, OrbitCamera, OrbitMode, Quat, Transform, Vec3, run_with_config};

fn main() {
    run_with_config(AppConfig::new().title("Black Hole"), |ctx| {
        // Setup: configure render pipeline and load assets
        ctx.default_font(16.0);

        // Hot-reloadable shaders - edit the files and see changes live!
        ctx.hot_effect_world("examples/shaders/black_hole.wgsl");

        // Enable 3D mesh rendering (after background, before post-process)
        ctx.enable_mesh_rendering();

        // Add post-processing after mesh rendering
        ctx.hot_post_process_world("examples/shaders/gravitational_lensing.wgsl");

        // Create a cube mesh
        let cube = ctx.mesh_cube();

        // Create a procedural Minecraft-style noise texture (16x16 for that blocky look)
        let texture = ctx.texture_minecraft_noise(16, 42);

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

            // Animate the cube: hover above the black hole and rotate
            let hover_height = 8.0 + (frame.time * 0.5).sin() * 0.5;
            let rotation = Quat::from_euler(
                glam::EulerRot::YXZ,
                frame.time * 0.7,
                frame.time * 0.5,
                frame.time * 0.3,
            );

            // Draw the textured cube
            frame.draw_mesh_textured_white(
                cube,
                Transform::new()
                    .position(Vec3::new(0.0, hover_height, 0.0))
                    .rotation(rotation)
                    .uniform_scale(10.0),
                texture,
            );

            // Draw debug overlay
            let y = frame.panel_titled(10.0, 10.0, 220.0, 100.0, "Debug Overlay");
            frame.text(18.0, y + 8.0, &format!("FPS: {:.1}", frame.fps()));
            frame.text_color(
                18.0,
                y + 28.0,
                &format!("Time: {:.1}s", frame.time),
                Color::rgba(0.7, 0.7, 0.7, 1.0),
            );
            frame.text_color(
                18.0,
                y + 48.0,
                "Minecraft cube near black hole",
                Color::rgba(0.5, 0.8, 0.5, 1.0),
            );
        }
    });
}
