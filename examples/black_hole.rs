use hoplite::{AppConfig, Color, OrbitCamera, OrbitMode, Quat, Transform, Vec3, run_with_config};
use winit::event::MouseButton;

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

        // Create a procedural blocky noise texture (16x16 for that pixelated look)
        let texture = ctx.texture_blocky_noise(16, 42);

        // Create a 2D sprite for the UI (same procedural texture, rendered in 2D layer)
        let sprite = ctx.sprite_blocky_noise(32, 123);

        // Camera: auto-rotate or interactive orbit
        let mut orbit = OrbitCamera::new()
            .target(Vec3::ZERO)
            .distance(32.0)
            .elevation(0.3)
            .fov(80.0)
            .mode(OrbitMode::Interactive);

        // State for toggling cube visibility
        let mut cube_visible = true;

        // Frame loop
        move |frame| {
            orbit.update(frame.input, frame.dt);
            frame.set_camera(orbit.camera());

            // Animate the cubes: hover above the black hole and rotate
            let hover_height = 8.0 + (frame.time * 0.5).sin() * 0.5;

            // Draw a 3x3 grid of textured cubes (if visible)
            if cube_visible {
                let spacing = 12.0;
                for row in 0..3 {
                    for col in 0..3 {
                        let idx = row * 3 + col;
                        // Offset rotation per cube for visual variety
                        let phase = idx as f32 * 0.3;
                        let rotation = Quat::from_euler(
                            glam::EulerRot::YXZ,
                            frame.time * 0.7 + phase,
                            frame.time * 0.5 + phase,
                            frame.time * 0.3 + phase,
                        );

                        let x = (col as f32 - 1.0) * spacing;
                        let z = (row as f32 - 1.0) * spacing;

                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(Vec3::new(x, hover_height, z))
                                    .rotation(rotation)
                                    .uniform_scale(3.0),
                            )
                            .texture(texture)
                            .draw();
                    }
                }
            }

            // Draw debug overlay
            let y = frame.panel_titled(10.0, 10.0, 300.0, 120.0, "Debug Overlay");
            frame.text(18.0, y + 8.0, &format!("FPS: {:.1}", frame.fps()));
            frame.text_color(
                18.0,
                y + 28.0,
                &format!("Time: {:.1}s", frame.time),
                Color::rgba(0.7, 0.7, 0.7, 1.0),
            );

            // Draw the 2D sprite in the bottom-right corner (clickable toggle for cube visibility)
            let sprite_x = frame.width() as f32 - 80.0;
            let sprite_y = frame.height() as f32 - 80.0;
            let sprite_w = 64.0;
            let sprite_h = 64.0;

            // Check for click on sprite
            if frame.input.mouse_pressed(MouseButton::Left) {
                let mouse_pos = frame.input.mouse_position();
                if mouse_pos.x >= sprite_x
                    && mouse_pos.x <= sprite_x + sprite_w
                    && mouse_pos.y >= sprite_y
                    && mouse_pos.y <= sprite_y + sprite_h
                {
                    cube_visible = !cube_visible;
                }
            }

            // Draw sprite with tint based on cube visibility
            let tint = if cube_visible {
                Color::WHITE
            } else {
                Color::rgba(0.4, 0.4, 0.4, 1.0) // Grayed out when cube is hidden
            };
            frame.sprite_scaled_tinted(sprite, sprite_x, sprite_y, sprite_w, sprite_h, tint);

            // Draw label above the sprite
            let label = if cube_visible {
                "Cubes: Shown"
            } else {
                "Cubes: Hidden"
            };
            frame.text(sprite_x - 40.0, sprite_y - 20.0, label);
        }
    });
}
