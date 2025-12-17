//! STL Model Loading Demo - demonstrates loading 3D models from STL files.
//!
//! This example shows the fluent geometry loading API:
//! - Load STL files with `ctx.load("path.stl")`
//! - Center, normalize, and scale models with chained methods
//! - Orbit around the model with mouse drag or auto-rotate

use hoplite::{AppConfig, Color, OrbitCamera, OrbitMode, Quat, Transform, Vec3, run_with_config};

fn main() {
    run_with_config(
        AppConfig::new().title("STL Loader Demo").size(1280, 720),
        |ctx| {
            ctx.default_font(18.0);
            ctx.background_color(Color::rgb(0.08, 0.08, 0.12));
            ctx.enable_mesh_rendering();

            // Load the chess piece STL with the fluent API
            // - centered() moves the bounding box center to origin
            // - normalized() scales to fit in a unit cube
            // - scaled() then scales to desired size
            let chess_piece = ctx
                .load("examples/assets/models/chess.stl")
                .centered()
                .upright()
                .normalized()
                .scaled(2.0)
                .unwrap();

            // Create some primitive meshes for the scene
            let plane = ctx.mesh_plane(10.0);
            let cube = ctx.mesh_cube();

            // Orbit camera for viewing the model
            let mut orbit = OrbitCamera::new()
                .target(Vec3::new(0.0, 1.0, 0.0))
                .distance(5.0)
                .elevation(0.4)
                .fov(60.0)
                .mode(OrbitMode::AutoRotate { speed: 0.3 });

            let mut time = 0.0f32;

            move |frame| {
                time += frame.dt;

                // Update camera
                orbit.update(frame.input, frame.dt);
                frame.set_camera(orbit.camera());

                // Draw the floor
                frame
                    .mesh(plane)
                    .transform(Transform::new())
                    .color(Color::rgb(0.15, 0.15, 0.18))
                    .draw();

                // Draw the chess piece on a pedestal
                // Pedestal base
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(Vec3::new(0.0, 0.15, 0.0))
                            .scale(Vec3::new(1.5, 0.3, 1.5)),
                    )
                    .color(Color::rgb(0.2, 0.2, 0.25))
                    .draw();

                // The chess piece - positioned on top of pedestal
                // Slowly bobbing up and down
                let bob = (time * 1.5).sin() * 0.05;
                frame
                    .mesh(chess_piece)
                    .transform(
                        Transform::new()
                            .position(Vec3::new(0.0, 1.3 + bob, 0.0))
                            .rotation(Quat::from_rotation_y(time * 0.2)),
                    )
                    .color(Color::rgb(0.9, 0.85, 0.7))
                    .draw();

                // Some decorative elements around the scene
                let decoration_color = Color::rgb(0.3, 0.25, 0.35);
                for i in 0..4 {
                    let angle = (i as f32) * std::f32::consts::FRAC_PI_2 + time * 0.1;
                    let radius = 3.0;
                    let x = angle.cos() * radius;
                    let z = angle.sin() * radius;
                    let height = 0.3 + (time + i as f32).sin().abs() * 0.2;

                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(x, height, z))
                                .scale(Vec3::new(0.4, height * 2.0, 0.4))
                                .rotation(Quat::from_rotation_y(time + i as f32)),
                        )
                        .color(decoration_color)
                        .draw();
                }

                // UI text
                frame.text(10.0, 10.0, "STL Model Loading Demo");
                frame.text_color(
                    10.0,
                    35.0,
                    "Drag mouse to orbit, scroll to zoom",
                    Color::rgb(0.6, 0.6, 0.6),
                );
                frame.text_color(
                    10.0,
                    55.0,
                    &format!("FPS: {:.0}", frame.fps()),
                    Color::rgb(0.5, 0.5, 0.5),
                );
            }
        },
    );
}
