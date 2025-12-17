//! Seated camera demo - demonstrates FreelookCamera in seated mode.
//!
//! The player is seated in a chair and can look around within a constrained
//! view range. Press Space to stand up and walk around, Space again to sit back down.
//! Click to capture the mouse, Escape to release it.

use hoplite::{
    AppConfig, Color, FreelookCamera, FreelookMode, KeyCode, MouseButton, Quat, Transform, Vec3,
    run_with_config,
};

fn main() {
    run_with_config(
        AppConfig::new().title("Seated Camera Demo").size(1280, 720),
        |ctx| {
            ctx.default_font(18.0);
            ctx.background_color(Color::rgb(0.05, 0.05, 0.08));
            ctx.enable_mesh_rendering();

            // Create meshes for the scene
            let cube = ctx.mesh_cube();
            let plane = ctx.mesh_plane(20.0);

            // Chair position in the room
            let chair_pos = Vec3::new(0.0, 0.0, 0.0);
            let seat_height = 0.5;
            let eye_height = 1.2; // Seated eye level

            // Create camera in seated mode
            // Player sits at chair position, can look ±60° horizontally and ±30° vertically
            // Facing toward -Z (forward in the room)
            let mut camera = FreelookCamera::new()
                .position(chair_pos + Vec3::Y * eye_height)
                .sensitivity(0.003)
                .speed(3.0)
                .fov(75.0);

            // Start seated, facing the table (toward -Z)
            camera.seat(
                FreelookMode::seated(chair_pos + Vec3::Y * eye_height)
                    .yaw_range(-60.0_f32.to_radians(), 60.0_f32.to_radians())
                    .pitch_range(-30.0_f32.to_radians(), 45.0_f32.to_radians())
                    .facing(Vec3::NEG_Z),
            );

            // Track cursor capture state
            let mut cursor_captured = false;

            move |frame| {
                // Capture cursor on click
                if frame.input.mouse_pressed(MouseButton::Left) && !cursor_captured {
                    frame.capture_cursor();
                    cursor_captured = true;
                }

                // Release cursor on Escape
                if frame.input.key_pressed(KeyCode::Escape) && cursor_captured {
                    frame.release_cursor();
                    cursor_captured = false;
                }

                // Toggle seated/standing with Space (only when cursor is captured)
                if frame.input.key_pressed(KeyCode::Space) && cursor_captured {
                    if camera.is_seated() {
                        camera.unseat();
                    } else {
                        // Sit back down
                        camera.seat(
                            FreelookMode::seated(chair_pos + Vec3::Y * eye_height)
                                .yaw_range(-60.0_f32.to_radians(), 60.0_f32.to_radians())
                                .pitch_range(-30.0_f32.to_radians(), 45.0_f32.to_radians())
                                .facing(Vec3::NEG_Z),
                        );
                    }
                }

                // Update camera only when cursor is captured
                if cursor_captured {
                    camera.update(frame.input, frame.dt);
                }
                frame.set_camera(camera.camera());

                // === Draw the room ===

                // Floor
                frame
                    .mesh(plane)
                    .transform(Transform::new())
                    .color(Color::rgb(0.15, 0.15, 0.18))
                    .draw();

                // === Draw the chair ===
                let chair_color = Color::rgb(0.4, 0.25, 0.1);

                // Seat (flat cube)
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(chair_pos + Vec3::Y * seat_height)
                            .scale(Vec3::new(0.6, 0.1, 0.6)),
                    )
                    .color(chair_color)
                    .draw();

                // Back rest
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(chair_pos + Vec3::new(0.0, seat_height + 0.4, 0.25))
                            .scale(Vec3::new(0.6, 0.7, 0.08)),
                    )
                    .color(chair_color)
                    .draw();

                // Chair legs (4 corners)
                let leg_positions = [
                    Vec3::new(-0.25, seat_height * 0.5, -0.25),
                    Vec3::new(0.25, seat_height * 0.5, -0.25),
                    Vec3::new(-0.25, seat_height * 0.5, 0.25),
                    Vec3::new(0.25, seat_height * 0.5, 0.25),
                ];
                for leg_offset in leg_positions {
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(chair_pos + leg_offset)
                                .scale(Vec3::new(0.08, seat_height, 0.08)),
                        )
                        .color(chair_color)
                        .draw();
                }

                // === Objects to look at ===

                // Table in front
                let table_pos = Vec3::new(0.0, 0.0, -2.0);
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(table_pos + Vec3::Y * 0.4)
                            .scale(Vec3::new(1.5, 0.08, 0.8)),
                    )
                    .color(Color::rgb(0.3, 0.2, 0.1))
                    .draw();

                // Table legs
                for x in [-0.6, 0.6] {
                    for z in [-0.3, 0.3] {
                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(table_pos + Vec3::new(x, 0.2, z))
                                    .scale(Vec3::new(0.08, 0.4, 0.08)),
                            )
                            .color(Color::rgb(0.25, 0.15, 0.08))
                            .draw();
                    }
                }

                // Objects on table
                // Red cube (mug?)
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(table_pos + Vec3::new(-0.3, 0.55, 0.0))
                            .scale(Vec3::splat(0.15)),
                    )
                    .color(Color::rgb(0.8, 0.2, 0.2))
                    .draw();

                // Blue cube (book?)
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(table_pos + Vec3::new(0.2, 0.5, -0.1))
                            .scale(Vec3::new(0.3, 0.05, 0.2))
                            .rotation(Quat::from_rotation_y(0.2)),
                    )
                    .color(Color::rgb(0.2, 0.3, 0.7))
                    .draw();

                // Walls (to give sense of space)
                // Back wall
                frame
                    .mesh(cube)
                    .transform(
                        Transform::new()
                            .position(Vec3::new(0.0, 2.0, -5.0))
                            .scale(Vec3::new(10.0, 4.0, 0.1)),
                    )
                    .color(Color::rgb(0.12, 0.12, 0.15))
                    .draw();

                // Side walls
                for x in [-5.0, 5.0] {
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(x, 2.0, 0.0))
                                .scale(Vec3::new(0.1, 4.0, 10.0)),
                        )
                        .color(Color::rgb(0.1, 0.1, 0.12))
                        .draw();
                }

                // Some decorative cubes on the sides (things to look at)
                let decorations = [
                    (Vec3::new(-3.0, 0.5, -3.0), Color::rgb(0.2, 0.6, 0.3), 0.8),
                    (Vec3::new(3.0, 0.3, -2.0), Color::rgb(0.6, 0.5, 0.2), 0.5),
                    (Vec3::new(-2.0, 1.0, -4.0), Color::rgb(0.5, 0.2, 0.5), 0.6),
                    (Vec3::new(2.5, 0.7, -4.0), Color::rgb(0.2, 0.5, 0.6), 0.7),
                ];

                let time = frame.time;
                for (pos, color, size) in decorations {
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(pos)
                                .scale(Vec3::splat(size))
                                .rotation(Quat::from_rotation_y(time * 0.5)),
                        )
                        .color(color)
                        .draw();
                }

                // === UI ===
                if !cursor_captured {
                    frame.text(10.0, 10.0, "Click to capture mouse");
                } else {
                    let mode_text = if camera.is_seated() {
                        "SEATED - Press SPACE to stand"
                    } else {
                        "STANDING - WASD to move, SPACE to sit"
                    };

                    frame.text(10.0, 10.0, mode_text);
                    frame.text(10.0, 35.0, "ESC to release mouse");

                    if camera.is_seated() {
                        frame.text_color(
                            10.0,
                            60.0,
                            "View constrained: +/-60 deg horizontal, -30/+45 deg vertical",
                            Color::rgb(0.6, 0.6, 0.6),
                        );
                    }
                }
            }
        },
    );
}
