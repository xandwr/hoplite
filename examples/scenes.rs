//! Scene management demo - demonstrates switching between independent scenes.
//!
//! This example shows:
//! - Two separate scenes with independent cameras
//! - Per-scene render pipelines (different post-processing per scene)
//! - Instant and animated transitions (fade-to-black, crossfade)
//!
//! Controls:
//! - ENTER: Switch to cockpit scene
//! - ESCAPE: Switch back to external scene
//! - 1-3: Change transition type (instant, fade, crossfade)

use hoplite::{
    AppConfig, Color, KeyCode, OrbitCamera, OrbitMode, Quat, Transform, Vec3,
    run_with_scenes_config, scene::Transition,
};

fn main() {
    run_with_scenes_config(
        AppConfig::new()
            .title("Scene Management Demo")
            .size(1280, 720),
        |ctx| {
            ctx.default_font(18.0);

            // Shared assets - available to all scenes
            let cube = ctx.mesh_cube();
            let plane = ctx.mesh_plane(30.0);

            // Track current transition type
            let transition_type = std::rc::Rc::new(std::cell::RefCell::new(TransitionType::Fade));
            let transition_type_1 = transition_type.clone();
            let transition_type_2 = transition_type.clone();

            // =========================================================
            // Scene 1: External view (ship floating in space)
            // =========================================================
            ctx.scene("external", |scene| {
                scene.background_color(Color::rgb(0.02, 0.02, 0.05));
                scene.enable_mesh_rendering();

                let mut orbit = OrbitCamera::new()
                    .target(Vec3::ZERO)
                    .distance(8.0)
                    .elevation(0.3)
                    .fov(75.0)
                    .mode(OrbitMode::AutoRotate { speed: 0.2 });

                let cube = cube;
                let transition_type = transition_type_1;

                move |frame| {
                    orbit.update(frame.input, frame.dt);
                    frame.set_camera(orbit.camera());

                    // Update transition type based on key presses
                    if frame.input.key_pressed(KeyCode::Digit1) {
                        *transition_type.borrow_mut() = TransitionType::Instant;
                    }
                    if frame.input.key_pressed(KeyCode::Digit2) {
                        *transition_type.borrow_mut() = TransitionType::Fade;
                    }
                    if frame.input.key_pressed(KeyCode::Digit3) {
                        *transition_type.borrow_mut() = TransitionType::Crossfade;
                    }

                    // Capture time before building meshes
                    let time = frame.time;
                    let ship_rotation = Quat::from_rotation_y(time * 0.1);

                    // Draw a simple "ship" - main body
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .scale(Vec3::new(2.0, 0.5, 3.0))
                                .rotation(ship_rotation),
                        )
                        .color(Color::rgb(0.4, 0.4, 0.5))
                        .draw();

                    // Wings
                    for x in [-1.5, 1.5] {
                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(Vec3::new(x, 0.0, 0.0))
                                    .scale(Vec3::new(1.5, 0.1, 1.0))
                                    .rotation(ship_rotation),
                            )
                            .color(Color::rgb(0.3, 0.3, 0.4))
                            .draw();
                    }

                    // Cockpit (blue tinted)
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(0.0, 0.3, 0.8))
                                .scale(Vec3::new(0.6, 0.3, 0.4))
                                .rotation(ship_rotation),
                        )
                        .color(Color::rgb(0.2, 0.3, 0.5))
                        .draw();

                    // Engine glow
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(0.0, 0.0, -1.6))
                                .scale(Vec3::new(0.4, 0.3, 0.2))
                                .rotation(ship_rotation),
                        )
                        .color(Color::rgb(0.8, 0.4, 0.2))
                        .draw();

                    // Some "stars" (small cubes in the distance)
                    let star_positions = [
                        Vec3::new(10.0, 5.0, -15.0),
                        Vec3::new(-12.0, 8.0, -20.0),
                        Vec3::new(8.0, -3.0, -18.0),
                        Vec3::new(-6.0, 10.0, -25.0),
                        Vec3::new(15.0, -5.0, -22.0),
                    ];
                    for pos in star_positions {
                        frame
                            .mesh(cube)
                            .transform(Transform::new().position(pos).scale(Vec3::splat(0.1)))
                            .color(Color::rgb(0.9, 0.9, 1.0))
                            .draw();
                    }

                    // UI
                    frame.text(10.0, 10.0, "EXTERNAL VIEW");
                    frame.text(10.0, 35.0, "Press ENTER to enter cockpit");
                    frame.text_color(
                        10.0,
                        60.0,
                        &format!(
                            "Transition: {} (1-3 to change)",
                            transition_type.borrow().name()
                        ),
                        Color::rgb(0.6, 0.6, 0.6),
                    );

                    // Switch to cockpit on Enter
                    if frame.input.key_pressed(KeyCode::Enter) {
                        let transition = transition_type.borrow().to_transition();
                        frame.switch_to_with("cockpit", transition);
                    }
                }
            });

            // =========================================================
            // Scene 2: Cockpit interior view
            // =========================================================
            ctx.scene("cockpit", |scene| {
                scene.background_color(Color::rgb(0.05, 0.03, 0.03));
                scene.enable_mesh_rendering();

                let cube = cube;
                let plane = plane;
                let transition_type = transition_type_2;
                let mut look_yaw = 0.0_f32;

                move |frame| {
                    // Simple look around with arrow keys
                    if frame.input.key_down(KeyCode::ArrowLeft) {
                        look_yaw += 1.5 * frame.dt;
                    }
                    if frame.input.key_down(KeyCode::ArrowRight) {
                        look_yaw -= 1.5 * frame.dt;
                    }

                    // Build camera with current look direction
                    let camera = hoplite::Camera::new()
                        .at(Vec3::new(0.0, 1.0, 0.0))
                        .looking_at(Vec3::new(look_yaw.sin(), 1.0, -look_yaw.cos()))
                        .with_fov(90.0);
                    frame.set_camera(camera);

                    // Update transition type based on key presses
                    if frame.input.key_pressed(KeyCode::Digit1) {
                        *transition_type.borrow_mut() = TransitionType::Instant;
                    }
                    if frame.input.key_pressed(KeyCode::Digit2) {
                        *transition_type.borrow_mut() = TransitionType::Fade;
                    }
                    if frame.input.key_pressed(KeyCode::Digit3) {
                        *transition_type.borrow_mut() = TransitionType::Crossfade;
                    }

                    // Capture time for animations
                    let time = frame.time;

                    // Draw cockpit interior
                    // Floor
                    frame
                        .mesh(plane)
                        .transform(Transform::new().scale(Vec3::splat(0.3)))
                        .color(Color::rgb(0.15, 0.12, 0.12))
                        .draw();

                    // Console in front
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(0.0, 0.6, -1.5))
                                .scale(Vec3::new(2.0, 0.8, 0.3)),
                        )
                        .color(Color::rgb(0.2, 0.2, 0.25))
                        .draw();

                    // Display screens on console
                    let screen_positions = [-0.6, 0.0, 0.6];
                    for (i, x) in screen_positions.iter().enumerate() {
                        let glow = (time * 2.0 + i as f32).sin() * 0.1 + 0.5;
                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(Vec3::new(*x, 0.9, -1.3))
                                    .scale(Vec3::new(0.4, 0.3, 0.05)),
                            )
                            .color(Color::rgb(0.1, glow * 0.5, glow))
                            .draw();
                    }

                    // Side panels
                    for x in [-1.2, 1.2] {
                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(Vec3::new(x, 1.0, -0.5))
                                    .scale(Vec3::new(0.1, 1.5, 2.0)),
                            )
                            .color(Color::rgb(0.12, 0.1, 0.1))
                            .draw();
                    }

                    // Ceiling
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(0.0, 2.2, -0.5))
                                .scale(Vec3::new(2.5, 0.1, 3.0)),
                        )
                        .color(Color::rgb(0.08, 0.08, 0.1))
                        .draw();

                    // Window (brighter area showing "space")
                    frame
                        .mesh(cube)
                        .transform(
                            Transform::new()
                                .position(Vec3::new(0.0, 1.5, -2.0))
                                .scale(Vec3::new(1.8, 0.8, 0.02)),
                        )
                        .color(Color::rgb(0.02, 0.02, 0.08))
                        .draw();

                    // Blinking warning lights
                    let blink = if (time * 3.0).sin() > 0.0 { 0.8 } else { 0.2 };
                    for x in [-0.9, 0.9] {
                        frame
                            .mesh(cube)
                            .transform(
                                Transform::new()
                                    .position(Vec3::new(x, 1.8, -1.4))
                                    .scale(Vec3::splat(0.08)),
                            )
                            .color(Color::rgb(blink, blink * 0.2, 0.0))
                            .draw();
                    }

                    // UI
                    frame.text(10.0, 10.0, "COCKPIT VIEW");
                    frame.text(10.0, 35.0, "Arrow keys to look around");
                    frame.text(10.0, 60.0, "Press ESCAPE for external view");
                    frame.text_color(
                        10.0,
                        85.0,
                        &format!(
                            "Transition: {} (1-3 to change)",
                            transition_type.borrow().name()
                        ),
                        Color::rgb(0.6, 0.6, 0.6),
                    );

                    // Switch back to external on Escape
                    if frame.input.key_pressed(KeyCode::Escape) {
                        let transition = transition_type.borrow().to_transition();
                        frame.switch_to_with("external", transition);
                    }
                }
            });

            // Start in external view
            ctx.start_scene("external");
        },
    );
}

#[derive(Clone, Copy)]
enum TransitionType {
    Instant,
    Fade,
    Crossfade,
}

impl TransitionType {
    fn name(&self) -> &'static str {
        match self {
            TransitionType::Instant => "Instant",
            TransitionType::Fade => "Fade to Black",
            TransitionType::Crossfade => "Crossfade",
        }
    }

    fn to_transition(&self) -> Transition {
        match self {
            TransitionType::Instant => Transition::instant(),
            TransitionType::Fade => Transition::fade_to_black(0.5),
            TransitionType::Crossfade => Transition::crossfade(0.8),
        }
    }
}
