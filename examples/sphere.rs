//! Raymarched sphere with orbiting camera - demonstrates world-space effects.

use hoplite::{AppConfig, OrbitCamera, OrbitMode, Vec3, run_with_config};

fn main() {
    run_with_config(AppConfig::new().title("Raymarched Sphere"), |ctx| {
        ctx.hot_effect_world("examples/shaders/sphere.wgsl");

        let mut orbit = OrbitCamera::new()
            .target(Vec3::ZERO)
            .distance(3.0)
            .elevation(0.3)
            .fov(90.0)
            .mode(OrbitMode::AutoRotate { speed: 0.5 });

        move |frame| {
            orbit.update(frame.input, frame.dt);
            *frame.camera = orbit.camera();
        }
    });
}
