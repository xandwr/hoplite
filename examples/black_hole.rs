use hoplite::{AppConfig, EffectNode, EffectPass, RenderGraph, run_with_config};

fn main() {
    run_with_config(AppConfig::new().title("Black Hole"), |ctx| {
        let effect = EffectPass::new_world(&ctx.gpu, include_str!("shaders/black_hole.wgsl"));

        let graph = RenderGraph::builder()
            .node(EffectNode::new(effect))
            .build(&ctx.gpu);

        move |frame| {
            // Slow orbit around the black hole
            let orbit_speed = 0.15;
            let orbit_radius = 6.0;
            frame.camera.position = [
                orbit_radius * (frame.time * orbit_speed).cos(),
                2.0 * (frame.time * orbit_speed * 0.5).sin(),
                orbit_radius * (frame.time * orbit_speed).sin(),
            ];
            *frame.camera = frame.camera.looking_at(0.0, 0.0, 0.0);

            graph.execute(frame.gpu, frame.time, frame.camera);
        }
    });
}
