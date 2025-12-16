use hoplite::{
    AppConfig, EffectNode, EffectPass, OrbitCamera, OrbitMode, RenderGraph, WorldPostProcessNode,
    WorldPostProcessPass, run_with_config,
};

fn main() {
    run_with_config(AppConfig::new().title("Black Hole"), |ctx| {
        // Scene pass: render the black hole and nebula background
        let scene = EffectPass::new_world(ctx.gpu, include_str!("shaders/black_hole.wgsl"));

        // Post-process pass: apply gravitational lensing (world-space aware)
        let lensing =
            WorldPostProcessPass::new(ctx.gpu, include_str!("shaders/gravitational_lensing.wgsl"));

        let mut graph = RenderGraph::builder()
            .node(EffectNode::new(scene))
            .node(WorldPostProcessNode::new(lensing))
            .build(ctx.gpu);

        // Auto-rotate mode: slowly orbit around the black hole
        // Switch to OrbitMode::Interactive for mouse control
        let mut orbit = OrbitCamera::new()
            .target(0.0, 0.0, 0.0)
            .distance(6.0)
            .elevation(0.3)
            .fov(90.0)
            .mode(OrbitMode::AutoRotate { speed: 0.15 });

        move |frame| {
            orbit.update(frame.input, frame.dt);
            *frame.camera = orbit.camera();

            graph.execute(frame.gpu, frame.time, frame.camera);
        }
    });
}
