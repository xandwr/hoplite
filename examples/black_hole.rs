use hoplite::{
    AppConfig, Color, EffectNode, EffectPass, OrbitCamera, OrbitMode, RenderGraph,
    WorldPostProcessNode, WorldPostProcessPass, run_with_config,
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

        // Load default font for debug UI
        let font = ctx.assets.default_font(16.0);

        // Auto-rotate mode: slowly orbit around the black hole
        // Switch to OrbitMode::Interactive for mouse control
        let mut orbit = OrbitCamera::new()
            .target(0.0, 0.0, 0.0)
            .distance(32.0)
            .elevation(0.3)
            .fov(80.0)
            .mode(OrbitMode::Interactive);

        move |frame| {
            orbit.update(frame.input, frame.dt);
            *frame.camera = orbit.camera();

            // Draw debug overlay using Draw2d
            let panel_x = 10.0;
            let panel_y = 10.0;
            let panel_w = 180.0;
            let panel_h = 80.0;

            // Panel background
            frame.draw_2d.rect(
                panel_x,
                panel_y,
                panel_w,
                panel_h,
                Color::rgba(0.1, 0.1, 0.1, 0.85),
            );

            // Panel border
            frame.draw_2d.rect(
                panel_x,
                panel_y,
                panel_w,
                1.0,
                Color::rgba(0.4, 0.4, 0.4, 1.0),
            );
            frame.draw_2d.rect(
                panel_x,
                panel_y + panel_h - 1.0,
                panel_w,
                1.0,
                Color::rgba(0.4, 0.4, 0.4, 1.0),
            );
            frame.draw_2d.rect(
                panel_x,
                panel_y,
                1.0,
                panel_h,
                Color::rgba(0.4, 0.4, 0.4, 1.0),
            );
            frame.draw_2d.rect(
                panel_x + panel_w - 1.0,
                panel_y,
                1.0,
                panel_h,
                Color::rgba(0.4, 0.4, 0.4, 1.0),
            );

            // Title bar
            frame.draw_2d.rect(
                panel_x,
                panel_y,
                panel_w,
                22.0,
                Color::rgba(0.15, 0.15, 0.15, 0.95),
            );
            frame.draw_2d.text(
                frame.assets,
                font,
                panel_x + 8.0,
                panel_y + 4.0,
                "Debug Overlay",
                Color::WHITE,
            );

            // Stats
            let fps = frame.fps();
            frame.draw_2d.text(
                frame.assets,
                font,
                panel_x + 8.0,
                panel_y + 30.0,
                &format!("FPS: {:.1}", fps),
                Color::WHITE,
            );
            frame.draw_2d.text(
                frame.assets,
                font,
                panel_x + 8.0,
                panel_y + 50.0,
                &format!("Time: {:.1}s", frame.time),
                Color::rgba(0.7, 0.7, 0.7, 1.0),
            );

            // Execute render graph with Draw2d overlay
            graph.execute_with_ui(frame.gpu, frame.time, frame.camera, |gpu, pass| {
                frame.draw_2d.render(gpu, pass, frame.assets);
            });
        }
    });
}
