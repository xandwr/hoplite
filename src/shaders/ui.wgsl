// UI shader for 2D screenspace rendering
// Renders quads with vertex colors and alpha blending

struct Uniforms {
    resolution: vec2f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec4f,
}

@vertex
fn vs(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Convert pixel coordinates to clip space [-1, 1]
    // Y is flipped: 0 at top, resolution.y at bottom
    let clip_x = (in.position.x / u.resolution.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (in.position.y / u.resolution.y) * 2.0;

    out.position = vec4f(clip_x, clip_y, 0.0, 1.0);
    out.color = in.color;

    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}
