// 2D sprite/text rendering shader

struct Uniforms {
    resolution: vec2f,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

// Texture binding (group 1) - only used by textured pipeline
@group(1) @binding(0) var t_atlas: texture_2d<f32>;
@group(1) @binding(1) var s_atlas: sampler;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) uv: vec2f,
    @location(2) color: vec4f,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
}

@vertex
fn vs(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Convert pixel coordinates to clip space [-1, 1]
    // Y is flipped: 0 at top, resolution.y at bottom
    let clip_x = (in.position.x / u.resolution.x) * 2.0 - 1.0;
    let clip_y = 1.0 - (in.position.y / u.resolution.y) * 2.0;

    out.position = vec4f(clip_x, clip_y, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;

    return out;
}

// Fragment shader for colored (non-textured) quads
@fragment
fn fs_colored(in: VertexOutput) -> @location(0) vec4f {
    return in.color;
}

// Fragment shader for textured quads (fonts)
// Uses R8 texture as alpha mask
@fragment
fn fs_textured(in: VertexOutput) -> @location(0) vec4f {
    let alpha = textureSample(t_atlas, s_atlas, in.uv).r;
    return vec4f(in.color.rgb, in.color.a * alpha);
}

// Fragment shader for RGBA sprites
// Samples full color and multiplies by vertex color (tint)
@fragment
fn fs_sprite(in: VertexOutput) -> @location(0) vec4f {
    let tex_color = textureSample(t_atlas, s_atlas, in.uv);
    return tex_color * in.color;
}
