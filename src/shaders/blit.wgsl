// Simple blit shader - copies input texture to output

@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle (oversized to cover screen)
    var positions = array<vec2f, 3>(
        vec2f(-1.0, -1.0),
        vec2f(3.0, -1.0),
        vec2f(-1.0, 3.0)
    );
    let pos = positions[vertex_index];

    var out: VertexOutput;
    out.position = vec4f(pos, 0.0, 1.0);
    out.uv = vec2f((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(input_texture, input_sampler, in.uv);
}
