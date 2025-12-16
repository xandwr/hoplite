struct Uniforms {
    resolution: vec2f,
    time: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    // Full-screen triangle: vertices at (-1,-1), (3,-1), (-1,3)
    // This single triangle covers the entire [-1,1] clip space
    let x = f32(i32(i % 2u) * 4 - 1);
    let y = f32(i32(i / 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = pos.xy / u.resolution;

    // Simple animated gradient to prove it works
    let color = vec3f(
        uv.x,
        uv.y,
        0.5 + 0.5 * sin(u.time)
    );

    return vec4f(color, 1.0);
}
