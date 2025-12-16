// Black hole with dark blue nebula background

struct Uniforms {
    resolution: vec2f,
    time: f32,
    fov: f32,
    camera_pos: vec3f,
    _pad1: f32,
    camera_forward: vec3f,
    _pad2: f32,
    camera_right: vec3f,
    _pad3: f32,
    camera_up: vec3f,
    aspect: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32(i32(i % 2u) * 4 - 1);
    let y = f32(i32(i / 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// Hash function for noise
fn hash(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// 3D noise
fn noise(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash(i), hash(i + vec3f(1.0, 0.0, 0.0)), u.x),
            mix(hash(i + vec3f(0.0, 1.0, 0.0)), hash(i + vec3f(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash(i + vec3f(0.0, 0.0, 1.0)), hash(i + vec3f(1.0, 0.0, 1.0)), u.x),
            mix(hash(i + vec3f(0.0, 1.0, 1.0)), hash(i + vec3f(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// Fractal Brownian Motion for nebula
fn fbm(p: vec3f) -> f32 {
    var value = 0.0;
    var amplitude = 0.5;
    var frequency = 1.0;
    var pos = p;

    for (var i = 0; i < 5; i++) {
        value += amplitude * noise(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// Nebula color based on direction
fn nebula(rd: vec3f) -> vec3f {
    let p = rd * 3.0 + vec3f(u.time * 0.02);

    // Multiple layers of nebula
    let n1 = fbm(p);
    let n2 = fbm(p * 2.0 + vec3f(100.0));
    let n3 = fbm(p * 0.5 + vec3f(50.0));

    // Dark blue base colors
    let deep_blue = vec3f(0.02, 0.03, 0.08);
    let mid_blue = vec3f(0.05, 0.08, 0.15);
    let bright_blue = vec3f(0.1, 0.15, 0.3);
    let purple_hint = vec3f(0.08, 0.04, 0.12);

    // Mix colors based on noise
    var color = deep_blue;
    color = mix(color, mid_blue, n1 * 0.8);
    color = mix(color, bright_blue, n2 * n2 * 0.5);
    color = mix(color, purple_hint, n3 * 0.3);

    // Add some subtle stars
    let star_noise = noise(rd * 500.0);
    if (star_noise > 0.98) {
        let star_brightness = (star_noise - 0.98) * 50.0;
        color += vec3f(star_brightness);
    }

    return color;
}

// Ray-sphere intersection
fn intersect_sphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> f32 {
    let oc = ro - center;
    let b = dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - c;

    if (discriminant < 0.0) {
        return -1.0;
    }

    let t = -b - sqrt(discriminant);
    if (t > 0.0) {
        return t;
    }

    return -1.0;
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    // Convert pixel to normalized device coordinates
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;

    // Construct ray direction from camera
    let half_fov = u.fov * 0.5;
    let rd = normalize(
        u.camera_forward
        + u.camera_right * uv.x * u.aspect * tan(half_fov)
        + u.camera_up * uv.y * tan(half_fov)
    );

    let ro = u.camera_pos;

    // Black hole parameters
    let black_hole_center = vec3f(0.0, 0.0, 0.0);
    let event_horizon_radius = 1.0;

    // Check intersection with event horizon (black sphere)
    let t = intersect_sphere(ro, rd, black_hole_center, event_horizon_radius);

    if (t > 0.0) {
        // Hit the event horizon - pure black
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Background nebula
    let bg = nebula(rd);

    return vec4f(bg, 1.0);
}
