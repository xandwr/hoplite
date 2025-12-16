// Gravitational lensing post-process effect (world-space aware)
// Projects black hole position from world space to screen space

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
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var input_sampler: sampler;

// Black hole world position (at origin)
const BLACK_HOLE_POS = vec3f(0.0, 0.0, 0.0);
const SCHWARZSCHILD_RADIUS = 1.0;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32(i32(i % 2u) * 4 - 1);
    let y = f32(i32(i / 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// Project a world position to normalized screen coordinates (0-1)
fn world_to_screen(world_pos: vec3f) -> vec2f {
    // Vector from camera to point
    let to_point = world_pos - u.camera_pos;

    // Project onto camera basis
    let z = dot(to_point, u.camera_forward);

    // Behind camera check
    if (z <= 0.0) {
        return vec2f(-1.0, -1.0); // Off screen
    }

    let x = dot(to_point, u.camera_right);
    let y = dot(to_point, u.camera_up);

    // Perspective projection
    let half_fov = u.fov * 0.5;
    let tan_fov = tan(half_fov);

    let ndc_x = x / (z * tan_fov * u.aspect);
    let ndc_y = y / (z * tan_fov);

    // Convert from NDC (-1 to 1) to UV (0 to 1)
    return vec2f(ndc_x * 0.5 + 0.5, -ndc_y * 0.5 + 0.5);
}

// Get the apparent angular size of an object at a distance
fn angular_size(radius: f32, distance: f32) -> f32 {
    return atan(radius / max(distance, 0.001));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = pos.xy / u.resolution;

    // Project black hole center to screen space
    let bh_screen = world_to_screen(BLACK_HOLE_POS);

    // Distance from camera to black hole
    let dist_to_bh = length(BLACK_HOLE_POS - u.camera_pos);

    // Apparent angular size of the event horizon in screen space
    let angular_radius = angular_size(SCHWARZSCHILD_RADIUS, dist_to_bh);
    let half_fov = u.fov * 0.5;

    // Convert angular size to screen-space radius (approximate)
    let screen_radius = angular_radius / half_fov * 0.5;

    // Einstein ring radius (about 2.5x Schwarzschild for this approximation)
    let einstein_radius = screen_radius * 2.5;

    // Vector from black hole screen position to current pixel
    let to_pixel = uv - bh_screen;

    // Correct for aspect ratio
    let corrected = vec2f(to_pixel.x * u.aspect, to_pixel.y);
    let r = length(corrected);

    // Lensing strength scales with apparent size
    let lens_strength = 0.5;

    // Inside event horizon - pure black
    if (r < screen_radius * 0.9) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Calculate deflection using Newtonian approximation
    // Deflection ~ rs^2 / r^2
    let rs_sq = screen_radius * screen_radius;
    let deflection_magnitude = lens_strength * rs_sq / (r * r + 0.0001);

    // Direction of deflection (toward the black hole)
    let deflection_dir = normalize(corrected);

    // Apply deflection
    var sample_offset = deflection_dir * deflection_magnitude;
    sample_offset.x /= u.aspect;

    // Sample position
    let sample_uv = uv + sample_offset;

    // Sample the scene
    var color = textureSample(input_texture, input_sampler, sample_uv);

    // Einstein ring effect
    let ring_dist = abs(r - einstein_radius);
    let ring_width = screen_radius * 0.3;
    let ring_intensity = smoothstep(ring_width, 0.0, ring_dist) * 0.4;

    // Blue-shifted ring glow
    color = vec4f(
        color.r + ring_intensity * 0.2,
        color.g + ring_intensity * 0.5,
        color.b + ring_intensity * 1.0,
        color.a
    );

    // Darken approaching the event horizon (gravitational redshift)
    let horizon_fade = smoothstep(screen_radius * 0.9, screen_radius * 2.0, r);
    color = vec4f(color.rgb * horizon_fade, color.a);

    return color;
}
