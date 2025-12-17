// Gravitational lensing post-process effect for mesh objects
// Distorts the rendered scene (meshes) based on black hole gravity

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

// Black hole parameters (must match black_hole.wgsl)
const BLACK_HOLE_POS = vec3f(0.0, 0.0, 0.0);
const M: f32 = 1.0;
const SPIN: f32 = 0.6;
const RS: f32 = 2.0 * M;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32(i32(i % 2u) * 4 - 1);
    let y = f32(i32(i / 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

// Project world position to screen UV (0-1)
fn world_to_screen(world_pos: vec3f) -> vec2f {
    let to_point = world_pos - u.camera_pos;
    let z = dot(to_point, u.camera_forward);

    if (z <= 0.0) {
        return vec2f(-1.0, -1.0);
    }

    let x = dot(to_point, u.camera_right);
    let y = dot(to_point, u.camera_up);

    let half_fov = u.fov * 0.5;
    let tan_fov = tan(half_fov);

    let ndc_x = x / (z * tan_fov * u.aspect);
    let ndc_y = y / (z * tan_fov);

    return vec2f(ndc_x * 0.5 + 0.5, -ndc_y * 0.5 + 0.5);
}

fn r_horizon() -> f32 {
    let a = SPIN;
    return M + sqrt(M * M - a * a);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = pos.xy / u.resolution;

    // Project black hole center to screen
    let bh_screen = world_to_screen(BLACK_HOLE_POS);

    // If black hole is behind camera, no lensing
    if (bh_screen.x < -0.5) {
        return textureSample(input_texture, input_sampler, uv);
    }

    // Distance from camera to black hole in world units
    let dist_to_bh = length(BLACK_HOLE_POS - u.camera_pos);

    // Calculate apparent size of Schwarzschild radius on screen
    let half_fov = u.fov * 0.5;
    let tan_fov = tan(half_fov);
    let rs_angular = RS / dist_to_bh;  // Angular size in radians
    let rs_screen = rs_angular / (2.0 * tan_fov);  // As fraction of screen height

    // Vector from black hole screen position to this pixel
    let to_pixel = uv - bh_screen;

    // Correct for aspect ratio to get true angular distance
    let to_pixel_corrected = vec2f(to_pixel.x * u.aspect, to_pixel.y);
    let r_screen = length(to_pixel_corrected);

    // Convert screen distance to approximate impact parameter (in units of RS)
    let impact_param = r_screen / rs_screen;

    // Event horizon on screen
    let r_h = r_horizon();
    let rh_screen = (r_h / dist_to_bh) / (2.0 * tan_fov);

    // Inside event horizon - black
    if (r_screen < rh_screen * 0.9) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    // Gravitational deflection angle: α ≈ 2RS/b for light passing at impact parameter b
    // This bends light TOWARD the black hole
    // So we sample from a position further from the BH than current pixel
    let deflection_angle = 2.0 * RS / max(impact_param * RS, RS * 1.5);

    // Convert angular deflection to screen-space offset
    // Deflection pulls the apparent position toward the black hole
    // So to find where light came from, we look AWAY from the black hole
    let dir_from_bh = normalize(to_pixel_corrected);
    let offset_magnitude = deflection_angle * rs_screen;

    // The offset should be AWAY from black hole (we're finding the source of bent light)
    var sample_offset = dir_from_bh * offset_magnitude;

    // Un-correct aspect ratio for final UV offset
    sample_offset.x /= u.aspect;

    // Sample from the offset position
    let sample_uv = uv + sample_offset;

    // Clamp to valid UV range
    let clamped_uv = clamp(sample_uv, vec2f(0.0), vec2f(1.0));

    var color = textureSample(input_texture, input_sampler, clamped_uv);

    // Gravitational redshift/dimming near horizon
    let redshift = sqrt(max(1.0 - rh_screen / r_screen, 0.1));
    color = vec4f(color.rgb * redshift, color.a);

    return color;
}
