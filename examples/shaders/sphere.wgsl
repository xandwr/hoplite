// Raymarched sphere - demonstrates world-space camera uniforms

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

fn sd_sphere(p: vec3f, center: vec3f, radius: f32) -> f32 {
    return length(p - center) - radius;
}

fn raymarch(ro: vec3f, rd: vec3f) -> f32 {
    var t = 0.0;
    for (var i = 0; i < 64; i++) {
        let p = ro + rd * t;
        let d = sd_sphere(p, vec3f(0.0), 1.0);
        if (d < 0.001) {
            return t;
        }
        t += d;
        if (t > 100.0) {
            break;
        }
    }
    return -1.0;
}

fn calc_normal(p: vec3f) -> vec3f {
    let e = vec2f(0.001, 0.0);
    return normalize(vec3f(
        sd_sphere(p + e.xyy, vec3f(0.0), 1.0) - sd_sphere(p - e.xyy, vec3f(0.0), 1.0),
        sd_sphere(p + e.yxy, vec3f(0.0), 1.0) - sd_sphere(p - e.yxy, vec3f(0.0), 1.0),
        sd_sphere(p + e.yyx, vec3f(0.0), 1.0) - sd_sphere(p - e.yyx, vec3f(0.0), 1.0)
    ));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
    let uv = (pos.xy / u.resolution) * 2.0 - 1.0;

    let half_fov = u.fov * 0.5;
    let rd = normalize(
        u.camera_forward
        + u.camera_right * uv.x * u.aspect * tan(half_fov)
        + u.camera_up * uv.y * tan(half_fov)
    );

    let ro = u.camera_pos;
    let t = raymarch(ro, rd);

    if (t > 0.0) {
        let p = ro + rd * t;
        let n = calc_normal(p);

        let light_dir = normalize(vec3f(1.0, 1.0, 1.0));
        let diffuse = max(dot(n, light_dir), 0.0);
        let ambient = 0.1;

        let color = vec3f(0.2, 0.5, 1.0) * (diffuse + ambient);
        return vec4f(color, 1.0);
    }

    let bg = mix(vec3f(0.1, 0.1, 0.15), vec3f(0.02, 0.02, 0.05), uv.y * 0.5 + 0.5);
    return vec4f(bg, 1.0);
}
