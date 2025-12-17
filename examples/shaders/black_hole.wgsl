// Black hole with Kerr metric geodesic ray tracing
// Uses RK4 integration in Cartesian coordinates to avoid B-L singularities

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

// ============================================================================
// Black Hole Parameters
// ============================================================================

const M: f32 = 1.0;                    // Mass
const SPIN: f32 = 0.6;                 // Spin parameter a (0 = Schwarzschild, <1 for Kerr)
const RS: f32 = 2.0 * M;               // Schwarzschild radius

const MAX_STEPS: i32 = 150;
const STEP_SIZE: f32 = 0.4;
const ESCAPE_RADIUS: f32 = 60.0;
const DISK_INNER: f32 = 3.0;
const DISK_OUTER: f32 = 15.0;

// ============================================================================
// Noise functions for starfield
// ============================================================================

fn hash(p: vec3f) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn noise(p: vec3f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(mix(hash(i), hash(i + vec3f(1,0,0)), u.x),
            mix(hash(i + vec3f(0,1,0)), hash(i + vec3f(1,1,0)), u.x), u.y),
        mix(mix(hash(i + vec3f(0,0,1)), hash(i + vec3f(1,0,1)), u.x),
            mix(hash(i + vec3f(0,1,1)), hash(i + vec3f(1,1,1)), u.x), u.y),
        u.z
    );
}

fn fbm(p: vec3f) -> f32 {
    var val = 0.0;
    var amp = 0.5;
    var pos = p;
    for (var i = 0; i < 5; i++) {
        val += amp * noise(pos);
        amp *= 0.5;
        pos *= 2.0;
    }
    return val;
}

// ============================================================================
// Kerr metric helper: compute Boyer-Lindquist r from Cartesian position
// ============================================================================

fn kerr_r(pos: vec3f) -> f32 {
    let a = SPIN;
    let a2 = a * a;
    let x2 = pos.x * pos.x;
    let y2 = pos.y * pos.y;
    let z2 = pos.z * pos.z;

    // Solve r^4 - (x^2+y^2+z^2-a^2)*r^2 - a^2*z^2 = 0
    let sum = x2 + y2 + z2 - a2;
    let r2 = 0.5 * (sum + sqrt(sum * sum + 4.0 * a2 * z2));
    return sqrt(max(r2, 0.0001));
}

// Event horizon
fn r_plus() -> f32 {
    let a = SPIN;
    return M + sqrt(max(M * M - a * a, 0.0));
}

// ============================================================================
// Gravitational acceleration using Kerr-Schild formulation
// This gives the coordinate acceleration for a photon in Cartesian coords
// ============================================================================

fn gravitational_acceleration(pos: vec3f, vel: vec3f) -> vec3f {
    let a = SPIN;
    let r = kerr_r(pos);
    let r2 = r * r;
    let r3 = r2 * r;

    // Protect against singularity
    let r_h = r_plus();
    if (r < r_h * 0.8) {
        return vec3f(0.0);
    }

    // Use position for radial direction
    let r_vec = length(pos);
    let rhat = pos / max(r_vec, 0.001);

    // Photon sphere for Schwarzschild is at r = 1.5 * rs = 3M
    // For Kerr it ranges from ~1M (prograde) to ~4M (retrograde)
    let photon_sphere = 3.0 * M;

    // The key formula for light deflection comes from the geodesic equation
    // For a photon: d²r/dλ² = -dV_eff/dr where V_eff is the effective potential
    // This gives acceleration ~ 3*M/r² * (1 - rs/r) for the leading relativistic correction

    // Simplified: use 1.5 * M * rs / r³ as the leading GR correction
    // This produces the correct deflection angle at large r: Δφ ≈ 4GM/(c²b)
    // let accel_magnitude = 1.5 * M * RS / r3;
    let b_squared = length(cross(pos, vel)) * length(cross(pos, vel)) * r * r;
    let accel_magnitude = M / (r * r) * (1.0 - 1.5 * RS * RS / max(b_squared, 0.1));

    // Radial acceleration (toward black hole)
    var accel = -rhat * accel_magnitude;

    // Frame dragging for Kerr: adds tangential acceleration
    // Magnitude falls off as 1/r³
    if (a > 0.001) {
        let phi_dir = normalize(vec3f(-pos.y, pos.x, 0.0) + vec3f(0.0001, 0.0, 0.0));
        // let frame_drag = 2.0 * a * M / r3;
        let frame_drag = 4.0 * a * M * M / (r3 * r) * length(vel);
        accel += phi_dir * frame_drag;
    }

    return accel;
}

// ============================================================================
// RK4 integration step
// ============================================================================

fn rk4_step(pos: vec3f, vel: vec3f, h: f32) -> array<vec3f, 2> {
    // k1
    let a1 = gravitational_acceleration(pos, vel);
    let k1_pos = vel;
    let k1_vel = a1;

    // k2
    let p2 = pos + 0.5 * h * k1_pos;
    let v2 = vel + 0.5 * h * k1_vel;
    let a2 = gravitational_acceleration(p2, v2);
    let k2_pos = v2;
    let k2_vel = a2;

    // k3
    let p3 = pos + 0.5 * h * k2_pos;
    let v3 = vel + 0.5 * h * k2_vel;
    let a3 = gravitational_acceleration(p3, v3);
    let k3_pos = v3;
    let k3_vel = a3;

    // k4
    let p4 = pos + h * k3_pos;
    let v4 = vel + h * k3_vel;
    let a4 = gravitational_acceleration(p4, v4);
    let k4_pos = v4;
    let k4_vel = a4;

    // Combine
    let new_pos = pos + h * (k1_pos + 2.0*k2_pos + 2.0*k3_pos + k4_pos) / 6.0;
    let new_vel = vel + h * (k1_vel + 2.0*k2_vel + 2.0*k3_vel + k4_vel) / 6.0;

    return array<vec3f, 2>(new_pos, normalize(new_vel));
}

// ============================================================================
// Sky rendering
// ============================================================================

fn star_field(dir: vec3f) -> vec3f {
    var color = vec3f(0.0);

    // Layer 1: dense dim stars
    let c1 = dir * 800.0;
    let h1 = hash(floor(c1));
    if (h1 > 0.996) {
        let b = (h1 - 0.996) * 250.0;
        let temp = hash(floor(c1) + 100.0);
        var sc = vec3f(1.0);
        if (temp < 0.3) { sc = vec3f(1.0, 0.7, 0.5); }
        else if (temp > 0.7) { sc = vec3f(0.6, 0.8, 1.0); }
        color += sc * b;
    }

    // Layer 2: bright stars
    let c2 = dir * 150.0;
    let h2 = hash(floor(c2));
    if (h2 > 0.998) {
        color += vec3f(1.0, 0.98, 0.9) * (h2 - 0.998) * 500.0;
    }

    return color;
}

fn sky(dir: vec3f) -> vec3f {
    let p = dir * 2.5;

    let n1 = fbm(p);
    let n2 = fbm(p * 2.0 + 50.0);

    // Dark space with subtle color variation
    var color = vec3f(0.008, 0.01, 0.02);
    color = mix(color, vec3f(0.02, 0.025, 0.06), n1 * 0.4);
    color = mix(color, vec3f(0.04, 0.015, 0.05), n2 * n2 * 0.2);

    color += star_field(dir);

    return color;
}

// ============================================================================
// Accretion disk - volumetric sampling
// ============================================================================

fn disk_density(pos: vec3f) -> f32 {
    let r = length(pos.xy);

    // Radial bounds with soft edges
    if (r < DISK_INNER * 0.8 || r > DISK_OUTER * 1.2) {
        return 0.0;
    }

    // Gaussian profile in z (thin disk)
    let disk_height = 0.1 + 0.05 * (r / DISK_INNER);  // Slightly flared
    let z_density = exp(-pos.z * pos.z / (2.0 * disk_height * disk_height));

    // Radial density: peaks at inner edge, falls off outward
    let r_norm = (r - DISK_INNER) / (DISK_OUTER - DISK_INNER);
    var r_density = 1.0;
    if (r < DISK_INNER) {
        r_density = smoothstep(DISK_INNER * 0.8, DISK_INNER, r);
    } else if (r > DISK_OUTER) {
        r_density = smoothstep(DISK_OUTER * 1.2, DISK_OUTER, r);
    } else {
        r_density = exp(-r_norm * 0.5);  // Gradual falloff
    }

    return z_density * r_density;
}

fn disk_emission(pos: vec3f, vel: vec3f, step_size: f32) -> vec3f {
    let density = disk_density(pos);
    if (density < 0.001) {
        return vec3f(0.0);
    }

    let r = length(pos.xy);

    // Temperature profile: T ~ r^(-3/4) (Shakura-Sunyaev thin disk)
    // Normalized so inner edge = 1.0
    let temp = pow(DISK_INNER / max(r, DISK_INNER), 0.75);

    // Orbital velocity of disk material (Keplerian, prograde around z-axis)
    // At inner edge of a black hole disk, v can be ~0.5c
    let v_orb = sqrt(M / max(r, DISK_INNER * 0.5)) * 0.5;

    // Orbital direction: counterclockwise when viewed from +z (prograde with BH spin)
    let phi = atan2(pos.y, pos.x);
    let orbit_dir = vec3f(-sin(phi), cos(phi), 0.0);

    // Direction toward camera (ray travels camera->point, so -vel points to camera)
    let to_camera = normalize(-vel);

    // Component of orbital velocity toward camera
    let v_toward_camera = dot(orbit_dir, to_camera) * v_orb;

    // Relativistic Doppler factor: δ = sqrt((1+β)/(1-β)) for motion toward observer
    // Simplified: δ ≈ 1 + v_los/c for v << c, but we want more drama
    let beta = v_toward_camera;
    let doppler = sqrt((1.0 + beta) / max(1.0 - beta, 0.1));

    // Clamp doppler to reasonable range
    let doppler_clamped = clamp(doppler, 0.4, 2.5);

    // Relativistic beaming: I_obs = I_emit * δ^3
    let beaming = doppler_clamped * doppler_clamped * doppler_clamped;

    // Base emission from temperature (L ~ T^4, we use T^3 for visible contrast)
    let luminosity = temp * temp * temp;

    // Color based on observed (Doppler-shifted) temperature
    // Blueshift increases apparent temp, redshift decreases it
    let observed_temp = temp * doppler_clamped;

    // Blackbody-ish color mapping
    var color: vec3f;
    if (observed_temp > 1.2) {
        // Very hot / blueshifted: white-blue
        let t = clamp((observed_temp - 1.2) / 0.8, 0.0, 1.0);
        color = mix(vec3f(1.0, 1.0, 1.0), vec3f(0.6, 0.8, 1.0), t);
    } else if (observed_temp > 0.8) {
        // Hot: white-yellow
        let t = (observed_temp - 0.8) / 0.4;
        color = mix(vec3f(1.0, 0.9, 0.7), vec3f(1.0, 1.0, 1.0), t);
    } else if (observed_temp > 0.4) {
        // Warm: orange
        let t = (observed_temp - 0.4) / 0.4;
        color = mix(vec3f(1.0, 0.5, 0.2), vec3f(1.0, 0.9, 0.7), t);
    } else {
        // Cool / redshifted: deep red
        let t = observed_temp / 0.4;
        color = mix(vec3f(0.8, 0.2, 0.1), vec3f(1.0, 0.5, 0.2), t);
    }

    // Final emission = color * luminosity * beaming * density * step
    let emission = density * luminosity * beaming * step_size * 2.5;

    return color * emission;
}

// ============================================================================
// Main ray trace
// ============================================================================

fn trace(origin: vec3f, direction: vec3f) -> vec3f {
    var pos = origin;
    var vel = direction;

    let r_h = r_plus();
    var disk_accum = vec3f(0.0);
    var min_r = 1000.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let r = kerr_r(pos);
        min_r = min(min_r, r);

        // Fell into black hole
        if (r < r_h * 1.02) {
            return disk_accum;
        }

        // Escaped to infinity - check if moving outward
        let moving_outward = dot(pos, vel) > 0.0;
        if (r > ESCAPE_RADIUS && moving_outward) {
            let photon_r = 1.5 * RS;
            if (min_r < photon_r * 1.3 && min_r > r_h * 1.1) {
                let ring_factor = exp(-pow((min_r - photon_r) / (0.3 * RS), 2.0));
                disk_accum *= (1.0 + ring_factor * 2.0);
            }
            return sky(vel) + disk_accum;
        }

        // Also escape if we're far and past closest approach
        if (r > 20.0 && r > min_r * 1.5 && moving_outward) {
            return sky(vel) + disk_accum;
        }

        // Step size: larger when far, smaller when close
        // But not too small or we'll never escape
        let h = STEP_SIZE * max(r * 0.15, 0.3);

        // Accumulate disk emission (volumetric)
        disk_accum += disk_emission(pos, vel, h);

        // RK4 step
        let result = rk4_step(pos, vel, h);
        pos = result[0];
        vel = result[1];
    }

    // Ran out of steps - return sky based on current direction
    // This prevents the dark halo artifact
    return sky(vel) + disk_accum;
}

// ============================================================================
// Fragment shader
// ============================================================================

@fragment
fn fs(@builtin(position) frag_pos: vec4f) -> @location(0) vec4f {
    let uv = (frag_pos.xy / u.resolution) * 2.0 - 1.0;

    let half_fov = u.fov * 0.5;
    let tan_fov = tan(half_fov);

    let ray_dir = normalize(
        u.camera_forward +
        u.camera_right * uv.x * u.aspect * tan_fov +
        u.camera_up * uv.y * tan_fov
    );

    var color = trace(u.camera_pos, ray_dir);

    // Tone mapping
    color = color / (color + 1.0);

    // Gamma
    color = pow(color, vec3f(0.85));

    return vec4f(color, 1.0);
}
