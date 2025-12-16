// Standard 3D mesh shader with basic lighting

struct CameraUniforms {
    view_proj: mat4x4f,
    view: mat4x4f,
    proj: mat4x4f,
    camera_pos: vec3f,
    time: f32,
}

struct ModelUniforms {
    model: mat4x4f,
    normal_matrix: mat4x4f,
    color: vec4f,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> model: ModelUniforms;

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) world_pos: vec3f,
    @location(1) world_normal: vec3f,
    @location(2) uv: vec2f,
}

@vertex
fn vs(in: VertexInput) -> VertexOutput {
    let world_pos = model.model * vec4f(in.position, 1.0);
    let world_normal = normalize((model.normal_matrix * vec4f(in.normal, 0.0)).xyz);

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_pos = world_pos.xyz;
    out.world_normal = world_normal;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs(in: VertexOutput) -> @location(0) vec4f {
    let normal = normalize(in.world_normal);
    let view_dir = normalize(camera.camera_pos - in.world_pos);

    // Simple directional light from above-right
    let light_dir = normalize(vec3f(0.5, 1.0, 0.3));
    let light_color = vec3f(1.0, 0.98, 0.95);

    // Ambient
    let ambient = 0.15;

    // Diffuse (half-lambert for softer look)
    let ndotl = dot(normal, light_dir);
    let diffuse = ndotl * 0.5 + 0.5;

    // Specular (Blinn-Phong)
    let half_vec = normalize(light_dir + view_dir);
    let spec = pow(max(dot(normal, half_vec), 0.0), 32.0) * 0.5;

    // Rim light for edge definition
    let rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0) * 0.2;

    let base_color = model.color.rgb;
    let lighting = ambient + diffuse * light_color + spec + rim;
    let final_color = base_color * lighting;

    return vec4f(final_color, model.color.a);
}
