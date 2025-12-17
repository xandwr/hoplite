# Hoplite

**A creative coding framework for Rust that gets out of your way.**

Write shaders, render 3D scenes, and build visualizations with a single closure. No boilerplate, no ceremony—just code for the screen.

```rust
use hoplite::*;

fn main() {
    run(|ctx| {
        ctx.default_font(16.0);
        ctx.hot_effect_world("shaders/nebula.wgsl");

        move |frame| {
            frame.text(10.0, 10.0, &format!("FPS: {:.0}", frame.fps()));
        }
    });
}
```

## Philosophy

Hoplite is built on three principles:

1. **One closure, one call** — Your setup and frame logic live in closures. No trait implementations, no engine lifecycle to memorize.

2. **Hot reload everything** — Edit your WGSL shaders and watch them update instantly. No restart required.

3. **Escape hatches everywhere** — Start simple, access the full wgpu API when you need it.

## Features

### Shader-First Rendering

```rust
run(|ctx| {
    // Background effect rendered every frame
    ctx.hot_effect_world("shaders/starfield.wgsl");

    // Post-processing that reads the previous pass
    ctx.hot_post_process("shaders/bloom.wgsl");

    move |frame| { /* your frame logic */ }
});
```

Effects and post-process passes chain automatically. The render graph handles ping-pong buffers, texture binding, and presentation.

### 3D Mesh Rendering

```rust
run(|ctx| {
    ctx.enable_mesh_rendering();
    let cube = ctx.mesh_cube();
    let sphere = ctx.mesh_sphere(32, 16);

    move |frame| {
        frame.draw_mesh(cube, Transform::new()
            .position(Vec3::new(0.0, 2.0, 0.0))
            .rotation(Quat::from_rotation_y(frame.time))
            .uniform_scale(1.5),
            Color::rgb(0.9, 0.3, 0.2)
        );
    }
});
```

Meshes render with depth testing, respecting effect passes and post-processing in the pipeline.

### Textured Meshes

```rust
run(|ctx| {
    ctx.enable_mesh_rendering();
    let cube = ctx.mesh_cube();
    let tex = ctx.texture_blocky_stone(16, 42);

    move |frame| {
        frame.draw_mesh_textured(cube, Transform::new(), Color::WHITE, tex);
    }
});
```

### 2D Sprites

```rust
run(|ctx| {
    let sprite = ctx.sprite_from_file("assets/icon.png").unwrap();

    move |frame| {
        // Draw at position
        frame.sprite(sprite, 10.0, 10.0);

        // Draw scaled with tint
        frame.sprite_scaled_tinted(sprite, 100.0, 100.0, 64.0, 64.0, Color::rgb(1.0, 0.5, 0.5));

        // Draw a region (for sprite sheets)
        frame.sprite_region(sprite, 200.0, 100.0, 32.0, 32.0, 0.0, 0.0, 16.0, 16.0);
    }
});
```

Sprites render in the 2D overlay layer on top of all 3D content and effects.

### Orbit Camera

```rust
let mut orbit = OrbitCamera::new()
    .target(Vec3::ZERO)
    .distance(10.0)
    .fov(75.0)
    .mode(OrbitMode::Interactive);  // or AutoRotate { speed: 0.5 }

move |frame| {
    orbit.update(frame.input, frame.dt);
    *frame.camera = orbit.camera();
}
```

Interactive mode: drag to rotate, scroll to zoom. Auto-rotate mode for demos and visualizations.

### Entity Component System (ECS)

```rust
run(|ctx| {
    ctx.enable_mesh_rendering();
    let cube = ctx.mesh_cube();

    // Spawn entities during setup
    ctx.world.spawn((
        Transform::new().position(Vec3::new(0.0, 0.0, -5.0)),
        RenderMesh::new(MeshHandle(cube), Color::RED),
    ));

    move |frame| {
        // Query and update entities
        for (_, transform) in frame.world.query::<&mut Transform>().iter() {
            transform.rotation *= Quat::from_rotation_y(frame.dt);
        }

        // Render all entities with mesh components
        frame.render_world();
    }
});
```

Built on [hecs](https://crates.io/crates/hecs) — a fast, minimal ECS. Use it for game objects, particles, or any dynamic entity management. The immediate-mode API still works alongside ECS.

### Immediate-Mode 2D

```rust
move |frame| {
    // Simple primitives
    frame.rect(10.0, 10.0, 100.0, 50.0, Color::rgba(0.2, 0.2, 0.2, 0.8));
    frame.text(20.0, 20.0, "Hello, Hoplite!");

    // Debug panels with title bars
    let y = frame.panel_titled(10.0, 100.0, 200.0, 150.0, "Debug");
    frame.text(18.0, y + 8.0, &format!("Time: {:.1}s", frame.time));
}
```

All 2D draws are batched and rendered as an overlay after your render pipeline completes.

### Runtime Hot Reload

Edit any `.wgsl` file passed to `hot_effect*` or `hot_post_process*` methods. Hoplite watches the filesystem and recompiles shaders on change. If compilation fails, the previous working shader stays active.

```
[hot-reload] Reloading shader: "shaders/nebula.wgsl"
[hot-reload] Shader compiled successfully
```

## Quick Start

```toml
[dependencies]
hoplite = { git = "https://github.com/xandwr/hoplite" }
```

```rust
use hoplite::*;

fn main() {
    run_with_config(
        AppConfig::new().title("My App").size(1280, 720),
        |ctx| {
            ctx.default_font(16.0);

            // Your setup here

            move |frame| {
                // Your frame logic here
            }
        }
    );
}
```

## Examples

Run the black hole demo with gravitational lensing:

```bash
cargo run --example black_hole
```

## API Reference

### Setup Context (`SetupContext`)

| Method | Description |
|--------|-------------|
| `default_font(size)` | Load the default font at given pixel size |
| `effect(shader)` | Add a screen-space effect pass |
| `effect_world(shader)` | Add a world-space effect with camera uniforms |
| `post_process(shader)` | Add screen-space post-processing |
| `post_process_world(shader)` | Add world-space post-processing |
| `hot_effect(path)` | Hot-reloadable screen-space effect |
| `hot_effect_world(path)` | Hot-reloadable world-space effect |
| `hot_post_process(path)` | Hot-reloadable screen-space post-process |
| `hot_post_process_world(path)` | Hot-reloadable world-space post-process |
| `enable_mesh_rendering()` | Enable 3D mesh pipeline |
| `mesh_cube()` | Create a unit cube mesh |
| `mesh_sphere(segments, rings)` | Create a UV sphere mesh |
| `mesh_plane(size)` | Create a flat plane mesh |
| `add_texture(texture)` | Add a texture, returns index |
| `texture_from_file(path)` | Load texture from file |
| `texture_from_bytes(bytes, label)` | Load texture from memory |
| `texture_blocky_noise(size, seed)` | Procedural dirt/stone texture |
| `texture_blocky_grass(size, seed)` | Procedural grass texture |
| `texture_blocky_stone(size, seed)` | Procedural stone texture |
| `add_sprite(sprite)` | Add a sprite, returns SpriteId |
| `sprite_from_file(path)` | Load sprite from file (linear filtering) |
| `sprite_from_file_nearest(path)` | Load sprite from file (pixel art) |
| `sprite_from_bytes(bytes, label)` | Load sprite from memory |

### Frame Context (`Frame`)

| Method | Description |
|--------|-------------|
| `fps()` | Current frames per second |
| `width()` / `height()` | Screen dimensions in pixels |
| `text(x, y, str)` | Draw text at position |
| `text_color(x, y, str, color)` | Draw colored text |
| `rect(x, y, w, h, color)` | Draw filled rectangle |
| `panel(x, y, w, h)` | Draw a bordered panel |
| `panel_titled(x, y, w, h, title)` | Panel with title bar |
| `draw_mesh(index, transform, color)` | Draw a 3D mesh |
| `draw_mesh_textured(index, transform, color, tex)` | Draw a textured 3D mesh |
| `sprite(id, x, y)` | Draw sprite at position |
| `sprite_tinted(id, x, y, tint)` | Draw sprite with color tint |
| `sprite_scaled(id, x, y, w, h)` | Draw sprite at custom size |
| `sprite_scaled_tinted(id, x, y, w, h, tint)` | Draw scaled sprite with tint |
| `sprite_region(id, x, y, w, h, sx, sy, sw, sh)` | Draw sprite sub-region |
| `render_world()` | Render all ECS entities with `Transform` + `RenderMesh` |

### Frame Fields

| Field | Type | Description |
|-------|------|-------------|
| `time` | `f32` | Total elapsed time in seconds |
| `dt` | `f32` | Delta time since last frame |
| `input` | `&Input` | Keyboard and mouse state |
| `camera` | `&mut Camera` | Current camera (modify to change view) |
| `world` | `&mut World` | ECS world for entity management |
| `gpu` | `&GpuContext` | Low-level GPU access |
| `draw` | `&mut Draw2d` | Low-level 2D API |

## Shader Uniforms

World-space shaders receive these uniforms:

```wgsl
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
```

Post-process shaders also get the input texture:

```wgsl
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var input_sampler: sampler;
```

## Dependencies

Hoplite builds on solid foundations:

- **wgpu** — Cross-platform GPU abstraction
- **winit** — Window creation and input handling
- **glam** — Fast math types (Vec3, Mat4, Quat)
- **hecs** — Fast, minimal Entity Component System
- **fontdue** — Font rasterization
- **bytemuck** — Safe casting for GPU buffers
- **image** — Image loading and handling

## License

MIT
