# Features

## Core

- **Closure-based API** — Setup and frame logic in closures, no traits to implement
- **Window creation** via winit with configurable title and dimensions
- **wgpu rendering** — Surface, device, queue, and automatic resize handling
- **Cross-platform** — Runs anywhere wgpu does (Windows, macOS, Linux, WebGPU)

## Render Graph

- **Automatic pass chaining** — Effects and post-process passes chain together seamlessly
- **Ping-pong buffers** — Managed internally for multi-pass rendering
- **Flexible node system** — Effect nodes, post-process nodes, mesh nodes
- **Screen-space passes** — Full-screen shaders with resolution and time uniforms
- **World-space passes** — Shaders receive full camera state (position, orientation, FOV)
- **UI overlay pass** — 2D content rendered on top of all effects

## Shader Hot Reload

- **Filesystem watching** — Automatic detection of .wgsl file changes
- **Instant recompilation** — See changes without restarting the app
- **Graceful failure** — Invalid shaders keep the previous working version
- **Console feedback** — Clear messages about reload status

## 3D Rendering

- **Mesh pipeline** — Depth-tested 3D mesh rendering
- **Textured meshes** — Apply textures to 3D meshes with UV mapping
- **Built-in primitives** — Cube, sphere, and plane meshes
- **Custom meshes** — Create meshes from vertex data
- **Transform system** — Position, rotation, scale via builder pattern
- **Per-mesh coloring** — Tint meshes at draw time
- **Pipeline integration** — Meshes respect effect and post-process passes

## Textures & Sprites

- **Texture loading** — Load from files, bytes, or raw RGBA data
- **Procedural textures** — Working on this one...
- **2D Sprites** — Screen-space sprite rendering with the 2D layer
- **Sprite regions** — Draw sub-regions of sprites for sprite sheets/atlases
- **Filtering modes** — Linear (smooth) or nearest-neighbor (pixel art) filtering
- **Tinting** — Color-multiply sprites at draw time

## Camera System

- **Camera struct** — Position, forward, up, FOV, near/far planes
- **OrbitCamera controller** — Ready-to-use orbiting camera
  - Interactive mode (mouse drag + scroll zoom)
  - Auto-rotate mode for demos
  - Configurable sensitivity, distance limits, FOV
- **Direct access** — Modify `frame.camera` for custom camera logic

## 2D Rendering

- **Immediate mode** — Draw commands each frame, batched automatically
- **Colored rectangles** — `rect(x, y, w, h, color)`
- **Text rendering** — Fontdue-powered with configurable font sizes
- **Panel builder** — Bordered panels with optional title bars
- **Efficient batching** — All 2D draws batched into minimal draw calls

## Input Handling

- **Keyboard state** — `key_pressed()`, `key_down()`, `key_released()`
- **Mouse state** — Button state, position, delta movement
- **Scroll wheel** — `scroll_delta()` for zoom and scroll interactions
- **Per-frame semantics** — Clear distinction between pressed/down/released

## Entity Component System (ECS)

- **First-class hecs integration** — `World` available in both setup and frame contexts
- **Built-in render components**:
  - `MeshHandle` — Reference to a mesh in the queue
  - `TextureHandle` — Reference to a texture in the queue
  - `RenderMesh` — Component for renderable entities (mesh + color + optional texture)
- **One-call rendering** — `frame.render_world()` renders all entities with `Transform` + `RenderMesh`
- **Hybrid approach** — ECS and immediate-mode APIs work together seamlessly
- **Dynamic entities** — Spawn, query, update, and despawn entities at runtime

## Uniforms & Shaders

Screen-space shaders:
```wgsl
struct Uniforms {
    resolution: vec2f,
    time: f32,
}
```

World-space shaders:
```wgsl
struct Uniforms {
    resolution: vec2f,
    time: f32,
    fov: f32,
    camera_pos: vec3f,
    camera_forward: vec3f,
    camera_right: vec3f,
    camera_up: vec3f,
    aspect: f32,
}
```

Post-process shaders also receive:
```wgsl
@group(0) @binding(1) var input_texture: texture_2d<f32>;
@group(0) @binding(2) var input_sampler: sampler;
```

## Architecture

```
src/
├── app.rs          # Application lifecycle, Frame/SetupContext
├── render_graph.rs # Multi-pass render graph with nodes
├── effect_pass.rs  # Full-screen shader effects
├── post_process.rs # Post-processing with input texture
├── hot_shader.rs   # Runtime shader hot-reload
├── mesh.rs         # 3D vertex data and primitives
├── texture.rs      # Texture rendering
├── mesh_pass.rs    # Mesh rendering pipeline
├── draw2d.rs       # Immediate-mode 2D rendering
├── ecs.rs          # ECS components (MeshHandle, RenderMesh, etc.)
├── camera.rs       # Camera state
├── orbit_camera.rs # Orbit camera controller
├── input.rs        # Keyboard/mouse input
├── assets.rs       # Font loading and atlas management
├── gpu.rs          # wgpu context wrapper
└── lib.rs          # Public API re-exports
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| wgpu | GPU abstraction |
| winit | Window and input |
| glam | Math types |
| fontdue | Font rasterization |
| bytemuck | GPU buffer casting |
| pollster | Async blocking |
| hecs | Entity Component System |
| image | Image rendering |

## Planned

- [ ] Audio playback
- [ ] More mesh primitives (cylinder, torus, custom OBJ loading)
- [ ] Render-to-texture for offscreen rendering
- [ ] Instanced mesh rendering
- [ ] Shadow mapping
- [ ] PBR materials
