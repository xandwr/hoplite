//! Mesh rendering system with deferred draw queuing.

use std::cell::RefCell;
use std::rc::Rc;

use crate::draw2d::Color;
use crate::ecs::{MeshId, TextureId};
use crate::gpu::GpuContext;
use crate::mesh::{Mesh, Transform};
use crate::mesh_pass::{DrawCall, MeshPass};
use crate::render_graph::{RenderContext, RenderNode};
use crate::texture::Texture;

/// A queued mesh draw call stored in the shared mesh queue.
///
/// This struct represents a single mesh instance to be rendered, with its
/// transform, color tint, and optional texture. Draw calls are accumulated
/// in a [`MeshQueue`] during the frame and processed by [`MeshNode`].
///
/// # Fields
///
/// * `mesh` - Type-safe handle to the mesh geometry
/// * `transform` - World-space transformation (position, rotation, scale)
/// * `color` - RGBA color tint applied to the mesh
/// * `texture` - Optional type-safe texture handle
pub struct QueuedMesh {
    /// Handle to the mesh in the queue's mesh array.
    pub mesh: MeshId,
    /// World-space transformation for this instance.
    pub transform: Transform,
    /// Color tint multiplied with vertex colors and textures.
    pub color: Color,
    /// Optional texture handle. `None` uses vertex colors only.
    pub texture: Option<TextureId>,
}

/// Shared storage for meshes, textures, and the per-frame draw queue.
///
/// `MeshQueue` provides a central repository for 3D assets and a queue for
/// draw calls. It is typically wrapped in `Rc<RefCell<>>` and shared between
/// the render graph and application code.
///
/// # Usage Pattern
///
/// ```ignore
/// // Setup: create queue and register meshes
/// let queue = Rc::new(RefCell::new(MeshQueue::new()));
/// let cube_idx = queue.borrow_mut().add_mesh(cube_mesh);
/// let tex_idx = queue.borrow_mut().add_texture(wood_texture);
///
/// // Each frame: queue draw calls
/// queue.borrow_mut().draw(cube_idx, transform, Color::WHITE);
/// queue.borrow_mut().draw_textured(cube_idx, transform2, Color::WHITE, tex_idx);
///
/// // Render graph processes the queue automatically
/// graph.execute(&gpu, time, &camera);
///
/// // After frame: clear for next frame
/// queue.borrow_mut().clear_queue();
/// ```
///
/// # Thread Safety
///
/// This struct is not thread-safe. Use `Rc<RefCell<>>` for single-threaded
/// applications or `Arc<Mutex<>>` for multi-threaded scenarios.
pub struct MeshQueue {
    /// Registered meshes, indexed by the values returned from [`add_mesh`](Self::add_mesh).
    pub meshes: Vec<Mesh>,
    /// Registered textures, indexed by the values returned from [`add_texture`](Self::add_texture).
    pub textures: Vec<Texture>,
    /// Per-frame draw queue, cleared at the end of each frame.
    pub draw_queue: Vec<QueuedMesh>,
}

impl MeshQueue {
    /// Creates a new empty mesh queue.
    pub fn new() -> Self {
        Self {
            meshes: Vec::new(),
            textures: Vec::new(),
            draw_queue: Vec::new(),
        }
    }

    /// Registers a mesh and returns a type-safe handle for later use.
    ///
    /// Meshes are stored permanently until the queue is dropped. Use the
    /// returned [`MeshId`] with [`draw`](Self::draw) or [`draw_textured`](Self::draw_textured).
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh geometry to register
    ///
    /// # Returns
    ///
    /// A [`MeshId`] that can be used to reference this mesh in draw calls.
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let idx = self.meshes.len();
        self.meshes.push(mesh);
        MeshId(idx)
    }

    /// Registers a texture and returns a type-safe handle for later use.
    ///
    /// Textures are stored permanently until the queue is dropped. Use the
    /// returned [`TextureId`] with [`draw_textured`](Self::draw_textured).
    ///
    /// # Arguments
    ///
    /// * `texture` - The texture to register
    ///
    /// # Returns
    ///
    /// A [`TextureId`] that can be used to reference this texture in draw calls.
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        let idx = self.textures.len();
        self.textures.push(texture);
        TextureId(idx)
    }

    /// Queues a mesh for rendering this frame without a texture.
    ///
    /// The mesh will be rendered using vertex colors multiplied by the
    /// provided color tint.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle from [`add_mesh`](Self::add_mesh)
    /// * `transform` - World-space transformation
    /// * `color` - Color tint (use `Color::WHITE` for no tinting)
    pub fn draw(&mut self, mesh: MeshId, transform: Transform, color: Color) {
        self.draw_queue.push(QueuedMesh {
            mesh,
            transform,
            color,
            texture: None,
        });
    }

    /// Queues a textured mesh for rendering this frame.
    ///
    /// The mesh will be rendered with the specified texture, with colors
    /// multiplied by the color tint.
    ///
    /// # Arguments
    ///
    /// * `mesh` - Handle from [`add_mesh`](Self::add_mesh)
    /// * `transform` - World-space transformation
    /// * `color` - Color tint (use `Color::WHITE` for no tinting)
    /// * `texture` - Handle from [`add_texture`](Self::add_texture)
    pub fn draw_textured(
        &mut self,
        mesh: MeshId,
        transform: Transform,
        color: Color,
        texture: TextureId,
    ) {
        self.draw_queue.push(QueuedMesh {
            mesh,
            transform,
            color,
            texture: Some(texture),
        });
    }

    /// Clears the draw queue for the next frame.
    ///
    /// Call this at the end of each frame after the render graph has executed.
    /// Registered meshes and textures are preserved.
    pub fn clear_queue(&mut self) {
        self.draw_queue.clear();
    }
}

impl Default for MeshQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Render node for 3D mesh rendering with depth testing.
///
/// `MeshNode` renders all meshes queued in its associated [`MeshQueue`].
/// It supports:
///
/// - Depth testing for correct occlusion
/// - Per-instance transforms and color tints
/// - Optional texturing
/// - Compositing over previous pass output (background blitting)
///
/// # Integration with Render Graph
///
/// When used after other nodes (e.g., an effect pass for a background),
/// `MeshNode` first blits the input texture, then renders meshes on top
/// with depth testing. This allows 3D objects to be composited over
/// procedural backgrounds.
///
/// # Clear Behavior
///
/// By default, `MeshNode` does **not** clear the target, allowing it to
/// render on top of the previous pass. Use [`with_clear`](Self::with_clear)
/// if you want to clear to a solid color first.
///
/// # Example
///
/// ```ignore
/// let queue = Rc::new(RefCell::new(MeshQueue::new()));
/// let cube_idx = queue.borrow_mut().add_mesh(Mesh::cube(&gpu));
///
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(sky_effect))                    // Background
///     .node(MeshNode::new(&gpu, Rc::clone(&queue)))         // 3D meshes on top
///     .node(PostProcessNode::new(tonemap))                  // Post-process
///     .build(&gpu);
///
/// // In render loop:
/// queue.borrow_mut().draw(cube_idx, Transform::default(), Color::WHITE);
/// graph.execute(&gpu, time, &camera);
/// queue.borrow_mut().clear_queue();
/// ```
pub struct MeshNode {
    /// The mesh rendering pass with pipeline and depth buffer.
    pub pass: MeshPass,
    /// Shared queue containing meshes, textures, and draw calls.
    pub queue: Rc<RefCell<MeshQueue>>,
    /// Optional clear color. `None` preserves previous pass output.
    pub clear_color: Option<wgpu::Color>,
}

impl MeshNode {
    /// Creates a new mesh render node.
    ///
    /// The node is configured to render on top of the previous pass by default
    /// (no clearing). The depth buffer is created at the current GPU surface size.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for creating the mesh pass and depth buffer
    /// * `queue` - Shared mesh queue (typically `Rc<RefCell<MeshQueue>>`)
    pub fn new(gpu: &GpuContext, queue: Rc<RefCell<MeshQueue>>) -> Self {
        Self {
            pass: MeshPass::new(gpu),
            queue,
            clear_color: None, // Don't clear by default - render on top of previous pass
        }
    }

    /// Sets a clear color, causing the target to be cleared before rendering.
    ///
    /// Use this when `MeshNode` is the first node in the graph or when you
    /// don't want to preserve the previous pass output.
    ///
    /// # Arguments
    ///
    /// * `color` - The color to clear to before rendering meshes
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    pub fn with_clear(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }
}

impl RenderNode for MeshNode {
    fn execute(
        &self,
        ctx: &mut RenderContext,
        target: &wgpu::TextureView,
        input: Option<&wgpu::TextureView>,
    ) {
        let queue = self.queue.borrow();

        // Build draw calls from the queue
        let draw_calls: Vec<DrawCall> = queue
            .draw_queue
            .iter()
            .filter_map(|q| {
                queue.meshes.get(q.mesh.0).map(|mesh| DrawCall {
                    mesh,
                    transform: q.transform,
                    color: q.color,
                    texture: q.texture.and_then(|t| queue.textures.get(t.0)),
                })
            })
            .collect();

        // If there's an input texture, we need to blit it first as the background
        if let Some(input_view) = input {
            // First pass: blit the input texture to the target (no depth)
            let mut blit_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mesh Blit Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            self.pass.blit(ctx.gpu, &mut blit_pass, input_view);
        }

        // If no meshes to draw, we're done (background is already blitted)
        if draw_calls.is_empty() {
            // If there was no input either, we need to at least clear the target
            if input.is_none() {
                let _clear_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Mesh Clear Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(
                                self.clear_color.unwrap_or(wgpu::Color::BLACK),
                            ),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
            }
            return;
        }

        // Second pass: render meshes on top with depth testing
        // Use Load since we already blitted (or Clear if no input and clear_color is set)
        let load_op = if input.is_some() {
            wgpu::LoadOp::Load
        } else {
            match self.clear_color {
                Some(color) => wgpu::LoadOp::Clear(color),
                None => wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            }
        };

        let mut render_pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mesh Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.pass.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.pass
            .render(ctx.gpu, &mut render_pass, ctx.camera, ctx.time, &draw_calls);
    }

    fn check_hot_reload(&mut self, gpu: &GpuContext) {
        self.pass.ensure_depth_size(gpu);
    }
}
