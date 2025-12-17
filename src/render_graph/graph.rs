//! The main render graph and builder for composing render pipelines.

use crate::camera::Camera;
use crate::gpu::GpuContext;
use crate::render_graph::{RenderContext, RenderNode, RenderTarget};

/// Builder for constructing render graphs with a fluent API.
///
/// `RenderGraphBuilder` provides a chainable interface for assembling render
/// pipelines. Nodes are executed in the order they are added.
///
/// # Example
///
/// ```ignore
/// let graph = RenderGraph::builder()
///     .node(EffectNode::new(scene_effect))      // First: render scene
///     .node(PostProcessNode::new(bloom))        // Then: apply bloom
///     .node(PostProcessNode::new(tonemap))      // Finally: tonemap
///     .build(&gpu);
/// ```
///
/// # Node Ordering
///
/// Nodes execute in insertion order. The first node receives no input
/// (`input` is `None`), while subsequent nodes receive the previous
/// node's output. The final node renders directly to the screen.
pub struct RenderGraphBuilder {
    nodes: Vec<Box<dyn RenderNode>>,
}

impl RenderGraphBuilder {
    /// Creates a new empty render graph builder.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Adds a render node to the graph.
    ///
    /// Nodes are executed in the order they are added. Any type implementing
    /// [`RenderNode`] can be added.
    ///
    /// # Arguments
    ///
    /// * `node` - The render node to add
    ///
    /// # Returns
    ///
    /// Self for method chaining (builder pattern).
    ///
    /// # Type Parameters
    ///
    /// * `N` - Any type implementing `RenderNode + 'static`
    pub fn node<N: RenderNode + 'static>(mut self, node: N) -> Self {
        self.nodes.push(Box::new(node));
        self
    }

    /// Builds the render graph, allocating ping-pong buffers.
    ///
    /// This method finalizes the graph and creates the intermediate render
    /// targets needed for multi-pass rendering. Two render targets are
    /// allocated at the current GPU surface size.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for creating render targets
    ///
    /// # Returns
    ///
    /// A ready-to-use [`RenderGraph`].
    pub fn build(self, gpu: &GpuContext) -> RenderGraph {
        // Create ping-pong buffers for multi-pass rendering
        let target_a = RenderTarget::new(gpu, "RenderGraph Target A");
        let target_b = RenderTarget::new(gpu, "RenderGraph Target B");

        RenderGraph {
            nodes: self.nodes,
            target_a,
            target_b,
        }
    }
}

impl Default for RenderGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A composable render graph that executes a chain of render passes.
///
/// `RenderGraph` manages a sequence of render nodes and the intermediate
/// buffers needed for multi-pass rendering. It handles:
///
/// - Ping-pong buffer management for pass chaining
/// - Automatic render target resizing on window resize
/// - Hot-reload checking for all nodes
/// - Final presentation to the screen
/// - Optional UI overlay compositing
///
/// # Buffer Management
///
/// For multi-pass rendering, the graph uses two intermediate render targets
/// (ping-pong buffers). Each pass alternates between reading from one buffer
/// and writing to the other, with the final pass writing directly to the screen.
///
/// ```text
/// Pass 0: None → Target A
/// Pass 1: Target A → Target B
/// Pass 2: Target B → Target A
/// Pass 3: Target A → Screen
/// ```
///
/// For single-node graphs, no intermediate buffers are used.
///
/// # Example
///
/// ```ignore
/// // Create a render graph
/// let mut graph = RenderGraph::builder()
///     .node(EffectNode::new(scene))
///     .node(PostProcessNode::new(bloom))
///     .build(&gpu);
///
/// // Simple rendering
/// graph.execute(&gpu, time, &camera);
///
/// // Or with UI overlay
/// graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
///     ui.render(gpu, pass);
/// });
/// ```
pub struct RenderGraph {
    /// The sequence of render nodes to execute.
    nodes: Vec<Box<dyn RenderNode>>,
    /// First ping-pong buffer for intermediate results.
    target_a: RenderTarget,
    /// Second ping-pong buffer for intermediate results.
    target_b: RenderTarget,
}

impl RenderGraph {
    /// Creates a new render graph builder.
    ///
    /// This is the recommended way to construct a render graph. Use the builder's
    /// fluent API to add nodes, then call [`build`](RenderGraphBuilder::build).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let graph = RenderGraph::builder()
    ///     .node(EffectNode::new(scene))
    ///     .node(PostProcessNode::new(bloom))
    ///     .build(&gpu);
    /// ```
    pub fn builder() -> RenderGraphBuilder {
        RenderGraphBuilder::new()
    }

    /// Adds a node to an existing render graph.
    ///
    /// This method allows dynamically extending a render graph after construction.
    /// The render targets are resized if needed to match the current GPU surface.
    ///
    /// # Arguments
    ///
    /// * `node` - The render node to add
    /// * `gpu` - GPU context for potential target resizing
    ///
    /// # Returns
    ///
    /// Self for method chaining.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Add a new post-process effect at runtime
    /// graph = graph.with_node(PostProcessNode::new(new_effect), &gpu);
    /// ```
    pub fn with_node<N: RenderNode + 'static>(mut self, node: N, gpu: &GpuContext) -> Self {
        self.nodes.push(Box::new(node));
        // Ensure we have render targets
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");
        self
    }

    /// Executes the render graph and presents to the screen.
    ///
    /// This is the main method called each frame. It:
    /// 1. Checks all nodes for hot-reload changes
    /// 2. Ensures render targets match the current window size
    /// 3. Executes each node in sequence with ping-pong buffering
    /// 4. Presents the final result to the screen
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `time` - Elapsed time in seconds (passed to shaders)
    /// * `camera` - Current camera state
    ///
    /// # Panics
    ///
    /// Panics if the surface texture cannot be acquired.
    pub fn execute(&mut self, gpu: &GpuContext, time: f32, camera: &Camera) {
        self.execute_with_ui(gpu, time, camera, |_, _| {});
    }

    /// Checks all nodes for hot-reload changes.
    ///
    /// This is called automatically by [`execute`](Self::execute) and
    /// [`execute_with_ui`](Self::execute_with_ui), but can be invoked manually
    /// if you need to trigger hot-reload checks outside the normal render loop.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context for shader recompilation
    pub fn check_hot_reload(&mut self, gpu: &GpuContext) {
        for node in &mut self.nodes {
            node.check_hot_reload(gpu);
        }
    }

    /// Executes the render graph with a UI overlay pass.
    ///
    /// Similar to [`execute`](Self::execute), but allows rendering UI elements
    /// on top of the final output. The UI closure is called after all render
    /// nodes have executed, with the render pass targeting the screen.
    ///
    /// The UI pass uses `LoadOp::Load` to preserve the rendered scene, so UI
    /// elements are composited on top.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `time` - Elapsed time in seconds
    /// * `camera` - Current camera state
    /// * `ui_fn` - Closure that receives `(&GpuContext, &mut wgpu::RenderPass)` for UI rendering
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut ui = UiPass::new(&gpu);
    ///
    /// // In render loop:
    /// ui.clear();
    /// ui.add(&GuiComponent::floating_pane(10.0, 10.0, 200.0, 100.0).build());
    /// graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
    ///     ui.render(gpu, pass);
    /// });
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if the surface texture cannot be acquired.
    pub fn execute_with_ui<F>(&mut self, gpu: &GpuContext, time: f32, camera: &Camera, ui_fn: F)
    where
        F: FnOnce(&GpuContext, &mut wgpu::RenderPass),
    {
        // Check for hot-reload changes before rendering
        self.check_hot_reload(gpu);

        // Ensure render targets are the right size
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");

        let output = gpu.surface.get_current_texture().unwrap();
        let screen_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph Encoder"),
            });

        let node_count = self.nodes.len();

        {
            let mut ctx = RenderContext {
                gpu,
                encoder: &mut encoder,
                time,
                camera,
            };

            // For single node, render directly to screen
            if node_count == 1 {
                self.nodes[0].execute(&mut ctx, &screen_view, None);
            } else {
                // Multi-pass: ping-pong between targets, final pass goes to screen
                let mut current_input: Option<&wgpu::TextureView> = None;

                for (i, node) in self.nodes.iter().enumerate() {
                    let is_last = i == node_count - 1;

                    let target = if is_last {
                        &screen_view
                    } else if i % 2 == 0 {
                        &self.target_a.view
                    } else {
                        &self.target_b.view
                    };

                    node.execute(&mut ctx, target, current_input);

                    // Set up input for next pass
                    if !is_last {
                        current_input = Some(if i % 2 == 0 {
                            &self.target_a.view
                        } else {
                            &self.target_b.view
                        });
                    }
                }
            }
        }

        // Render UI on top (if any)
        {
            let mut ui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("UI Overlay Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &screen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            ui_fn(gpu, &mut ui_pass);
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Executes the render graph to an arbitrary target texture (not the screen).
    ///
    /// This is used during scene transitions to capture a scene's output to a
    /// render target for crossfade effects. The scene is rendered normally through
    /// all its passes, but the final output goes to the provided target instead
    /// of the screen.
    ///
    /// Unlike [`execute_with_ui`](Self::execute_with_ui), this does NOT present
    /// to the screen and does NOT include a UI overlay pass.
    ///
    /// # Arguments
    ///
    /// * `gpu` - GPU context
    /// * `time` - Elapsed time in seconds
    /// * `camera` - Current camera state
    /// * `target` - The texture view to render to
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Capture scene output to a render target for compositing
    /// let capture_target = RenderTarget::new(&gpu, "Scene Capture");
    /// graph.execute_to_target(&gpu, time, &camera, &capture_target.view);
    /// ```
    pub fn execute_to_target(
        &mut self,
        gpu: &GpuContext,
        time: f32,
        camera: &Camera,
        target: &wgpu::TextureView,
    ) {
        // Check for hot-reload changes before rendering
        self.check_hot_reload(gpu);

        // Ensure render targets are the right size
        self.target_a.ensure_size(gpu, "RenderGraph Target A");
        self.target_b.ensure_size(gpu, "RenderGraph Target B");

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RenderGraph To Target Encoder"),
            });

        let node_count = self.nodes.len();

        {
            let mut ctx = RenderContext {
                gpu,
                encoder: &mut encoder,
                time,
                camera,
            };

            // For single node, render directly to provided target
            if node_count == 1 {
                self.nodes[0].execute(&mut ctx, target, None);
            } else {
                // Multi-pass: ping-pong between targets, final pass goes to provided target
                let mut current_input: Option<&wgpu::TextureView> = None;

                for (i, node) in self.nodes.iter().enumerate() {
                    let is_last = i == node_count - 1;

                    let node_target = if is_last {
                        target
                    } else if i % 2 == 0 {
                        &self.target_a.view
                    } else {
                        &self.target_b.view
                    };

                    node.execute(&mut ctx, node_target, current_input);

                    // Set up input for next pass
                    if !is_last {
                        current_input = Some(if i % 2 == 0 {
                            &self.target_a.view
                        } else {
                            &self.target_b.view
                        });
                    }
                }
            }
        }

        gpu.queue.submit(std::iter::once(encoder.finish()));
    }
}
