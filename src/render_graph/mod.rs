//! Composable render graph system for multi-pass rendering pipelines.
//!
//! This module provides a flexible, node-based render graph architecture that allows
//! chaining multiple render passes together with automatic ping-pong buffer management.
//! Each pass can read from the previous pass's output and write to its own render target,
//! enabling complex post-processing chains and multi-stage rendering effects.
//!
//! # Architecture
//!
//! The render graph uses a linear pipeline model:
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
//! │  EffectNode │───▶│ PostProcess │───▶│ PostProcess │───▶│   Screen    │
//! │  (Scene)    │    │   Node 1    │    │   Node 2    │    │  (Final)    │
//! └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
//!       │                  │                  │
//!       ▼                  ▼                  ▼
//!   Target A ◀────────▶ Target B        (ping-pong)
//! ```
//!
//! # Node Types
//!
//! - [`EffectNode`] / [`HotEffectNode`]: Full-screen shader effects (backgrounds, procedural scenes)
//! - [`PostProcessNode`] / [`HotPostProcessNode`]: Screen-space post-processing (blur, bloom, color grading)
//! - [`WorldPostProcessNode`] / [`HotWorldPostProcessNode`]: Post-processing with camera/world data (raymarching, fog)
//! - [`MeshNode`]: 3D mesh rendering with depth testing
//!
//! Hot-reload variants automatically watch shader files and recompile on changes.
//!
//! # Example
//!
//! ```ignore
//! use hoplite::{RenderGraph, EffectNode, PostProcessNode};
//!
//! // Build a render graph with a scene and post-processing
//! let mut graph = RenderGraph::builder()
//!     .node(EffectNode::new(scene_effect))           // Render procedural scene
//!     .node(PostProcessNode::new(bloom_pass))        // Apply bloom
//!     .node(PostProcessNode::new(tonemap_pass))      // Tonemap to screen
//!     .build(&gpu);
//!
//! // In render loop:
//! loop {
//!     graph.execute(&gpu, time, &camera);
//! }
//! ```
//!
//! # With UI Overlay
//!
//! ```ignore
//! graph.execute_with_ui(&gpu, time, &camera, |gpu, pass| {
//!     ui.render(gpu, pass);  // UI composited on top of final output
//! });
//! ```

mod effect_nodes;
mod graph;
mod mesh_queue;
mod post_process_nodes;
mod render_node;
mod render_target;

pub use effect_nodes::{EffectNode, HotEffectNode};
pub use graph::{RenderGraph, RenderGraphBuilder};
pub use mesh_queue::{MeshNode, MeshQueue};
pub use post_process_nodes::{
    HotPostProcessNode, HotWorldPostProcessNode, PostProcessNode, WorldPostProcessNode,
};
pub use render_node::RenderNode;
pub use render_target::{RenderContext, RenderTarget};
