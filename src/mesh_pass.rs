//! 3D mesh rendering pass with depth testing and texture support.
//!
//! This module provides the [`MeshPass`] render pass, which handles rendering of 3D meshes
//! with proper depth testing, camera transformations, and optional texturing. It is designed
//! to work within the render graph system and can composite its output over previous passes.
//!
//! # Architecture
//!
//! The mesh pass uses three bind groups:
//! - **Group 0**: Camera uniforms (view/projection matrices, camera position, time)
//! - **Group 1**: Model uniforms (model matrix, normal matrix, color)
//! - **Group 2**: Texture and sampler for the mesh surface
//!
//! # Example
//!
//! ```ignore
//! use hoplite::{MeshPass, DrawCall, Camera, Mesh, Transform};
//!
//! // Create the mesh pass
//! let mesh_pass = MeshPass::new(&gpu);
//!
//! // Queue draw calls
//! let draw_calls = vec![
//!     DrawCall {
//!         mesh: &my_mesh,
//!         transform: Transform::from_position([0.0, 0.0, 0.0]),
//!         color: Color::WHITE,
//!         texture: Some(&my_texture),
//!     },
//! ];
//!
//! // Render in a render pass
//! mesh_pass.render(&gpu, &mut render_pass, &camera, time, &draw_calls);
//! ```
//!
//! # Depth Buffer
//!
//! The mesh pass maintains its own depth buffer that automatically resizes to match
//! the screen dimensions. Call [`MeshPass::ensure_depth_size`] before rendering if
//! the window may have been resized.
//!
//! # Blitting
//!
//! The [`MeshPass::blit`] method allows compositing an input texture as the background
//! before rendering meshes. This is useful for layering the 3D scene over 2D content
//! from previous render passes.

use crate::camera::Camera;
use crate::draw2d::Color;
use crate::gpu::GpuContext;
use crate::mesh::{Mesh, Transform, Vertex3d};
use crate::texture::Texture;

/// Camera uniforms for 3D rendering.
///
/// This structure is uploaded to the GPU each frame and provides the shader
/// with all necessary camera and timing information.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniforms {
    /// Combined view-projection matrix for transforming world positions to clip space.
    pub view_proj: [[f32; 4]; 4],
    /// View matrix (world to camera space transformation).
    pub view: [[f32; 4]; 4],
    /// Projection matrix (camera to clip space transformation).
    pub proj: [[f32; 4]; 4],
    /// Camera position in world space, useful for lighting calculations.
    pub camera_pos: [f32; 3],
    /// Elapsed time in seconds, for animated shaders.
    pub time: f32,
}

/// Per-instance model uniforms.
///
/// This structure is uploaded to the GPU for each draw call and provides
/// the shader with model-specific transformation and color data.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelUniforms {
    /// Model matrix (object to world space transformation).
    pub model: [[f32; 4]; 4],
    /// Normal matrix (inverse transpose of model matrix) for correct normal transformation.
    pub normal_matrix: [[f32; 4]; 4],
    /// RGBA color multiplier applied to the mesh.
    pub color: [f32; 4],
}

/// A draw call queued for rendering.
///
/// Represents a single mesh to be rendered with its associated transform,
/// color tint, and optional texture. Draw calls are batched and processed
/// by [`MeshPass::render`].
///
/// # Example
///
/// ```ignore
/// let draw_call = DrawCall {
///     mesh: &cube_mesh,
///     transform: Transform::from_position([0.0, 1.0, 0.0])
///         .with_scale([2.0, 2.0, 2.0]),
///     color: Color::RED,
///     texture: None, // Uses default white texture
/// };
/// ```
pub struct DrawCall<'a> {
    /// Reference to the mesh geometry to render.
    pub mesh: &'a Mesh,
    /// World-space transform (position, rotation, scale) for the mesh.
    pub transform: Transform,
    /// Color tint applied to the mesh (multiplied with texture color).
    pub color: Color,
    /// Optional texture to apply. If `None`, a default white texture is used.
    pub texture: Option<&'a Texture>,
}

/// Handles 3D mesh rendering with depth testing.
///
/// `MeshPass` is a GPU render pass optimized for rendering textured 3D meshes
/// with proper depth sorting. It manages all necessary GPU resources including
/// pipelines, uniform buffers, and depth textures.
///
/// # Features
///
/// - **Depth testing**: Proper occlusion with a 32-bit floating point depth buffer
/// - **Texturing**: Per-mesh texture binding with a default white fallback
/// - **Color tinting**: Per-mesh color multiplier for variety without texture changes
/// - **Blitting**: Composite previous render pass output as background
/// - **Auto-resize**: Depth buffer automatically resizes to match screen dimensions
///
/// # Pipeline Configuration
///
/// - Back-face culling enabled (counter-clockwise front faces)
/// - Alpha blending for transparent meshes
/// - Depth write and Less-than comparison
///
/// # Usage
///
/// The typical render loop involves:
/// 1. Call [`ensure_depth_size`](Self::ensure_depth_size) if the window may have resized
/// 2. Optionally call [`blit`](Self::blit) to composite a background texture
/// 3. Call [`render`](Self::render) with your camera and draw calls
pub struct MeshPass {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    model_buffer: wgpu::Buffer,
    model_bind_group: wgpu::BindGroup,
    /// The depth texture used for depth testing.
    pub(crate) depth_texture: wgpu::Texture,
    /// View into the depth texture for render pass attachment.
    pub(crate) depth_view: wgpu::TextureView,
    depth_size: (u32, u32),
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_sampler: wgpu::Sampler,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    default_texture: Texture,
}

impl MeshPass {
    /// Creates a new mesh rendering pass.
    ///
    /// This initializes all GPU resources including:
    /// - The mesh rendering pipeline with depth testing
    /// - Camera and model uniform buffers
    /// - A blit pipeline for background compositing
    /// - A default 1x1 white texture for untextured meshes
    /// - A depth buffer sized to the current screen dimensions
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context containing the device, queue, and surface configuration
    pub fn new(gpu: &GpuContext) -> Self {
        let device = &gpu.device;

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mesh Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
        });

        // Camera uniform buffer (group 0)
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniforms"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Model uniform buffer (group 1)
        let model_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Model Uniforms"),
            size: std::mem::size_of::<ModelUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let model_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Model Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Model Bind Group"),
            layout: &model_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            }],
        });

        // Texture bind group layout (group 2)
        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create a 1x1 white default texture for untextured meshes
        let default_texture =
            Texture::from_rgba(gpu, &[255, 255, 255, 255], 1, 1, "Default White Texture");

        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mesh Pipeline Layout"),
            bind_group_layouts: &[
                &camera_bind_group_layout,
                &model_bind_group_layout,
                &texture_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        // Depth texture
        let (depth_texture, depth_view) = Self::create_depth_texture(gpu);

        // Blit pipeline for compositing input texture as background
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Render pipeline
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mesh Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[Vertex3d::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            camera_buffer,
            camera_bind_group,
            model_buffer,
            model_bind_group,
            depth_texture,
            depth_view,
            depth_size: (gpu.width(), gpu.height()),
            blit_pipeline,
            blit_bind_group_layout,
            blit_sampler,
            texture_bind_group_layout,
            default_texture,
        }
    }

    /// Creates a bind group for a texture.
    ///
    /// This creates a GPU bind group that binds a texture and its sampler
    /// for use in the mesh shader. The bind group is assigned to group 2
    /// in the pipeline layout.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `texture` - The texture to create a bind group for
    ///
    /// # Returns
    ///
    /// A `wgpu::BindGroup` ready to be bound during rendering.
    pub fn create_texture_bind_group(
        &self,
        gpu: &GpuContext,
        texture: &Texture,
    ) -> wgpu::BindGroup {
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mesh Texture Bind Group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture.sampler),
                },
            ],
        })
    }

    fn create_depth_texture(gpu: &GpuContext) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width: gpu.width(),
                height: gpu.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Ensures the depth buffer matches the current screen size.
    ///
    /// Call this method at the start of each frame if the window may have been
    /// resized. If the depth buffer dimensions don't match the GPU context's
    /// current dimensions, a new depth texture is created.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context with the current screen dimensions
    pub fn ensure_depth_size(&mut self, gpu: &GpuContext) {
        if self.depth_size != (gpu.width(), gpu.height()) {
            let (texture, view) = Self::create_depth_texture(gpu);
            self.depth_texture = texture;
            self.depth_view = view;
            self.depth_size = (gpu.width(), gpu.height());
        }
    }

    /// Blits (copies) an input texture to the render target.
    ///
    /// This method renders a fullscreen quad textured with the input texture,
    /// effectively copying it to the current render target. It's typically used
    /// to composite the output of a previous render pass (such as 2D content)
    /// as the background before rendering 3D meshes on top.
    ///
    /// The blit uses linear filtering and replaces the destination completely
    /// (no blending).
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `render_pass` - The active render pass to draw into
    /// * `input_view` - The texture view to copy from
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Composite the 2D pass output as background
    /// mesh_pass.blit(&gpu, &mut render_pass, &draw2d_output_view);
    /// // Then render 3D meshes on top
    /// mesh_pass.render(&gpu, &mut render_pass, &camera, time, &draw_calls);
    /// ```
    pub fn blit(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        input_view: &wgpu::TextureView,
    ) {
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &self.blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                },
            ],
        });

        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }

    /// Renders a list of draw calls.
    ///
    /// This is the main rendering method for the mesh pass. It updates camera
    /// uniforms once per frame, then iterates through all draw calls, updating
    /// per-instance uniforms and issuing draw commands for each mesh.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context
    /// * `render_pass` - The active render pass to draw into (must have depth attachment)
    /// * `camera` - The camera providing view and projection matrices
    /// * `time` - Elapsed time in seconds (passed to shaders for animation)
    /// * `draw_calls` - Slice of draw calls to render
    ///
    /// # Behavior
    ///
    /// - Returns early if `draw_calls` is empty
    /// - Camera uniforms are updated once at the start
    /// - For each draw call:
    ///   - Model and normal matrices are computed from the transform
    ///   - A texture bind group is created (using default white if no texture specified)
    ///   - The mesh is drawn with indexed rendering
    ///
    /// # Performance Note
    ///
    /// Currently, model uniforms and texture bind groups are updated per draw call,
    /// which may cause GPU stalls for large numbers of meshes. For better performance
    /// with many instances, consider batching meshes with the same texture.
    pub fn render(
        &self,
        gpu: &GpuContext,
        render_pass: &mut wgpu::RenderPass,
        camera: &Camera,
        time: f32,
        draw_calls: &[DrawCall],
    ) {
        if draw_calls.is_empty() {
            return;
        }

        // Update camera uniforms
        let view = camera.view_matrix();
        let proj = camera.projection_matrix(gpu.aspect(), 0.1, 1000.0);
        let view_proj = proj * view;

        let camera_uniforms = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            proj: proj.to_cols_array_2d(),
            camera_pos: camera.position.to_array(),
            time,
        };

        gpu.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[camera_uniforms]),
        );

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.camera_bind_group, &[]);

        // Render each mesh
        for call in draw_calls {
            let model_matrix = call.transform.matrix();
            // Normal matrix is inverse transpose of model matrix (for non-uniform scaling)
            let normal_matrix = model_matrix.inverse().transpose();

            let model_uniforms = ModelUniforms {
                model: model_matrix.to_cols_array_2d(),
                normal_matrix: normal_matrix.to_cols_array_2d(),
                color: [call.color.r, call.color.g, call.color.b, call.color.a],
            };

            gpu.queue.write_buffer(
                &self.model_buffer,
                0,
                bytemuck::cast_slice(&[model_uniforms]),
            );

            render_pass.set_bind_group(1, &self.model_bind_group, &[]);

            // Bind texture (use default white texture if none provided)
            let texture = call.texture.unwrap_or(&self.default_texture);
            let texture_bind_group = self.create_texture_bind_group(gpu, texture);
            render_pass.set_bind_group(2, &texture_bind_group, &[]);

            render_pass.set_vertex_buffer(0, call.mesh.vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(call.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..call.mesh.index_count, 0, 0..1);
        }
    }
}
