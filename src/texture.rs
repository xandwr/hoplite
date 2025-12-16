//! GPU texture and sprite management.
//!
//! This module provides abstractions for working with GPU textures in wgpu:
//!
//! - [`Texture`] - General-purpose textures for 3D rendering (e.g., block textures)
//! - [`Sprite`] - 2D textures optimized for UI/HUD overlay rendering
//!
//! # Texture vs Sprite
//!
//! Both types wrap wgpu textures but are configured differently:
//!
//! | Feature | Texture | Sprite |
//! |---------|---------|--------|
//! | Filtering | Nearest (pixelated) | Linear (smooth) or Nearest |
//! | Address mode | Repeat (tiling) | Clamp to edge |
//! | Use case | 3D world textures | UI elements, HUD |
//!
//! # Examples
//!
//! ```ignore
//! // Load a texture from a file
//! let texture = Texture::from_file(&gpu, "assets/stone.png")?;
//!
//! // Generate a procedural blocky texture
//! let grass = Texture::blocky_grass(&gpu, 16, 42);
//!
//! // Load a sprite for UI
//! let icon = Sprite::from_bytes(&gpu, include_bytes!("icon.png"), "icon")?;
//! ```

use crate::gpu::GpuContext;

/// A GPU texture that can be bound to shaders.
///
/// Textures are configured with nearest-neighbor filtering and repeating address mode,
/// making them ideal for Minecraft-style blocky/pixelated 3D rendering where textures
/// tile across surfaces.
#[derive(Debug)]
pub struct Texture {
    /// The underlying wgpu texture resource.
    #[allow(dead_code)]
    pub(crate) texture: wgpu::Texture,
    /// View into the texture for shader binding.
    pub(crate) view: wgpu::TextureView,
    /// Sampler defining how the texture is filtered and addressed.
    pub(crate) sampler: wgpu::Sampler,
    /// Width of the texture in pixels.
    pub width: u32,
    /// Height of the texture in pixels.
    pub height: u32,
}

/// A 2D sprite for UI/HUD rendering.
///
/// Sprites are rendered in the 2D layer on top of the 3D scene using screen-space
/// pixel coordinates. Unlike [`Texture`], sprites use clamp-to-edge addressing
/// (no tiling) and offer both linear (smooth) and nearest (pixelated) filtering options.
///
/// # Filtering Modes
///
/// - `from_rgba` / `from_file` / `from_bytes` - Linear filtering for smooth scaling
/// - `from_rgba_nearest` / `from_file_nearest` / `from_bytes_nearest` - Nearest-neighbor
///   filtering for pixel art that should stay crisp
#[derive(Debug)]
pub struct Sprite {
    /// View into the texture for shader binding.
    pub(crate) view: wgpu::TextureView,
    /// Sampler defining how the sprite is filtered and addressed.
    pub(crate) sampler: wgpu::Sampler,
    /// Width of the sprite in pixels.
    pub width: u32,
    /// Height of the sprite in pixels.
    pub height: u32,
}

impl Texture {
    /// Create a texture from raw RGBA data.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `data` - Raw pixel data in RGBA8 format (4 bytes per pixel)
    /// * `width` - Texture width in pixels
    /// * `height` - Texture height in pixels
    /// * `label` - Debug label for the texture (visible in graphics debuggers)
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 4`.
    pub fn from_rgba(gpu: &GpuContext, data: &[u8], width: u32, height: u32, label: &str) -> Self {
        use wgpu::util::DeviceExt;

        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use nearest-neighbor filtering for that crispy Minecraft look
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} Sampler", label)),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            view,
            sampler,
            width,
            height,
        }
    }

    /// Load a texture from an image file.
    ///
    /// Supports common image formats (PNG, JPEG, etc.) via the `image` crate.
    /// The image is automatically converted to RGBA8 format.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `path` - Path to the image file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or decoded.
    pub fn from_file(gpu: &GpuContext, path: &str) -> Result<Self, image::ImageError> {
        let img = image::open(path)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba(gpu, &img, width, height, path))
    }

    /// Load a texture from embedded bytes.
    ///
    /// Useful for loading textures embedded in the binary via `include_bytes!`.
    /// Supports common image formats (PNG, JPEG, etc.).
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `bytes` - Raw image file bytes (not raw pixels - this is decoded as an image)
    /// * `label` - Debug label for the texture
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be decoded as an image.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let texture = Texture::from_bytes(
    ///     &gpu,
    ///     include_bytes!("../assets/stone.png"),
    ///     "stone",
    /// )?;
    /// ```
    pub fn from_bytes(
        gpu: &GpuContext,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, image::ImageError> {
        let img = image::load_from_memory(bytes)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba(gpu, &img, width, height, label))
    }

    /// Generate a procedural Minecraft-style noise texture.
    ///
    /// Creates a blocky, pixelated texture with earthy colors reminiscent of
    /// Minecraft dirt/stone blocks. Uses a hash-based noise function for
    /// deterministic, reproducible results.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `size` - Texture dimensions (creates a `size × size` square texture)
    /// * `seed` - Random seed for reproducible generation
    ///
    /// # Color Palette
    ///
    /// Uses earthy tones: browns, grays, and tans typical of dirt/stone blocks.
    pub fn blocky_noise(gpu: &GpuContext, size: u32, seed: u32) -> Self {
        let mut data = vec![0u8; (size * size * 4) as usize];

        // Minecraft-style color palette (earthy tones)
        let colors: &[[u8; 3]] = &[
            [139, 90, 43],   // Brown (dirt)
            [128, 128, 128], // Gray (stone)
            [85, 85, 85],    // Dark gray
            [160, 120, 60],  // Light brown
            [100, 70, 40],   // Dark brown
            [90, 90, 90],    // Medium gray
            [120, 100, 70],  // Tan
            [70, 60, 50],    // Very dark brown
        ];

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;

                // Simple hash-based noise for blocky look
                let hash = Self::hash(x, y, seed);

                // Pick a base color from palette
                let color_idx = (hash % colors.len() as u32) as usize;
                let base = colors[color_idx];

                // Add some variation
                let variation = ((Self::hash(x + 1000, y + 1000, seed) % 30) as i32) - 15;

                data[idx] = (base[0] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 1] = (base[1] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 2] = (base[2] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 3] = 255;
            }
        }

        Self::from_rgba(gpu, &data, size, size, "Minecraft Noise Texture")
    }

    /// Generate a procedural grass-top block texture.
    ///
    /// Creates a Minecraft-style grass texture using various shades of green.
    /// Suitable for the top face of grass blocks.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `size` - Texture dimensions (creates a `size × size` square texture)
    /// * `seed` - Random seed for reproducible generation
    pub fn blocky_grass(gpu: &GpuContext, size: u32, seed: u32) -> Self {
        let mut data = vec![0u8; (size * size * 4) as usize];

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;
                let hash = Self::hash(x, y, seed);

                // Green grass colors
                let greens: &[[u8; 3]] = &[
                    [86, 125, 70], // Grass green
                    [75, 115, 60], // Darker grass
                    [95, 135, 75], // Lighter grass
                    [80, 120, 65], // Medium grass
                ];

                let color_idx = (hash % greens.len() as u32) as usize;
                let base = greens[color_idx];

                let variation = ((Self::hash(x + 500, y + 500, seed) % 20) as i32) - 10;

                data[idx] = (base[0] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 1] = (base[1] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 2] = (base[2] as i32 + variation).clamp(0, 255) as u8;
                data[idx + 3] = 255;
            }
        }

        Self::from_rgba(gpu, &data, size, size, "Minecraft Grass Texture")
    }

    /// Generate a procedural stone texture.
    ///
    /// Creates a Minecraft-style stone texture with blocky stone patterns
    /// and subtle cracks between stones. Uses 4×4 pixel blocks to create the
    /// characteristic stone appearance.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `size` - Texture dimensions (creates a `size × size` square texture)
    /// * `seed` - Random seed for reproducible generation
    pub fn blocky_stone(gpu: &GpuContext, size: u32, seed: u32) -> Self {
        let mut data = vec![0u8; (size * size * 4) as usize];

        for y in 0..size {
            for x in 0..size {
                let idx = ((y * size + x) * 4) as usize;

                // Create a blocky pattern with "stones" of varying sizes
                let block_x = x / 4;
                let block_y = y / 4;
                let block_hash = Self::hash(block_x, block_y, seed);

                // Gray stone colors
                let grays: &[[u8; 3]] = &[
                    [128, 128, 128], // Medium gray
                    [100, 100, 100], // Dark gray
                    [150, 150, 150], // Light gray
                    [90, 90, 90],    // Darker
                    [140, 140, 140], // Lighter
                    [110, 110, 110], // Medium dark
                ];

                let color_idx = (block_hash % grays.len() as u32) as usize;
                let base = grays[color_idx];

                // Add per-pixel variation for texture
                let pixel_hash = Self::hash(x, y, seed + 12345);
                let variation = ((pixel_hash % 16) as i32) - 8;

                // Add "cracks" between blocks
                let in_crack = (x % 4 == 0 || y % 4 == 0) && Self::hash(x, y, seed + 999) % 3 == 0;
                let crack_darken = if in_crack { -20 } else { 0 };

                data[idx] = (base[0] as i32 + variation + crack_darken).clamp(0, 255) as u8;
                data[idx + 1] = (base[1] as i32 + variation + crack_darken).clamp(0, 255) as u8;
                data[idx + 2] = (base[2] as i32 + variation + crack_darken).clamp(0, 255) as u8;
                data[idx + 3] = 255;
            }
        }

        Self::from_rgba(gpu, &data, size, size, "Minecraft Cobblestone Texture")
    }

    /// Simple hash function for procedural generation.
    ///
    /// Combines x, y coordinates with a seed to produce a deterministic pseudo-random
    /// value. Uses multiplicative hashing with carefully chosen prime constants.
    ///
    /// This is not cryptographically secure but provides good distribution for
    /// texture generation purposes.
    fn hash(x: u32, y: u32, seed: u32) -> u32 {
        let mut h = seed;
        h = h.wrapping_add(x.wrapping_mul(374761393));
        h = h.wrapping_add(y.wrapping_mul(668265263));
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177);
        h ^= h >> 16;
        h
    }
}

impl Sprite {
    /// Create a sprite from raw RGBA data with linear filtering.
    ///
    /// Linear filtering produces smooth results when the sprite is scaled,
    /// suitable for most UI elements. For pixel art, use [`from_rgba_nearest`](Self::from_rgba_nearest).
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `data` - Raw pixel data in RGBA8 format (4 bytes per pixel)
    /// * `width` - Sprite width in pixels
    /// * `height` - Sprite height in pixels
    /// * `label` - Debug label for the sprite
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 4`.
    pub fn from_rgba(gpu: &GpuContext, data: &[u8], width: u32, height: u32, label: &str) -> Self {
        use wgpu::util::DeviceExt;

        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Use linear filtering for smooth sprites (can be changed to Nearest for pixel art)
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} Sampler", label)),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            view,
            sampler,
            width,
            height,
        }
    }

    /// Create a sprite with nearest-neighbor filtering (pixel art style).
    ///
    /// Nearest-neighbor filtering preserves sharp pixel edges when scaling,
    /// ideal for pixel art sprites that should remain crisp.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `data` - Raw pixel data in RGBA8 format (4 bytes per pixel)
    /// * `width` - Sprite width in pixels
    /// * `height` - Sprite height in pixels
    /// * `label` - Debug label for the sprite
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height * 4`.
    pub fn from_rgba_nearest(
        gpu: &GpuContext,
        data: &[u8],
        width: u32,
        height: u32,
        label: &str,
    ) -> Self {
        use wgpu::util::DeviceExt;

        let texture = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some(&format!("{} Sampler", label)),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            view,
            sampler,
            width,
            height,
        }
    }

    /// Load a sprite from an image file with linear filtering.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `path` - Path to the image file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or decoded.
    pub fn from_file(gpu: &GpuContext, path: &str) -> Result<Self, image::ImageError> {
        let img = image::open(path)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba(gpu, &img, width, height, path))
    }

    /// Load a sprite from an image file with nearest-neighbor filtering.
    ///
    /// Use this for pixel art sprites that should remain crisp when scaled.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `path` - Path to the image file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or decoded.
    pub fn from_file_nearest(gpu: &GpuContext, path: &str) -> Result<Self, image::ImageError> {
        let img = image::open(path)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba_nearest(gpu, &img, width, height, path))
    }

    /// Load a sprite from embedded bytes with linear filtering.
    ///
    /// Useful for loading sprites embedded in the binary via `include_bytes!`.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `bytes` - Raw image file bytes (decoded as an image, not raw pixels)
    /// * `label` - Debug label for the sprite
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be decoded as an image.
    pub fn from_bytes(
        gpu: &GpuContext,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, image::ImageError> {
        let img = image::load_from_memory(bytes)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba(gpu, &img, width, height, label))
    }

    /// Load a sprite from embedded bytes with nearest-neighbor filtering.
    ///
    /// Use this for pixel art sprites that should remain crisp when scaled.
    /// Useful for loading sprites embedded in the binary via `include_bytes!`.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for resource creation
    /// * `bytes` - Raw image file bytes (decoded as an image, not raw pixels)
    /// * `label` - Debug label for the sprite
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be decoded as an image.
    pub fn from_bytes_nearest(
        gpu: &GpuContext,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self, image::ImageError> {
        let img = image::load_from_memory(bytes)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba_nearest(gpu, &img, width, height, label))
    }
}
