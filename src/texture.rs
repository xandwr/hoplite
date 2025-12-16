use crate::gpu::GpuContext;

/// A GPU texture that can be bound to shaders.
#[derive(Debug)]
pub struct Texture {
    #[allow(dead_code)]
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
    pub(crate) sampler: wgpu::Sampler,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    /// Create a texture from raw RGBA data.
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
    pub fn from_file(gpu: &GpuContext, path: &str) -> Result<Self, image::ImageError> {
        let img = image::open(path)?.to_rgba8();
        let (width, height) = img.dimensions();
        Ok(Self::from_rgba(gpu, &img, width, height, path))
    }

    /// Load a texture from embedded bytes.
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
    /// Minecraft dirt/stone blocks.
    pub fn minecraft_noise(gpu: &GpuContext, size: u32, seed: u32) -> Self {
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
    pub fn minecraft_grass(gpu: &GpuContext, size: u32, seed: u32) -> Self {
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

    /// Generate a procedural cobblestone texture.
    pub fn minecraft_cobblestone(gpu: &GpuContext, size: u32, seed: u32) -> Self {
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
