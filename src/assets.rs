use crate::gpu::GpuContext;
use fontdue::{Font, FontSettings};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Opaque identifier for a loaded font.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub(crate) usize);

/// Information about a single glyph in the font atlas.
#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    /// UV coordinates in the atlas (x, y, width, height) normalized to [0, 1].
    pub uv: [f32; 4],
    /// Size of the glyph in pixels.
    pub width: u32,
    pub height: u32,
    /// Offset from the cursor position to where the glyph should be drawn.
    pub offset_x: f32,
    pub offset_y: f32,
    /// How far to advance the cursor after this glyph.
    pub advance: f32,
}

/// A font atlas containing pre-rasterized glyphs.
pub struct FontAtlas {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    glyphs: HashMap<char, GlyphInfo>,
    #[allow(dead_code)]
    font: Font, // Kept for potential dynamic glyph rasterization
    size: f32,
    line_height: f32,
}

impl FontAtlas {
    /// Create a new font atlas from TTF/OTF data.
    pub fn new(gpu: &GpuContext, font_data: &[u8], size: f32) -> Self {
        let font =
            Font::from_bytes(font_data, FontSettings::default()).expect("Failed to parse font");

        // Characters to pre-rasterize
        let chars: Vec<char> = (32u8..=126u8).map(|c| c as char).collect();

        // First pass: rasterize all glyphs to get their sizes
        let rasterized: Vec<(char, fontdue::Metrics, Vec<u8>)> = chars
            .iter()
            .map(|&c| {
                let (metrics, bitmap) = font.rasterize(c, size);
                (c, metrics, bitmap)
            })
            .collect();

        // Calculate atlas dimensions using a simple row packing
        let padding = 1u32;
        let mut atlas_width = 512u32;
        let mut atlas_height = 512u32;

        // Try to fit everything, increase size if needed
        loop {
            let mut x = padding;
            let mut y = padding;
            let mut row_height = 0u32;
            let mut fits = true;

            for (_, metrics, _) in &rasterized {
                let glyph_w = metrics.width as u32;
                let glyph_h = metrics.height as u32;

                if x + glyph_w + padding > atlas_width {
                    x = padding;
                    y += row_height + padding;
                    row_height = 0;
                }

                if y + glyph_h + padding > atlas_height {
                    fits = false;
                    break;
                }

                x += glyph_w + padding;
                row_height = row_height.max(glyph_h);
            }

            if fits {
                break;
            }

            // Double the smaller dimension
            if atlas_width <= atlas_height {
                atlas_width *= 2;
            } else {
                atlas_height *= 2;
            }
        }

        // Create atlas bitmap
        let mut atlas_data = vec![0u8; (atlas_width * atlas_height) as usize];
        let mut glyphs = HashMap::new();

        let mut x = padding;
        let mut y = padding;
        let mut row_height = 0u32;

        for (c, metrics, bitmap) in &rasterized {
            let glyph_w = metrics.width as u32;
            let glyph_h = metrics.height as u32;

            // Move to next row if needed
            if x + glyph_w + padding > atlas_width {
                x = padding;
                y += row_height + padding;
                row_height = 0;
            }

            // Copy glyph bitmap to atlas
            for gy in 0..glyph_h {
                for gx in 0..glyph_w {
                    let src_idx = (gy * glyph_w + gx) as usize;
                    let dst_idx = ((y + gy) * atlas_width + (x + gx)) as usize;
                    atlas_data[dst_idx] = bitmap[src_idx];
                }
            }

            // Store glyph info with normalized UVs
            let uv = [
                x as f32 / atlas_width as f32,
                y as f32 / atlas_height as f32,
                glyph_w as f32 / atlas_width as f32,
                glyph_h as f32 / atlas_height as f32,
            ];

            glyphs.insert(
                *c,
                GlyphInfo {
                    uv,
                    width: glyph_w,
                    height: glyph_h,
                    offset_x: metrics.xmin as f32,
                    offset_y: metrics.ymin as f32,
                    advance: metrics.advance_width,
                },
            );

            x += glyph_w + padding;
            row_height = row_height.max(glyph_h);
        }

        // Create GPU texture
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Font Atlas"),
            size: wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas_width),
                rows_per_image: Some(atlas_height),
            },
            wgpu::Extent3d {
                width: atlas_width,
                height: atlas_height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Font Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Calculate line height from font metrics
        let line_metrics = font.horizontal_line_metrics(size);
        let line_height = line_metrics.map(|m| m.new_line_size).unwrap_or(size * 1.2);

        Self {
            texture,
            view,
            sampler,
            glyphs,
            font,
            size,
            line_height,
        }
    }

    /// Get glyph info for a character.
    pub fn glyph(&self, c: char) -> Option<&GlyphInfo> {
        self.glyphs.get(&c)
    }

    /// Get the font size this atlas was created with.
    pub fn size(&self) -> f32 {
        self.size
    }

    /// Get the line height for this font.
    pub fn line_height(&self) -> f32 {
        self.line_height
    }

    /// Measure the width of a string.
    pub fn measure(&self, text: &str) -> f32 {
        text.chars()
            .filter_map(|c| self.glyphs.get(&c))
            .map(|g| g.advance)
            .sum()
    }
}

/// Built-in embedded font (JetBrains Mono subset for debug UI).
/// Using a minimal embedded font avoids file dependencies.
const EMBEDDED_FONT: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");

/// Asset manager for loading and caching resources.
pub struct Assets {
    gpu: *const GpuContext,
    pub(crate) fonts: Vec<Arc<FontAtlas>>,
    #[allow(dead_code)]
    default_font: Option<FontId>, // Reserved for caching
}

impl Assets {
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        Self {
            gpu: gpu as *const GpuContext,
            fonts: Vec::new(),
            default_font: None,
        }
    }

    fn gpu(&self) -> &GpuContext {
        // SAFETY: Assets lifetime is tied to the app lifetime,
        // and GpuContext outlives Assets
        unsafe { &*self.gpu }
    }

    /// Load a font from a file path.
    pub fn load_font(&mut self, path: impl AsRef<Path>, size: f32) -> FontId {
        let data = std::fs::read(path.as_ref()).expect("Failed to read font file");
        self.load_font_bytes(&data, size)
    }

    /// Load a font from raw TTF/OTF bytes.
    pub fn load_font_bytes(&mut self, data: &[u8], size: f32) -> FontId {
        let atlas = FontAtlas::new(self.gpu(), data, size);
        let id = FontId(self.fonts.len());
        self.fonts.push(Arc::new(atlas));
        id
    }

    /// Get or load the default embedded font at the given size.
    pub fn default_font(&mut self, size: f32) -> FontId {
        // For simplicity, always create at requested size
        // A more sophisticated impl would cache by size
        self.load_font_bytes(EMBEDDED_FONT, size)
    }

    /// Get a font atlas by ID.
    pub fn font(&self, id: FontId) -> Option<Arc<FontAtlas>> {
        self.fonts.get(id.0).cloned()
    }
}
