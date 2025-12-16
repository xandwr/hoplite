//! Asset management for the Hoplite engine.
//!
//! This module provides resource loading and caching functionality, with a focus on
//! font rendering through GPU-accelerated font atlases. The primary types are:
//!
//! - [`Assets`] - The central asset manager for loading and caching resources
//! - [`FontAtlas`] - A pre-rasterized font texture for efficient text rendering
//! - [`FontId`] - An opaque handle to a loaded font
//! - [`GlyphInfo`] - Metrics and UV coordinates for individual glyphs
//!
//! # Font Rendering Pipeline
//!
//! Fonts are loaded using the `fontdue` library, rasterized at a specified size, and
//! packed into a texture atlas. The atlas uses a simple row-packing algorithm that
//! automatically grows to accommodate all ASCII printable characters (32-126).
//!
//! # Example
//!
//! ```ignore
//! // Load a custom font
//! let font_id = assets.load_font("fonts/MyFont.ttf", 24.0);
//!
//! // Or use the embedded default font
//! let default_id = assets.default_font(16.0);
//!
//! // Access the font atlas for rendering
//! if let Some(atlas) = assets.font(font_id) {
//!     let width = atlas.measure("Hello, world!");
//! }
//! ```

use crate::gpu::GpuContext;
use fontdue::{Font, FontSettings};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Opaque identifier for a loaded font.
///
/// This handle is returned when loading fonts through [`Assets`] and can be used
/// to retrieve the corresponding [`FontAtlas`] for rendering. The handle is lightweight
/// (just an index) and can be freely copied and stored.
///
/// # Example
///
/// ```ignore
/// let font_id = assets.load_font("my_font.ttf", 24.0);
/// let atlas = assets.font(font_id).expect("Font should exist");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontId(pub(crate) usize);

/// Information about a single glyph in the font atlas.
///
/// Contains all the data needed to render a character from the atlas texture:
/// - UV coordinates for sampling the correct region of the atlas
/// - Pixel dimensions for quad sizing
/// - Positioning offsets for proper baseline alignment
/// - Advance width for cursor movement
///
/// These metrics come directly from the `fontdue` rasterizer and follow standard
/// font metric conventions where positive Y points upward from the baseline.
#[derive(Clone, Copy, Debug)]
pub struct GlyphInfo {
    /// UV coordinates in the atlas as `[x, y, width, height]`, normalized to `[0, 1]`.
    ///
    /// Use these to calculate texture sampling coordinates:
    /// - Top-left: `(uv[0], uv[1])`
    /// - Bottom-right: `(uv[0] + uv[2], uv[1] + uv[3])`
    pub uv: [f32; 4],
    /// Width of the glyph in pixels.
    pub width: u32,
    /// Height of the glyph in pixels.
    pub height: u32,
    /// Horizontal offset from the cursor position to where the glyph should be drawn.
    ///
    /// This is the bearing/origin offset that positions the glyph correctly relative
    /// to the text cursor. Can be negative for glyphs that extend left of the cursor.
    pub offset_x: f32,
    /// Vertical offset from the baseline to the top of the glyph.
    ///
    /// Combined with `height`, this determines vertical positioning relative to the
    /// text baseline.
    pub offset_y: f32,
    /// How far to advance the cursor horizontally after rendering this glyph.
    ///
    /// This includes the glyph width plus any inter-character spacing defined by
    /// the font.
    pub advance: f32,
}

/// A font atlas containing pre-rasterized glyphs.
///
/// Font atlases pack multiple glyph bitmaps into a single GPU texture for efficient
/// text rendering. Instead of rendering each character individually, text renderers
/// can draw quads that sample different regions of the atlas texture.
///
/// # Atlas Generation
///
/// When created, the atlas:
/// 1. Rasterizes all ASCII printable characters (32-126) at the specified size
/// 2. Packs glyphs into a texture using row-based bin packing
/// 3. Starts at 512x512 and doubles dimensions as needed to fit all glyphs
/// 4. Stores glyph metrics for text layout calculations
///
/// # Texture Format
///
/// The atlas uses `R8Unorm` format (single-channel grayscale) to minimize memory.
/// Shaders should sample the red channel and use it as alpha for anti-aliased rendering.
///
/// # Thread Safety
///
/// The atlas is not `Send` or `Sync` due to the wgpu texture handles. Access should
/// be confined to the render thread.
pub struct FontAtlas {
    /// The GPU texture containing packed glyph bitmaps.
    pub texture: wgpu::Texture,
    /// Texture view for binding to shaders.
    pub view: wgpu::TextureView,
    /// Linear-filtered sampler for smooth text rendering.
    pub sampler: wgpu::Sampler,
    /// Mapping from characters to their glyph information.
    glyphs: HashMap<char, GlyphInfo>,
    /// The original font, retained for potential dynamic glyph rasterization.
    #[allow(dead_code)]
    font: Font,
    /// Font size in pixels that this atlas was rasterized at.
    size: f32,
    /// Recommended line height for this font and size.
    line_height: f32,
}

impl FontAtlas {
    /// Creates a new font atlas from TTF/OTF font data.
    ///
    /// This rasterizes all ASCII printable characters (codes 32-126) at the specified
    /// pixel size and packs them into a GPU texture atlas.
    ///
    /// # Arguments
    ///
    /// * `gpu` - The GPU context for creating textures
    /// * `font_data` - Raw bytes of a TTF or OTF font file
    /// * `size` - Font size in pixels (e.g., 16.0, 24.0)
    ///
    /// # Panics
    ///
    /// Panics if the font data cannot be parsed by `fontdue`.
    ///
    /// # Performance
    ///
    /// Atlas creation involves CPU-side rasterization of ~95 glyphs and a texture
    /// upload. This should be done during loading, not per-frame.
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

    /// Returns glyph information for a character.
    ///
    /// Returns `None` if the character is not in the atlas (i.e., not in ASCII 32-126).
    /// For missing glyphs, callers typically substitute a fallback like `'?'` or skip
    /// rendering.
    #[inline]
    pub fn glyph(&self, c: char) -> Option<&GlyphInfo> {
        self.glyphs.get(&c)
    }

    /// Returns the font size in pixels that this atlas was rasterized at.
    ///
    /// Text should be rendered at this size for optimal quality. Scaling the rendered
    /// quads will work but may appear blurry or pixelated.
    #[inline]
    pub fn size(&self) -> f32 {
        self.size
    }

    /// Returns the recommended line height for this font.
    ///
    /// This value comes from the font's horizontal line metrics and represents the
    /// distance between baselines for multi-line text. Falls back to `size * 1.2`
    /// if the font doesn't provide line metrics.
    #[inline]
    pub fn line_height(&self) -> f32 {
        self.line_height
    }

    /// Measures the total horizontal advance width of a string.
    ///
    /// This sums the advance widths of all characters in the string, giving the
    /// cursor position after rendering the full text. Characters not in the atlas
    /// are skipped (contribute zero width).
    ///
    /// # Note
    ///
    /// This does not account for kerning pairs. For precise text layout, a more
    /// sophisticated text shaping solution would be needed.
    pub fn measure(&self, text: &str) -> f32 {
        text.chars()
            .filter_map(|c| self.glyphs.get(&c))
            .map(|g| g.advance)
            .sum()
    }
}

/// Built-in embedded font data (JetBrains Mono Regular).
///
/// This font is compiled into the binary using `include_bytes!`, eliminating
/// runtime file dependencies. JetBrains Mono is used for its excellent readability
/// at small sizes, making it ideal for debug overlays and developer UI.
const EMBEDDED_FONT: &[u8] = include_bytes!("fonts/JetBrainsMono-Regular.ttf");

/// Asset manager for loading and caching resources.
///
/// The `Assets` struct is the central hub for loading and managing game resources.
/// Currently focused on font management, it provides methods to load fonts from
/// files or raw bytes, and access them via lightweight [`FontId`] handles.
///
/// # Lifetime Management
///
/// Assets holds a raw pointer to [`GpuContext`] for texture creation. This is safe
/// because:
/// - `Assets` is created by and owned by the application
/// - `GpuContext` outlives `Assets` in the application lifecycle
/// - Access is confined to the main/render thread
///
/// # Font Caching
///
/// Fonts are stored in an `Arc` to allow shared access from multiple renderers.
/// Each font loaded at a different size creates a separate atlas (no runtime scaling).
///
/// # Example
///
/// ```ignore
/// let mut assets = Assets::new(&gpu);
///
/// // Load fonts at initialization
/// let title_font = assets.load_font("fonts/title.ttf", 48.0);
/// let body_font = assets.load_font("fonts/body.ttf", 16.0);
/// let debug_font = assets.default_font(12.0);
///
/// // Use fonts for rendering
/// if let Some(atlas) = assets.font(title_font) {
///     // Render text using atlas.glyph(), atlas.measure(), etc.
/// }
/// ```
pub struct Assets {
    /// Raw pointer to the GPU context for creating textures.
    gpu: *const GpuContext,
    /// Loaded font atlases, indexed by [`FontId`].
    pub(crate) fonts: Vec<Arc<FontAtlas>>,
    /// Reserved field for caching the default font at a specific size.
    #[allow(dead_code)]
    default_font: Option<FontId>,
}

impl Assets {
    /// Creates a new asset manager.
    ///
    /// This is called internally by the application during initialization.
    /// The `gpu` reference must remain valid for the lifetime of the `Assets` instance.
    pub(crate) fn new(gpu: &GpuContext) -> Self {
        Self {
            gpu: gpu as *const GpuContext,
            fonts: Vec::new(),
            default_font: None,
        }
    }

    /// Returns a reference to the GPU context.
    ///
    /// # Safety
    ///
    /// This is safe because `Assets` lifetime is tied to the application lifetime,
    /// and `GpuContext` is guaranteed to outlive `Assets`.
    fn gpu(&self) -> &GpuContext {
        unsafe { &*self.gpu }
    }

    /// Loads a font from a file path.
    ///
    /// Reads the font file from disk and creates a [`FontAtlas`] at the specified size.
    /// The returned [`FontId`] can be used to retrieve the atlas for rendering.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to a TTF or OTF font file
    /// * `size` - Font size in pixels
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be read or the font data is invalid.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let font_id = assets.load_font("assets/fonts/Roboto.ttf", 24.0);
    /// ```
    pub fn load_font(&mut self, path: impl AsRef<Path>, size: f32) -> FontId {
        let data = std::fs::read(path.as_ref()).expect("Failed to read font file");
        self.load_font_bytes(&data, size)
    }

    /// Loads a font from raw TTF/OTF bytes.
    ///
    /// Useful for fonts embedded in the binary or loaded from custom sources
    /// (archives, network, etc.).
    ///
    /// # Arguments
    ///
    /// * `data` - Raw bytes of a TTF or OTF font file
    /// * `size` - Font size in pixels
    ///
    /// # Panics
    ///
    /// Panics if the font data cannot be parsed.
    pub fn load_font_bytes(&mut self, data: &[u8], size: f32) -> FontId {
        let atlas = FontAtlas::new(self.gpu(), data, size);
        let id = FontId(self.fonts.len());
        self.fonts.push(Arc::new(atlas));
        id
    }

    /// Gets or loads the default embedded font at the specified size.
    ///
    /// Uses the built-in JetBrains Mono font, which is ideal for debug text,
    /// developer consoles, and other UI elements requiring a monospace font.
    ///
    /// # Note
    ///
    /// Currently creates a new atlas for each call. For repeated use at the same
    /// size, store the returned [`FontId`] rather than calling this repeatedly.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load once during initialization
    /// let debug_font = assets.default_font(14.0);
    ///
    /// // Reuse the ID for rendering
    /// let atlas = assets.font(debug_font).unwrap();
    /// ```
    pub fn default_font(&mut self, size: f32) -> FontId {
        self.load_font_bytes(EMBEDDED_FONT, size)
    }

    /// Retrieves a font atlas by its ID.
    ///
    /// Returns `None` if the ID is invalid (e.g., from a different `Assets` instance
    /// or after the assets were cleared).
    ///
    /// The returned `Arc<FontAtlas>` can be cloned cheaply for use across multiple
    /// render passes or threads.
    #[inline]
    pub fn font(&self, id: FontId) -> Option<Arc<FontAtlas>> {
        self.fonts.get(id.0).cloned()
    }
}
