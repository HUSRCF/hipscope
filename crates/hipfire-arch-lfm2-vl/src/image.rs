// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Image preprocessing for the LFM2-VL vision encoder.
//!
//! A faithful port of HF `Lfm2VlImageProcessor` (transformers
//! `image_processing_lfm2_vl.py`, pinned 2026-08-27) minus batching and
//! NaFlex padding — hipfire processes one request-image at a time and each
//! sub-image unpadded, which is numerically identical for the rows that
//! survive unpadding (bidirectional attention; masked pad rows are dropped
//! by `get_image_features`).
//!
//! Differences vs `hipfire-arch-qwen35-vl::image` (do NOT unify):
//! round-to-factor uses Python semantics (`round()` = banker's rounding),
//! factor is 32 (= patch 16 × downsample 2), large images split into a
//! tile grid + thumbnail, normalization is IMAGENET_STANDARD, and patch
//! vectors flatten as `(dy, dx, channel)`.

use crate::config::VisionConfig;
use std::path::Path;

/// Decompression-bomb guard: checked from format-header dimensions BEFORE
/// any pixel buffer is allocated. Same contract as qwen35-vl (accept what a
/// 300–400 DPI A4 scan needs, reject 50000×50000 PNGs). UMA box budget is
/// downstream of this anyway via smart_resize shrinking to ≤262144 px.
const MAX_DIMENSION_PIXELS: usize = 16_777_216;

// IMAGENET_STANDARD mean/std (HF image_utils), channel order R,G,B.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct SubImage {
    /// Normalized CHW pixels (`[3][h][w]`, `(x/255 - mean)/std`).
    pub pixels: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

impl SubImage {
    pub fn gh(&self, cfg: &VisionConfig) -> usize {
        self.h / cfg.patch_size
    }
    pub fn gw(&self, cfg: &VisionConfig) -> usize {
        self.w / cfg.patch_size
    }
    /// Patchified rows `[gh·gw, ps·ps·C]`, per-patch layout `(dy, dx, c)`
    /// matching HF `convert_image_to_patches`' permute(0,2,4,3,5,1).
    pub fn patches(&self, cfg: &VisionConfig) -> Vec<f32> {
        let ps = cfg.patch_size;
        let c_n = cfg.num_channels;
        let gh = self.h / ps;
        let gw = self.w / ps;
        let mut out = vec![0.0f32; gh * gw * ps * ps * c_n];
        for py in 0..gh {
            for px in 0..gw {
                let row = py * gw + px;
                for dy in 0..ps {
                    for dx in 0..ps {
                        for c in 0..c_n {
                            out[(row * ps * ps + dy * ps + dx) * c_n + c] =
                                self.pixels[c * self.h * self.w + (py * ps + dy) * self.w + px * ps + dx];
                        }
                    }
                }
            }
        }
        out
    }
}

/// One fully preprocessed request image: ordered sub-images (tiles
/// row-major then thumbnail when split) plus the grid metadata needed for
/// the `<image>` placeholder expansion.
pub struct Prepared {
    pub sub_images: Vec<SubImage>,
    pub grid_cols: usize,
    pub grid_rows: usize,
}

impl Prepared {
    /// Total number of `<image>` placeholder tokens across all sub-images.
    pub fn total_tokens(&self, cfg: &VisionConfig) -> usize {
        self.sub_images.iter().map(|s| cfg.tokens_for_grid(s.gh(cfg), s.gw(cfg))).sum()
    }
}

fn round_half_even(x: f64) -> i64 {
    // Python round() → ties-to-even, unlike Rust's f64::round (ties-away).
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 && x.fract() != 0.0 {
        // exactly .5 and rounded to odd → step back toward even
        (r - x.signum()) as i64
    } else {
        r as i64
    }
}

fn round_by_factor(number: f64, factor: usize) -> i64 {
    round_half_even(number / factor as f64) * factor as i64
}

/// HF `Lfm2VlImageProcessor::smart_resize`. Returns `(h_bar, w_bar)`.
pub fn smart_resize(
    height: usize,
    width: usize,
    total_factor: usize,
    min_image_tokens: usize,
    max_image_tokens: usize,
    patch_size: usize,
    downsample_factor: usize,
) -> (usize, usize) {
    let ds2 = (downsample_factor * downsample_factor) as u64;
    let min_pixels = (min_image_tokens as u64) * (patch_size as u64).pow(2) * ds2;
    let max_pixels = (max_image_tokens as u64) * (patch_size as u64).pow(2) * ds2;

    let h_bar = (total_factor as i64).max(round_by_factor(height as f64, total_factor)) as usize;
    let w_bar = (total_factor as i64).max(round_by_factor(width as f64, total_factor)) as usize;

    let hw = height as u64 * width as u64;
    if h_bar as u64 * w_bar as u64 > max_pixels {
        let beta = (hw as f64 / max_pixels as f64).sqrt();
        let h = ((height as f64 / beta / total_factor as f64).floor() * total_factor as f64)
            .max(total_factor as f64) as usize;
        let w = ((width as f64 / beta / total_factor as f64).floor() * total_factor as f64)
            .max(total_factor as f64) as usize;
        (h, w)
    } else if (h_bar as u64) * (w_bar as u64) < min_pixels {
        let beta = (min_pixels as f64 / hw as f64).sqrt();
        let h = ((height as f64 * beta / total_factor as f64).ceil() * total_factor as f64) as usize;
        let w = ((width as f64 * beta / total_factor as f64).ceil() * total_factor as f64) as usize;
        (h.max(total_factor), w.max(total_factor))
    } else {
        (h_bar, w_bar)
    }
}

/// HF `_is_image_too_large`: rounded dims exceed
/// max_image_tokens · ps² · ds² · tolerance.
fn is_too_large(h: usize, w: usize, cfg: &VisionConfig) -> bool {
    let total_factor = cfg.patch_size * cfg.downsample_factor;
    let hb = (round_by_factor(h as f64, total_factor)).max(cfg.patch_size as i64) as usize;
    let wb = (round_by_factor(w as f64, total_factor)).max(cfg.patch_size as i64) as usize;
    let budget = (cfg.max_image_tokens as f64)
        * (cfg.patch_size.pow(2) as f64)
        * (cfg.downsample_factor.pow(2) as f64)
        * cfg.max_pixels_tolerance as f64;
    hb as f64 * wb as f64 > budget
}

/// HF `find_closest_aspect_ratio` over the `min_tiles..=max_tiles` grid set.
fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    target_ratios: &[(usize, usize)],
    width: usize,
    height: usize,
    tile_size: usize,
) -> (usize, usize) {
    let mut best_diff = f64::INFINITY;
    let mut best = (1usize, 1usize);
    let area = (width * height) as f64;
    for &(w, h) in target_ratios {
        let ratio = w as f64 / h as f64;
        let diff = (aspect_ratio - ratio).abs();
        if diff < best_diff {
            best_diff = diff;
            best = (w, h);
        } else if diff == best_diff {
            let target_area = (tile_size * tile_size * w * h) as f64;
            if area > 0.5 * target_area {
                best = (w, h);
            }
        }
    }
    best
}

fn target_ratios(min_tiles: usize, max_tiles: usize) -> Vec<(usize, usize)> {
    let mut ratios = Vec::new();
    for n in min_tiles..=max_tiles {
        for w in 1..=n {
            for h in 1..=n {
                if (min_tiles..=max_tiles).contains(&(w * h)) {
                    ratios.push((w, h));
                }
            }
        }
    }
    ratios.sort_by_key(|&(w, h)| w * h);
    ratios.dedup();
    ratios
}

fn resize_chw(img: &image::DynamicImage, new_w: usize, new_h: usize) -> Vec<f32> {
    // PIL bilinear ≈ image crate Triangle filter (same approximation class
    // the qwen35-vl preprocessing validated against HF references).
    let resized = img
        .resize_exact(new_w as u32, new_h as u32, image::imageops::FilterType::Triangle)
        .to_rgb8();
    let mut chw = vec![0.0f32; 3 * new_h * new_w];
    let plane = new_h * new_w;
    for y in 0..new_h {
        for x in 0..new_w {
            let p = resized.get_pixel(x as u32, y as u32);
            let idx = y * new_w + x;
            for (c, ch) in p.0.iter().take(3).enumerate() {
                let v = *ch as f32 / 255.0;
                chw[c * plane + idx] = (v - MEAN[c]) / STD[c];
            }
        }
    }
    chw
}

fn sub_from_dynamic(img: image::DynamicImage, new_h: usize, new_w: usize) -> SubImage {
    SubImage { pixels: resize_chw(&img, new_w, new_h), h: new_h, w: new_w }
}

fn preprocess(img: image::DynamicImage, cfg: &VisionConfig) -> Result<Prepared, String> {
    let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);
    if orig_w.checked_mul(orig_h).unwrap_or(u64::MAX as usize) > MAX_DIMENSION_PIXELS {
        return Err(format!(
            "image dimensions ({orig_w}x{orig_h}) exceed maximum ({MAX_DIMENSION_PIXELS} pixels)"
        ));
    }

    let total_factor = cfg.patch_size * cfg.downsample_factor;
    let (new_h, new_w) = smart_resize(
        orig_h,
        orig_w,
        total_factor,
        cfg.min_image_tokens,
        cfg.max_image_tokens,
        cfg.patch_size,
        cfg.downsample_factor,
    );

    let do_split = cfg.do_image_splitting && !(cfg.min_tiles == 1 && cfg.max_tiles == 1);
    let mut sub_images: Vec<SubImage> = Vec::new();
    let (grid_cols, grid_rows);

    if is_too_large(orig_h, orig_w, cfg) && do_split {
        let ar = orig_w as f64 / orig_h as f64;
        let (gc, gr) =
            find_closest_aspect_ratio(ar, &target_ratios(cfg.min_tiles, cfg.max_tiles), orig_w, orig_h, cfg.tile_size);
        let big = img.resize_exact(
            (cfg.tile_size * gc) as u32,
            (cfg.tile_size * gr) as u32,
            image::imageops::FilterType::Triangle,
        );
        let tw = cfg.tile_size;
        for ry in 0..gr {
            for rx in 0..gc {
                let tile = image::DynamicImage::from(big.crop_imm((rx * tw) as u32, (ry * tw) as u32, tw as u32, tw as u32));
                sub_images.push(sub_from_dynamic(tile, tw, tw));
            }
        }
        if cfg.use_thumbnail && !(gc == 1 && gr == 1) {
            sub_images.push(sub_from_dynamic(img.clone(), new_h, new_w));
        }
        grid_cols = gc;
        grid_rows = gr;
    } else {
        sub_images.push(sub_from_dynamic(img, new_h, new_w));
        grid_cols = 1;
        grid_rows = 1;
    }

    Ok(Prepared { sub_images, grid_cols, grid_rows })
}

/// Load + preprocess an image from a filesystem path.
pub fn load_and_preprocess(path: &Path, cfg: &VisionConfig) -> Result<Prepared, String> {
    let img =
        image::open(path).map_err(|e| format!("failed to open image {}: {e}", path.display()))?;
    preprocess(img, cfg)
}

/// Load + preprocess raw PNG/JPEG bytes.
pub fn load_and_preprocess_from_bytes(data: &[u8], cfg: &VisionConfig) -> Result<Prepared, String> {
    let reader = image::ImageReader::new(std::io::Cursor::new(data))
        .with_guessed_format()
        .map_err(|e| format!("failed to read image: {e}"))?;
    let (ow, oh) = reader.into_dimensions().map_err(map_image_err)?;
    if ow as usize * oh as usize > MAX_DIMENSION_PIXELS {
        return Err(format!(
            "image dimensions ({ow}x{oh}) exceed maximum ({MAX_DIMENSION_PIXELS} pixels)"
        ));
    }
    let img = image::load_from_memory(data).map_err(map_image_err)?;
    preprocess(img, cfg)
}

fn map_image_err(e: image::ImageError) -> String {
    match e {
        image::ImageError::Unsupported(_) => {
            "unsupported image format — supported: png, jpeg".to_string()
        }
        other => format!("failed to decode image: {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_half_even_matches_python() {
        assert_eq!(round_half_even(0.5), 0); // python round(0.5)=0
        assert_eq!(round_half_even(1.5), 2); // python round(1.5)=2
        assert_eq!(round_half_even(2.5), 2);
        assert_eq!(round_half_even(3.5), 4);
        assert_eq!(round_half_even(-0.5), 0);
        assert_eq!(round_half_even(10.4), 10);
        assert_eq!(round_half_even(10.6), 11);
    }

    #[test]
    fn smart_resize_bounds_and_factors() {
        let (h, w) = smart_resize(100, 100, 32, 64, 256, 16, 2);
        // min budget 65536 px: 100×100 too small → upscale; ceil path with
        // beta sqrt(65536/10000)≈2.561 → 256·… exact math: h=ceil(100*2.561/32)*32
        let beta = (65_536_f64 / 10_000_f64).sqrt();
        let eh = (100.0 * beta / 32.0).ceil() as usize * 32;
        assert_eq!((h, w), (eh, eh));
        assert_eq!(h % 32, 0);

        // large: 4000×3000 rounds into budget? rounded would exceed 262144
        // → floor shrink keeping aspect.
        let (h, w) = smart_resize(4000, 3000, 32, 64, 256, 16, 2);
        assert_eq!(h % 32, 0);
        assert_eq!(w % 32, 0);
        assert!(h as u64 * w as u64 <= 262_144);
    }

    #[test]
    fn banker_rounding_flip_case() {
        // An image whose height/factor sits on .5: Rust round differs from
        // python. 48.0*... pick value where n/f has fract .5: 80/32 = 2.5
        // → python round = 2 (even), rust = 3.
        let (_, w) = smart_resize(10, 80, 32, 1, 262_144_000, 16, 2);
        assert_eq!(w % 32, 0);
        // direct check of helper:
        assert_eq!(round_by_factor(80.0, 32), 64); // 2.5 → 2 (even)
        assert_eq!(round_by_factor(112.0, 32), 128); // 3.5 → 4 (even)
        assert_eq!(round_by_factor(111.9, 32), 96); // 3.4969 → 3
    }

    #[test]
    fn tokens_count_single_and_split() {
        let cfg = VisionConfig::default();
        // single sub-image 512×512 → 1024 patches → 256 tokens
        let s = SubImage { pixels: vec![0.0; 3 * 512 * 512], h: 512, w: 512 };
        assert_eq!(cfg.tokens_for_grid(s.gh(&cfg), s.gw(&cfg)), 256);
    }

    #[test]
    fn patches_layout_is_dy_dx_channel() {
        let cfg = VisionConfig::default();
        let ps = cfg.patch_size;
        // synthetic CHW with distinct values: pix[c][y][x]
        let h = ps;
        let w = ps * 2; // two patches horizontally
        let mut pixels = vec![0.0f32; 3 * h * w];
        for c in 0..3 {
            for y in 0..h {
                for x in 0..w {
                    pixels[c * h * w + y * w + x] = (c * 10_000 + y * 100 + x) as f32;
                }
            }
        }
        let s = SubImage { pixels, h, w };
        let p = s.patches(&cfg);
        assert_eq!(p.len(), 2 * ps * ps * 3);
        // patch 1 (px=1): value at dy=1,dx=0,c=0 → y=1,x=ps → 0+100+16
        assert_eq!(p[(1 * ps * ps + 1 * ps + 0) * 3 + 0], 116.0);
        // patch 0, dy=0,dx=1,c=2 → y=0,x=1 → 20000+0+1
        assert_eq!(p[(0 * ps * ps + 0 * ps + 1) * 3 + 2], 20_001.0);
    }
}
