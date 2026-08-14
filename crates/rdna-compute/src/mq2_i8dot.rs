//! Host-side conversion helpers for the gfx90a MQ2-I8DOT kernels.

use half::f16;
use std::sync::OnceLock;

#[derive(Default)]
struct LayerSelection {
    enabled: bool,
    ranges: Option<Vec<(usize, usize)>>,
}

fn parse_layer_ranges(selector: &str) -> Vec<(usize, usize)> {
    selector
        .split(',')
        .filter_map(|item| {
            let item = item.trim();
            if let Some((start, end)) = item.split_once('-') {
                let start = start.trim().parse::<usize>().ok()?;
                let end = end.trim().parse::<usize>().ok()?;
                (start <= end).then_some((start, end))
            } else {
                item.parse::<usize>().ok().map(|layer| (layer, layer))
            }
        })
        .collect()
}

fn layer_selection() -> &'static LayerSelection {
    static SELECTION: OnceLock<LayerSelection> = OnceLock::new();
    SELECTION.get_or_init(|| {
        let enabled =
            hipfire_config::developer_var("HIPFIRE_GFX90A_MQ2_I8DOT").as_deref() == Ok("1");
        let ranges = hipfire_config::developer_var("HIPFIRE_GFX90A_MQ2_I8DOT_LAYERS")
            .ok()
            .map(|selector| parse_layer_ranges(&selector));
        LayerSelection { enabled, ranges }
    })
}

/// Whether the opt-in I8DOT path is enabled for a specific transformer layer.
/// The process-start selector is parsed once because this runs for every layer
/// on every token.
pub fn layer_enabled(layer: Option<usize>) -> bool {
    let selection = layer_selection();
    if !selection.enabled {
        return false;
    }
    match (&selection.ranges, layer) {
        (None, _) => true,
        (Some(_), None) => false,
        (Some(ranges), Some(layer)) => ranges
            .iter()
            .any(|&(start, end)| start <= layer && layer <= end),
    }
}

/// Re-encode row-major MQ2-Lloyd groups as affine-I8 metadata. Both formats
/// occupy 72 bytes per 256 weights, so the model footprint is unchanged.
pub fn transcode_affine(weights: &[u8]) -> Result<Vec<u8>, String> {
    if weights.len() % 72 != 0 {
        return Err(format!(
            "MQ2-Lloyd byte count {} is not group aligned",
            weights.len()
        ));
    }
    let mut out = Vec::with_capacity(weights.len());
    for group in weights.chunks_exact(72) {
        let c: [f32; 4] = std::array::from_fn(|i| {
            f16::from_bits(u16::from_le_bytes([group[2 * i], group[2 * i + 1]])).to_f32()
        });
        let mut n = [0.0_f32; 4];
        for &packed in &group[8..] {
            for shift in [0, 2, 4, 6] {
                n[((packed >> shift) & 3) as usize] += 1.0;
            }
        }
        let s0 = ((c[3] - c[0]) / 254.0).max(1.0e-12);
        let b0 = 0.5 * (c[3] + c[0]);
        let mut q = [-127_i32, 0, 0, 127];
        q[1] = (((c[1] - b0) / s0).round() as i32).clamp(-126, 125);
        q[2] = (((c[2] - b0) / s0).round() as i32).clamp(q[1] + 1, 126);
        let sn = n.iter().sum::<f32>();
        let snq = (0..4).map(|i| n[i] * q[i] as f32).sum::<f32>();
        let snqq = (0..4).map(|i| n[i] * (q[i] * q[i]) as f32).sum::<f32>();
        let snc = (0..4).map(|i| n[i] * c[i]).sum::<f32>();
        let snqc = (0..4).map(|i| n[i] * q[i] as f32 * c[i]).sum::<f32>();
        let det = snqq * sn - snq * snq;
        let scale = if det.abs() > 1.0e-20 {
            (snqc * sn - snc * snq) / det
        } else {
            s0
        };
        let bias = if sn > 0.0 {
            (snc - scale * snq) / sn
        } else {
            b0
        };
        out.extend(q.map(|value| value as i8 as u8));
        out.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        out.extend_from_slice(&f16::from_f32(bias).to_bits().to_le_bytes());
        out.extend_from_slice(&group[8..]);
    }
    Ok(out)
}

/// Reorder one affine-I8 matrix into the row8/SG8 compute order consumed by
/// the gfx90a PIPE2 kernel. `m` includes both gate and up rows.
pub fn tile_sg8(weights: &[u8], m: usize, k: usize) -> Result<Vec<u8>, String> {
    if m % 8 != 0 || k % 256 != 0 || weights.len() != m * (k / 256) * 72 {
        return Err(format!(
            "invalid MQ2-I8DOT tile shape: bytes={} m={m} k={k}",
            weights.len()
        ));
    }
    let groups = k / 256;
    let row_bytes = groups * 72;
    let mut tiled = Vec::with_capacity(weights.len());
    for tile in 0..m / 8 {
        for group in 0..groups {
            for row_local in 0..8 {
                let base = (tile * 8 + row_local) * row_bytes + group * 72;
                tiled.extend_from_slice(&weights[base..base + 8]);
            }
            for batch in 0..2 {
                for subgroup in 0..8 {
                    let row_local = batch * 4 + subgroup / 2;
                    let half = subgroup & 1;
                    let base = (tile * 8 + row_local) * row_bytes + group * 72;
                    for lane8 in 0..8 {
                        for chunk in 0..4 {
                            tiled.push(weights[base + 8 + half * 32 + lane8 + chunk * 8]);
                        }
                    }
                }
            }
        }
    }
    Ok(tiled)
}
