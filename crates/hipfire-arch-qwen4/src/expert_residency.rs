// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Routed-expert residency for cards whose VRAM cannot hold every expert.
//!
//! Flash-Next's routed experts are ~65 GB, the rest of the model ~5 GB. On a
//! discrete card the experts of trunk layers at or past the VRAM layer count
//! ([`EXPERT_VRAM_LAYERS_ENV`]) and the MTP layer's are fulfilled into pinned,
//! device-mapped host RAM instead of VRAM. The sealed MoE kernels already read
//! every expert through per-layer pointer tables, so a host-mapped layer is
//! read over PCIe (zero-copy) with no kernel or dispatch change: a decode
//! token reads only its routed experts' bytes from host RAM.

use hipfire_runtime::weight_manifest::{ShardPolicy, WeightEntry, WeightResidency};

/// `N` keeps the routed experts of trunk layers `0..N` in VRAM; `auto` picks
/// the largest `N` that fits the card's free VRAM. Unset keeps every expert
/// resident (the fully resident load).
pub const EXPERT_VRAM_LAYERS_ENV: &str = "HIPFIRE_QWEN4_EXPERT_VRAM_LAYERS";

/// VRAM left free by `auto` beyond the resident non-expert weights: forward
/// scratch, KV and state for the 2048-token contract plus headroom. Measured
/// on gfx1201: a load with every routed expert in host RAM used 5.28 GiB
/// beyond the non-expert weights.
pub const AUTO_VRAM_RESERVE_BYTES: u64 = 6656 << 20;

/// Host RAM that must remain available after the pinned experts are placed.
pub const HOST_RAM_HEADROOM_BYTES: u64 = 4 << 30;

/// Slack `Gpu::upload_raw_host_mapped` adds to every host-mapped tensor.
pub const HOST_MAPPED_PAD_BYTES: u64 = 1 << 20;

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertVramLayers {
    Layers(usize),
    Auto,
}

/// Parse [`EXPERT_VRAM_LAYERS_ENV`]. `Ok(None)` = unset (fully resident).
pub fn expert_vram_layers_from_env() -> Result<Option<ExpertVramLayers>, String> {
    match hipfire_config::developer_var(EXPERT_VRAM_LAYERS_ENV) {
        Ok(value) => parse_expert_vram_layers(&value).map(Some),
        Err(_) => Ok(None),
    }
}

fn parse_expert_vram_layers(value: &str) -> Result<ExpertVramLayers, String> {
    let value = value.trim();
    if value == "auto" {
        return Ok(ExpertVramLayers::Auto);
    }
    value
        .parse::<usize>()
        .map(ExpertVramLayers::Layers)
        .map_err(|_| format!("{EXPERT_VRAM_LAYERS_ENV}={value:?} is neither `auto` nor a layer count"))
}

fn is_routed_expert(name: &str) -> bool {
    name.ends_with(".mlp.experts.gate_up_proj") || name.ends_with(".mlp.experts.down_proj")
}

/// Resident bytes outside the routed experts, and the routed-expert bytes of
/// one trunk layer (layer 0), from each entry's payload size. Tied aliases
/// have no payload of their own and external rows stay in the file.
pub fn resident_split(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<(u64, u64), String> {
    let mut non_expert = 0u64;
    let mut layer_experts = 0u64;
    for entry in weights {
        if entry.residency.is_external() || matches!(entry.policy, ShardPolicy::Tied { .. }) {
            continue;
        }
        let bytes = bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
        if !is_routed_expert(&entry.name) {
            non_expert += bytes;
        } else if !entry.name.starts_with("mtp.") && entry.layer == Some(0) {
            layer_experts += bytes;
        }
    }
    Ok((non_expert, layer_experts))
}

/// The largest trunk-layer count whose routed experts fit in `free_vram` after
/// the non-expert weights and [`AUTO_VRAM_RESERVE_BYTES`].
pub fn auto_vram_layers(
    free_vram: u64,
    non_expert_bytes: u64,
    layer_expert_bytes: u64,
    num_layers: usize,
) -> usize {
    let budget = free_vram
        .saturating_sub(non_expert_bytes)
        .saturating_sub(AUTO_VRAM_RESERVE_BYTES);
    match budget.checked_div(layer_expert_bytes) {
        Some(layers) => (layers as usize).min(num_layers),
        None => num_layers,
    }
}

/// Mark the routed-expert entries of trunk layers `>= vram_layers`, and of
/// the MTP layer, [`WeightResidency::HostMapped`]. Returns the number of
/// entries moved to host RAM.
pub fn place_routed_experts(weights: &mut [WeightEntry], vram_layers: usize) -> usize {
    let mut moved = 0;
    for entry in weights.iter_mut() {
        if entry.residency != WeightResidency::Resident || !is_routed_expert(&entry.name) {
            continue;
        }
        let trunk_resident = !entry.name.starts_with("mtp.")
            && entry.layer.is_some_and(|layer| layer < vram_layers);
        if !trunk_resident {
            entry.residency = WeightResidency::HostMapped;
            moved += 1;
        }
    }
    moved
}

/// Pinned host bytes the [`WeightResidency::HostMapped`] entries will take.
pub fn host_mapped_bytes(
    weights: &[WeightEntry],
    bytes_of: impl Fn(&WeightEntry) -> Option<u64>,
) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in weights {
        if entry.residency == WeightResidency::HostMapped {
            let bytes =
                bytes_of(entry).ok_or_else(|| format!("no payload size for '{}'", entry.name))?;
            total += bytes + HOST_MAPPED_PAD_BYTES;
        }
    }
    Ok(total)
}

/// Refuse before any allocation when the pinned experts would not leave
/// [`HOST_RAM_HEADROOM_BYTES`] of `MemAvailable`. Pinned pages cannot be
/// reclaimed, so over-committing them starves the rest of the host.
pub fn check_host_ram(host_bytes: u64, mem_available: Option<u64>) -> Result<(), String> {
    if host_bytes == 0 {
        return Ok(());
    }
    let Some(available) = mem_available else {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM, but MemAvailable is unreadable",
            host_bytes as f64 / GIB
        ));
    };
    let needed = host_bytes + HOST_RAM_HEADROOM_BYTES;
    if available < needed {
        return Err(format!(
            "routed experts need {:.1} GiB of pinned host RAM plus {:.0} GiB headroom, but \
             MemAvailable is {:.1} GiB; free host memory or keep more expert layers in VRAM \
             ({EXPERT_VRAM_LAYERS_ENV})",
            host_bytes as f64 / GIB,
            HOST_RAM_HEADROOM_BYTES as f64 / GIB,
            available as f64 / GIB
        ));
    }
    Ok(())
}

/// TTM's page limit, in pages, which caps every GTT allocation on the host.
const TTM_PAGES_LIMIT: &str = "/sys/module/ttm/parameters/pages_limit";

/// TTM's page size on x86_64, the only host ROCm supports for discrete GPUs.
const TTM_PAGE_BYTES: u64 = 4096;

/// GTT room on the host: TTM's cap and what amdgpu devices already hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GttBudget {
    pub limit_bytes: u64,
    pub used_bytes: u64,
}

/// The GTT budget when host-mapped memory is GTT-backed, i.e. when
/// `HSA_USERPTR_FOR_PAGED_MEM` is `0` (hip-bridge's default). It is `None`
/// under userptr, or when sysfs does not expose TTM's limit.
///
/// Host allocations may count against another device than the one the
/// process runs on (on a 5-card gfx1201 host, a card-2 process's host-mapped
/// experts show in card 0's `mem_info_gtt_used`), so this sums
/// `mem_info_gtt_used` over every amdgpu device.
pub fn gtt_budget() -> Option<GttBudget> {
    if std::env::var("HSA_USERPTR_FOR_PAGED_MEM").ok()?.trim() != "0" {
        return None;
    }
    let read_u64 = |path: &std::path::Path| -> Option<u64> {
        std::fs::read_to_string(path).ok()?.trim().parse().ok()
    };
    let limit_bytes = read_u64(TTM_PAGES_LIMIT.as_ref())?.checked_mul(TTM_PAGE_BYTES)?;
    let mut used_bytes = 0u64;
    for card in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let name = card.file_name();
        let is_card = name
            .to_str()
            .and_then(|name| name.strip_prefix("card"))
            .is_some_and(|index| !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()));
        if is_card {
            used_bytes += read_u64(&card.path().join("device/mem_info_gtt_used")).unwrap_or(0);
        }
    }
    Some(GttBudget { limit_bytes, used_bytes })
}

/// Refuse before any allocation when GTT-backed host-mapped experts would not
/// fit under TTM's `pages_limit` beside what amdgpu devices already hold.
/// Past it `hipHostMalloc` fails, after the load has uploaded the VRAM
/// weights. `None` (userptr, or no TTM limit in sysfs) skips the check.
pub fn check_gtt_cap(host_bytes: u64, budget: Option<GttBudget>) -> Result<(), String> {
    let Some(GttBudget { limit_bytes, used_bytes }) = budget else {
        return Ok(());
    };
    let free = limit_bytes.saturating_sub(used_bytes);
    if host_bytes > free {
        return Err(format!(
            "routed experts need {:.1} GiB of GTT-backed host RAM, but TTM's GTT cap \
             ({TTM_PAGES_LIMIT} = {} pages, {:.1} GiB) has {:.1} GiB left beside the {:.1} GiB \
             amdgpu devices already hold; keep more expert layers in VRAM \
             ({EXPERT_VRAM_LAYERS_ENV}), raise ttm.pages_limit, or set \
             HSA_USERPTR_FOR_PAGED_MEM=1 for pageable userptr host memory, which host-memory \
             pressure can stall (ROCm/rocm-systems#12528)",
            host_bytes as f64 / GIB,
            limit_bytes / TTM_PAGE_BYTES,
            limit_bytes as f64 / GIB,
            free as f64 / GIB,
            used_bytes as f64 / GIB
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::weight_manifest::ShardPolicy;
    use rdna_compute::DType;

    fn entry(name: &str, layer: usize) -> WeightEntry {
        let mut entry =
            WeightEntry::model(name, vec![4, 4, 256], DType::MQ4G256V2, ShardPolicy::Replicate);
        entry.layer = Some(layer);
        entry
    }

    #[test]
    fn layers_past_the_budget_and_mtp_go_to_host() {
        let mut weights = vec![
            entry("model.language_model.layers.0.mlp.experts.gate_up_proj", 0),
            entry("model.language_model.layers.1.mlp.experts.down_proj", 1),
            entry("model.language_model.layers.1.mlp.shared_expert.down_proj.weight", 1),
            entry("model.language_model.layers.2.mlp.experts.gate_up_proj", 2),
            entry("mtp.layers.0.mlp.experts.down_proj", 0),
        ];
        assert_eq!(place_routed_experts(&mut weights, 2), 2);
        let host: Vec<_> = weights
            .iter()
            .filter(|entry| entry.residency == WeightResidency::HostMapped)
            .map(|entry| entry.name.as_str())
            .collect();
        assert_eq!(
            host,
            [
                "model.language_model.layers.2.mlp.experts.gate_up_proj",
                "mtp.layers.0.mlp.experts.down_proj"
            ]
        );
    }

    #[test]
    fn auto_fits_the_measured_gfx1201_card() {
        // R9700 before load: 32548 MiB free; Flash-Next non-expert weights
        // 5.364 GB; one trunk layer's routed experts 1.3369 GB.
        let free = 32548u64 << 20;
        assert_eq!(auto_vram_layers(free, 5_364_000_000, 1_336_900_000, 48), 16);
        // A card that holds everything keeps every layer resident.
        assert_eq!(auto_vram_layers(u64::MAX / 2, 5_364_000_000, 1_336_900_000, 48), 48);
        // No room past the reserve places every expert in host RAM.
        assert_eq!(auto_vram_layers(8 << 30, 5_364_000_000, 1_336_900_000, 48), 0);
    }

    #[test]
    fn host_ram_check_refuses_below_headroom() {
        let host = 60u64 << 30;
        assert!(check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES)).is_ok());
        let error = check_host_ram(host, Some(host + HOST_RAM_HEADROOM_BYTES - 1)).unwrap_err();
        assert!(error.contains("MemAvailable"), "{error}");
        assert!(check_host_ram(host, None).is_err());
        assert!(check_host_ram(0, None).is_ok());
    }

    #[test]
    fn gtt_cap_counts_what_devices_already_hold() {
        // This host: pages_limit 16108144 (61.4 GiB); N=12 host-maps 46.1 GiB.
        let limit_bytes = 16_108_144 * TTM_PAGE_BYTES;
        let host = 46u64 << 30;
        let fresh = GttBudget { limit_bytes, used_bytes: 16 << 20 };
        assert!(check_gtt_cap(host, Some(fresh)).is_ok());
        assert!(check_gtt_cap(limit_bytes - fresh.used_bytes, Some(fresh)).is_ok());
        let error = check_gtt_cap(limit_bytes - fresh.used_bytes + 1, Some(fresh)).unwrap_err();
        assert!(error.contains("pages_limit") && error.contains("16108144 pages"), "{error}");
        // A second load beside one already holding 46 GiB of GTT is refused.
        let beside = GttBudget { limit_bytes, used_bytes: 46 << 30 };
        assert!(check_gtt_cap(host, Some(beside)).is_err());
        // Userptr host memory, or no TTM limit in sysfs, is not GTT-capped.
        assert!(check_gtt_cap(u64::MAX, None).is_ok());
    }

    #[test]
    fn parse_accepts_auto_and_counts_only() {
        assert_eq!(parse_expert_vram_layers(" auto "), Ok(ExpertVramLayers::Auto));
        assert_eq!(parse_expert_vram_layers("16"), Ok(ExpertVramLayers::Layers(16)));
        assert!(parse_expert_vram_layers("16GB").is_err());
    }
}
