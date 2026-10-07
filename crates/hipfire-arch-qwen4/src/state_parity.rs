// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact GPU state/rollback parity scenarios used by the Qwen4 parity CLI.
//!
//! Model logits are intentionally not produced here: the CLI's state mode
//! obtains those from the real loaded bundle/forward/spec path.  This module
//! only orchestrates compact state transitions and serializes device-backed
//! family digests.

use crate::config::compact_test_config;
use crate::gpu_forward::{
    qwen4_profile_enable, qwen4_profile_reset, qwen4_profile_snapshot, Qwen4GpuForwardScratch,
    Qwen4ProfilePhase, Qwen4ProfileStats,
};
use crate::kv_backend::Qwen4KvBackend;
use crate::mtp_gpu::{MtpGpuState, MtpStateParityMetadata};
use crate::mtp_spec::validate_native_mtp_prefill_request;
use crate::state::Qwen4State;
use hip_bridge::launch_counters;
use hipfire_runtime::external_rows::RowCacheStats;
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use rdna_compute::tensor_ops::{
    indexed_attention_cache_append, indexed_attention_pool_rope, indexed_attention_reuse_selection,
    indexed_attention_select, IndexedAttentionCacheAppend, IndexedAttentionPoolRope,
    IndexedAttentionReuseSelection, IndexedAttentionSelect,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_SEQ: usize = 8;
const PREFIX: usize = 4;
const DRAFTS: [u32; 2] = [1001, 1002];
const BONUS: u32 = 1003;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug)]
struct Family {
    hash: u64,
    numel: usize,
}

impl Family {
    fn new() -> Self {
        Self {
            hash: FNV_OFFSET,
            numel: 0,
        }
    }

    fn bytes(&mut self, bytes: &[u8], numel: usize) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(FNV_PRIME);
        }
        self.numel = self.numel.saturating_add(numel);
    }

    fn finish(self) -> Value {
        json!({"digest": format!("{:016x}", self.hash), "numel": self.numel})
    }
}

type Families = BTreeMap<String, Value>;

pub fn run_compact(gpu: &mut Gpu) -> Result<Value, String> {
    let config = compact_test_config();
    // The product's automatic QSA storage (VMM where supported), mapped for
    // the whole parity context: the oracle writes and reads whole arenas.
    let backend = Qwen4KvBackend::automatic(gpu);
    let format = crate::state::Qwen4StateFormat::F32;
    let mut ar = Qwen4State::new_with_backend(gpu, &config, MAX_SEQ, format, backend)
        .map_err(|error| format!("allocate compact AR state: {error}"))?;
    let mut native = match Qwen4State::new_with_backend(gpu, &config, MAX_SEQ, format, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            return Err(format!("allocate compact native target state: {error}"));
        }
    };
    let mut direct = match MtpGpuState::new_with_backend(gpu, &config, MAX_SEQ, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            return Err(format!("allocate compact direct MTP state: {error}"));
        }
    };
    let mut mtp = match MtpGpuState::new_with_backend(gpu, &config, MAX_SEQ, backend) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            let _ = direct.free_gpu(gpu);
            return Err(format!("allocate compact native MTP state: {error}"));
        }
    };
    let mapped = (|| -> Result<(), String> {
        for state in [&mut ar, &mut native] {
            state
                .ensure_mapped_capacity(gpu, MAX_SEQ)
                .map_err(|error| error.to_string())?;
        }
        for state in [&mut direct, &mut mtp] {
            state
                .ensure_mapped_capacity(gpu, MAX_SEQ)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    })()
    .map_err(|error| format!("map compact {} QSA context: {error}", backend.name()));
    let result = mapped
        .and_then(|()| run_inner(gpu, &config, &mut ar, &mut native, &mut direct, &mut mtp))
        .map(|mut value| {
            value["qsa_backend"] = json!(backend.name());
            value
        });
    let cleanup = [
        ar.free_gpu(gpu).err().map(|error| error.to_string()),
        native.free_gpu(gpu).err().map(|error| error.to_string()),
        direct.free_gpu(gpu).map(|error| error.to_string()),
        mtp.free_gpu(gpu).map(|error| error.to_string()),
    ]
    .into_iter()
    .flatten()
    .next();
    result.and_then(|value| cleanup.map_or(Ok(value), Err))
}

fn run_inner(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    let boundary = prepare(gpu, config, ar, native, direct, mtp)?;
    let mut cases = Vec::new();
    for accepted in 0..=DRAFTS.len() {
        cases.push(acceptance_case(
            gpu, config, ar, native, direct, mtp, accepted,
        )?);
    }
    let scenarios = vec![
        boundary,
        stale_ticket(gpu, ar, direct)?,
        terminal_seed(gpu, config, ar, direct)?,
        rollback_failure(gpu, config, ar, direct, "cancellation", None)?,
        rollback_failure(
            gpu,
            config,
            ar,
            direct,
            "injected_replay_failure",
            Some("injected replay error"),
        )?,
        cache_suffix_refusal(),
    ];
    let pass = cases.iter().all(is_pass) && scenarios.iter().all(is_pass);
    Ok(json!({
        "schema": "hipfire.qwen4.state_parity.compact.v1",
        "fixture": "canonical_compact_state",
        "gpu_arch": gpu.arch,
        "acceptance_cases": cases,
        "scenarios": scenarios,
        "status": if pass {"pass"} else {"fail"},
    }))
}

fn is_pass(value: &Value) -> bool {
    value.get("status") == Some(&Value::String("pass".into()))
}

fn prepare(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    reset_target(gpu, ar)?;
    reset_target(gpu, native)?;
    reset_mtp(gpu, direct)?;
    reset_mtp(gpu, mtp)?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_pair(gpu, config, ar, native, position, token)?;
        append_mtp_pair(gpu, config, direct, mtp, position, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX)?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX)?;
    reuse(gpu, config, direct)?;
    reuse(gpu, config, mtp)?;
    let direct_buf = direct.parity_buffers();
    let mtp_buf = mtp.parity_buffers();
    let direct_len = download_i32(gpu, direct_buf.selected_len_out, 1)?[0];
    let mtp_len = download_i32(gpu, mtp_buf.selected_len_out, 1)?[0];
    let direct_selection = download_i32(gpu, direct_buf.selected_indices, PREFIX + 1)?;
    let mtp_selection = download_i32(gpu, mtp_buf.selected_indices, PREFIX + 1)?;
    let direct_meta = direct.parity_metadata();
    let mtp_meta = mtp.parity_metadata();
    let target_meta = ar.qsa.first().map(|qsa| {
        json!({
            "full_len": qsa.full_len,
            "raw_len": qsa.raw_len,
            "pooled_len": qsa.pooled_len,
            "selected_len": qsa.selected_len,
            "position": qsa.position,
        })
    });
    let pass = target_meta.is_some()
        && direct_meta == mtp_meta
        && direct_meta.full_len == PREFIX
        && direct_meta.raw_len == PREFIX
        && direct_meta.pooled_len == 1
        && direct_meta.selected_len == PREFIX
        && direct_len == (PREFIX + 1) as i32
        && direct_len == mtp_len
        && direct_selection == [0, 1, 2, 3, 4]
        && direct_selection == mtp_selection;
    Ok(json!({
        "case": "qsa_pooling_boundary",
        "status": if pass {"pass"} else {"fail"},
        "target_metadata": target_meta,
        "mtp_direct_metadata": metadata_json(direct_meta),
        "mtp_native_metadata": metadata_json(mtp_meta),
        "reuse_selected_len": {"direct": direct_len, "native": mtp_len, "expected": PREFIX + 1},
        "reuse_selected_indices": {"direct": direct_selection, "native": mtp_selection},
    }))
}

fn acceptance_case(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
    accepted: usize,
) -> Result<Value, String> {
    prepare(gpu, config, ar, native, direct, mtp)?;
    let target_ticket = native.snapshot(gpu).map_err(|error| error.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|error| error.to_string())?;
    for (index, &token) in DRAFTS.iter().enumerate() {
        append_target_single(gpu, config, native, PREFIX + index, token)?;
        append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
    }
    let committed = committed(accepted);
    if accepted < DRAFTS.len() {
        mutate_target(gpu, native)?;
        mutate_mtp(gpu, mtp)?;
        native
            .restore_retain(gpu, target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.restore_retain(gpu, mtp_ticket)
            .map_err(|error| error.to_string())?;
        for (index, &token) in committed.iter().enumerate() {
            append_target_single(gpu, config, native, PREFIX + index, token)?;
            append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
        }
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
    } else {
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
        append_target_single(gpu, config, native, PREFIX + DRAFTS.len(), BONUS)?;
        append_mtp_single(gpu, config, mtp, PREFIX + DRAFTS.len(), BONUS)?;
    }
    for (index, &token) in committed.iter().enumerate() {
        append_target_single(gpu, config, ar, PREFIX + index, token)?;
        append_mtp_single(gpu, config, direct, PREFIX + index, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX + committed.len())?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX + committed.len())?;
    let target = compare_family_maps(
        &target_families(gpu, config, ar)?,
        &target_families(gpu, config, native)?,
    );
    let mtp_families = compare_family_maps(
        &mtp_families(gpu, config, direct)?,
        &mtp_families(gpu, config, mtp)?,
    );
    let pass = target.0 && mtp_families.0;
    Ok(json!({
        "case": match accepted {0 => "zero", 1 => "one", _ => "all"},
        "accepted_drafts": accepted,
        "committed_tokens": committed,
        "status": if pass {"pass"} else {"fail"},
        "target_families": target.1,
        "mtp_families": mtp_families.1,
    }))
}

fn committed(accepted: usize) -> Vec<u32> {
    let mut result = DRAFTS[..accepted].to_vec();
    result.push(BONUS);
    result
}

fn reset_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn reset_mtp(gpu: &mut Gpu, state: &mut MtpGpuState) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn append_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_target_single(gpu, config, first, position, token)?;
    append_target_single(gpu, config, second, position, token)
}

fn append_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_mtp_single(gpu, config, first, position, token)?;
    append_mtp_single(gpu, config, second, position, token)
}

fn append_target_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    for (layer_index, qsa) in state.qsa.iter_mut().enumerate() {
        let key_values = row(full_width, layer_index, position, token, 0.01);
        let value_values = row(full_width, layer_index, position, token, -0.02);
        let raw_values = row(raw_width, layer_index, position, token, 0.03);
        let key = gpu
            .upload_f32(&key_values, &[full_width])
            .map_err(|e| e.to_string())?;
        let value = match gpu.upload_f32(&value_values, &[full_width]) {
            Ok(value) => value,
            Err(error) => {
                let _ = gpu.free_tensor(key);
                return Err(error.to_string());
            }
        };
        let result = (|| {
            indexed_attention_cache_append(
                gpu,
                &IndexedAttentionCacheAppend {
                    key: &key,
                    value: &value,
                    full_keys: &qsa.full_keys,
                    full_values: &qsa.full_values,
                    position,
                    kv_width: full_width,
                },
            )
            .map_err(|e| e.to_string())?;
            write_raw(gpu, &qsa.raw_index_keys, position * raw_width, &raw_values)?;
            qsa.full_len = position + 1;
            qsa.raw_len = position + 1;
            qsa.position = position + 1;
            Ok::<(), String>(())
        })();
        let _ = gpu.free_tensor(key);
        let _ = gpu.free_tensor(value);
        result?;
    }
    state.position = position + 1;
    Ok(())
}

fn append_mtp_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let key_values = row(full_width, 0, position, token, 0.01);
    let value_values = row(full_width, 0, position, token, -0.02);
    let raw_values = row(raw_width, 0, position, token, 0.03);
    let key = gpu
        .upload_f32(&key_values, &[full_width])
        .map_err(|e| e.to_string())?;
    let value = match gpu.upload_f32(&value_values, &[full_width]) {
        Ok(value) => value,
        Err(error) => {
            let _ = gpu.free_tensor(key);
            return Err(error.to_string());
        }
    };
    let result = (|| {
        let metadata = state.parity_metadata();
        let buffers = state.parity_buffers();
        indexed_attention_cache_append(
            gpu,
            &IndexedAttentionCacheAppend {
                key: &key,
                value: &value,
                full_keys: buffers.full_keys,
                full_values: buffers.full_values,
                position,
                kv_width: full_width,
            },
        )
        .map_err(|e| e.to_string())?;
        write_raw(
            gpu,
            buffers.raw_index_keys,
            position * raw_width,
            &raw_values,
        )?;
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: position + 1,
            raw_len: position + 1,
            pooled_len: metadata.pooled_len,
            selected_len: metadata.selected_len,
            position: position + 1,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(key);
    let _ = gpu.free_tensor(value);
    result
}

fn row(width: usize, layer: usize, position: usize, token: u32, scale: f32) -> Vec<f32> {
    (0..width)
        .map(|index| {
            let phase = (layer * 17 + position * 31 + index * 7 + token as usize) as f32;
            (phase.sin() * 0.25 + 0.5) * scale
        })
        .collect()
}

fn select_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    select_target(gpu, config, first, visible)?;
    select_target(gpu, config, second, visible)
}

fn select_target(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        for qsa in &mut state.qsa {
            indexed_attention_pool_rope(
                gpu,
                &IndexedAttentionPoolRope {
                    raw_keys: &qsa.raw_index_keys,
                    pooled: &qsa.pooled_keys,
                    norm: None,
                    block_count: visible / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    index_dim: raw_width,
                    position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                        position_start: visible.saturating_sub(1),
                        rows: 1,
                    }),
                    grid_bound: visible / config.indexer_compress_ratio,
                },
            )
            .map_err(|e| e.to_string())?;
            indexed_attention_select(
                gpu,
                &IndexedAttentionSelect {
                    query: &query,
                    pooled: &qsa.pooled_keys,
                    selected: &qsa.selected_indices,
                    block_count: visible / config.indexer_compress_ratio,
                    index_heads: config.indexer_n_heads,
                    index_dim: config.indexer_head_dim,
                    budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    visible,
                    capacity: config.qsa_selected_capacity(),
                },
            )
            .map_err(|e| e.to_string())?;
            qsa.pooled_len = visible / config.indexer_compress_ratio;
            qsa.selected_len = visible;
        }
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn select_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    select_mtp(gpu, config, first, visible)?;
    select_mtp(gpu, config, second, visible)
}

fn select_mtp(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        let buffers = state.parity_buffers();
        indexed_attention_pool_rope(
            gpu,
            &IndexedAttentionPoolRope {
                raw_keys: buffers.raw_index_keys,
                pooled: buffers.pooled_keys,
                norm: None,
                block_count: visible / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                index_dim: raw_width,
                position: Some(rdna_compute::tensor_ops::QsaPositionBinding {
                    position_start: visible.saturating_sub(1),
                    rows: 1,
                }),
                grid_bound: visible / config.indexer_compress_ratio,
            },
        )
        .map_err(|e| e.to_string())?;
        indexed_attention_select(
            gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: buffers.pooled_keys,
                selected: buffers.selected_indices,
                block_count: visible / config.indexer_compress_ratio,
                index_heads: config.indexer_n_heads,
                index_dim: config.indexer_head_dim,
                budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                visible,
                capacity: config.qsa_selected_capacity(),
            },
        )
        .map_err(|e| e.to_string())?;
        let metadata = state.parity_metadata();
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: metadata.full_len,
            raw_len: metadata.raw_len,
            pooled_len: visible / config.indexer_compress_ratio,
            selected_len: visible,
            position: metadata.position,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn reuse(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<(), String> {
    let buffers = state.parity_buffers();
    indexed_attention_reuse_selection(
        gpu,
        &IndexedAttentionReuseSelection {
            selected: buffers.selected_indices,
            selected_len: PREFIX,
            position: PREFIX,
            capacity: config.qsa_selected_capacity(),
            selected_len_out: buffers.selected_len_out,
        },
    )
    .map_err(|e| e.to_string())
}

fn query(config: &crate::config::Qwen4Config) -> Vec<f32> {
    (0..config.indexer_n_heads * config.indexer_head_dim)
        .map(|index| (index as f32 * 0.013).cos() * 0.125)
        .collect()
}

fn write_raw(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
    offset_elements: usize,
    values: &[f32],
) -> Result<(), String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    let view = tensor.sub_offset(offset_elements, values.len());
    gpu.hip
        .memcpy_htod(&view.buf, bytes)
        .map_err(|error| error.to_string())
}

fn write_scalar_f32(gpu: &mut Gpu, tensor: &GpuTensor, value: f32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 1);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn write_scalar_i32(gpu: &mut Gpu, tensor: &GpuTensor, value: i32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 4);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn mutate_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    for layer in &state.gdn {
        write_scalar_f32(gpu, &layer.recurrent, 91.0)?;
        write_scalar_f32(gpu, &layer.conv, 92.0)?;
    }
    for layer in &state.qsa {
        write_scalar_f32(gpu, &layer.partial_keys, 93.0)?;
        write_scalar_f32(gpu, &layer.partial_values, 94.0)?;
        write_scalar_i32(gpu, &layer.selected_indices, -93)?;
    }
    write_scalar_f32(gpu, &state.ple_conv, 95.0)?;
    write_scalar_f32(gpu, &state.hyper_feedback, 96.0)
}

fn mutate_mtp(gpu: &mut Gpu, state: &MtpGpuState) -> Result<(), String> {
    let buffers = state.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, -97)?;
    write_scalar_i32(gpu, buffers.selected_len_out, 97)?;
    write_scalar_f32(gpu, buffers.wide_hidden, 98.0)
}

fn target_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &Qwen4State,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let mut families = BTreeMap::new();
    let mut recurrent = Family::new();
    let mut conv = Family::new();
    for layer in &state.gdn {
        append_f32(
            gpu,
            &mut recurrent,
            &layer.recurrent,
            layer.recurrent.numel(),
        )?;
        append_f32(gpu, &mut conv, &layer.conv, layer.conv.numel())?;
    }
    families.insert("gdn_recurrent".into(), recurrent.finish());
    families.insert("gdn_conv".into(), conv.finish());
    let mut qsa_families = [
        ("qsa_full_keys", Family::new()),
        ("qsa_full_values", Family::new()),
        ("qsa_raw_index_keys", Family::new()),
        ("qsa_pooled_keys", Family::new()),
        ("qsa_partial_keys", Family::new()),
        ("qsa_partial_values", Family::new()),
        ("qsa_selected_indices", Family::new()),
    ];
    let mut metadata = Family::new();
    for qsa in &state.qsa {
        append_f32(
            gpu,
            &mut qsa_families[0].1,
            &qsa.full_keys,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[1].1,
            &qsa.full_values,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[2].1,
            &qsa.raw_index_keys,
            qsa.raw_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[3].1,
            &qsa.pooled_keys,
            qsa.pooled_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[4].1,
            &qsa.partial_keys,
            qsa.partial_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[5].1,
            &qsa.partial_values,
            qsa.partial_len * full_width,
        )?;
        append_raw(
            gpu,
            &mut qsa_families[6].1,
            &qsa.selected_indices,
            qsa.selected_len * 4,
            qsa.selected_len,
        )?;
        metadata_usize(
            &mut metadata,
            [
                qsa.full_len,
                qsa.raw_len,
                qsa.pooled_len,
                qsa.partial_len,
                qsa.selected_len,
                qsa.position,
            ],
        );
    }
    metadata_usize(&mut metadata, [state.position, state.max_seq_len]);
    for (name, family) in qsa_families {
        families.insert(name.into(), family.finish());
    }
    families.insert("ple_conv".into(), tensor(gpu, &state.ple_conv)?);
    families.insert("hyper_feedback".into(), tensor(gpu, &state.hyper_feedback)?);
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn mtp_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let metadata_values = state.parity_metadata();
    let buffers = state.parity_buffers();
    let mut families = BTreeMap::new();
    families.insert(
        "full_keys".into(),
        prefix(
            gpu,
            buffers.full_keys,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "full_values".into(),
        prefix(
            gpu,
            buffers.full_values,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "raw_index_keys".into(),
        prefix(
            gpu,
            buffers.raw_index_keys,
            metadata_values.raw_len * raw_width,
        )?,
    );
    families.insert(
        "pooled_keys".into(),
        prefix(
            gpu,
            buffers.pooled_keys,
            metadata_values.pooled_len * raw_width,
        )?,
    );
    families.insert(
        "selected_indices".into(),
        raw_prefix(
            gpu,
            buffers.selected_indices,
            metadata_values.selected_len,
            metadata_values.selected_len,
        )?,
    );
    families.insert(
        "selected_len_out".into(),
        raw_prefix(gpu, buffers.selected_len_out, 1, 1)?,
    );
    families.insert("wide_hidden".into(), tensor(gpu, buffers.wide_hidden)?);
    let mut metadata = Family::new();
    metadata_usize(
        &mut metadata,
        [
            metadata_values.full_len,
            metadata_values.raw_len,
            metadata_values.pooled_len,
            metadata_values.selected_len,
            metadata_values.position,
            metadata_values.step_index,
        ],
    );
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn tensor(gpu: &Gpu, tensor: &GpuTensor) -> Result<Value, String> {
    prefix(gpu, tensor, tensor.numel())
}

fn prefix(gpu: &Gpu, tensor: &GpuTensor, elements: usize) -> Result<Value, String> {
    let mut family = Family::new();
    append_f32(gpu, &mut family, tensor, elements)?;
    Ok(family.finish())
}

fn raw_prefix(
    gpu: &Gpu,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<Value, String> {
    let mut family = Family::new();
    append_raw(gpu, &mut family, tensor, elements, numel)?;
    Ok(family.finish())
}

fn append_f32(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let values = gpu.download_f32(&view).map_err(|error| error.to_string())?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    family.bytes(bytes, values.len());
    Ok(())
}

fn append_raw(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let mut bytes = vec![0u8; view.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &view.buf)
        .map_err(|error| error.to_string())?;
    family.bytes(&bytes, numel);
    Ok(())
}

fn metadata_usize<const N: usize>(family: &mut Family, values: [usize; N]) {
    let mut bytes = Vec::with_capacity(N * std::mem::size_of::<usize>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    family.bytes(&bytes, N);
}

fn compare_family_maps(left: &Families, right: &Families) -> (bool, Value) {
    let keys = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut result = Map::new();
    let mut pass = true;
    for key in keys {
        let left_value = left.get(&key);
        let right_value = right.get(&key);
        let left_digest = left_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let right_digest = right_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let matches = left_digest == right_digest
            && left_value.and_then(|value| value.get("numel"))
                == right_value.and_then(|value| value.get("numel"));
        pass &= matches;
        result.insert(
            key,
            json!({
                "ar_or_direct": left_value,
                "native": right_value,
                "match": matches,
            }),
        );
    }
    (pass, Value::Object(result))
}

fn metadata_json(metadata: MtpStateParityMetadata) -> Value {
    json!({
        "full_len": metadata.full_len,
        "raw_len": metadata.raw_len,
        "pooled_len": metadata.pooled_len,
        "selected_len": metadata.selected_len,
        "position": metadata.position,
        "step_index": metadata.step_index,
    })
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, count: usize) -> Result<Vec<i32>, String> {
    let mut values = vec![0i32; count];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, count * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .map_err(|e| e.to_string())?;
    Ok(values)
}

fn stale_ticket(
    gpu: &mut Gpu,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_refusal = target
        .restore(gpu, target_ticket)
        .err()
        .map(|e| e.to_string());
    let mtp_refusal = mtp.restore(gpu, mtp_ticket).err().map(|e| e.to_string());
    let unchanged = target.position == 0 && mtp.parity_metadata().position == 0;
    Ok(json!({
        "case": "stale_ticket",
        "status": if target_refusal.is_some() && mtp_refusal.is_some() && unchanged {"pass"} else {"fail"},
        "target_refusal": target_refusal,
        "mtp_refusal": mtp_refusal,
        "state_unchanged": unchanged,
    }))
}

fn terminal_seed(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    append_target_single(gpu, config, target, 0, config.eos_token_id)?;
    append_mtp_single(gpu, config, mtp, 0, config.eos_token_id)?;
    for qsa in &mut target.qsa {
        write_scalar_i32(gpu, &qsa.selected_indices, 0)?;
        qsa.selected_len = 1;
        qsa.position = 1;
    }
    target.position = 1;
    let metadata = mtp.parity_metadata();
    let buffers = mtp.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, 0)?;
    mtp.parity_set_metadata(MtpStateParityMetadata {
        full_len: 1,
        raw_len: 1,
        pooled_len: 0,
        selected_len: 1,
        position: 1,
        step_index: metadata.step_index,
    });
    let pass = target.position == 1 && mtp.parity_metadata().position == 1;
    Ok(json!({
        "case": "forced_terminal_seed",
        "status": if pass {"pass"} else {"fail"},
        "forced_token": config.eos_token_id,
        "terminal": true,
        "target_metadata": {"position": target.position, "selected_len": target.qsa.first().map(|qsa| qsa.selected_len)},
        "mtp_metadata": metadata_json(mtp.parity_metadata()),
    }))
}

fn rollback_failure(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
    case: &str,
    injected: Option<&str>,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_single(gpu, config, target, position, token)?;
        append_mtp_single(gpu, config, mtp, position, token)?;
    }
    select_target(gpu, config, target, PREFIX)?;
    select_mtp(gpu, config, mtp, PREFIX)?;
    let before_target = target_families(gpu, config, target)?;
    let before_mtp = mtp_families(gpu, config, mtp)?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    mutate_target(gpu, target)?;
    mutate_mtp(gpu, mtp)?;
    let failure = injected.unwrap_or("cancelled by caller").to_string();
    target
        .restore(gpu, target_ticket)
        .map_err(|e| e.to_string())?;
    mtp.restore(gpu, mtp_ticket).map_err(|e| e.to_string())?;
    let target_compare =
        compare_family_maps(&before_target, &target_families(gpu, config, target)?);
    let mtp_compare = compare_family_maps(&before_mtp, &mtp_families(gpu, config, mtp)?);
    let pass = target_compare.0 && mtp_compare.0;
    Ok(json!({
        "case": case,
        "status": if pass {"pass"} else {"fail"},
        "replay_error": failure,
        "target_rollback": target_compare.1,
        "mtp_rollback": mtp_compare.1,
    }))
}

fn cache_suffix_refusal() -> Value {
    let refusal = validate_native_mtp_prefill_request(&[1, 2, 3], &[1, 2, 3], 0, true)
        .expect_err("cache-hit suffix must be refused");
    json!({
        "case": "cache_suffix_refusal",
        "status": "pass",
        "refusal": refusal,
        "cache_hit": true,
        "prompt_len": 3,
        "fill_len": 3,
    })
}

/// JSON report returned by the state parity orchestration entrypoint.
pub struct StateParityReport(Value);

impl StateParityReport {
    pub fn into_json(self) -> Value {
        self.0
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&self.0).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

const PROFILE_CHECKPOINT_SCHEMA: &str = "hipfire.qwen4.profile_checkpoint.v1";
const PROFILE_CHECKPOINT_PREFIX: &str = "HIPFIRE_QWEN4_PROFILE_CHECKPOINT ";

const PROFILE_SOURCE_CALLBACK_SCHEMA: &str = "hipfire.qwen4.profile_source_callback.v1";
const PROFILE_SOURCE_CALLBACK_PREFIX: &str = "HIPFIRE_QWEN4_PROFILE_SOURCE_CALLBACK ";

fn profile_elapsed_between_ns(started: Instant, ended: Instant) -> u64 {
    ended
        .duration_since(started)
        .as_nanos()
        .min(u128::from(u64::MAX)) as u64
}

fn profile_source_residency(residency: WeightResidency) -> Value {
    match residency {
        WeightResidency::Resident => json!("resident"),
        WeightResidency::HostMapped => json!("host_mapped"),
        WeightResidency::ExternalRows {
            row_bytes,
            valid_rows,
        } => json!({
            "kind": "external_rows",
            "row_bytes": row_bytes,
            "valid_rows": valid_rows,
        }),
    }
}

fn emit_profile_source_callback(
    index: u64,
    entry: &WeightEntry,
    elapsed_ns: u64,
    cumulative_ns: u64,
) {
    let checkpoint = json!({
        "schema": PROFILE_SOURCE_CALLBACK_SCHEMA,
        "index": index,
        "name": &entry.name,
        "layer": entry.layer,
        "residency": profile_source_residency(entry.residency),
        "elapsed_ns": elapsed_ns,
        "cumulative_ns": cumulative_ns,
    });
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{PROFILE_SOURCE_CALLBACK_PREFIX}{checkpoint}");
    let _ = stderr.flush();
}

fn emit_profile_checkpoint(
    phase: &str,
    wall_ns: u64,
    hip: &Value,
    internal: Option<&Value>,
    ple: Option<&Value>,
) {
    let checkpoint = json!({
        "schema": PROFILE_CHECKPOINT_SCHEMA,
        "phase": phase,
        "wall_ns": wall_ns,
        "hip": hip,
        "internal": internal.cloned().unwrap_or(Value::Null),
        "ple": ple.cloned().unwrap_or(Value::Null),
    });
    let mut stderr = std::io::stderr().lock();
    let _ = writeln!(stderr, "{PROFILE_CHECKPOINT_PREFIX}{checkpoint}");
    let _ = stderr.flush();
}

fn emit_profile_load_checkpoint(phase: &str, wall_ns: u64) {
    let hip = hip_counter_snapshot();
    emit_profile_checkpoint(phase, wall_ns, &hip, None, None);
}

fn profile_duration_ns(started: Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

fn hip_counter_entry(calls: u64, ffi_ns: u64, bytes: u64) -> Value {
    json!({
        "calls": calls,
        "ffi_ns": ffi_ns,
        "bytes": bytes,
    })
}

fn hip_counter_snapshot() -> Value {
    json!({
        "ffi_total_calls": launch_counters::count(),
        "ffi_total_ns": launch_counters::time_ns(),
        "launch_kernel": hip_counter_entry(
            launch_counters::launch_kernel::count(),
            launch_counters::launch_kernel::time_ns(),
            launch_counters::launch_kernel::bytes(),
        ),
        "memcpy_dtod": hip_counter_entry(
            launch_counters::memcpy_dtod::count(),
            launch_counters::memcpy_dtod::time_ns(),
            launch_counters::memcpy_dtod::bytes(),
        ),
        "memcpy_htod": hip_counter_entry(
            launch_counters::memcpy_htod::count(),
            launch_counters::memcpy_htod::time_ns(),
            launch_counters::memcpy_htod::bytes(),
        ),
        "memcpy_dtoh": hip_counter_entry(
            launch_counters::memcpy_dtoh::count(),
            launch_counters::memcpy_dtoh::time_ns(),
            launch_counters::memcpy_dtoh::bytes(),
        ),
        "memset": hip_counter_entry(
            launch_counters::memset::count(),
            launch_counters::memset::time_ns(),
            launch_counters::memset::bytes(),
        ),
        "ensure_kernel_lookup": hip_counter_entry(
            launch_counters::ensure_kernel_lookup::count(),
            launch_counters::ensure_kernel_lookup::time_ns(),
            launch_counters::ensure_kernel_lookup::bytes(),
        ),
        "stream_sync": hip_counter_entry(
            launch_counters::stream_sync::count(),
            launch_counters::stream_sync::time_ns(),
            launch_counters::stream_sync::bytes(),
        ),
        "event_sync": hip_counter_entry(
            launch_counters::event_sync::count(),
            launch_counters::event_sync::time_ns(),
            launch_counters::event_sync::bytes(),
        ),
        "device_sync": hip_counter_entry(
            launch_counters::device_sync::count(),
            launch_counters::device_sync::time_ns(),
            launch_counters::device_sync::bytes(),
        ),
        "graph_launch": hip_counter_entry(
            launch_counters::graph_launch::count(),
            launch_counters::graph_launch::time_ns(),
            launch_counters::graph_launch::bytes(),
        ),
    })
}

fn qwen4_profile_stats_json(stats: Qwen4ProfileStats) -> Value {
    Value::Object(
        Qwen4ProfilePhase::ALL
            .into_iter()
            .map(|phase| {
                let slot = phase as usize;
                (
                    phase.name().to_string(),
                    json!({ "calls": stats.calls[slot], "host_ns": stats.ns[slot] }),
                )
            })
            .collect(),
    )
}
const PROFILE_DIRTY_BYTE: i32 = 0x3f;

fn dirty_qwen4_moe_scratch(gpu: &mut Gpu, scratch: &Qwen4GpuForwardScratch) -> Result<(), String> {
    for tensor in [
        &scratch.router_logits,
        &scratch.moe_x_rot,
        &scratch.moe_gate_up,
        &scratch.moe_gate,
        &scratch.moe_up,
        &scratch.moe_hidden,
        &scratch.moe_output,
        &scratch.moe_gate_batch,
        &scratch.moe_up_batch,
        &scratch.moe_rot_batch,
        &scratch.moe_topk_indices,
        &scratch.moe_topk_weights,
        &scratch.moe_down_expanded,
        &scratch.moe_scalar,
    ] {
        gpu.hip
            .memset(&tensor.buf, PROFILE_DIRTY_BYTE, tensor.buf.size())
            .map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn dirty_profile_target_moe_reuse(
    gpu: &mut Gpu,
    bundle: &crate::bundle::Qwen4Bundle,
) -> Result<(), String> {
    let forward = bundle
        .execution
        .as_ref()
        .ok_or_else(|| "qwen4 profile forward resources are not attached".to_string())?;
    dirty_qwen4_moe_scratch(gpu, &forward.scratch)
}

fn dirty_profile_mtp_moe_reuse(
    gpu: &mut Gpu,
    bundle: &crate::bundle::Qwen4Bundle,
) -> Result<(), String> {
    let mtp = bundle
        .mtp
        .as_ref()
        .ok_or_else(|| "qwen4 profile MTP resources are not attached".to_string())?;
    mtp.dirty_moe_reuse(gpu).map_err(|error| error.to_string())
}

fn sealed_moe_profile_evidence(stats: &Value, dirty_reuse: bool) -> Result<Value, String> {
    let seal = stats
        .get("moe_seal")
        .ok_or_else(|| "qwen4 profile has no MoE seal counters".to_string())?;
    let calls = seal
        .get("calls")
        .and_then(Value::as_u64)
        .ok_or_else(|| "qwen4 profile MoE seal counter is malformed".to_string())?;
    if calls == 0 {
        return Err("qwen4 profile executed no sealed MoE calls".to_string());
    }
    Ok(json!({
        "calls": calls,
        "host_ns": seal.get("host_ns").cloned().unwrap_or(Value::Null),
        "route": "BoundMoeExperts::from_cache -> seal_decode -> execute_steps(Step::Moe)",
        "sealed_route_executed": true,
        "dirty_reuse": dirty_reuse,
        "dirty_pattern_byte": PROFILE_DIRTY_BYTE,
    }))
}

fn ple_cache_stats_json(stats: RowCacheStats) -> Value {
    json!({
        "capacity_bytes": stats.capacity_bytes,
        "resident_bytes": stats.resident_bytes,
        "resident_pages": stats.resident_pages,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "reads": stats.reads,
        "coalesced_reads": stats.coalesced_reads,
        "read_bytes": stats.read_bytes,
        "evictions": stats.evictions,
        "queue_depth": stats.queue_depth,
        "outstanding_readers": stats.outstanding_readers,
        "outstanding_leases": stats.outstanding_leases,
        "staging_in_use": stats.staging_in_use,
        "staging_high_water": stats.staging_high_water,
    })
}

fn ple_cache_delta(before: RowCacheStats, after: RowCacheStats) -> Value {
    json!({
        "cache_hits": after.cache_hits.saturating_sub(before.cache_hits),
        "cache_misses": after.cache_misses.saturating_sub(before.cache_misses),
        "reads": after.reads.saturating_sub(before.reads),
        "coalesced_reads": after.coalesced_reads.saturating_sub(before.coalesced_reads),
        "read_bytes": after.read_bytes.saturating_sub(before.read_bytes),
        "evictions": after.evictions.saturating_sub(before.evictions),
    })
}

fn target_layer_families_json(config: &crate::config::Qwen4Config) -> Value {
    json!({
        "linear_attention_layers": config.n_linear_layers(),
        "full_attention_layers": config.n_full_layers(),
        "moe_layers": config.num_hidden_layers,
        "ple_layer_ids": config.ple_layer_ids,
        "attribution": "rocprofv3 kernel trace by symbol; HIP counters are phase totals",
    })
}

/// Map the bundle's target and MTP QSA context arenas for all `tokens` of
/// the (small) parity context: the probes read whole arenas, not only the
/// rows a forward has mapped.
fn map_bundle_context(
    gpu: &mut Gpu,
    bundle: &mut crate::bundle::Qwen4Bundle,
    tokens: usize,
) -> Result<(), String> {
    bundle
        .state
        .ensure_mapped_capacity(gpu, tokens)
        .map_err(|error| format!("map qwen4 target QSA context: {error}"))?;
    if let Some(mtp) = bundle.mtp.as_mut() {
        mtp.ensure_mapped_capacity(gpu, tokens)
            .map_err(|error| format!("map qwen4 MTP QSA context: {error}"))?;
    }
    Ok(())
}

fn cleanup_profile_resources(
    gpu: &mut Gpu,
    bundle: crate::bundle::Qwen4Bundle,
    pending: Option<GpuTensor>,
) -> Option<String> {
    let mut errors = Vec::new();
    if let Some(pending) = pending {
        if let Err(error) = gpu.free_tensor(pending) {
            errors.push(format!("free profile pending hidden: {error}"));
        }
    }
    if let Err(error) = bundle.free_gpu(gpu) {
        errors.push(format!("free profile bundle: {error}"));
    }
    if errors.is_empty() {
        None
    } else {
        Some(errors.join("; "))
    }
}

/// JSON report for one bounded production target token and one native MTP step.
pub struct ProfileReport(Value);

impl ProfileReport {
    pub fn into_json(self) -> Value {
        self.0
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&self.0).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

fn qwen4_range_payload(
    source: &hipfire_runtime::hfq::HfqModelSource,
    entry: &WeightEntry,
) -> Result<hipfire_runtime::model_source::SourcePayload<'static>, String> {
    source
        .tensor_range(&entry.name)
        .map_err(|error| error.to_string())?
        .map(hipfire_runtime::model_source::SourcePayload::Range)
        .ok_or_else(|| format!("missing tensor '{}'", entry.name))
}

/// Load the admitted production artifact, execute exactly one target token and
/// one native MTP token, and emit host/HIP/rocprof attribution context.
pub fn run_profile(model_path: &Path, corpus_path: &Path) -> Result<ProfileReport, String> {
    let profile_enabled = true;
    qwen4_profile_enable(false);
    let result = run_profile_inner(model_path, corpus_path, profile_enabled);
    qwen4_profile_enable(false);
    result
}

fn run_profile_inner(
    model_path: &Path,
    corpus_path: &Path,
    profile_enabled: bool,
) -> Result<ProfileReport, String> {
    let (tokens, corpus) = read_state_tokens(corpus_path)?;
    let input_token = *tokens
        .first()
        .ok_or_else(|| "profile corpus has no input token".to_string())?;

    launch_counters::reset();
    let load_started = Instant::now();
    let open_started = Instant::now();
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let open_ns = profile_duration_ns(open_started);
    emit_profile_load_checkpoint("open_hfq", open_ns);

    let admission_started = Instant::now();
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let admission_ns = profile_duration_ns(admission_started);
    emit_profile_load_checkpoint("admission", admission_ns);
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();

    let gpu_init_started = Instant::now();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    let gpu_init_ns = profile_duration_ns(gpu_init_started);
    emit_profile_load_checkpoint("gpu_init", gpu_init_ns);
    if gpu.is_uma() {
        hfq.drop_mmap();
    }

    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let manifest_started = Instant::now();
    let source_progress = RefCell::new((0_u64, None::<Instant>, None::<Instant>));
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            let callback_started = Instant::now();
            let (index, elapsed_ns, cumulative_ns) = {
                let mut progress = source_progress.borrow_mut();
                let first_started = progress.1.as_ref().copied();
                let previous_started = progress.2.as_ref().copied();
                let index = progress.0;
                progress.0 = progress.0.saturating_add(1);
                progress.1.get_or_insert(callback_started);
                progress.2 = Some(callback_started);
                (
                    index,
                    previous_started
                        .map(|started| profile_elapsed_between_ns(started, callback_started))
                        .unwrap_or(0),
                    first_started
                        .map(|started| profile_elapsed_between_ns(started, callback_started))
                        .unwrap_or(0),
                )
            };
            emit_profile_source_callback(index, entry, elapsed_ns, cumulative_ns);

            qwen4_range_payload(&source, entry)
        },
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let manifest_ns = profile_duration_ns(manifest_started);
    emit_profile_load_checkpoint("manifest_fulfillment", manifest_ns);

    let assemble_started = Instant::now();
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        1,
        metadata,
        crate::state::Qwen4StateFormat::F32,
        backend,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let assemble_ns = profile_duration_ns(assemble_started);
    emit_profile_load_checkpoint("bundle_assembly", assemble_ns);

    let attach_forward_started = Instant::now();
    let setup_result = (|| -> Result<(), String> {
        bundle
            .attach_forward(&mut gpu, 1)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        Ok(())
    })();
    let attach_forward_ns = profile_duration_ns(attach_forward_started);
    emit_profile_load_checkpoint("attach_forward", attach_forward_ns);
    if let Err(error) = setup_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }

    let attach_mtp_started = Instant::now();
    let attach_mtp_result = bundle
        .attach_mtp(&mut gpu, 1)
        .map_err(|error| format!("qwen4 MTP setup failed: {error}"));
    let attach_mtp_ns = profile_duration_ns(attach_mtp_started);
    emit_profile_load_checkpoint("attach_mtp", attach_mtp_ns);
    if let Err(error) = attach_mtp_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }

    let setup_started = Instant::now();
    let setup_result = map_bundle_context(&mut gpu, &mut bundle, 1).and_then(|()| {
        bundle
            .ensure_spec_hidden(&mut gpu, 1)
            .map_err(|error| format!("qwen4 profile hidden setup failed: {error}"))
    });
    if let Err(error) = setup_result {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }
    let hidden_width = match config.hc_count.checked_mul(config.hidden_size) {
        Some(width) => width,
        None => {
            let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
            let error = "qwen4 profile hidden width overflow".to_string();
            return Err(match cleanup {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    let pending = match gpu.zeros(&[hidden_width], DType::F32) {
        Ok(pending) => pending,
        Err(error) => {
            let cleanup = cleanup_profile_resources(&mut gpu, bundle, None);
            let error = format!("allocate qwen4 profile pending hidden: {error}");
            return Err(match cleanup {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    if let Err(error) = bundle
        .reset(&mut gpu)
        .map_err(|error| format!("reset qwen4 profile state: {error}"))
    {
        let cleanup = cleanup_profile_resources(&mut gpu, bundle, Some(pending));
        return Err(match cleanup {
            Some(cleanup) => format!("{error}; {cleanup}"),
            None => error,
        });
    }
    let setup_ns = profile_duration_ns(setup_started);
    emit_profile_load_checkpoint("state_reset_and_scratch", setup_ns);
    let load_ns = profile_duration_ns(load_started);
    let load_hip = hip_counter_snapshot();
    emit_profile_checkpoint("load", load_ns, &load_hip, None, None);
    let layer_families = target_layer_families_json(&config);

    qwen4_profile_enable(profile_enabled);
    let execution_result = (|| -> Result<Value, String> {
        qwen4_profile_reset();
        launch_counters::reset();
        let ple_before = bundle.ple_rows().cache_stats();
        let target_started = Instant::now();
        dirty_profile_target_moe_reuse(&mut gpu, &bundle)?;
        let target_token = bundle
            .spec_capture_token(&mut gpu, input_token)
            .map_err(|error| format!("qwen4 profile target token: {error}"))?;
        let target_ns = profile_duration_ns(target_started);
        let target_hip = hip_counter_snapshot();
        let target_internal = qwen4_profile_stats_json(qwen4_profile_snapshot());
        let target_sealed = sealed_moe_profile_evidence(&target_internal, true)?;
        let target_ple_after = bundle.ple_rows().cache_stats();
        let target_d2h = target_hip
            .get("memcpy_dtoh")
            .cloned()
            .unwrap_or(Value::Null);
        let target_ple = json!({
            "before": ple_cache_stats_json(ple_before),
            "after": ple_cache_stats_json(target_ple_after),
            "delta": ple_cache_delta(ple_before, target_ple_after),
        });
        emit_profile_checkpoint(
            "target",
            target_ns,
            &target_hip,
            Some(&target_internal),
            Some(&target_ple),
        );

        qwen4_profile_reset();
        launch_counters::reset();
        let mtp_position = bundle
            .mtp_position()
            .map_err(|error| format!("read qwen4 profile MTP position: {error}"))?;
        let mtp_started = Instant::now();
        bundle
            .copy_spec_hidden_row_to(&mut gpu, 0, &pending)
            .map_err(|error| format!("qwen4 profile pending hidden copy: {error}"))?;
        dirty_profile_mtp_moe_reuse(&mut gpu, &bundle)?;
        let mtp_token = bundle
            .mtp_forward_token(&mut gpu, input_token, Some(&pending), mtp_position, true)
            .map_err(|error| format!("qwen4 profile native MTP token: {error}"))?;
        let mtp_ns = profile_duration_ns(mtp_started);
        let mtp_hip = hip_counter_snapshot();
        let mtp_internal = qwen4_profile_stats_json(qwen4_profile_snapshot());
        let mtp_sealed = sealed_moe_profile_evidence(&mtp_internal, true)?;
        let mtp_position_after = bundle
            .mtp_position()
            .map_err(|error| format!("read qwen4 profile MTP end position: {error}"))?;
        let mtp_d2h = mtp_hip.get("memcpy_dtoh").cloned().unwrap_or(Value::Null);
        emit_profile_checkpoint("mtp", mtp_ns, &mtp_hip, Some(&mtp_internal), None);

        Ok(json!({
            "target": {
                "input_token": input_token,
                "output_token": target_token,
                "state_position_after": bundle.state.position,
                "wall_ns": target_ns,
                "hip": target_hip,
                "internal": target_internal,
                "ple": target_ple,
            },
            "mtp": {
                "input_token": input_token,
                "output_token": mtp_token,
                "position": mtp_position,
                "position_after": mtp_position_after,
                "wall_ns": mtp_ns,
                "hip": mtp_hip,
                "internal": mtp_internal,
            },
            "d2h": {
                "target": target_d2h,
                "mtp": mtp_d2h,
                "attribution": "memcpy_dtoh counters are nested in the target/MTP phase that issued them",
            },
            "sealed_moe_validation": {
                "target": target_sealed,
                "mtp": mtp_sealed,
            },
        }))
    })();

    qwen4_profile_enable(false);
    let teardown_started = Instant::now();
    launch_counters::reset();
    let pending_cleanup = gpu
        .free_tensor(pending)
        .err()
        .map(|error| format!("free qwen4 profile pending hidden: {error}"));
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("free qwen4 profile bundle: {error}"));
    let teardown_ns = profile_duration_ns(teardown_started);
    let teardown_hip = hip_counter_snapshot();
    let teardown_error = pending_cleanup.or(bundle_cleanup);

    let execution = match execution_result {
        Ok(execution) => execution,
        Err(error) => {
            return Err(match teardown_error {
                Some(cleanup) => format!("{error}; {cleanup}"),
                None => error,
            });
        }
    };
    if let Some(error) = teardown_error {
        return Err(error);
    }
    Ok(ProfileReport(json!({
        "schema": "hipfire.qwen4.profile.v1",
        "gpu_arch": gpu.arch,
        "model": model_path.display().to_string(),
        "corpus": corpus,
        "input_token": input_token,
        "hipfire_profile": {
            "enabled": profile_enabled,
            "environment": "profile mode enables seal/PLE counters locally",
            "scope": "Qwen4 seal/PLE host hooks; rocprofv3 is authoritative for kernel timing",
        },
        "load": {
            "wall_ns": load_ns,
            "hip": load_hip,
            "steps": {
                "open_hfq_ns": open_ns,
                "admission_ns": admission_ns,
                "gpu_init_ns": gpu_init_ns,
                "manifest_fulfillment_ns": manifest_ns,
                "bundle_assembly_ns": assemble_ns,
                "attach_forward_ns": attach_forward_ns,
                "attach_mtp_ns": attach_mtp_ns,
                "state_reset_and_scratch_ns": setup_ns,
            },
        },
        "bounded_work": {
            "target_tokens": 1,
            "native_mtp_steps": 1,
            "compact_state_parity": false,
            "quality_probe": false,
        },
        "target_layer_families": layer_families,
        "execution": execution,
        "teardown": {
            "wall_ns": teardown_ns,
            "hip": teardown_hip,
        },
        "status": "pass",
    })))
}

/// Load one real HFQ model, run the production AR/native-MTP path, and append
/// compact state-arena scenarios to the resulting JSON report.
pub fn run_state_parity(
    model_path: &Path,
    corpus_path: &Path,
) -> Result<StateParityReport, String> {
    let (tokens, corpus) = read_state_tokens(corpus_path)?;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_range_payload(&source, entry),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        2048,
        metadata,
        crate::state::Qwen4StateFormat::F32,
        backend,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let real = (|| {
        bundle
            .attach_forward(&mut gpu, 2048)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, 2048)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        map_bundle_context(&mut gpu, &mut bundle, 2048)?;
        real_model_probe(&mut gpu, &mut bundle, &tokens, &corpus)
    })();
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let real = real?;
    if let Some(error) = bundle_cleanup {
        return Err(error);
    }
    let compact = run_compact(&mut gpu)?;
    let status = if compact.get("status") == Some(&Value::String("pass".into()))
        && real.get("status") == Some(&Value::String("pass".into()))
    {
        "pass"
    } else {
        "fail"
    };
    Ok(StateParityReport(json!({
        "schema": "hipfire.qwen4.state_parity.v1",
        "gpu_arch": gpu.arch,
        "model": model_path,
        "corpus": corpus,
        "real_model": real,
        "compact_state": compact,
        "status": status,
    })))
}

fn real_model_probe(
    gpu: &mut Gpu,
    bundle: &mut crate::bundle::Qwen4Bundle,
    tokens: &[u32],
    corpus: &Value,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab_size;
    let ar_elements = tokens
        .len()
        .checked_mul(vocab)
        .ok_or_else(|| "real-model AR logits shape overflow".to_string())?;
    let ar_buffer = gpu
        .zeros(&[ar_elements], DType::F32)
        .map_err(|error| format!("allocate real-model AR logits: {error}"))?;
    let ar_result = (|| {
        bundle.reset(gpu).map_err(|error| error.to_string())?;
        bundle
            .forward_chunk(gpu, tokens, &ar_buffer, None)
            .map_err(|error| format!("real AR forward: {error}"))?;
        let last_row_start = tokens
            .len()
            .checked_sub(1)
            .and_then(|row| row.checked_mul(vocab))
            .ok_or_else(|| "real-model AR logits row is empty".to_string())?;
        download_host_logits(gpu, &ar_buffer.sub_offset(last_row_start, vocab))
    })();
    let ar_logits = match ar_result {
        Ok(logits) => logits,
        Err(error) => {
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    if let Err(error) = bundle.reset(gpu) {
        let _ = gpu.free_tensor(ar_buffer);
        return Err(error.to_string());
    }
    let mtp_buffer = match gpu.zeros(&[vocab], DType::F32) {
        Ok(buffer) => buffer,
        Err(error) => {
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("allocate real-model MTP logits: {error}"));
        }
    };
    let mut drafter = crate::mtp_spec::Qwen4MtpDrafter::new(DRAFTS.len(), 2048, None);
    use hipfire_runtime::spec::MtpDrafter;
    let seed = match drafter.mtp_prefill(gpu, bundle, tokens, tokens, 0, false, &|| false) {
        Ok(seed) => seed,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("real native MTP prefill: {error}"));
        }
    };
    let eos = bundle.config.eos_token_id;
    let window = match drafter.mtp_step(
        gpu,
        bundle,
        tokens.len(),
        seed,
        &[],
        DRAFTS.len(),
        eos,
        None,
    ) {
        Ok(window) => window,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("real native MTP step: {error}"));
        }
    };
    let probe_token = window.committed.last().copied().unwrap_or(seed);
    let mtp_position = match bundle.mtp_position() {
        Ok(position) => position,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(format!("read native MTP position: {error}"));
        }
    };
    if let Err(error) =
        bundle.mtp_forward_token_logits(gpu, probe_token, mtp_position, true, &mtp_buffer)
    {
        MtpDrafter::mtp_free(Box::new(drafter), gpu);
        let _ = gpu.free_tensor(mtp_buffer);
        let _ = gpu.free_tensor(ar_buffer);
        return Err(format!("real native MTP logits: {error}"));
    }
    let mtp_logits = match download_host_logits(gpu, &mtp_buffer) {
        Ok(logits) => logits,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    MtpDrafter::mtp_free(Box::new(drafter), gpu);
    let families = match bundle_family_json(gpu, bundle) {
        Ok(families) => families,
        Err(error) => {
            let _ = gpu.free_tensor(mtp_buffer);
            let _ = gpu.free_tensor(ar_buffer);
            return Err(error);
        }
    };
    let max_abs = ar_logits
        .values
        .iter()
        .zip(&mtp_logits.values)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let finite = ar_logits.values.iter().all(|value| value.is_finite())
        && mtp_logits.values.iter().all(|value| value.is_finite());
    let result = json!({
        "status": if finite {"pass"} else {"fail"},
        "corpus": corpus,
        "ar_logit_row": tokens.len() - 1,
        "ar_token_count": tokens.len(),
        "accepted_drafts": window.accepted,
        "drafts_generated": window.drafts_generated,
        "committed": window.committed,
        "seed": seed,
        "probe_token": probe_token,
        "ar_logits": host_logits_json(&ar_logits),
        "native_mtp_logits": host_logits_json(&mtp_logits),
        "ar_vs_native_mtp": {
            "max_abs": max_abs,
            "same_within_1e-5": max_abs <= 1.0e-5,
            "ar_digest": ar_logits.digest,
            "native_mtp_digest": mtp_logits.digest,
        },
        "state_families": families,
    });
    gpu.free_tensor(mtp_buffer)
        .map_err(|error| error.to_string())?;
    gpu.free_tensor(ar_buffer)
        .map_err(|error| error.to_string())?;
    Ok(result)
}

#[derive(Clone, Debug)]
struct HostLogits {
    values: Vec<f32>,
    digest: String,
    top1: usize,
}

fn download_host_logits(gpu: &Gpu, tensor: &GpuTensor) -> Result<HostLogits, String> {
    let values = gpu
        .download_f32(tensor)
        .map_err(|error| error.to_string())?;
    let mut family = Family::new();
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    family.bytes(bytes, values.len());
    let digest = family
        .finish()
        .get("digest")
        .and_then(Value::as_str)
        .ok_or("host logits digest missing")?
        .to_string();
    let top1 = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Ok(HostLogits {
        values,
        digest,
        top1,
    })
}
fn host_logits_json(logits: &HostLogits) -> Value {
    json!({"digest": logits.digest, "numel": logits.values.len(), "top1": logits.top1})
}

fn bundle_family_json(gpu: &Gpu, bundle: &crate::bundle::Qwen4Bundle) -> Result<Value, String> {
    let target = target_families(gpu, &bundle.config, &bundle.state)?;
    let mtp = bundle
        .mtp
        .as_ref()
        .map(|mtp| mtp_families(gpu, &bundle.config, mtp.parity_state()))
        .transpose()?;
    Ok(json!({"target": target, "mtp": mtp}))
}

fn read_state_tokens(path: &Path) -> Result<(Vec<u32>, Value), String> {
    const COUNT: usize = 17;
    const SHA256: &str = "e53de8c7b501eaaea637648feb6f569dd17cd564c2f669b2924ccdf1b7e52e2f";
    let metadata_path = if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    };
    let metadata_text = std::fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: Value = serde_json::from_str(&metadata_text)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    if metadata.get("count").and_then(Value::as_u64) != Some(COUNT as u64) {
        return Err("state parity requires the canonical 17-token corpus".to_string());
    }
    let relative = metadata
        .get("path")
        .and_then(Value::as_str)
        .ok_or("state corpus metadata has no path")?;
    let payload = metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(PathBuf::from(relative));
    let bytes =
        std::fs::read(&payload).map_err(|error| format!("read {}: {error}", payload.display()))?;
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(&bytes);
    if bytes.len() != COUNT * 4 || format!("{:x}", digest.finalize()) != SHA256 {
        return Err(
            "state corpus byte count or SHA256 does not match canonical corpus".to_string(),
        );
    }
    let tokens = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok((
        tokens,
        json!({
            "metadata_path": metadata_path,
            "payload_path": payload,
            "count": COUNT,
            "sha256": SHA256
        }),
    ))
}

/// Native-MTP prompt fill digests for each prompt length: the MTP head and
/// target state families after `mtp_prefill`, the pending hidden row, the
/// seed, and one following draft window. Run once per
/// `HIPFIRE_QWEN4_MTP_BATCHED_FILL` setting and compare the two outputs; the
/// route is read once per process.
///
/// `warm_chunks == 0` fills each length cold. Otherwise each length is the
/// suffix of a prefix-cache hit: a cold fill of an exact warm prompt of
/// `P = warm_chunks * chunk_rows` tokens captures the end-of-prompt
/// checkpoint at `P`, then a prompt sharing those `P` tokens and followed by
/// `length` fresh ones is filled from the restored checkpoint
/// (`start_pos = P`) and digested.
pub fn run_mtp_fill_digest(
    model_path: &Path,
    lengths: &[usize],
    max_seq: usize,
    warm_chunks: usize,
) -> Result<Value, String> {
    use hipfire_runtime::spec::MtpDrafter;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_range_payload(&source, entry),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let warm = warm_chunks > 0;
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        max_seq,
        metadata,
        crate::state::Qwen4StateFormat::F32,
        backend,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let run = (|| -> Result<Value, String> {
        if warm {
            bundle
                .attach_prefix_cache(&mut gpu)
                .map_err(|error| format!("qwen4 prefix cache setup failed: {error}"))?;
        }
        bundle
            .attach_forward(&mut gpu, max_seq)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, max_seq)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        if warm {
            bundle
                .attach_prefix_cache(&mut gpu)
                .map_err(|error| format!("qwen4 prefix cache setup failed: {error}"))?;
        }
        map_bundle_context(&mut gpu, &mut bundle, max_seq)?;
        let vocab = config.vocab_size as u64;
        // A deterministic spread over the real vocabulary.
        let tokens = |salt: u64, count: usize| -> Vec<u32> {
            let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ salt;
            (0..count)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (10 + (state >> 33) % (vocab - 10)) as u32
                })
                .collect()
        };
        let prefix_len = if warm {
            let chunk = bundle
                .spec_chunk_rows()
                .ok_or("qwen4 forward resources are not attached")?;
            warm_chunks
                .checked_mul(chunk)
                .filter(|&len| len < max_seq)
                .ok_or("warm prefix does not fit MAX_SEQ")?
        } else {
            0
        };
        let mut drafter = crate::mtp_spec::Qwen4MtpDrafter::new(3, max_seq, None);
        let mut rows = Vec::new();
        for &length in lengths {
            let seed = if warm {
                // Warm with exactly the shared `prefix_len` tokens: the cold
                // fill captures the end-of-prompt checkpoint at `prefix_len`
                // and an empty-history commit publishes that checkpoint only.
                // A prompt with a fresh suffix then restores it.
                let shared = tokens(0x5eed ^ prefix_len as u64, prefix_len);
                drafter
                    .mtp_prefill(&mut gpu, &mut bundle, &shared, &shared, 0, false, &|| false)?;
                bundle.commit_prefix(&[]);
                let mut prompt = shared;
                prompt.extend(tokens(0xa11 ^ length as u64, length));
                let plan = bundle
                    .bind_prefix_plan(
                        &prompt,
                        prefix_len,
                        crate::bundle::Qwen4PrefixMode::NativeMtp,
                    )
                    .map_err(|error| format!("warm length {length}: {error}"))?;
                if plan.source() != crate::bundle::Qwen4PrefixSource::Prompt
                    || plan.start_pos != prefix_len
                {
                    return Err(format!(
                        "warm length {length}: prefix plan is {:?} at {}, expected Prompt at {prefix_len}",
                        plan.source(),
                        plan.start_pos
                    ));
                }
                drafter.mtp_prefill(
                    &mut gpu,
                    &mut bundle,
                    &prompt,
                    &prompt[prefix_len..],
                    prefix_len,
                    true,
                    &|| false,
                )?
            } else {
                let prompt = tokens(length as u64, length);
                drafter.mtp_prefill(&mut gpu, &mut bundle, &prompt, &prompt, 0, false, &|| false)?
            };
            let after_fill = bundle_family_json(&gpu, &bundle)?;
            let mut pending = Family::new();
            append_f32(
                &gpu,
                &mut pending,
                drafter.pending_hidden_for_parity()?,
                drafter.pending_hidden_for_parity()?.numel(),
            )?;
            let position = bundle.state.position;
            let eos = bundle.config.eos_token_id;
            let window =
                drafter.mtp_step(&mut gpu, &mut bundle, position, seed, &[seed], 3, eos, None)?;
            let after_window = bundle_family_json(&gpu, &bundle)?;
            rows.push(json!({
                "length": length,
                "seed": seed,
                "pending_hidden": pending.finish(),
                "families_after_fill": after_fill,
                "window_committed": window.committed,
                "families_after_window": after_window,
            }));
            eprintln!("mtp-fill digest: length {length} start {prefix_len} seed {seed}");
        }
        Ok(json!({
            "batched_fill": hipfire_config::developer_bool("HIPFIRE_QWEN4_MTP_BATCHED_FILL", true),
            "max_seq": max_seq,
            "warm_prefix": prefix_len,
            "rows": rows,
        }))
    })();
    let cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let value = run?;
    if let Some(error) = cleanup {
        return Err(error);
    }
    Ok(value)
}

use crate::bundle::{Qwen4Bundle, Qwen4PrefixMode, Qwen4PrefixPlan, Qwen4PrefixSource};
use crate::mtp_spec::Qwen4MtpDrafter;

/// Prompt lengths `P` of the prefix-cache session oracle.
const SESSION_PROMPT_LENGTHS: [usize; 11] = [1, 5, 63, 64, 65, 511, 513, 1025, 1900, 4097, 8193];
/// Fixed suffix lengths; a multi-chunk suffix (`2 * chunk_rows + 1`) is added.
const SESSION_SUFFIX_LENGTHS: [usize; 4] = [1, 3, 64, 513];
/// Tokens decoded before a request commits (crosses compress-4 pooling
/// boundaries and GDN ring flips).
const SESSION_DECODE_TOKENS: usize = 9;
/// Greedy ids compared after each suffix prefill.
const SESSION_NEXT_IDS: usize = 6;
const SESSION_MTP_K: usize = 3;
/// Rows past `prompt + suffix` a case may use for decode and verify windows.
const SESSION_HEADROOM: usize = 64;
const SESSION_NEGATIVE_PROMPT: usize = 65;
const SESSION_NEGATIVE_TAIL: usize = 3;

type CaseOutcome = Result<(Option<String>, Value), String>;
type NegativeCase =
    fn(&mut SessionRig, &mut Gpu, &mut Option<Qwen4MtpDrafter>, Qwen4PrefixMode) -> CaseOutcome;
type SuffixCase = fn(&mut SessionRig, &mut Gpu, Qwen4PrefixMode, usize, usize) -> CaseOutcome;

fn mode_name(mode: Qwen4PrefixMode) -> &'static str {
    match mode {
        Qwen4PrefixMode::Ar => "ar",
        Qwen4PrefixMode::NativeMtp => "native_mtp",
    }
}

/// Digest `units` elements of `tensor` byte for byte (GDN Q8 codes and scales
/// included): the generic readers above widen to F32.
fn append_units(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    units: usize,
) -> Result<(), String> {
    if units == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, units);
    let bytes = gpu
        .download_raw_bytes(&view)
        .map_err(|error| error.to_string())?;
    family.bytes(&bytes, units);
    Ok(())
}

/// Every target state family in the state's own storage formats: GDN
/// recurrent bytes and scales, conv, active QSA rows and marks, PLE conv and
/// history, HC feedback.
fn session_target_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &Qwen4State,
) -> Result<Families, String> {
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let full_width = config.num_key_value_heads * config.head_dim;
    let mut recurrent = Family::new();
    let mut conv = Family::new();
    for layer in &state.gdn {
        append_units(
            gpu,
            &mut recurrent,
            &layer.recurrent,
            layer.recurrent.numel(),
        )?;
        append_units(gpu, &mut conv, &layer.conv, layer.conv.numel())?;
    }
    let mut full_keys = Family::new();
    let mut full_values = Family::new();
    let mut raw_index_keys = Family::new();
    let mut pooled_keys = Family::new();
    let mut partial_keys = Family::new();
    let mut partial_values = Family::new();
    let mut selected_indices = Family::new();
    let mut metadata = Family::new();
    for qsa in &state.qsa {
        let kv_units = qsa.full_len * qsa.full_row_units;
        append_units(gpu, &mut full_keys, &qsa.full_keys, kv_units)?;
        append_units(gpu, &mut full_values, &qsa.full_values, kv_units)?;
        append_units(
            gpu,
            &mut raw_index_keys,
            &qsa.raw_index_keys,
            qsa.raw_len * raw_width,
        )?;
        append_units(
            gpu,
            &mut pooled_keys,
            &qsa.pooled_keys,
            qsa.pooled_len * raw_width,
        )?;
        append_units(
            gpu,
            &mut partial_keys,
            &qsa.partial_keys,
            qsa.partial_len * raw_width,
        )?;
        append_units(
            gpu,
            &mut partial_values,
            &qsa.partial_values,
            qsa.partial_len * full_width,
        )?;
        append_units(
            gpu,
            &mut selected_indices,
            &qsa.selected_indices,
            qsa.selected_len * std::mem::size_of::<i32>(),
        )?;
        metadata_usize(
            &mut metadata,
            [
                qsa.full_len,
                qsa.raw_len,
                qsa.pooled_len,
                qsa.partial_len,
                qsa.selected_len,
                qsa.position,
            ],
        );
    }
    metadata_usize(&mut metadata, [state.position, state.max_seq_len]);
    let mut ple_conv = Family::new();
    append_units(gpu, &mut ple_conv, &state.ple_conv, state.ple_conv.numel())?;
    let mut hyper_feedback = Family::new();
    append_units(
        gpu,
        &mut hyper_feedback,
        &state.hyper_feedback,
        state.hyper_feedback.numel(),
    )?;
    let mut ple_history = Family::new();
    ple_history.bytes(format!("{:?}", state.ple_history).as_bytes(), 1);
    let mut families = Families::new();
    for (name, family) in [
        ("gdn_recurrent", recurrent),
        ("gdn_conv", conv),
        ("qsa_full_keys", full_keys),
        ("qsa_full_values", full_values),
        ("qsa_raw_index_keys", raw_index_keys),
        ("qsa_pooled_keys", pooled_keys),
        ("qsa_partial_keys", partial_keys),
        ("qsa_partial_values", partial_values),
        ("qsa_selected_indices", selected_indices),
        ("ple_conv", ple_conv),
        ("ple_history", ple_history),
        ("hyper_feedback", hyper_feedback),
        ("metadata", metadata),
    ] {
        families.insert(name.into(), family.finish());
    }
    Ok(families)
}

/// Target families, and for native MTP the head families and the drafter's
/// pending target hidden row.
struct SessionState {
    target: Families,
    mtp: Option<Families>,
    pending: Option<Value>,
}

fn family_digest(value: Option<&Value>) -> &str {
    value
        .and_then(|value| value.get("digest"))
        .and_then(Value::as_str)
        .unwrap_or("-")
}

fn first_family_mismatch(scope: &str, left: &Families, right: &Families) -> Option<String> {
    let keys = left.keys().chain(right.keys()).collect::<BTreeSet<_>>();
    for key in keys {
        let (a, b) = (left.get(key), right.get(key));
        if a != b {
            return Some(format!(
                "{scope}.{key}: {} vs {}",
                family_digest(a),
                family_digest(b)
            ));
        }
    }
    None
}

fn first_state_mismatch(stage: &str, left: &SessionState, right: &SessionState) -> Option<String> {
    first_family_mismatch(&format!("{stage}:target"), &left.target, &right.target)
        .or_else(|| match (&left.mtp, &right.mtp) {
            (Some(a), Some(b)) => first_family_mismatch(&format!("{stage}:mtp"), a, b),
            (None, None) => None,
            _ => Some(format!("{stage}:mtp head presence differs")),
        })
        .or_else(|| {
            (left.pending != right.pending).then(|| {
                format!(
                    "{stage}:pending_hidden: {} vs {}",
                    family_digest(left.pending.as_ref()),
                    family_digest(right.pending.as_ref())
                )
            })
        })
}

/// Everything one suffix prefill and the greedy decode after it produce.
struct FlowOut {
    source: Qwen4PrefixSource,
    start: usize,
    /// Native MTP's first emitted token.
    seed: Option<u32>,
    /// Digest of the AR final logits row (native MTP exposes none).
    logits: Option<String>,
    after_fill: SessionState,
    ids: Vec<u32>,
    after_decode: SessionState,
}

fn first_flow_mismatch(left: &FlowOut, right: &FlowOut) -> Option<String> {
    first_state_mismatch("after_fill", &left.after_fill, &right.after_fill)
        .or_else(|| {
            (left.logits != right.logits).then(|| {
                format!(
                    "after_fill:final_logits: {:?} vs {:?}",
                    left.logits, right.logits
                )
            })
        })
        .or_else(|| {
            (left.seed != right.seed)
                .then(|| format!("after_fill:seed: {:?} vs {:?}", left.seed, right.seed))
        })
        .or_else(|| {
            (left.ids != right.ids)
                .then(|| format!("decode:ids: {:?} vs {:?}", left.ids, right.ids))
        })
        .or_else(|| first_state_mismatch("after_decode", &left.after_decode, &right.after_decode))
}

fn expect_refusal<T, E: std::fmt::Display>(
    what: &str,
    result: Result<T, E>,
) -> Result<String, String> {
    match result {
        Ok(_) => Err(format!("{what}: expected a refusal, got success")),
        Err(error) => Ok(error.to_string()),
    }
}

fn case_value(
    case: &str,
    mode: Qwen4PrefixMode,
    prompt_len: usize,
    suffix_len: usize,
    outcome: CaseOutcome,
) -> Value {
    let (mismatch, detail) = match outcome {
        Ok(outcome) => outcome,
        Err(error) => (Some(format!("error: {error}")), Value::Null),
    };
    json!({
        "case": case,
        "mode": mode_name(mode),
        "prompt_len": prompt_len,
        "suffix_len": suffix_len,
        "status": if mismatch.is_none() {"pass"} else {"fail"},
        "first_mismatch": mismatch,
        "detail": detail,
    })
}

/// The committed first request of a session.
struct Committed {
    /// Host history passed to `commit_prefix`: prompt plus consumed tokens.
    full: Vec<u32>,
    /// Tokens the decode emitted (empty without decode).
    decode_ids: Vec<u32>,
    /// The bundle published `full` as its live lineage.
    live: bool,
}

struct Decoded {
    ids: Vec<u32>,
    /// Tokens the target consumed (all of `ids` for AR; all but the pending
    /// last one for native MTP).
    consumed: Vec<u32>,
}

/// One bundle driven through sequential prefix-cache session arms.
struct SessionRig {
    bundle: Qwen4Bundle,
    logits: GpuTensor,
    max_seq: usize,
    vocab: usize,
    eos: u32,
}

impl SessionRig {
    /// Deterministic spread over the real vocabulary, never the EOS id.
    fn tokens(&self, salt: u64, count: usize) -> Vec<u32> {
        let span = self.vocab as u64 - 10;
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ salt.wrapping_mul(0xff51_afd7_ed55_8ccd);
        let mut out = Vec::with_capacity(count);
        while out.len() < count {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let token = (10 + (state >> 33) % span) as u32;
            if token != self.eos {
                out.push(token);
            }
        }
        out
    }

    /// A real-vocabulary token that is neither `token` nor EOS.
    fn other_token(&self, token: u32) -> u32 {
        let mut candidate = token;
        loop {
            candidate = if candidate as usize + 1 >= self.vocab {
                10
            } else {
                candidate + 1
            };
            if candidate != token && candidate != self.eos {
                return candidate;
            }
        }
    }

    fn with_drafter<T>(
        &mut self,
        gpu: &mut Gpu,
        mode: Qwen4PrefixMode,
        f: impl FnOnce(&mut Self, &mut Gpu, &mut Option<Qwen4MtpDrafter>) -> Result<T, String>,
    ) -> Result<T, String> {
        use hipfire_runtime::spec::MtpDrafter;
        let mut drafter = (mode == Qwen4PrefixMode::NativeMtp)
            .then(|| Qwen4MtpDrafter::new(SESSION_MTP_K, self.max_seq, None));
        let result = f(self, gpu, &mut drafter);
        if let Some(drafter) = drafter {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
        }
        result
    }

    fn state(
        &self,
        gpu: &Gpu,
        drafter: &Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
    ) -> Result<SessionState, String> {
        let target = session_target_families(gpu, &self.bundle.config, &self.bundle.state)?;
        if mode == Qwen4PrefixMode::Ar {
            return Ok(SessionState {
                target,
                mtp: None,
                pending: None,
            });
        }
        let head = self
            .bundle
            .mtp
            .as_ref()
            .ok_or("native MTP head is not attached")?;
        let mtp = mtp_families(gpu, &self.bundle.config, head.parity_state())?;
        let pending = match drafter {
            Some(drafter) => {
                let hidden = drafter.pending_hidden_for_parity()?;
                Some(prefix(gpu, hidden, hidden.numel())?)
            }
            None => None,
        };
        Ok(SessionState {
            target,
            mtp: Some(mtp),
            pending,
        })
    }

    /// Prefill `prompt`: cold without a plan, else the hit `plan` binds.
    fn prefill(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        prompt: &[u32],
        plan: Option<Qwen4PrefixPlan>,
    ) -> Result<Option<u32>, String> {
        use hipfire_runtime::spec::MtpDrafter;
        match mode {
            Qwen4PrefixMode::Ar => {
                self.bundle
                    .prefill_final(gpu, prompt, plan.unwrap_or_default(), &self.logits)
                    .map_err(|error| error.to_string())?;
                Ok(None)
            }
            Qwen4PrefixMode::NativeMtp => {
                let drafter = drafter
                    .as_mut()
                    .ok_or("native MTP drafter is not allocated")?;
                let seed = match plan {
                    None => drafter.mtp_prefill(
                        gpu,
                        &mut self.bundle,
                        prompt,
                        prompt,
                        0,
                        false,
                        &|| false,
                    )?,
                    Some(plan) => drafter.mtp_prefill(
                        gpu,
                        &mut self.bundle,
                        prompt,
                        &prompt[plan.start_pos..],
                        plan.start_pos,
                        true,
                        &|| false,
                    )?,
                };
                Ok(Some(seed))
            }
        }
    }

    /// Greedy decode of at least `count` tokens: AR forwards each argmax;
    /// native MTP runs verify windows from the prefill `seed`.
    fn decode(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        seed: Option<u32>,
        count: usize,
    ) -> Result<Decoded, String> {
        use hipfire_runtime::spec::MtpDrafter;
        match mode {
            Qwen4PrefixMode::Ar => {
                let mut ids = Vec::with_capacity(count);
                for _ in 0..count {
                    ids.push(
                        self.bundle
                            .forward_token_or_argmax(gpu, None, &self.logits)
                            .map_err(|error| error.to_string())?,
                    );
                }
                Ok(Decoded {
                    consumed: ids.clone(),
                    ids,
                })
            }
            Qwen4PrefixMode::NativeMtp => {
                let drafter = drafter
                    .as_mut()
                    .ok_or("native MTP drafter is not allocated")?;
                let mut seed = seed.ok_or("native MTP decode needs the prefill seed")?;
                let mut ids = vec![seed];
                while ids.len() <= count {
                    let position = self.bundle.state.position;
                    // No token is EOS: the oracle compares fixed-length streams.
                    let window = drafter.mtp_step(
                        gpu,
                        &mut self.bundle,
                        position,
                        seed,
                        &ids,
                        SESSION_MTP_K,
                        u32::MAX,
                        None,
                    )?;
                    seed = *window
                        .committed
                        .last()
                        .ok_or("native MTP window committed nothing")?;
                    ids.extend_from_slice(&window.committed);
                }
                let consumed = ids[..ids.len() - 1].to_vec();
                Ok(Decoded { ids, consumed })
            }
        }
    }

    /// Cold prefill of `prompt`, optional decode, then the client's terminal
    /// commit of the full consumed history.
    fn first_session(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        prompt: &[u32],
        decode: bool,
    ) -> Result<Committed, String> {
        let seed = self.prefill(gpu, drafter, mode, prompt, None)?;
        let decoded = if decode {
            self.decode(gpu, drafter, mode, seed, SESSION_DECODE_TOKENS)?
        } else {
            Decoded {
                ids: Vec::new(),
                consumed: Vec::new(),
            }
        };
        let mut full = prompt.to_vec();
        full.extend_from_slice(&decoded.consumed);
        if self.bundle.state.position != full.len() {
            return Err(format!(
                "consumed history is {} tokens but the target is at {}",
                full.len(),
                self.bundle.state.position
            ));
        }
        self.bundle.commit_prefix(&full);
        let live = self.bundle.prefix_cache_tokens(mode) == Some(full.as_slice());
        Ok(Committed {
            full,
            decode_ids: decoded.ids,
            live,
        })
    }

    fn require_live(&self, committed: &Committed) -> Result<(), String> {
        if committed.live {
            return Ok(());
        }
        Err(format!(
            "live lineage of {} tokens was not published (target at {}, head at {:?})",
            committed.full.len(),
            self.bundle.state.position,
            self.bundle.mtp_position().ok()
        ))
    }

    /// Bind the planner's `start` and require the receipt's source.
    fn bind(
        &self,
        prompt: &[u32],
        start: usize,
        mode: Qwen4PrefixMode,
        expect: Qwen4PrefixSource,
    ) -> Result<Qwen4PrefixPlan, String> {
        let plan = self
            .bundle
            .bind_prefix_plan(prompt, start, mode)
            .map_err(|error| format!("bind at {start}: {error}"))?;
        if plan.source() != expect || plan.start_pos != start {
            return Err(format!(
                "bind at {start} produced {:?} at {}, expected {expect:?}",
                plan.source(),
                plan.start_pos
            ));
        }
        Ok(plan)
    }

    /// The hit's suffix prefill, then the next greedy ids; state families
    /// are digested after each.
    fn run_suffix(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        prompt: &[u32],
        plan: Qwen4PrefixPlan,
    ) -> Result<FlowOut, String> {
        let seed = self.prefill(gpu, drafter, mode, prompt, Some(plan))?;
        if self.bundle.state.position != prompt.len() {
            return Err(format!(
                "suffix prefill left the target at {}, expected {}",
                self.bundle.state.position,
                prompt.len()
            ));
        }
        let logits = match mode {
            Qwen4PrefixMode::Ar => Some(download_host_logits(gpu, &self.logits)?.digest),
            Qwen4PrefixMode::NativeMtp => None,
        };
        let after_fill = self.state(gpu, drafter, mode)?;
        let decoded = self.decode(gpu, drafter, mode, seed, SESSION_NEXT_IDS)?;
        let after_decode = self.state(gpu, drafter, mode)?;
        Ok(FlowOut {
            source: plan.source(),
            start: plan.start_pos,
            seed,
            logits,
            after_fill,
            ids: decoded.ids,
            after_decode,
        })
    }

    /// Production: cold `P1`, decode, commit, then a prompt `P1 + S2` that
    /// diverges from the decoded tokens at `P` restores the end-of-prompt
    /// checkpoint (`Prompt`) and prefills `S2`. Returns the flow, `S2`, and
    /// whether the live lineage was also published.
    fn divergence_production(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        p1: &[u32],
        suffix_len: usize,
    ) -> Result<(FlowOut, Vec<u32>, bool), String> {
        let committed = self.first_session(gpu, drafter, mode, p1, true)?;
        let first_generated = committed.full[p1.len()];
        let mut s2 = self.tokens(
            0x52 ^ ((p1.len() as u64) << 20) ^ suffix_len as u64,
            suffix_len,
        );
        if s2[0] == first_generated {
            s2[0] = self.other_token(first_generated);
        }
        let mut prompt = p1.to_vec();
        prompt.extend_from_slice(&s2);
        let plan = self.bind(&prompt, p1.len(), mode, Qwen4PrefixSource::Prompt)?;
        if self.bundle.prefix_checkpoint_position(mode) != Some(p1.len()) {
            return Err(format!(
                "end-of-prompt checkpoint is {:?}, expected {}",
                self.bundle.prefix_checkpoint_position(mode),
                p1.len()
            ));
        }
        let out = self.run_suffix(gpu, drafter, mode, &prompt, plan)?;
        Ok((out, s2, committed.live))
    }

    /// Oracle: cold `P1`, commit with no decode (live at `P`), then the same
    /// `P1 + S2` continues the live state (`Live`).
    fn divergence_oracle(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
        p1: &[u32],
        s2: &[u32],
    ) -> Result<FlowOut, String> {
        let committed = self.first_session(gpu, drafter, mode, p1, false)?;
        self.require_live(&committed)?;
        let mut prompt = p1.to_vec();
        prompt.extend_from_slice(s2);
        let plan = self.bind(&prompt, p1.len(), mode, Qwen4PrefixSource::Live)?;
        self.run_suffix(gpu, drafter, mode, &prompt, plan)
    }

    fn case_divergence(
        &mut self,
        gpu: &mut Gpu,
        mode: Qwen4PrefixMode,
        p: usize,
        s: usize,
    ) -> CaseOutcome {
        let p1 = self.tokens(0xd1 ^ p as u64, p);
        let (production, s2, live_published) = self.with_drafter(gpu, mode, |rig, gpu, d| {
            rig.divergence_production(gpu, d, mode, &p1, s)
        })?;
        let oracle = self.with_drafter(gpu, mode, |rig, gpu, d| {
            rig.divergence_oracle(gpu, d, mode, &p1, &s2)
        })?;
        let mismatch = first_flow_mismatch(&production, &oracle);
        Ok((
            mismatch,
            json!({
                "production_source": format!("{:?}", production.source),
                "production_start": production.start,
                "oracle_source": format!("{:?}", oracle.source),
                "oracle_start": oracle.start,
                "live_also_published": live_published,
                "seed": production.seed,
                "ids": production.ids,
            }),
        ))
    }

    /// Cold `P1`, decode, commit, then `consumed + S3` continues the live
    /// state; returns the flow and the first request's decoded ids.
    fn live_run(
        &mut self,
        gpu: &mut Gpu,
        mode: Qwen4PrefixMode,
        p1: &[u32],
        suffix_len: usize,
    ) -> Result<(FlowOut, Vec<u32>), String> {
        self.with_drafter(gpu, mode, |rig, gpu, drafter| {
            let committed = rig.first_session(gpu, drafter, mode, p1, true)?;
            rig.require_live(&committed)?;
            let mut prompt = committed.full.clone();
            prompt.extend(rig.tokens(
                0x11fe ^ ((p1.len() as u64) << 20) ^ suffix_len as u64,
                suffix_len,
            ));
            let plan = rig.bind(&prompt, committed.full.len(), mode, Qwen4PrefixSource::Live)?;
            let out = rig.run_suffix(gpu, drafter, mode, &prompt, plan)?;
            Ok((out, committed.decode_ids))
        })
    }

    /// The same live-continuation session twice from reset must agree.
    fn case_live_determinism(
        &mut self,
        gpu: &mut Gpu,
        mode: Qwen4PrefixMode,
        p: usize,
        s: usize,
    ) -> CaseOutcome {
        let p1 = self.tokens(0xd1 ^ p as u64, p);
        let (first, first_decode) = self.live_run(gpu, mode, &p1, s)?;
        let (second, second_decode) = self.live_run(gpu, mode, &p1, s)?;
        let mismatch = (first_decode != second_decode)
            .then(|| format!("first_request:ids: {first_decode:?} vs {second_decode:?}"))
            .or_else(|| first_flow_mismatch(&first, &second));
        Ok((
            mismatch,
            json!({
                "source": format!("{:?}", first.source),
                "start": first.start,
                "seed": first.seed,
                "ids": first.ids,
            }),
        ))
    }

    /// `begin_prefix` with a `Live` receipt must leave every state family
    /// untouched; replaying the consumed receipt must be refused before any
    /// write.
    fn case_live_noop(&mut self, gpu: &mut Gpu, mode: Qwen4PrefixMode, p: usize) -> CaseOutcome {
        let p1 = self.tokens(0xd1 ^ p as u64, p);
        self.with_drafter(gpu, mode, |rig, gpu, drafter| {
            let committed = rig.first_session(gpu, drafter, mode, &p1, true)?;
            rig.require_live(&committed)?;
            let live_end = committed.full.len();
            let mut prompt = committed.full.clone();
            prompt.extend(rig.tokens(0x71 ^ p as u64, 1));
            let plan = rig.bind(&prompt, live_end, mode, Qwen4PrefixSource::Live)?;
            let before = rig.state(gpu, drafter, mode)?;
            rig.bundle
                .begin_prefix(gpu, &prompt, plan, mode)
                .map_err(|error| format!("Live begin: {error}"))?;
            let after = rig.state(gpu, drafter, mode)?;
            let mut mismatch = first_state_mismatch("live_begin", &before, &after);
            if mismatch.is_none() && rig.bundle.state.position != live_end {
                mismatch = Some(format!(
                    "live_begin:position: {live_end} vs {}",
                    rig.bundle.state.position
                ));
            }
            let refusal = expect_refusal(
                "second begin with the consumed Live receipt",
                rig.bundle.begin_prefix(gpu, &prompt, plan, mode),
            )?;
            let after_stale = rig.state(gpu, drafter, mode)?;
            if mismatch.is_none() {
                mismatch = first_state_mismatch("stale_live_begin", &after, &after_stale);
            }
            Ok((
                mismatch,
                json!({"live_end": live_end, "stale_refusal": refusal}),
            ))
        })
    }

    /// A commit whose history is longer than, shorter than, or empty against
    /// the target position must not publish a live lineage: no live bind is
    /// accepted, only the end-of-prompt checkpoint.
    fn negative_misaligned(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
    ) -> CaseOutcome {
        let p1 = self.tokens(0x91, SESSION_NEGATIVE_PROMPT);
        let extra = self.tokens(0x92, 1)[0];
        let tail = self.tokens(0x93, SESSION_NEGATIVE_TAIL);
        let mut detail = Map::new();
        for variant in ["longer", "shorter", "empty"] {
            let seed = self.prefill(gpu, drafter, mode, &p1, None)?;
            let decoded = self.decode(gpu, drafter, mode, seed, SESSION_DECODE_TOKENS)?;
            let live_end = p1.len() + decoded.consumed.len();
            if self.bundle.state.position != live_end {
                return Err(format!(
                    "{variant}: target at {}, expected {live_end}",
                    self.bundle.state.position
                ));
            }
            let mut record = p1.clone();
            record.extend_from_slice(&decoded.consumed);
            let mut history = record.clone();
            match variant {
                "longer" => history.push(extra),
                "shorter" => {
                    history.pop();
                }
                _ => history.clear(),
            }
            self.bundle.commit_prefix(&history);
            let mut prompt = record;
            prompt.push(extra);
            prompt.extend_from_slice(&tail);
            if self.bundle.prefix_cache_tokens(mode) != Some(p1.as_slice()) {
                return Err(format!(
                    "{variant}: published lineage is not the end-of-prompt record"
                ));
            }
            if self.bundle.prefix_checkpoint_position(mode) != Some(p1.len()) {
                return Err(format!(
                    "{variant}: checkpoint position is {:?}, expected {}",
                    self.bundle.prefix_checkpoint_position(mode),
                    p1.len()
                ));
            }
            let live_refusal = expect_refusal(
                &format!("{variant}: bind at the unpublished live end {live_end}"),
                self.bundle.bind_prefix_plan(&prompt, live_end, mode),
            )?;
            let history_refusal = if history.len() > p1.len() && history.len() != live_end {
                Some(expect_refusal(
                    &format!("{variant}: bind at the history end {}", history.len()),
                    self.bundle.bind_prefix_plan(&prompt, history.len(), mode),
                )?)
            } else {
                None
            };
            self.bind(&prompt, p1.len(), mode, Qwen4PrefixSource::Prompt)?;
            detail.insert(
                variant.to_string(),
                json!({"live_refusal": live_refusal, "history_refusal": history_refusal}),
            );
        }
        Ok((None, Value::Object(detail)))
    }

    /// A checkpoint of one mode must not bind or begin under the other.
    fn negative_mode_mismatch(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
    ) -> CaseOutcome {
        let other = match mode {
            Qwen4PrefixMode::Ar => Qwen4PrefixMode::NativeMtp,
            Qwen4PrefixMode::NativeMtp => Qwen4PrefixMode::Ar,
        };
        let p1 = self.tokens(0x94, SESSION_NEGATIVE_PROMPT);
        let mut prompt = p1.clone();
        prompt.extend(self.tokens(0x95, SESSION_NEGATIVE_TAIL));
        self.prefill(gpu, drafter, mode, &p1, None)?;
        self.bundle.commit_prefix(&p1);
        let bind_refusal = expect_refusal(
            "bind under the other mode",
            self.bundle.bind_prefix_plan(&prompt, p1.len(), other),
        )?;
        if self.bundle.prefix_cache_tokens(other).is_some()
            || self.bundle.prefix_checkpoint_position(other).is_some()
        {
            return Err("the other mode sees a published lineage or checkpoint".to_string());
        }
        let plan = self.bind(&prompt, p1.len(), mode, Qwen4PrefixSource::Live)?;
        let before = self.state(gpu, drafter, mode)?;
        let begin_refusal = expect_refusal(
            "begin under the other mode",
            self.bundle.begin_prefix(gpu, &prompt, plan, other),
        )?;
        let after = self.state(gpu, drafter, mode)?;
        Ok((
            first_state_mismatch("refused_mode_begin", &before, &after),
            json!({"bind_refusal": bind_refusal, "begin_refusal": begin_refusal}),
        ))
    }

    /// A consumed receipt, or one overtaken by another begin, is refused
    /// before any device write.
    fn negative_stale_receipt(
        &mut self,
        gpu: &mut Gpu,
        drafter: &mut Option<Qwen4MtpDrafter>,
        mode: Qwen4PrefixMode,
    ) -> CaseOutcome {
        let p1 = self.tokens(0x96, SESSION_NEGATIVE_PROMPT);
        let mut detail = Map::new();
        let mut mismatch = None;
        for scenario in ["replayed_prompt_receipt", "after_cold_begin"] {
            let committed = self.first_session(gpu, drafter, mode, &p1, true)?;
            let mut s2 = self.tokens(0x97, SESSION_NEGATIVE_TAIL);
            if s2[0] == committed.full[p1.len()] {
                s2[0] = self.other_token(s2[0]);
            }
            let mut prompt = p1.clone();
            prompt.extend_from_slice(&s2);
            let plan = self.bind(&prompt, p1.len(), mode, Qwen4PrefixSource::Prompt)?;
            let baseline = if scenario == "replayed_prompt_receipt" {
                self.bundle
                    .begin_prefix(gpu, &prompt, plan, mode)
                    .map_err(|error| format!("{scenario}: Prompt begin: {error}"))?;
                self.state(gpu, drafter, mode)?
            } else {
                self.bundle
                    .begin_prefix(gpu, &prompt, Qwen4PrefixPlan::default(), mode)
                    .map_err(|error| format!("{scenario}: cold begin: {error}"))?;
                self.state(gpu, drafter, mode)?
            };
            let refusal =
                expect_refusal(scenario, self.bundle.begin_prefix(gpu, &prompt, plan, mode))?;
            let after = self.state(gpu, drafter, mode)?;
            if mismatch.is_none() {
                mismatch = first_state_mismatch(scenario, &baseline, &after);
            }
            detail.insert(scenario.to_string(), json!({"refusal": refusal}));
        }
        Ok((mismatch, Value::Object(detail)))
    }
}

/// Prefix-cache session oracle. One bundle carries sequential arms on the
/// product's default context and shipped state formats, for AR and native
/// MTP, over prompt lengths `P`, suffix lengths `S` and a multi-chunk suffix:
///
/// - `prompt_divergence`: cold `P1`, decode, commit; `P1 + S2` diverging at
///   `P` binds the end-of-prompt checkpoint (`Prompt`). The oracle is cold
///   `P1` committed with no decode, then the same `P1 + S2` on the live state
///   (`Live`). Every target family, head family, pending hidden row, final
///   logits or seed, and the next greedy ids must match byte for byte.
/// - `live_continuation`: cold `P1`, decode, commit; `consumed + S3` continues
///   the live state. Run twice from reset: digests, logits and ids must be
///   identical. `live_begin_noop` digests every family around the `Live`
///   `begin_prefix` and requires them unchanged.
/// - Negative: misaligned commits never bind `Live`, a checkpoint of one mode
///   refuses the other, and stale receipts are refused before any write.
///
/// Native MTP exposes no final logits row; its comparison covers the seed,
/// pending hidden row, head families and ids.
pub fn run_prefix_cache_session_parity(
    model_path: &Path,
    corpus_path: &Path,
) -> Result<StateParityReport, String> {
    let (_, corpus) = read_state_tokens(corpus_path)?;
    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_range_payload(&source, entry),
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let state_format = crate::state::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        "",
        &gpu,
        &config,
    )?;
    let backend = Qwen4KvBackend::automatic(&gpu);
    // The product's default context; every case's rows must fit it.
    let max_seq = crate::QWEN4_DEFAULT_CONTEXT.min(config.max_position_embeddings);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        max_seq,
        metadata,
        state_format,
        backend,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    // The loader's order: target prefix arena, forward, MTP head, then the
    // head's part of the prefix arena.
    let attach = (|| -> Result<(), String> {
        bundle
            .attach_prefix_cache(&mut gpu)
            .map_err(|error| format!("qwen4 prefix cache setup failed: {error}"))?;
        bundle
            .attach_forward(&mut gpu, max_seq)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, max_seq)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        bundle
            .attach_prefix_cache(&mut gpu)
            .map_err(|error| format!("qwen4 prefix cache setup failed: {error}"))?;
        if !bundle.prefix_cache_attached() {
            return Err("qwen4 prefix cache is not attached".to_string());
        }
        Ok(())
    })();
    let logits = match attach.and_then(|()| {
        gpu.zeros(&[config.vocab_size], DType::F32)
            .map_err(|error| format!("allocate session logits: {error}"))
    }) {
        Ok(logits) => logits,
        Err(error) => {
            let _ = bundle.free_gpu(&mut gpu);
            return Err(error);
        }
    };
    let mut rig = SessionRig {
        bundle,
        logits,
        max_seq,
        vocab: config.vocab_size,
        eos: config.eos_token_id,
    };
    let run = (|| -> Result<Value, String> {
        let chunk = rig
            .bundle
            .spec_chunk_rows()
            .ok_or("qwen4 forward resources are not attached")?;
        let mut suffixes = SESSION_SUFFIX_LENGTHS.to_vec();
        suffixes.push(2 * chunk + 1);
        let mut cases: Vec<Value> = Vec::new();
        let mut record = |value: Value| {
            eprintln!(
                "prefix-cache session: {} {} P={} S={}: {}{}",
                value["mode"].as_str().unwrap_or("?"),
                value["case"].as_str().unwrap_or("?"),
                value["prompt_len"],
                value["suffix_len"],
                value["status"].as_str().unwrap_or("?"),
                value["first_mismatch"]
                    .as_str()
                    .map(|mismatch| format!(" ({mismatch})"))
                    .unwrap_or_default(),
            );
            cases.push(value);
        };
        for mode in [Qwen4PrefixMode::Ar, Qwen4PrefixMode::NativeMtp] {
            let negatives: [(&str, NegativeCase); 3] = [
                (
                    "negative_misaligned_commit",
                    SessionRig::negative_misaligned,
                ),
                ("negative_mode_mismatch", SessionRig::negative_mode_mismatch),
                ("negative_stale_receipt", SessionRig::negative_stale_receipt),
            ];
            for (name, case) in negatives {
                let started = Instant::now();
                let outcome =
                    rig.with_drafter(&mut gpu, mode, |rig, gpu, d| case(rig, gpu, d, mode));
                let mut value = case_value(name, mode, SESSION_NEGATIVE_PROMPT, 0, outcome);
                value["elapsed_s"] = json!(started.elapsed().as_secs_f64());
                record(value);
            }
            for &p in &SESSION_PROMPT_LENGTHS {
                let started = Instant::now();
                let outcome = match session_case_fits(max_seq, p, 1) {
                    Ok(()) => rig.case_live_noop(&mut gpu, mode, p),
                    Err(error) => Err(error),
                };
                let mut value = case_value("live_begin_noop", mode, p, 1, outcome);
                value["elapsed_s"] = json!(started.elapsed().as_secs_f64());
                record(value);
                for &s in &suffixes {
                    let suffix_cases: [(&str, SuffixCase); 2] = [
                        ("prompt_divergence", SessionRig::case_divergence),
                        ("live_continuation", SessionRig::case_live_determinism),
                    ];
                    for (name, case) in suffix_cases {
                        let started = Instant::now();
                        let outcome = match session_case_fits(max_seq, p, s) {
                            Ok(()) => case(&mut rig, &mut gpu, mode, p, s),
                            Err(error) => Err(error),
                        };
                        let mut value = case_value(name, mode, p, s, outcome);
                        value["elapsed_s"] = json!(started.elapsed().as_secs_f64());
                        record(value);
                    }
                }
            }
        }
        let failed = cases
            .iter()
            .filter(|case| !is_pass(case))
            .map(|case| {
                format!(
                    "{} {} P={} S={}: {}",
                    case["mode"].as_str().unwrap_or("?"),
                    case["case"].as_str().unwrap_or("?"),
                    case["prompt_len"],
                    case["suffix_len"],
                    case["first_mismatch"].as_str().unwrap_or("?"),
                )
            })
            .collect::<Vec<_>>();
        Ok(json!({
            "status": if failed.is_empty() {"pass"} else {"fail"},
            "max_seq": max_seq,
            "chunk_rows": chunk,
            "state_format": format!("{state_format:?}"),
            "total": cases.len(),
            "failed": failed,
            "cases": cases,
        }))
    })();
    let SessionRig { bundle, logits, .. } = rig;
    let logits_cleanup = gpu
        .free_tensor(logits)
        .err()
        .map(|error| format!("free session logits: {error}"));
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let session = run?;
    if let Some(error) = logits_cleanup.or(bundle_cleanup) {
        return Err(error);
    }
    Ok(StateParityReport(json!({
        "schema": "hipfire.qwen4.prefix_cache_session_parity.v1",
        "gpu_arch": gpu.arch,
        "model": model_path,
        "corpus": corpus,
        "status": session["status"],
        "session": session,
    })))
}

fn session_case_fits(max_seq: usize, prompt: usize, suffix: usize) -> Result<(), String> {
    if prompt + suffix + SESSION_HEADROOM > max_seq {
        return Err(format!(
            "prompt {prompt} + suffix {suffix} + {SESSION_HEADROOM} rows of decode headroom exceeds max_seq {max_seq}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gpu() -> Option<Gpu> {
        Gpu::init().ok().or_else(|| {
            eprintln!("skip: Qwen4 compact state parity requires a GPU");
            None
        })
    }

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        value
            .get(name)
            .unwrap_or_else(|| panic!("report is missing field {name:?}: {value}"))
    }

    fn assert_pass(value: &Value) {
        assert_eq!(
            field(value, "status").as_str(),
            Some("pass"),
            "failed report case: {value}"
        );
    }

    fn assert_family_matches(value: &Value, name: &str) {
        let families = field(value, name)
            .as_object()
            .unwrap_or_else(|| panic!("{name} is not an object: {}", field(value, name)));
        assert!(!families.is_empty(), "{name} has no state families");
        for (family, comparison) in families {
            assert_eq!(
                field(comparison, "match").as_bool(),
                Some(true),
                "{name}.{family} did not match: {comparison}"
            );
        }
    }

    #[test]
    fn compact_runner_reports_device_backed_state_parity_scenarios() {
        let Some(mut gpu) = try_gpu() else {
            return;
        };
        let report = run_compact(&mut gpu).expect("compact state parity runner");

        assert_eq!(
            field(&report, "schema").as_str(),
            Some("hipfire.qwen4.state_parity.compact.v1")
        );
        assert_eq!(field(&report, "gpu_arch").as_str(), Some(gpu.arch.as_str()));
        assert_pass(&report);

        let cases = field(&report, "acceptance_cases")
            .as_array()
            .expect("acceptance_cases array");
        assert_eq!(cases.len(), 3);
        let expected_cases = [
            ("zero", 0usize, json!([1003u32])),
            ("one", 1usize, json!([1001u32, 1003u32])),
            ("all", 2usize, json!([1001u32, 1002u32, 1003u32])),
        ];
        let mut case_names = BTreeSet::new();
        for case in cases {
            assert_pass(case);
            let name = field(case, "case").as_str().expect("acceptance case name");
            let (_, accepted, committed_tokens) = expected_cases
                .iter()
                .find(|(expected, _, _)| *expected == name)
                .unwrap_or_else(|| panic!("unexpected acceptance case {name:?}"));
            assert_eq!(
                field(case, "accepted_drafts").as_u64(),
                Some(*accepted as u64)
            );
            assert_eq!(field(case, "committed_tokens"), committed_tokens);
            assert_family_matches(case, "target_families");
            assert_family_matches(case, "mtp_families");
            case_names.insert(name.to_string());
        }
        assert_eq!(
            case_names,
            expected_cases
                .iter()
                .map(|(name, _, _)| (*name).to_string())
                .collect()
        );

        let scenarios = field(&report, "scenarios")
            .as_array()
            .expect("scenarios array");
        assert_eq!(scenarios.len(), 6);
        let mut scenario_names = BTreeSet::new();
        for scenario in scenarios {
            assert_pass(scenario);
            scenario_names.insert(
                field(scenario, "case")
                    .as_str()
                    .expect("scenario name")
                    .to_string(),
            );
        }
        assert_eq!(
            scenario_names,
            [
                "qsa_pooling_boundary",
                "stale_ticket",
                "forced_terminal_seed",
                "cancellation",
                "injected_replay_failure",
                "cache_suffix_refusal",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        );

        let boundary = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("qsa_pooling_boundary"))
            .expect("QSA pooling boundary scenario");
        assert_eq!(
            field(field(boundary, "target_metadata"), "full_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "raw_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "pooled_len").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "selected_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "direct").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "native").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "expected").as_u64(),
            Some(5)
        );
        assert_eq!(
            field(boundary, "reuse_selected_indices"),
            &json!({
                "direct": [0, 1, 2, 3, 4],
                "native": [0, 1, 2, 3, 4],
            })
        );

        let stale = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("stale_ticket"))
            .expect("stale ticket scenario");
        assert!(field(stale, "target_refusal").as_str().is_some());
        assert!(field(stale, "mtp_refusal").as_str().is_some());
        assert_eq!(field(stale, "state_unchanged").as_bool(), Some(true));

        let terminal = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("forced_terminal_seed"))
            .expect("forced terminal seed scenario");
        assert_eq!(field(terminal, "terminal").as_bool(), Some(true));
        assert_eq!(
            field(terminal, "forced_token").as_u64(),
            Some(compact_test_config().eos_token_id as u64)
        );
        assert_eq!(
            field(field(terminal, "target_metadata"), "position").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(terminal, "mtp_metadata"), "position").as_u64(),
            Some(1)
        );

        for name in ["cancellation", "injected_replay_failure"] {
            let rollback = scenarios
                .iter()
                .find(|scenario| field(scenario, "case").as_str() == Some(name))
                .unwrap_or_else(|| panic!("{name} scenario"));
            assert_family_matches(rollback, "target_rollback");
            assert_family_matches(rollback, "mtp_rollback");
        }
        let injected = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("injected_replay_failure"))
            .expect("injected replay failure scenario");
        assert_eq!(
            field(injected, "replay_error").as_str(),
            Some("injected replay error")
        );

        let cache = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("cache_suffix_refusal"))
            .expect("cache suffix refusal scenario");
        assert_eq!(field(cache, "cache_hit").as_bool(), Some(true));
        assert!(field(cache, "refusal")
            .as_str()
            .unwrap()
            .contains("cache-hit suffix"));
    }
}

/// Hardware takeover oracle used by `mtp_takeover_fill_hw`. Route knobs must
/// be set before starting the process. Qwen4 pairs token p with hidden p,
/// including the seed row; an accepted EOS stays pending, not head-filled.
pub fn run_mtp_takeover_fill_oracle(
    model_path: &Path,
    max_seq: usize,
) -> Result<Value, String> {
    use crate::bundle::Qwen4Bundle;
    use crate::mtp_spec::Qwen4MtpDrafter;
    use hipfire_runtime::ngram_mod::NgramModConfig;
    use hipfire_runtime::prompt_frame::JinjaChatFrame;
    use hipfire_runtime::spec::{MtpDrafter, SpecRequestConfig};
    use hipfire_runtime::tokenizer::Tokenizer;

    // Preserve the existing family/metadata report, and additionally hash
    // storage bytes: dequantized equality alone is not byte identity.
    fn snapshot(gpu: &Gpu, bundle: &Qwen4Bundle, drafter: &Qwen4MtpDrafter) -> Result<Value, String> {
        let mut families = bundle_family_json(gpu, bundle)?;
        let mut target = Family::new();
        let mut head = Family::new();
        let append = |family: &mut Family, tensor: &GpuTensor, elements: usize| {
            append_raw(gpu, family, tensor, elements, elements)
        };
        for layer in &bundle.state.gdn {
            append(&mut target, &layer.recurrent, layer.recurrent.numel())?;
            append(&mut target, &layer.conv, layer.conv.numel())?;
        }
        let full = bundle.config.num_key_value_heads * bundle.config.head_dim;
        let raw = bundle.config.indexer_kv_heads * bundle.config.indexer_head_dim;
        for qsa in &bundle.state.qsa {
            for (tensor, count) in [
                (&qsa.full_keys, qsa.full_len * full),
                (&qsa.full_values, qsa.full_len * full),
                (&qsa.raw_index_keys, qsa.raw_len * raw),
                (&qsa.pooled_keys, qsa.pooled_len * raw),
                (&qsa.partial_keys, qsa.partial_len * raw),
                (&qsa.partial_values, qsa.partial_len * full),
                (&qsa.selected_indices, qsa.selected_len),
            ] {
                append(&mut target, tensor, count)?;
            }
        }
        append(&mut target, &bundle.state.ple_conv, bundle.state.ple_conv.numel())?;
        append(&mut target, &bundle.state.hyper_feedback, bundle.state.hyper_feedback.numel())?;
        let state = bundle.mtp.as_ref().ok_or("missing MTP head")?.parity_state();
        let meta = state.parity_metadata();
        let buffers = state.parity_buffers();
        for (tensor, count) in [
            (buffers.full_keys, meta.full_len * full),
            (buffers.full_values, meta.full_len * full),
            (buffers.raw_index_keys, meta.raw_len * raw),
            (buffers.pooled_keys, meta.pooled_len * raw),
            (buffers.selected_indices, meta.selected_len),
            (buffers.selected_len_out, 1),
            (buffers.wide_hidden, buffers.wide_hidden.numel()),
        ] {
            append(&mut head, tensor, count)?;
        }
        families["raw_storage"] = json!({"target": target.finish(), "mtp": head.finish()});
        let pending = drafter.pending_hidden_for_parity()?;
        Ok(json!({
            "families": families,
            "pending_hidden": raw_prefix(gpu, pending, pending.numel(), pending.numel())?,
            "target_position": bundle.state.position,
            "mtp_position": bundle.mtp_position().map_err(|error| error.to_string())?,
        }))
    }

    fn head_rows(gpu: &Gpu, bundle: &Qwen4Bundle, start: usize, rows: usize) -> Result<Value, String> {
        let state = bundle.mtp.as_ref().ok_or("missing MTP head")?.parity_state();
        let buffers = state.parity_buffers();
        let full = bundle.config.num_key_value_heads * bundle.config.head_dim;
        let raw = bundle.config.indexer_kv_heads * bundle.config.indexer_head_dim;
        let mut family = Family::new();
        for (tensor, width) in [
            (buffers.full_keys, full), (buffers.full_values, full),
            (buffers.raw_index_keys, raw),
        ] {
            let view = tensor.sub_offset(start * width, rows * width);
            append_raw(gpu, &mut family, &view, rows * width, rows * width)?;
        }
        Ok(family.finish())
    }

    fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len(), "hidden dimensions");
        let numerator: f64 = a.iter().zip(b).map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2)).sum();
        let denominator: f64 = b.iter().map(|&x| f64::from(x).powi(2)).sum();
        (numerator / denominator.max(1e-30)).sqrt()
    }

    fn prefill(
        gpu: &mut Gpu,
        bundle: &mut Qwen4Bundle,
        drafter: &mut Qwen4MtpDrafter,
        prompt: &[u32],
    ) -> Result<u32, String> {
        // Arming again clears request-local history without reallocating the
        // pool. Cold prefill resets both device owners and pending hidden.
        drafter.configure_request(SpecRequestConfig {
            allow_ngram_modifier: true,
            ..SpecRequestConfig::default()
        });
        drafter.mtp_prefill(gpu, bundle, prompt, prompt, 0, false, &|| false)
    }

    let mut hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .map_err(|error| error.to_string())?;
    let template = hfq.chat_template().ok_or("artifact has no chat template")?;
    let rendered = JinjaChatFrame {
        tokenizer: &tokenizer,
        template: &template,
        system: None,
        user: "Write a Rust function fn parse_kv(line: &str) -> Option<(String, String)> that splits on the first '=', trims both sides, and rejects an empty key. Add three unit tests.",
        enable_thinking: false,
        bos_token: None,
        reasoning_strength: None,
        reasoning_effort: None,
    }.render()?;
    let prompt = tokenizer.encode(&rendered);
    const REF_STEPS: usize = 40;
    assert!(!prompt.is_empty() && prompt.len() + REF_STEPS + 8 < max_seq, "oracle context");
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let mut gpu = Gpu::init().map_err(|error| format!("hardware oracle requires HIP GPU: {error}"))?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single().map_err(|error| error.to_string())?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| qwen4_range_payload(&source, entry),
    ).map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    // Match run_mtp_fill_digest: the reusable family walker uses logical
    // F32 row widths. This is a head-fill/rollback oracle, not a KV-format
    // admission test; retain the automatic VMM/legacy storage backend.
    let format = crate::state::Qwen4StateFormat::F32;
    let backend = Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config, transaction, &receipt.placements, &mut gpu, max_seq,
        receipt.ple, format, backend,
    ).map_err(|error| error.to_string())?;
    // Longer than the entire oracle history: the armed pool is provably
    // empty/missing. Explicit takeovers supply the hit spans under test.
    let mut drafter = Qwen4MtpDrafter::new(3, max_seq, None).with_ngram(Some(NgramModConfig {
        capacity: 1024,
        n_match: prompt.len() + REF_STEPS + 8,
        n_min: 3,
        n_max: 3,
    }));
    let run = (|| -> Result<Value, String> {
        bundle.attach_forward(&mut gpu, max_seq).map_err(|error| error.to_string())?;
        bundle.attach_mtp(&mut gpu, max_seq).map_err(|error| error.to_string())?;
        map_bundle_context(&mut gpu, &mut bundle, max_seq)?;
        let dim = bundle.config.hc_count.checked_mul(bundle.config.hidden_size)
            .ok_or("oracle hidden width overflow")?;
        let eos = bundle.config.eos_token_id;
        let interleaved = hipfire_config::developer_var("HIPFIRE_MTP_INCREMENTAL")
            .is_ok_and(|value| value == "1");

        // Fresh prefill, then target-only single-row greedy forwards. No
        // speculative head predictions enter the reference continuation.
        let seed0 = prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?;
        let mut reference = vec![seed0];
        let mut reference_hidden = Vec::with_capacity(REF_STEPS);
        for i in 0..REF_STEPS {
            assert_ne!(reference[i], eos, "reference stopped before oracle sequence");
            let pick = bundle.spec_capture_token(&mut gpu, reference[i])
                .map_err(|error| error.to_string())?;
            let hidden = bundle.spec_hidden.as_ref().ok_or("missing reference hidden")?;
            reference_hidden.push(gpu.download_f32(&hidden.sub_offset(0, dim))
                .map_err(|error| error.to_string())?);
            reference.push(pick);
        }
        let wrong = |token: u32| if token == 0 { 1 } else { token - 1 };
        assert_eq!(prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?, seed0);
        let mut index = 0usize;
        let mut windows = Vec::new();
        // The first EOS candidate is index 1, so two drafts are emitted but
        // only seed and candidate 0 are consumed.
        for (label, expected, stop) in [
            ("partial", 1, false), ("full", 3, false), ("zero", 0, false),
            ("accepted_eos_index1", 2, true),
            ("reject_index0", 0, false), ("reject_index1", 1, false),
            ("reject_index2", 2, false),
        ] {
            let position = prompt.len() + index;
            let mut candidates = reference[index + 1..index + 4].to_vec();
            if !stop && expected < 3 {
                candidates[expected] = wrong(candidates[expected]);
            }
            let window_eos = if stop { candidates[1] } else { eos };
            let consumed = expected + 1 - usize::from(stop);
            let untouched = head_rows(&gpu, &bundle, position + consumed, 4 - consumed)?;
            let window = drafter.mtp_takeover_step(
                &mut gpu, &mut bundle, position, reference[index],
                &reference[..=index], &candidates, window_eos,
            )?;
            let emitted = if stop { expected } else { expected + 1 };
            assert_eq!(window.accepted, expected, "{label}: accepted prefix");
            assert_eq!(window.drafts_generated, 3, "{label}: candidates offered");
            assert_eq!(window.committed, reference[index + 1..index + 1 + emitted], "{label}: greedy IDs");
            assert_eq!(bundle.state.position, position + consumed, "{label}: target position");
            assert_eq!(bundle.mtp_position().map_err(|error| error.to_string())?, position + consumed, "{label}: head position");
            assert_eq!(
                head_rows(&gpu, &bundle, position + consumed, 4 - consumed)?,
                untouched, "{label}: unconsumed head rows must never be written",
            );
            // Interleaved capture overwrites row 0 each time; that physical
            // row is the logical verify row consumed-1.
            let capture_row = if interleaved { 0 } else { consumed - 1 };
            let captured = bundle.spec_hidden.as_ref().ok_or("missing verify hidden")?
                .sub_offset(capture_row * dim, dim);
            let pending = drafter.pending_hidden_for_parity()?;
            assert_eq!(
                raw_prefix(&gpu, pending, dim, dim)?,
                raw_prefix(&gpu, &captured, dim, dim)?,
                "{label}: pending hidden must equal last consumed verify row",
            );
            let mut report = snapshot(&gpu, &bundle, &drafter)?;
            report["label"] = json!(label);
            report["accepted"] = json!(window.accepted);
            report["consumed"] = json!(consumed);
            report["committed"] = json!(window.committed);
            windows.push(report);
            index += consumed;
        }
        // No reset/demotion between the explicit hits and this real native
        // miss. The same request remains armed; history is too short to hit.
        assert!(drafter.request_stats().mtp_ngram, "modifier request must remain armed");
        let before_stats = drafter.request_stats();
        let miss = drafter.mtp_step(
            &mut gpu, &mut bundle, prompt.len() + index, reference[index],
            &reference[..=index], 3, eos, None,
        )?;
        assert_eq!(miss.committed.first(), Some(&reference[index + 1]), "native miss resumes at reference token");
        if !interleaved {
            assert!(miss.drafts_generated >= 1, "native miss must draft, not retire to AR");
        }
        let after_stats = drafter.request_stats();
        assert_eq!(after_stats.ngram_mod_windows, before_stats.ngram_mod_windows, "empty pool must miss");
        assert!(after_stats.mtp_windows > before_stats.mtp_windows, "native window counter must advance");

        // Two identical cold states, same first rejection, different rejected
        // tail. Compare live device families, raw storage and pending bytes.
        let mut starts = Vec::new();
        let mut tails = Vec::new();
        for variant in 0..2 {
            assert_eq!(prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?, seed0);
            starts.push(snapshot(&gpu, &bundle, &drafter)?);
            let untouched = head_rows(&gpu, &bundle, prompt.len() + 1, 3)?;
            let mut candidates = reference[1..4].to_vec();
            candidates[0] = wrong(candidates[0]);
            if variant == 1 {
                candidates[1] = wrong(candidates[1]);
                candidates[2] = wrong(candidates[2]);
            }
            let window = drafter.mtp_takeover_step(
                &mut gpu, &mut bundle, prompt.len(), seed0, &[seed0], &candidates, eos,
            )?;
            assert_eq!(window.accepted, 0, "tail fixture first rejection");
            assert_eq!(window.committed, reference[1..2], "tail fixture committed IDs");
            assert_eq!(head_rows(&gpu, &bundle, prompt.len() + 1, 3)?, untouched,
                "tail fixture: rejected head rows must remain untouched");
            tails.push(snapshot(&gpu, &bundle, &drafter)?);
        }
        assert_eq!(starts[0], starts[1], "tail fixture must start byte-identically");
        assert_eq!(tails[0], tails[1], "rejected tail must not enter committed target/head/pending state");

        // Compare token-p/hidden-p teacher forcing against the existing route.
        assert_eq!(prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?, seed0);
        let pairing_start = snapshot(&gpu, &bundle, &drafter)?;
        let full = drafter.mtp_takeover_step(
            &mut gpu, &mut bundle, prompt.len(), seed0, &[seed0], &reference[1..4], eos,
        )?;
        assert_eq!(full.accepted, 3, "pairing takeover must fully accept");
        let takeover = gpu.download_f32(drafter.pending_hidden_for_parity()?)
            .map_err(|error| error.to_string())?;
        let takeover_state = snapshot(&gpu, &bundle, &drafter)?;
        assert_eq!(prefill(&mut gpu, &mut bundle, &mut drafter, &prompt)?, seed0);
        assert_eq!(snapshot(&gpu, &bundle, &drafter)?, pairing_start, "pairing starts must match");
        assert!(drafter.mtp_forced_advance(
            &mut gpu, &mut bundle, &reference[..4], prompt.len(), &|| false,
        )?, "teacher-forced route must handle all rows");
        let forced_state = snapshot(&gpu, &bundle, &drafter)?;
        assert_eq!(takeover_state["target_position"], forced_state["target_position"]);
        assert_eq!(takeover_state["mtp_position"], forced_state["mtp_position"]);
        let forced = gpu.download_f32(drafter.pending_hidden_for_parity()?)
            .map_err(|error| error.to_string())?;
        let correct = rel_l2(&takeover, &forced);
        // h_(P+2), rather than the required h_(P+3), is the off-by-one
        // Qwen3.5-style pairing. Reference uses the same one-row forwards.
        let shifted = rel_l2(&takeover, &reference_hidden[2]);
        assert!(correct.is_finite() && shifted.is_finite() && correct < 0.1 * shifted,
            "teacher-forced token-p/hidden-p pairing: rel-L2={correct:e}, shifted={shifted:e}");
        Ok(json!({
            "schema": "hipfire.qwen4.mtp_takeover_fill.v1",
            "gpu_arch": gpu.arch,
            "max_seq": max_seq,
            "prompt_tokens": prompt.len(),
            "state_format": "f32",
            "kv_backend": backend.name(),
            "reference": reference,
            "windows": windows,
            "rejected_tail": {"same_state": true, "before": starts[0], "after": tails[0]},
            "pairing": {
                "rel_l2": correct, "shifted_rel_l2": shifted,
                "shifted_row": -1,
                "pending_byte_identical": takeover_state["pending_hidden"] == forced_state["pending_hidden"],
                "families_byte_identical": takeover_state["families"] == forced_state["families"],
                "takeover": takeover_state, "forced": forced_state,
            },
            "native_miss": {"drafts_generated": miss.drafts_generated, "committed": miss.committed},
        }))
    })();
    Box::new(drafter).mtp_free(&mut gpu);
    let cleanup = bundle.free_gpu(&mut gpu).map_err(|error| error.to_string());
    let value = run?;
    cleanup?;
    Ok(value)
}
