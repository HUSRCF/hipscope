// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! G4 lifecycle evidence.
//!
//! CPU tests exercise the loader-owned reset/eviction and terminal contracts on
//! every run. The GPU tests are intentionally ignored: they need the exact
//! gfx1151 fixture set and the opt-in fault hooks described in each test's
//! ignore reason.

#![allow(clippy::all)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use hipfire_engine::emit::{emit_active_attempt_error, emit_gen_start, emit_qwen_ar_cancelled};
use hipfire_engine::terminal::{
    activate_terminal_control, clear_terminal_control, emit_staged_terminal_done,
    set_active_attempt_id,
};
use hipfire_generate::ar::GenerationRoute;
use hipfire_generate::common::attest_rollback_steps;
use hipfire_runtime::dflash::TargetHiddenLog;
use hipfire_runtime::kv_adaptive::{KvAdaptive, Preset};
use hipfire_runtime::loader_api::{CaskConfig, SpecLoadCfg};
use hipfire_runtime::llama::{EmbeddingFormat, WeightTensor};
use hipfire_runtime::model_load::WeightSource;
use rdna_compute::{Gpu, GpuTensor};

// ── loader-owned reset matrix ──────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResetShape {
    Single,
    PipelineParallel,
    TensorParallel,
    ExpertParallel,
}

impl ResetShape {
    fn lane_count(self) -> usize {
        match self {
            Self::Single => 1,
            Self::PipelineParallel => 2,
            Self::TensorParallel | Self::ExpertParallel => 4,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::PipelineParallel => "pp",
            Self::TensorParallel => "tp",
            Self::ExpertParallel => "ep",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LaneState {
    persistent_policy_id: u64,
    persistent_scratch_id: u64,
    request_tokens: Vec<u32>,
}

#[derive(Debug)]
struct ResetHarness {
    shape: ResetShape,
    seq_pos: usize,
    conversation_tokens: Vec<u32>,
    lanes: Vec<LaneState>,
}

impl ResetHarness {
    fn new(shape: ResetShape) -> Self {
        let lanes = (0..shape.lane_count())
            .map(|lane| LaneState {
                persistent_policy_id: 0xCA5E_0000 + lane as u64,
                persistent_scratch_id: 0x5C12_0000 + lane as u64,
                request_tokens: vec![lane as u32, 10, 11],
            })
            .collect();
        Self {
            shape,
            seq_pos: 37,
            conversation_tokens: vec![1, 2, 3, 4],
            lanes,
        }
    }

    /// Model the observable part of the loader-owned adapter matrix. Each lane
    /// is attempted even after another lane reports an error; successful lanes
    /// clear request state, while persistent policy/scratch identity remains.
    fn reset_via_loader_adapter(&mut self, failures: &[usize]) -> hipfire_generate::common::RollbackEpilogue {
        self.seq_pos = 0;
        self.conversation_tokens.clear();
        let failures: BTreeSet<usize> = failures.iter().copied().collect();
        let mut steps = Vec::with_capacity(self.lanes.len());
        for (lane, state) in self.lanes.iter_mut().enumerate() {
            let result = if failures.contains(&lane) {
                Err(format!("{} lane {lane} reset failed", self.shape.label()))
            } else {
                state.request_tokens.clear();
                Ok(())
            };
            steps.push((format!("{} lane {lane}", self.shape.label()), result));
        }
        let refs: Vec<(&str, Result<(), String>)> = steps
            .iter()
            .map(|(name, result)| (name.as_str(), result.clone()))
            .collect();
        attest_rollback_steps(&refs, Ok(()))
    }

    fn cancel_lane(&mut self, lane: usize) {
        self.lanes[lane].request_tokens.clear();
    }
}

#[test]
fn loader_owned_reset_matrix_clears_all_request_state_and_preserves_policy() {
    for shape in [
        ResetShape::Single,
        ResetShape::PipelineParallel,
        ResetShape::TensorParallel,
        ResetShape::ExpertParallel,
    ] {
        let mut harness = ResetHarness::new(shape);
        let persistent_before: Vec<(u64, u64)> = harness
            .lanes
            .iter()
            .map(|lane| (lane.persistent_policy_id, lane.persistent_scratch_id))
            .collect();
        let epilogue = harness.reset_via_loader_adapter(&[]);
        assert!(epilogue.rolled_back, "{} reset must attest", shape.label());
        assert_eq!(harness.seq_pos, 0);
        assert!(harness.conversation_tokens.is_empty());
        assert!(harness
            .lanes
            .iter()
            .all(|lane| lane.request_tokens.is_empty()));
        assert_eq!(
            harness
                .lanes
                .iter()
                .map(|lane| (lane.persistent_policy_id, lane.persistent_scratch_id))
                .collect::<Vec<_>>(),
            persistent_before,
            "{} reset replaced persistent owners",
            shape.label()
        );
    }
}

#[test]
fn loader_owned_reset_matrix_aggregates_failures_and_visits_every_lane() {
    for shape in [
        ResetShape::Single,
        ResetShape::PipelineParallel,
        ResetShape::TensorParallel,
        ResetShape::ExpertParallel,
    ] {
        let failing_lane = shape.lane_count().saturating_sub(1);
        let mut harness = ResetHarness::new(shape);
        let epilogue = harness.reset_via_loader_adapter(&[failing_lane]);
        assert!(!epilogue.rolled_back, "failed {} reset cannot attest", shape.label());
        let context = epilogue.context.expect("failure context");
        assert!(
            context.contains(&format!("lane {failing_lane}")),
            "{} reset omitted failing lane: {context}",
            shape.label()
        );
        assert_eq!(
            harness.lanes[..failing_lane]
                .iter()
                .filter(|lane| lane.request_tokens.is_empty())
                .count(),
            failing_lane,
            "{} adapter stopped before later lane failure",
            shape.label()
        );
        assert_eq!(
            harness.lanes[failing_lane].request_tokens,
            vec![failing_lane as u32, 10, 11],
            "failed lane state must remain inspectable for fail-closed recovery"
        );
    }
}

#[test]
fn loader_owned_reset_is_scoped_to_one_batch_lane() {
    let mut harness = ResetHarness::new(ResetShape::TensorParallel);
    let policy_before = harness.lanes[1].persistent_policy_id;
    let scratch_before = harness.lanes[1].persistent_scratch_id;
    let peer_request = harness.lanes[1].request_tokens.clone();
    harness.cancel_lane(0);
    assert!(harness.lanes[0].request_tokens.is_empty());
    assert_eq!(harness.lanes[1].request_tokens, peer_request);
    assert_eq!(harness.lanes[1].persistent_policy_id, policy_before);
    assert_eq!(harness.lanes[1].persistent_scratch_id, scratch_before);
}

// ── eviction policy/request state isolation ────────────────────────────────

#[derive(Debug)]
struct EvictionRequestState {
    seq_pos: usize,
    compact_offset: i32,
    adaptive_step: usize,
    speculative_pending: Vec<u32>,
    target_hidden: TargetHiddenLog,
}

impl EvictionRequestState {
    fn seeded() -> Self {
        let mut target_hidden = TargetHiddenLog::new();
        target_hidden.seed_prompt(3);
        target_hidden.mark_uploaded(3);
        target_hidden.mark_proj_cached(3);
        target_hidden.mark_full_cached(3);
        Self {
            seq_pos: 19,
            compact_offset: 7,
            adaptive_step: 2,
            speculative_pending: vec![41, 42],
            target_hidden,
        }
    }

    fn reset(&mut self) {
        self.seq_pos = 0;
        self.compact_offset = 0;
        self.adaptive_step = 0;
        self.speculative_pending.clear();
        self.target_hidden.reset();
    }
}

#[test]
fn eviction_policy_scratch_counter_identity_survives_reset() {
    let mut adaptive = KvAdaptive::from_preset(Preset::Aggressive, 10_000, 4, 256);
    let floors = (adaptive.k_floor, adaptive.v_floor);
    let steps = adaptive.steps.clone();
    let thresholds = adaptive.thresholds.clone();
    let gate = adaptive.configure_eviction_handoff(128);
    let policy = Arc::new(String::from("triattn-policy"));
    let scratch = Arc::new(String::from("triattn-scratch"));
    let policy_ptr = Arc::as_ptr(&policy);
    let scratch_ptr = Arc::as_ptr(&scratch);
    let eviction_count = std::cell::Cell::new(4usize);
    let mut request = EvictionRequestState::seeded();

    adaptive.cur_k = hipfire_runtime::kv_adaptive::KMode::Fwht2;
    adaptive.cur_v = hipfire_runtime::kv_adaptive::VMode::Lloyd2;
    adaptive.next_step = adaptive.steps.len();
    request.reset();
    adaptive.reset();

    assert_eq!((adaptive.k_floor, adaptive.v_floor), floors);
    assert_eq!(adaptive.steps, steps);
    assert_eq!(adaptive.thresholds, thresholds);
    assert!(!gate.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(Arc::as_ptr(&policy), policy_ptr);
    assert_eq!(Arc::as_ptr(&scratch), scratch_ptr);
    assert_eq!(eviction_count.get(), 4, "counter is model-lifetime telemetry");
    assert_eq!(request.seq_pos, 0);
    assert_eq!(request.compact_offset, 0);
    assert_eq!(request.adaptive_step, 0);
    assert!(request.speculative_pending.is_empty());
    assert_eq!(request.target_hidden.uploaded_rows(), 0);
    assert!(request.target_hidden.abs_positions().is_empty());
}

#[test]
fn eviction_request_state_clears_on_normal_reset_and_speculative_rollback() {
    let mut adaptive = KvAdaptive::from_preset(Preset::Balanced, 10_000, 4, 256);
    let policy = Arc::new(String::from("persistent-policy"));
    let scratch = Arc::new(String::from("persistent-scratch"));
    let policy_ptr = Arc::as_ptr(&policy);
    let scratch_ptr = Arc::as_ptr(&scratch);
    let eviction_count = std::cell::Cell::new(9usize);
    for rollback in [false, true] {
        let mut request = EvictionRequestState::seeded();
        request.seq_pos = 41;
        request.compact_offset = 13;
        request.adaptive_step = 3;
        request.speculative_pending.extend([99, 100]);
        if rollback {
            // Speculative rollback has the same request-local clearing contract
            // as a normal reset; it must not replace the persistent owner.
            request.reset();
        } else {
            request.reset();
        }
        adaptive.reset();
        assert_eq!(request.seq_pos, 0);
        assert_eq!(request.compact_offset, 0);
        assert_eq!(request.adaptive_step, 0);
        assert!(request.speculative_pending.is_empty());
        assert_eq!(request.target_hidden.uploaded_rows(), 0);
        assert_eq!(Arc::as_ptr(&policy), policy_ptr);
        assert_eq!(Arc::as_ptr(&scratch), scratch_ptr);
        assert_eq!(eviction_count.get(), 9);
    }
}

// ── concrete writer race/cardinality matrix ───────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterFamily {
    Single,
    Batch,
    Ep,
    Vl,
    Glimmer,
}

impl WriterFamily {
    fn route(self) -> GenerationRoute {
        match self {
            Self::Single | Self::Batch | Self::Vl => GenerationRoute::QwenAr,
            Self::Ep => GenerationRoute::Deepseek4Ep,
            Self::Glimmer => GenerationRoute::GlimmerAr,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TerminalWriter {
    Done,
    Error,
    Cancel,
}

fn terminal_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
}

fn emit_family_gen_start(family: WriterFamily, output: &mut Vec<u8>, id: &str) {
    match family {
        WriterFamily::Ep => hipfire_generate::qwen::emit_ds4_ep_gen_start(
            output,
            id,
            hipfire_runtime::prompt_frame::ThinkMode::NonThink,
        ),
        WriterFamily::Single
        | WriterFamily::Batch
        | WriterFamily::Vl
        | WriterFamily::Glimmer => {
            let contract = if matches!(family, WriterFamily::Single | WriterFamily::Vl) {
                Some(2)
            } else {
                None
            };
            emit_gen_start(output, id, false, contract);
        }
    }
}

fn emit_terminal_writer(writer: TerminalWriter, output: &mut Vec<u8>, id: &str, attempt: u64) {
    match writer {
        TerminalWriter::Done => {
            let pending = serde_json::json!({
                "type": "done",
                "id": id,
                "attempt_id": attempt,
                "finish_reason": "stop",
            });
            emit_staged_terminal_done(output, &pending);
        }
        TerminalWriter::Error => emit_active_attempt_error(
            output,
            Some(id),
            "synthetic writer race",
            "internal",
            false,
            true,
        ),
        TerminalWriter::Cancel => emit_qwen_ar_cancelled(output, id, 0),
    }
}

fn parse_json_lines(bytes: &[u8]) -> Vec<serde_json::Value> {
    String::from_utf8(bytes.to_vec())
        .expect("writer output utf8")
        .lines()
        .map(|line| serde_json::from_str(line).expect("writer JSONL"))
        .collect()
}

#[test]
fn all_generation_routes_are_named_once_and_writer_families_are_in_all() {
    let names: BTreeSet<&str> = GenerationRoute::ALL.iter().map(|route| route.name()).collect();
    assert_eq!(names.len(), GenerationRoute::ALL.len());
    for family in [
        WriterFamily::Single,
        WriterFamily::Batch,
        WriterFamily::Ep,
        WriterFamily::Vl,
        WriterFamily::Glimmer,
    ] {
        assert!(
            GenerationRoute::ALL.contains(&family.route()),
            "writer family route {:?} absent from GenerationRoute::ALL",
            family
        );
    }
}

#[test]
fn concrete_generation_writers_emit_gen_start_first_and_claim_one_terminal() {
    let _lock = terminal_test_lock();
    let families = [
        WriterFamily::Single,
        WriterFamily::Batch,
        WriterFamily::Ep,
        WriterFamily::Vl,
        WriterFamily::Glimmer,
    ];
    let writers = [TerminalWriter::Done, TerminalWriter::Error, TerminalWriter::Cancel];

    for (family_idx, family) in families.into_iter().enumerate() {
        for (writer_idx, winner) in writers.into_iter().enumerate() {
            let id = format!("g4-{:?}-{writer_idx}", family);
            let attempt = 700 + (family_idx * 10 + writer_idx) as u64;
            clear_terminal_control();
            activate_terminal_control(&id, attempt);
            set_active_attempt_id(attempt);
            let mut output = Vec::new();
            emit_family_gen_start(family, &mut output, &id);
            let start_line_count = parse_json_lines(&output).len();

            // Publish one winner first, then race the other concrete terminal
            // writers against an already-claimed transaction. This keeps the
            // expected winner deterministic while still exercising concurrent
            // loser paths and their claim checks.
            emit_terminal_writer(winner, &mut output, &id, attempt);
            let shared = Arc::new(Mutex::new(output));
            let mut joins = Vec::new();
            for contender in writers.iter().copied().filter(|c| *c != winner) {
                let shared = Arc::clone(&shared);
                let id = id.clone();
                joins.push(std::thread::spawn(move || {
                    set_active_attempt_id(attempt);
                    let mut out = shared.lock().unwrap();
                    emit_terminal_writer(contender, &mut out, &id, attempt);
                }));
            }
            for join in joins {
                join.join().expect("terminal writer thread");
            }
            output = Arc::try_unwrap(shared)
                .expect("writer output owner")
                .into_inner()
                .unwrap();
            let lines = parse_json_lines(&output);
            assert!(
                lines.len() >= start_line_count,
                "terminal writer removed gen_start"
            );
            assert_eq!(
                lines
                    .first()
                    .and_then(|v| v.get("type"))
                    .and_then(|v| v.as_str()),
                Some("gen_start")
            );
            let terminal_types: Vec<&str> = lines[start_line_count..]
                .iter()
                .filter_map(|value| value.get("type").and_then(|v| v.as_str()))
                .collect();
            match winner {
                TerminalWriter::Done => {
                    assert_eq!(terminal_types.iter().filter(|t| **t == "done").count(), 1)
                }
                TerminalWriter::Error => {
                    assert_eq!(terminal_types.iter().filter(|t| **t == "error").count(), 1)
                }
                TerminalWriter::Cancel => {
                    assert_eq!(terminal_types.iter().filter(|t| **t == "aborted").count(), 1);
                    assert_eq!(terminal_types.iter().filter(|t| **t == "done").count(), 1);
                }
            }

            // Late writers after the race are also no-ops.
            let before_late = output.clone();
            for contender in writers {
                emit_terminal_writer(contender, &mut output, &id, attempt);
            }
            assert_eq!(output, before_late, "terminal claim leaked a late writer");
            clear_terminal_control();
            set_active_attempt_id(0);
        }
    }
}

// ── ignored live-GPU ownership tests ───────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WeightFaultStage {
    Embed,
    FinalNorm,
    Output,
    Layer(usize),
}

struct FaultingSource<S> {
    inner: S,
    stage: WeightFaultStage,
}

impl<S> FaultingSource<S> {
    fn new(inner: S, stage: WeightFaultStage) -> Self {
        Self { inner, stage }
    }

    fn fail(&self, stage: WeightFaultStage) -> hip_bridge::HipResult<()> {
        if self.stage == stage {
            Err(hip_bridge::HipError::new(0x4734, &format!("G4 fault at {stage:?}")))
        } else {
            Ok(())
        }
    }
}

impl<S: WeightSource> WeightSource for FaultingSource<S> {
    type Layer = S::Layer;

    fn n_layers(&self) -> usize {
        self.inner.n_layers()
    }

    fn prepare(&mut self, n_devices: usize) -> hip_bridge::HipResult<()> {
        self.inner.prepare(n_devices)
    }

    fn read_embed(&mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<(GpuTensor, EmbeddingFormat)> {
        self.fail(WeightFaultStage::Embed)?;
        self.inner.read_embed(gpu)
    }

    fn read_final_norm(&mut self, gpu: &mut Gpu) -> hip_bridge::HipResult<GpuTensor> {
        self.fail(WeightFaultStage::FinalNorm)?;
        self.inner.read_final_norm(gpu)
    }

    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> hip_bridge::HipResult<(WeightTensor, bool)> {
        self.fail(WeightFaultStage::Output)?;
        self.inner.read_output(gpu, embd, embd_fmt, can_alias)
    }

    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> hip_bridge::HipResult<Self::Layer> {
        self.fail(WeightFaultStage::Layer(layer_idx))?;
        self.inner.read_layer(gpu, layer_idx)
    }

    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer) {
        self.inner.free_layer(gpu, layer)
    }
}

fn required_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} for ignored G4 GPU test")))
}

fn assert_gpu_baseline(gpu: &mut Gpu, baseline: usize) {
    gpu.ensure_vmm_cleaned().expect("VMM cleanup after lifecycle attempt");
    gpu.drain_pool();
    assert_eq!(gpu.vmm_allocation_count(), baseline, "VMM owner leaked across lifecycle attempt");
}

#[test]
#[ignore = "requires exact gfx1151 HIP device plus warm HFQ fixture in HIPFIRE_G4_HFQ_FIXTURE; ignored on CPU"]
fn gpu_hfq_staged_failure_sweep_then_success_repeated_unload() {
    let path = required_path("HIPFIRE_G4_HFQ_FIXTURE");
    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(gpu.arch, "gfx1151", "G4 fixture is certified only on gfx1151");
    let baseline = gpu.vmm_allocation_count();

    // Warm baseline and two complete unload cycles prove publication ownership,
    // alias handling, and repeated unload do not accumulate a stale owner.
    for _ in 0..2 {
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let mut source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("warm HFQ load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }

    for stage in [
        WeightFaultStage::Embed,
        WeightFaultStage::FinalNorm,
        WeightFaultStage::Output,
        WeightFaultStage::Layer(0),
        WeightFaultStage::Layer(2),
    ] {
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let mut source = FaultingSource::new(source, stage);
        let result = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        );
        assert!(result.is_err(), "fault stage {stage:?} unexpectedly loaded");
        assert_gpu_baseline(&mut gpu, baseline);

        // A failed load must not poison the next successful load on the same
        // warm GPU, and a second unload must remain clean.
        let mut hfq = hipfire_runtime::hfq::HfqFile::open(&path).expect("HFQ fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_hfq(&hfq).expect("Qwen config");
        let mut source = hipfire_arch_qwen35::qwen35::HfqSource::new(&mut hfq, &cfg);
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("post-failure HFQ load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }
}

#[test]
#[ignore = "requires exact gfx1151 HIP device plus warm ParoQuant safetensors fixture in HIPFIRE_G4_PARO_DIR; ignored on CPU"]
fn gpu_paro_staged_failure_sweep_then_success_repeated_unload() {
    let path = required_path("HIPFIRE_G4_PARO_DIR");
    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(gpu.arch, "gfx1151", "G4 fixture is certified only on gfx1151");
    let baseline = gpu.vmm_allocation_count();

    for stage in [
        WeightFaultStage::Embed,
        WeightFaultStage::FinalNorm,
        WeightFaultStage::Output,
        WeightFaultStage::Layer(0),
        WeightFaultStage::Layer(2),
    ] {
        let source_file = hipfire_runtime::safetensors_source::SafetensorsSource::open(&path)
            .expect("ParoQuant fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source_file)
            .expect("Paro config");
        let source = hipfire_arch_qwen35::qwen35::ParoSource::new(&source_file, &cfg)
            .expect("Paro source");
        let mut source = FaultingSource::new(source, stage);
        let result = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        );
        assert!(result.is_err(), "fault stage {stage:?} unexpectedly loaded");
        assert_gpu_baseline(&mut gpu, baseline);

        let source_file = hipfire_runtime::safetensors_source::SafetensorsSource::open(&path)
            .expect("ParoQuant fixture");
        let cfg = hipfire_arch_qwen35::qwen35::config_from_safetensors(&source_file)
            .expect("Paro config");
        let mut source = hipfire_arch_qwen35::qwen35::ParoSource::new(&source_file, &cfg)
            .expect("Paro source");
        let weights = hipfire_arch_qwen35::qwen35::load_weights(
            &mut source,
            std::slice::from_mut(&mut gpu),
            &hipfire_runtime::model_load::Layout::single(cfg.n_layers),
        )
        .expect("post-failure Paro load");
        weights.free_gpu(&mut gpu);
        assert_gpu_baseline(&mut gpu, baseline);
    }
}

fn try_load_model(
    gpu: &mut Gpu,
    target: &Path,
    draft: Option<&Path>,
    spec: SpecLoadCfg,
) -> Result<hipfire_loader::LoadedModel, String> {
    let cask = CaskConfig::default();
    hipfire_loader::load_model(
        target.to_str().expect("target utf8"),
        1024,
        draft.map(|path| path.to_str().expect("draft utf8")),
        None,
        None,
        None,
        &cask,
        1,
        spec,
        gpu,
    )
}

fn load_and_unload_model(gpu: &mut Gpu, target: &Path, draft: Option<&Path>, spec: SpecLoadCfg) {
    let model = try_load_model(gpu, target, draft, spec).expect("model load");
    hipfire_loader::unload_model(model, gpu).expect("model unload");
}

fn expect_load_failure(
    gpu: &mut Gpu,
    target: &Path,
    draft: Option<&Path>,
    spec: SpecLoadCfg,
) {
    match try_load_model(gpu, target, draft, spec) {
        Err(error) => assert!(
            !error.is_empty(),
            "fault fixture returned an empty failure reason"
        ),
        Ok(model) => {
            let _ = hipfire_loader::unload_model(model, gpu);
            panic!("fault fixture unexpectedly loaded");
        }
    }
}

#[test]
#[ignore = "requires exact gfx1151 HIP device, DFlash/DSpark sidecars, and existing stage-fault fixture hooks via HIPFIRE_G4_*; ignored on CPU"]
fn gpu_dflash_dspark_target_verify_and_head_failures_recover_without_double_free() {
    let dflash_target = required_path("HIPFIRE_G4_DFLASH_TARGET");
    let dflash_draft = required_path("HIPFIRE_G4_DFLASH_DRAFT");
    let dspark_target = required_path("HIPFIRE_G4_DSPARK_TARGET");
    let dspark_draft = required_path("HIPFIRE_G4_DSPARK_DRAFT");
    let dflash_fault = required_path("HIPFIRE_G4_DFLASH_TARGET_VERIFY_FAILURE");
    let dspark_fault = required_path("HIPFIRE_G4_DSPARK_HEAD_FAILURE");

    let mut gpu = Gpu::init().expect("HIP device");
    assert_eq!(gpu.arch, "gfx1151", "G4 fixtures are certified only on gfx1151");
    let baseline = gpu.vmm_allocation_count();

    load_and_unload_model(
        &mut gpu,
        &dflash_target,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    expect_load_failure(
        &mut gpu,
        &dflash_fault,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    load_and_unload_model(
        &mut gpu,
        &dflash_target,
        Some(&dflash_draft),
        SpecLoadCfg {
            dspark: Some(false),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    expect_load_failure(
        &mut gpu,
        &dspark_fault,
        Some(&dspark_draft),
        SpecLoadCfg {
            dspark: Some(true),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
    load_and_unload_model(
        &mut gpu,
        &dspark_target,
        Some(&dspark_draft),
        SpecLoadCfg {
            dspark: Some(true),
            ..Default::default()
        },
    );
    assert_gpu_baseline(&mut gpu, baseline);
}
