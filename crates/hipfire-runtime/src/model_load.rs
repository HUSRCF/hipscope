//! Arch-generic whole-model weight orchestration (Tier-2). Sequences
//! embed → final-norm → output → per-device layer loop over a `WeightSource`,
//! whose impls own the format-specific reads (HFQ vs ParoQuant) and bake their
//! own config. Per-arch crates wrap the returned `LoadedWeights<L>` into their
//! own weights struct. Complements `weight_backend::WeightBackend` (Tier-3,
//! per-tensor dequant), which `WeightSource::read_layer` calls internally.

use crate::llama::{EmbeddingFormat, WeightTensor};
use crate::multi_gpu::Gpus;
use hip_bridge::HipResult;
use rdna_compute::{Gpu, GpuTensor};

/// Where each piece of the model lands across a device slice. `single` = the
/// n==1 degenerate case (everything on device 0). Moved verbatim from
/// `hipfire-arch-qwen35::qwen35::Layout` — arch-agnostic (depends only on `Gpus`).
pub struct Layout {
    output_device: usize,
    layer_to_device: Vec<usize>,
}
impl Layout {
    pub fn single(n_layers: usize) -> Self {
        Self {
            output_device: 0,
            layer_to_device: vec![0; n_layers],
        }
    }
    pub fn from_gpus(g: &Gpus, n_layers: usize) -> Self {
        Self {
            output_device: g.output_device,
            layer_to_device: (0..n_layers).map(|i| g.device_for_layer(i)).collect(),
        }
    }
    pub fn device_for_layer(&self, i: usize) -> usize {
        self.layer_to_device[i]
    }
    pub fn output_device(&self) -> usize {
        self.output_device
    }
}

/// Neutral result of the orchestrator. Each arch assembles its own weights
/// struct from this (qwen35 adds `pager`; llama drops `lm_head_aliases_embd`).
pub struct LoadedWeights<L> {
    pub token_embd: GpuTensor,
    pub embd_format: EmbeddingFormat,
    pub output_norm: GpuTensor,
    pub output: WeightTensor,
    pub layers: Vec<L>,
    /// True iff the tied lm_head aliases the embedding buffer (qwen35 single-GPU);
    /// llama always returns `false` (it reuploads).
    pub lm_head_aliases_embd: bool,
}

/// Whole-model weight source — the one place HFQ vs PaRo differs. Config is held
/// by the impl (not passed per-call) so the orchestrator stays config-agnostic.
/// `read_layer` reuses Tier-3 `load_layer<B: WeightBackend>` internally.
pub trait WeightSource {
    type Layer;
    fn n_layers(&self) -> usize;
    /// Pre-load hook. HFQ drops the mmap when n==1; PaRo rejects n>1; llama no-op.
    fn prepare(&mut self, n_devices: usize) -> HipResult<()>;
    fn read_embed(&mut self, gpu: &mut Gpu) -> HipResult<(GpuTensor, EmbeddingFormat)>;
    fn read_final_norm(&mut self, gpu: &mut Gpu) -> HipResult<GpuTensor>;
    /// `can_alias` is true iff embed and output share a device (n==1); the impl
    /// decides whether to use it (qwen35 aliases; llama ignores it and reuploads).
    fn read_output(
        &mut self,
        gpu: &mut Gpu,
        embd: &GpuTensor,
        embd_fmt: EmbeddingFormat,
        can_alias: bool,
    ) -> HipResult<(WeightTensor, bool)>;
    fn read_layer(&mut self, gpu: &mut Gpu, layer_idx: usize) -> HipResult<Self::Layer>;
    /// Release one successfully loaded layer during whole-model rollback.
    ///
    /// Layer ownership is architecture-specific (Qwen3.5 carries MoE
    /// pointer tables and shared Paro sidecars), so the source supplies the
    /// exact teardown instead of relying on a generic `Drop`.
    fn free_layer(&mut self, gpu: &mut Gpu, layer: Self::Layer);
}

/// Drive a `WeightSource` across a device slice. Single shared copy of the
/// embed → norm → output → per-device layer loop.
pub fn load_weights<S: WeightSource>(
    source: &mut S,
    devices: &mut [Gpu],
    layout: &Layout,
) -> HipResult<LoadedWeights<S::Layer>> {
    source.prepare(devices.len())?;
    let out_dev = layout.output_device();
    let can_alias = devices.len() == 1;

    // Every successful allocation is published into one of these staging
    // owners before the next fallible step. They remain local until the final
    // `LoadedWeights` move, so a later layer/global/head error drains the
    // complete prefix rather than orphaning earlier GPU tensors.
    let mut staged_embd: Option<(GpuTensor, EmbeddingFormat)> = None;
    let mut staged_output_norm: Option<GpuTensor> = None;
    let mut staged_output: Option<(WeightTensor, bool)> = None;
    let mut staged_layers: Vec<S::Layer> = Vec::with_capacity(source.n_layers());

    let result = (|| -> HipResult<LoadedWeights<S::Layer>> {
        let (token_embd, embd_format) = source.read_embed(&mut devices[0])?;
        staged_embd = Some((token_embd, embd_format));

        let output_norm = source.read_final_norm(&mut devices[out_dev])?;
        staged_output_norm = Some(output_norm);

        let (output, lm_head_aliases_embd) = source.read_output(
            &mut devices[out_dev],
            &staged_embd.as_ref().expect("embedding staged").0,
            embd_format,
            can_alias,
        )?;
        staged_output = Some((output, lm_head_aliases_embd));

        for i in 0..source.n_layers() {
            let d = layout.device_for_layer(i);
            staged_layers.push(source.read_layer(&mut devices[d], i)?);
        }

        let (token_embd, embd_format) = staged_embd.take().expect("embedding staged");
        let output_norm = staged_output_norm.take().expect("output norm staged");
        let (output, lm_head_aliases_embd) = staged_output.take().expect("output staged");
        Ok(LoadedWeights {
            token_embd,
            embd_format,
            output_norm,
            output,
            layers: std::mem::take(&mut staged_layers),
            lm_head_aliases_embd,
        })
    })();

    if result.is_err() {
        // Completed layers are architecture-owned and must be drained in
        // reverse publication order while their device placement is known.
        for (layer_idx, layer) in staged_layers.drain(..).enumerate().rev() {
            let device = layout.device_for_layer(layer_idx);
            source.free_layer(&mut devices[device], layer);
        }
        if let Some((output, aliases_embd)) = staged_output.take() {
            let gpu = &mut devices[out_dev];
            if aliases_embd {
                output.free_metadata_only(gpu);
            } else {
                output.free_all(gpu);
            }
        }
        if let Some(output_norm) = staged_output_norm.take() {
            let _ = devices[out_dev].free_tensor(output_norm);
        }
        if let Some((token_embd, _)) = staged_embd.take() {
            let _ = devices[0].free_tensor(token_embd);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_layout_all_on_device_0() {
        let l = Layout::single(5);
        assert_eq!(l.output_device(), 0);
        for i in 0..5 {
            assert_eq!(l.device_for_layer(i), 0);
        }
    }
}
