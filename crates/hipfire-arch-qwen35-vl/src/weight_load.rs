// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 HUSRCF
// hipfire — see LICENSE and NOTICE in the project root.

//! Transactional vision uploads: commit only after every upload succeeds.
//! The fake uploader exercises failure at every allocation without a GPU.

use super::{
    load_f16_gpu, load_f16_gpu_padded, load_f32_cpu, load_f32_gpu, vit_lds_route, VisionConfig,
    VisionLayerWeights, VisionWeights, VIT_LDS_K_ALIGN,
};
use hip_bridge::HipResult;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::{Gpu, GpuTensor};

trait Upload {
    type Tensor;
    fn f16(&mut self, name: &str) -> HipResult<Self::Tensor>;
    fn f16_padded(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        pitch: usize,
    ) -> HipResult<Self::Tensor>;
    fn f32(&mut self, name: &str, len: usize) -> HipResult<Self::Tensor>;
    fn free(&mut self, tensor: Self::Tensor) -> HipResult<()>;
}

struct Pending<'a, U: Upload> {
    uploader: &'a mut U,
    tensors: Vec<U::Tensor>,
}

impl<U: Upload> Pending<'_, U> {
    fn f16(&mut self, name: &str) -> HipResult<()> {
        let tensor = self.uploader.f16(name)?;
        self.tensors.push(tensor);
        Ok(())
    }
    fn f16_padded(&mut self, name: &str, rows: usize, cols: usize, pitch: usize) -> HipResult<()> {
        let tensor = self.uploader.f16_padded(name, rows, cols, pitch)?;
        self.tensors.push(tensor);
        Ok(())
    }
    fn f32(&mut self, name: &str, len: usize) -> HipResult<()> {
        let tensor = self.uploader.f32(name, len)?;
        self.tensors.push(tensor);
        Ok(())
    }
}

impl<U: Upload> Drop for Pending<'_, U> {
    fn drop(&mut self) {
        // Uploads are synchronous; no compute can still read these buffers.
        for tensor in self.tensors.drain(..).rev() {
            if let Err(error) = self.uploader.free(tensor) {
                eprintln!("vision weight rollback failed: {error}");
            }
        }
    }
}

fn stage<U: Upload>(
    config: &VisionConfig,
    fc2_ld: usize,
    uploader: &mut U,
) -> HipResult<Vec<U::Tensor>> {
    let h = config.hidden_size;
    let mut pending = Pending {
        uploader,
        tensors: Vec::with_capacity(8 + config.num_layers * 12),
    };
    pending.f16("model.visual.patch_embed.proj.weight")?;
    pending.f32("model.visual.patch_embed.proj.bias", h)?;
    for i in 0..config.num_layers {
        let p = format!("model.visual.blocks.{i}");
        pending.f32(&format!("{p}.norm1.weight"), h)?;
        pending.f32(&format!("{p}.norm1.bias"), h)?;
        pending.f16(&format!("{p}.attn.qkv.weight"))?;
        pending.f32(&format!("{p}.attn.qkv.bias"), 3 * h)?;
        pending.f16(&format!("{p}.attn.proj.weight"))?;
        pending.f32(&format!("{p}.attn.proj.bias"), h)?;
        pending.f32(&format!("{p}.norm2.weight"), h)?;
        pending.f32(&format!("{p}.norm2.bias"), h)?;
        pending.f16(&format!("{p}.mlp.linear_fc1.weight"))?;
        pending.f32(&format!("{p}.mlp.linear_fc1.bias"), config.mlp_dim)?;
        if fc2_ld != config.mlp_dim {
            pending.f16_padded(
                &format!("{p}.mlp.linear_fc2.weight"),
                h,
                config.mlp_dim,
                fc2_ld,
            )?;
        } else {
            pending.f16(&format!("{p}.mlp.linear_fc2.weight"))?;
        }
        pending.f32(&format!("{p}.mlp.linear_fc2.bias"), h)?;
    }
    let merge = h * config.spatial_merge_size * config.spatial_merge_size;
    pending.f32("model.visual.merger.norm.weight", h)?;
    pending.f32("model.visual.merger.norm.bias", h)?;
    pending.f16("model.visual.merger.linear_fc1.weight")?;
    pending.f32("model.visual.merger.linear_fc1.bias", merge)?;
    pending.f16("model.visual.merger.linear_fc2.weight")?;
    pending.f32(
        "model.visual.merger.linear_fc2.bias",
        config.out_hidden_size,
    )?;
    Ok(std::mem::take(&mut pending.tensors))
}

struct HfqUpload<'a> {
    hfq: &'a HfqFile,
    gpu: &'a mut Gpu,
}
impl Upload for HfqUpload<'_> {
    type Tensor = GpuTensor;
    fn f16(&mut self, name: &str) -> HipResult<GpuTensor> {
        load_f16_gpu(self.hfq, self.gpu, name)
    }
    fn f16_padded(
        &mut self,
        name: &str,
        rows: usize,
        cols: usize,
        pitch: usize,
    ) -> HipResult<GpuTensor> {
        load_f16_gpu_padded(self.hfq, self.gpu, name, rows, cols, pitch)
    }
    fn f32(&mut self, name: &str, len: usize) -> HipResult<GpuTensor> {
        load_f32_gpu(self.hfq, self.gpu, name, len)
    }
    fn free(&mut self, tensor: GpuTensor) -> HipResult<()> {
        self.gpu.free_tensor(tensor)
    }
}

pub(super) fn load(
    hfq: &HfqFile,
    config: &VisionConfig,
    gpu: &mut Gpu,
) -> HipResult<VisionWeights> {
    let pos_embed = load_f32_cpu(
        hfq,
        "model.visual.pos_embed.weight",
        config.num_position_embeddings * config.hidden_size,
    );
    eprintln!(
        "  loading vision weights (GPU, {} blocks)...",
        config.num_layers
    );
    // Match beta's LDS linear route: fc2 rows carry zero-padded K columns.
    let fc2_ld = if vit_lds_route(gpu, config.hidden_size) {
        config.mlp_dim.next_multiple_of(VIT_LDS_K_ALIGN)
    } else {
        config.mlp_dim
    };
    let mut uploaded = stage(config, fc2_ld, &mut HfqUpload { hfq, gpu })?.into_iter();
    // All fallible GPU work has completed. The fixed field order below is the
    // stage order above; consuming it transfers each owner exactly once.
    let mut take = || uploaded.next().expect("staged vision tensor");
    let patch_embed_w = take();
    let patch_embed_b = take();
    let layers = (0..config.num_layers)
        .map(|_| VisionLayerWeights {
            norm1_w: take(),
            norm1_b: take(),
            qkv_w: take(),
            qkv_b: take(),
            proj_w: take(),
            proj_b: take(),
            norm2_w: take(),
            norm2_b: take(),
            fc1_w: take(),
            fc1_b: take(),
            fc2_w: take(),
            fc2_b: take(),
            fc2_ld,
        })
        .collect();
    Ok(VisionWeights {
        patch_embed_w,
        patch_embed_b,
        pos_embed,
        layers,
        merger_norm_w: take(),
        merger_norm_b: take(),
        merger_fc1_w: take(),
        merger_fc1_b: take(),
        merger_fc2_w: take(),
        merger_fc2_b: take(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hip_bridge::HipError;
    struct FakeUpload {
        fail_at: usize,
        created: usize,
        freed: Vec<usize>,
    }
    impl FakeUpload {
        fn next(&mut self) -> HipResult<usize> {
            if self.created == self.fail_at {
                return Err(HipError::new(2, "injected upload failure"));
            }
            let value = self.created;
            self.created += 1;
            Ok(value)
        }
    }
    impl Upload for FakeUpload {
        type Tensor = usize;
        fn f16(&mut self, _: &str) -> HipResult<usize> {
            self.next()
        }
        fn f16_padded(&mut self, _: &str, _: usize, _: usize, _: usize) -> HipResult<usize> {
            self.next()
        }
        fn f32(&mut self, _: &str, _: usize) -> HipResult<usize> {
            self.next()
        }
        fn free(&mut self, tensor: usize) -> HipResult<()> {
            self.freed.push(tensor);
            Ok(())
        }
    }
    fn config() -> VisionConfig {
        VisionConfig {
            hidden_size: 1152,
            num_heads: 16,
            head_dim: 72,
            num_layers: 27,
            mlp_dim: 4304,
            patch_size: 16,
            temporal_patch_size: 2,
            out_hidden_size: 2560,
            spatial_merge_size: 2,
            num_position_embeddings: 2304,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }
    #[test]
    fn rollback_reclaims_every_successful_upload_at_every_failure_position() {
        for fail_at in 0..332 {
            let mut uploader = FakeUpload {
                fail_at,
                created: 0,
                freed: Vec::new(),
            };
            assert!(stage(&config(), config().mlp_dim, &mut uploader).is_err());
            assert_eq!(uploader.created, fail_at);
            assert_eq!(uploader.freed, (0..fail_at).rev().collect::<Vec<_>>());
        }
    }
    #[test]
    fn commit_transfers_all_owners_without_freeing_them() {
        let mut uploader = FakeUpload {
            fail_at: usize::MAX,
            created: 0,
            freed: Vec::new(),
        };
        assert_eq!(
            stage(&config(), config().mlp_dim, &mut uploader).unwrap(),
            (0..332).collect::<Vec<_>>()
        );
        assert!(uploader.freed.is_empty());
    }
}
