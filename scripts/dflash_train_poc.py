#!/usr/bin/env python3
"""
dflash_train_poc.py — minimal DFlash draft training loop.

Targets AMD MI300X (ROCm 7.2, PyTorch 2.11+rocm7.2). Dependencies installed
by scripts/amd_quickdeploy.sh.

What it does:
  1. Loads a Qwen3.5-family target from HF (frozen, bf16).
  2. Constructs a fresh DFlashDraftModel (num_layers configurable) using the
     reference architecture at .dflash-reference/dflash/model.py.
  3. Streams a plain-text corpus through the target tokenizer to produce
     fixed-length training sequences.
  4. For each training step, samples a mask-block start position, builds
     the noise_embedding (first position = real token, rest = mask), runs
     target forward with grad disabled to extract per-layer hidden states,
     runs the draft with grad enabled, computes cross-entropy loss vs the
     ground-truth tokens at positions 1..B-1, backprops.
  5. Saves checkpoint after each `--ckpt-every` steps. Final checkpoint is
     in a .safetensors format loadable by dflash_convert.rs.

Hardcoded for CLARITY, not flexibility. Scale knobs:
  --target-repo    HF repo of the target (e.g. Qwen/Qwen3.5-4B).
  --draft-layers   Number of decoder layers in the draft (paper uses 5).
  --block-size     B (paper uses 16).
  --seq-len        Training sequence length.
  --batch-size     Examples per gradient step.
  --lr             Peak LR for AdamW + cosine.
  --steps          Total training steps.
  --corpus         Plain-text corpus file (one doc per blank line).
  --out            Directory for checkpoints.
  --ckpt-every     Save every N steps.
  --log-every      Log loss every N steps.

This is a POC. It trains on `wikitext_calib.txt` by default — good for
validating the pipeline end-to-end, not for shipping a production draft.
Expect loss to drop from ~12 (random init cross-entropy ≈ log(vocab_size))
to ~2-3 within ~10K steps on MI300X at B=16, seq=1024, batch=4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.optim.lr_scheduler import LambdaLR

# Pull the reference model.py off .dflash-reference/ without having to
# `pip install -e` it (avoids transformers-version conflicts).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / ".dflash-reference"))
from dflash.model import DFlashDraftModel, build_target_layer_ids, extract_context_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--target-repo", default="Qwen/Qwen3.5-4B")
    p.add_argument("--draft-layers", type=int, default=5)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--corpus", default="/root/wikitext_calib.txt")
    p.add_argument("--out", default="/root/dflash_train_poc_out")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", default=None, help="Path to checkpoint safetensors to resume from.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_corpus_tokens(corpus_path: str, tokenizer) -> list[int]:
    """Tokenize the whole corpus into a flat list of IDs, respecting BOS at doc boundaries."""
    print(f"[data] tokenizing {corpus_path}...", flush=True)
    text = Path(corpus_path).read_text()
    # Slice into docs on blank lines to avoid one giant sequence.
    docs = [d.strip() for d in text.split("\n\n") if d.strip()]
    print(f"[data]   {len(docs):,} docs", flush=True)
    bos = tokenizer.bos_token_id
    ids: list[int] = []
    for d in docs:
        if bos is not None:
            ids.append(bos)
        ids.extend(tokenizer.encode(d, add_special_tokens=False))
    print(f"[data]   {len(ids):,} tokens", flush=True)
    return ids


def sample_batch(
    ids: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Random contiguous slices. Simple and fast; no packing tricks."""
    out = torch.empty(batch_size, seq_len, dtype=torch.long, device=device)
    max_start = len(ids) - seq_len - 1
    for b in range(batch_size):
        start = random.randint(0, max_start)
        out[b] = torch.tensor(ids[start : start + seq_len], dtype=torch.long, device=device)
    return out


def build_draft_config(target_config, draft_layers: int, block_size: int, mask_token_id: int):
    """Clone a DFlash draft config from the target config — same hidden/heads, fewer layers."""
    import copy

    cfg = copy.deepcopy(target_config)
    cfg.num_hidden_layers = draft_layers
    # Required by the DFlashDraftModel __init__.
    cfg.num_target_layers = target_config.num_hidden_layers
    cfg.block_size = block_size
    cfg.dflash_config = {
        "mask_token_id": mask_token_id,
        "target_layer_ids": build_target_layer_ids(target_config.num_hidden_layers, draft_layers),
    }
    return cfg


def cosine_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load target (frozen, bf16) ────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[target] loading {args.target_repo} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_repo)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_repo,
        torch_dtype=dtype,
        attn_implementation="eager",   # safer on ROCm; swap to sdpa once verified
    ).to(device)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    print(f"[target]   {target.config.num_hidden_layers} layers, "
          f"hidden={target.config.hidden_size}, vocab={target.config.vocab_size}", flush=True)

    # Pick a mask token id that's unlikely to appear in data; reference uses 248070.
    mask_token_id = min(248070, target.config.vocab_size - 1)

    # ── build draft ───────────────────────────────────────────────────
    draft_cfg = build_draft_config(
        target.config, args.draft_layers, args.block_size, mask_token_id
    )
    draft = DFlashDraftModel(draft_cfg).to(device=device, dtype=dtype)
    if args.resume:
        from safetensors.torch import load_file
        sd = load_file(args.resume)
        missing, unexpected = draft.load_state_dict(sd, strict=False)
        print(f"[draft] resumed from {args.resume}; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    n_draft_params = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    print(f"[draft]   {args.draft_layers} layers, {n_draft_params / 1e6:.1f}M params, block={args.block_size}", flush=True)

    # ── data ──────────────────────────────────────────────────────────
    ids = read_corpus_tokens(args.corpus, tokenizer)
    if len(ids) < args.seq_len + args.block_size + 16:
        print(f"[data] ERROR: corpus has only {len(ids)} tokens, need >{args.seq_len + args.block_size}", flush=True)
        return 2

    # ── optimizer ─────────────────────────────────────────────────────
    optim = torch.optim.AdamW(
        (p for p in draft.parameters() if p.requires_grad),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    )
    sched = LambdaLR(optim, lambda s: cosine_schedule(s, args.warmup, args.steps))

    # ── train ─────────────────────────────────────────────────────────
    print(f"[train] {args.steps} steps, batch={args.batch_size}, seq={args.seq_len}, lr={args.lr}", flush=True)
    loss_ema: Optional[float] = None
    t_start = time.time()
    draft.train()
    for step in range(args.steps + 1):
        optim.zero_grad(set_to_none=True)

        batch = sample_batch(ids, args.seq_len, args.batch_size, device)  # [B, L]
        # Pick a block start per example (must have >=1 context token AND B+1 trailing).
        starts = [
            random.randint(1, args.seq_len - args.block_size - 1)
            for _ in range(args.batch_size)
        ]

        # Target forward with hidden states. Run on the prefix up through the block.
        # For simplicity, process each example separately — cheap relative to backward.
        draft_logits_list = []
        labels_list = []
        for b in range(args.batch_size):
            s = starts[b]
            prefix = batch[b : b + 1, : s + args.block_size]  # [1, s+B]
            block_ids = prefix[:, s : s + args.block_size]

            with torch.no_grad():
                t_out = target(
                    input_ids=prefix,
                    output_hidden_states=True,
                    use_cache=False,
                )
                layer_ids = draft_cfg.dflash_config["target_layer_ids"]
                # hidden_states is tuple of (num_layers + 1) tensors; extract_context_feature
                # uses offset=1 to skip the embedding layer output.
                target_hidden = extract_context_feature(t_out.hidden_states, layer_ids)
                # target_hidden shape: [1, s+B, hidden*len(layer_ids)]
                target_hidden = target_hidden[:, :s, :]  # only the *context*, not the block

            # Noise embedding: position 0 of the block is the real token;
            # positions 1..B-1 are the mask token (to be predicted).
            masked_block = block_ids.clone()
            masked_block[:, 1:] = mask_token_id
            noise_embedding = target.model.embed_tokens(masked_block).to(dtype)

            position_ids = torch.arange(s, s + args.block_size, device=device).unsqueeze(0)

            draft_hidden = draft(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                position_ids=position_ids,
                use_cache=False,
            )
            # [1, B, hidden] → [1, B-1, vocab]
            logits = target.lm_head(draft_hidden[:, -args.block_size + 1 :, :])
            # Labels: the *actual* tokens at positions 1..B-1.
            labels = block_ids[:, 1:]  # [1, B-1]
            draft_logits_list.append(logits)
            labels_list.append(labels)

        draft_logits = torch.cat(draft_logits_list, dim=0)  # [B, B-1, vocab]
        labels = torch.cat(labels_list, dim=0)              # [B, B-1]
        loss = F.cross_entropy(
            draft_logits.reshape(-1, draft_logits.size(-1)).float(),
            labels.reshape(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optim.step()
        sched.step()

        lv = float(loss.item())
        loss_ema = lv if loss_ema is None else 0.99 * loss_ema + 0.01 * lv

        if step % args.log_every == 0:
            elapsed = time.time() - t_start
            rate = (step + 1) / max(1e-6, elapsed)
            print(
                f"[step {step:6d}] loss={lv:.4f} ema={loss_ema:.4f} "
                f"lr={sched.get_last_lr()[0]:.2e} rate={rate:.2f} step/s",
                flush=True,
            )

        if step > 0 and step % args.ckpt_every == 0:
            ckpt_path = out_dir / f"draft_step{step}.safetensors"
            save_file(draft.state_dict(), str(ckpt_path))
            meta = {
                "step": step,
                "loss_ema": loss_ema,
                "target_repo": args.target_repo,
                "draft_layers": args.draft_layers,
                "block_size": args.block_size,
                "mask_token_id": mask_token_id,
                "target_layer_ids": draft_cfg.dflash_config["target_layer_ids"],
            }
            (out_dir / f"draft_step{step}.json").write_text(json.dumps(meta, indent=2))
            print(f"[ckpt]   wrote {ckpt_path}", flush=True)

    final = out_dir / "draft_final.safetensors"
    save_file(draft.state_dict(), str(final))
    (out_dir / "draft_final.json").write_text(json.dumps({
        "steps": args.steps,
        "loss_ema": loss_ema,
        "target_repo": args.target_repo,
        "draft_layers": args.draft_layers,
        "block_size": args.block_size,
        "mask_token_id": mask_token_id,
    }, indent=2))
    print(f"[done] final ckpt at {final}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
