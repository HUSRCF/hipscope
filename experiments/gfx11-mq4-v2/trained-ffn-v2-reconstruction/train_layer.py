#!/usr/bin/env python3
"""Train one reduced-width Qwen3.6 dense FFN from captured residual tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


GROUP_SIZE = 256
BLOCK_SIZE = 128
QWEN35_NORM_BIAS = 1.0


@dataclass(frozen=True)
class Capture:
    residual_in: torch.Tensor
    residual_delta: torch.Tensor
    manifest: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--train-capture", type=Path, action="append", required=True)
    parser.add_argument("--heldout-capture", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--keep-groups", type=int, choices=(39, 41), required=True)
    parser.add_argument(
        "--teacher",
        choices=("captured-mq4", "source-fp8"),
        default="captured-mq4",
    )
    parser.add_argument("--train-tokens", type=int, default=4096)
    parser.add_argument("--heldout-tokens", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-chunk", type=int, default=32)
    parser.add_argument("--source-audit-tokens", type=int, default=64)
    parser.add_argument("--max-source-captured-relative-l2", type=float, default=0.25)
    parser.add_argument("--allow-source-mismatch", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def capture_tensor_path(root: Path, name: object) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in (".", "..")
        or Path(name).name != name
    ):
        raise ValueError(f"{root}: capture tensor filename must be a simple relative name")
    return root / name


def read_capture(root: Path, expected_layer: int, max_tokens: int) -> Capture:
    manifest_path = root / "tensor_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") not in (1, 2) or manifest.get("dtype") != "f16-le":
        raise ValueError(f"{root}: unsupported capture manifest")
    if manifest.get("layer") != expected_layer:
        raise ValueError(f"{root}: layer mismatch")
    hidden = int(manifest["hidden_dim"])
    remaining = max_tokens
    inputs: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    for chunk in manifest["chunks"]:
        chunk_tokens = int(chunk["tokens"])
        if chunk_tokens <= 0:
            raise ValueError(f"{root}: capture chunk has non-positive token count")
        take = min(chunk_tokens, remaining)
        count = int(chunk["tokens"]) * hidden
        in_path = capture_tensor_path(root, chunk["residual_in_file"])
        delta_name = chunk.get("ffn_delta_file")
        out_name = chunk.get("residual_out_file")
        if not delta_name and not out_name:
            raise ValueError(f"{root}: capture chunk has no FFN target")
        target_path = capture_tensor_path(root, delta_name or out_name)
        expected_bytes = count * 2
        if in_path.stat().st_size != expected_bytes or target_path.stat().st_size != expected_bytes:
            raise ValueError(f"{root}: capture tensor byte count mismatch")
        residual_in = np.fromfile(in_path, dtype="<f2", count=count).reshape(-1, hidden)
        target = np.fromfile(target_path, dtype="<f2", count=count).reshape(-1, hidden)
        inputs.append(torch.from_numpy(residual_in[:take].copy()))
        if delta_name:
            outputs.append(torch.from_numpy(target[:take].copy()))
        else:
            outputs.append(torch.from_numpy((target[:take] - residual_in[:take]).copy()))
        remaining -= take
    if not inputs:
        raise ValueError(f"{root}: no capture tokens")
    residual_in = torch.cat(inputs)
    return Capture(residual_in, torch.cat(outputs), manifest)


def combine_captures(paths: list[Path], layer: int, max_tokens: int) -> Capture:
    captures: list[Capture] = []
    remaining = max_tokens
    for path in paths:
        if remaining <= 0:
            break
        capture = read_capture(path, layer, remaining)
        captures.append(capture)
        remaining -= capture.residual_in.shape[0]
    if not captures:
        raise ValueError("empty capture list")
    return Capture(
        torch.cat([capture.residual_in for capture in captures]),
        torch.cat([capture.residual_delta for capture in captures]),
        {"sources": [str(path) for path in paths]},
    )


def tensor_names(layer: int) -> dict[str, str]:
    prefix = f"model.language_model.layers.{layer}"
    return {
        "norm": f"{prefix}.input_layernorm.weight",
        "gate": f"{prefix}.mlp.gate_proj.weight",
        "gate_scale": f"{prefix}.mlp.gate_proj.weight_scale_inv",
        "up": f"{prefix}.mlp.up_proj.weight",
        "up_scale": f"{prefix}.mlp.up_proj.weight_scale_inv",
        "down": f"{prefix}.mlp.down_proj.weight",
        "down_scale": f"{prefix}.mlp.down_proj.weight_scale_inv",
    }


def load_selection(
    path: Path,
    layer: int,
    keep_groups: int,
    capture_roots: list[Path],
) -> tuple[list[int], dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("kind") != "hipfire_dense_ffn_group_selection":
        raise ValueError("unsupported group-selection manifest")
    if int(manifest.get("group_size", 0)) != GROUP_SIZE:
        raise ValueError("group-selection size mismatch")
    if int(manifest.get("keep_groups", 0)) != keep_groups:
        raise ValueError("group-selection keep count mismatch")
    matches = [entry for entry in manifest.get("layers", []) if int(entry["layer"]) == layer]
    if len(matches) != 1:
        raise ValueError(f"selection manifest has {len(matches)} entries for layer {layer}")
    groups = [int(group) for group in matches[0]["groups"]]
    if len(groups) != keep_groups or groups != sorted(set(groups)):
        raise ValueError("selected groups must be sorted, unique, and complete")
    capture_hashes = set()
    for root in capture_roots:
        run_path = root / "run_manifest.json"
        if not run_path.is_file():
            raise ValueError(f"capture has no run manifest: {root}")
        capture_hashes.add(json.loads(run_path.read_text()).get("model_sha256"))
    if capture_hashes != {manifest.get("model_sha256")}:
        raise ValueError("selection and capture model SHA-256 mismatch")
    return groups, manifest


def load_source_layer(source_dir: Path, layer: int) -> tuple[dict[str, torch.Tensor], str]:
    shard = source_dir / f"layers-{layer}.safetensors"
    if not shard.is_file():
        raise ValueError(f"missing source shard: {shard}")
    names = tensor_names(layer)
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for key, name in names.items():
            if name not in keys:
                raise ValueError(f"source tensor missing: {name}")
            tensors[key] = source.get_tensor(name)
    return tensors, sha256_file(shard)


def dequant_fp8_blockwise(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must be matrices")
    rows, cols = weight.shape
    expected = (math.ceil(rows / BLOCK_SIZE), math.ceil(cols / BLOCK_SIZE))
    if tuple(scale.shape) != expected:
        raise ValueError(f"FP8 scale shape {tuple(scale.shape)} != {expected}")
    row_scale = scale.to(torch.float32).repeat_interleave(BLOCK_SIZE, 0)[:rows]
    expanded = row_scale.repeat_interleave(BLOCK_SIZE, 1)[:, :cols]
    return (weight.to(torch.float32) * expanded).to(torch.bfloat16)


def rank_groups(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> list[int]:
    if gate.shape != up.shape or down.shape != (gate.shape[1], gate.shape[0]):
        raise ValueError("incompatible FFN matrices")
    if gate.shape[0] % GROUP_SIZE:
        raise ValueError("FFN width must be 256-aligned")
    groups = gate.shape[0] // GROUP_SIZE
    gate_e = gate.float().square().reshape(groups, GROUP_SIZE, -1).sum((1, 2))
    up_e = up.float().square().reshape(groups, GROUP_SIZE, -1).sum((1, 2))
    down_e = down.float().square().reshape(down.shape[0], groups, GROUP_SIZE).sum((0, 2))
    tiny = torch.finfo(torch.float32).tiny
    scores = (gate_e.clamp_min(tiny).log() + up_e.clamp_min(tiny).log() + down_e.clamp_min(tiny).log()) / 3
    return sorted(range(groups), key=lambda group: (-float(scores[group]), group))


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
    return (normalized * (weight.float() + QWEN35_NORM_BIAS)).to(torch.bfloat16)


def ffn(x_norm: torch.Tensor, gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    with torch.autocast(
        device_type=gate.device.type,
        dtype=torch.bfloat16,
        enabled=gate.device.type == "cuda",
    ):
        gate_out = F.linear(x_norm, gate)
        up_out = F.linear(x_norm, up)
        return F.linear(F.silu(gate_out.float()) * up_out.float(), down).to(torch.float32)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = (prediction.float() - target.float()).square().sum()
    denominator = target.float().square().sum().clamp_min(torch.finfo(torch.float32).tiny)
    return float(torch.sqrt(numerator / denominator))


def enforce_source_teacher_contract(
    relative_error: float | None,
    limit: float,
    allow_mismatch: bool,
) -> None:
    if relative_error is None:
        raise ValueError("source-fp8 teacher requires a non-empty source audit")
    if relative_error > limit and not allow_mismatch:
        raise ValueError(
            "source-fp8 teacher disagrees with the captured production contract: "
            f"relative_l2={relative_error:.6f} exceeds {limit:.6f}; pass "
            "--allow-source-mismatch only for an explicit diagnostic run"
        )


def validate_capture_splits(train: list[Path], heldout: list[Path]) -> None:
    resolved_train = [path.resolve() for path in train]
    resolved_heldout = [path.resolve() for path in heldout]
    train_roots = set(resolved_train)
    heldout_roots = set(resolved_heldout)
    if len(train_roots) != len(resolved_train):
        raise ValueError("duplicate train capture path")
    if len(heldout_roots) != len(resolved_heldout):
        raise ValueError("duplicate held-out capture path")
    if train_roots & heldout_roots:
        raise ValueError("train and held-out captures must be disjoint")


def materialize_source_teacher(
    residual_in: torch.Tensor,
    norm: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    eps: float,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        norm_d = norm.to(device)
        gate_d = gate.to(device)
        up_d = up.to(device)
        down_d = down.to(device)
        for chunk in residual_in.split(chunk_size):
            x_norm = rms_norm(chunk.to(device), norm_d, eps)
            outputs.append(ffn(x_norm, gate_d, up_d, down_d).cpu())
    return torch.cat(outputs)


def evaluate(
    residual_in: torch.Tensor,
    target: torch.Tensor,
    norm: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    eps: float,
    device: torch.device,
    batch_size: int,
) -> float:
    predictions = []
    with torch.no_grad():
        for chunk in residual_in.split(batch_size):
            predictions.append(
                ffn(
                    rms_norm(chunk.to(device), norm, eps),
                    gate,
                    up,
                    down,
                ).cpu()
            )
    return relative_l2(torch.cat(predictions), target)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    if args.train_tokens <= 0 or args.heldout_tokens <= 0:
        raise ValueError("token limits must be positive")
    validate_capture_splits(args.train_capture, args.heldout_capture)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    config = json.loads((args.source_dir / "config.json").read_text())["text_config"]
    hidden = int(config["hidden_size"])
    intermediate = int(config["intermediate_size"])
    eps = float(config["rms_norm_eps"])
    if hidden != 5120 or intermediate != 17408:
        raise ValueError("this probe currently supports Qwen3.6-27B dense FFNs only")

    train = combine_captures(args.train_capture, args.layer, args.train_tokens)
    heldout = combine_captures(args.heldout_capture, args.layer, args.heldout_tokens)
    if train.residual_in.shape[1] != hidden or heldout.residual_in.shape[1] != hidden:
        raise ValueError("capture hidden dimension mismatch")
    selected_groups, selection_manifest = load_selection(
        args.selection_manifest,
        args.layer,
        args.keep_groups,
        args.train_capture + args.heldout_capture,
    )
    source, source_shard_sha256 = load_source_layer(args.source_dir, args.layer)
    norm_cpu = source["norm"].to(torch.bfloat16)
    gate_full = dequant_fp8_blockwise(source["gate"], source["gate_scale"])
    up_full = dequant_fp8_blockwise(source["up"], source["up_scale"])
    down_full = dequant_fp8_blockwise(source["down"], source["down_scale"])
    source_ranking = rank_groups(gate_full, up_full, down_full)
    source_selected = set(source_ranking[: args.keep_groups])
    selection_overlap = len(source_selected.intersection(selected_groups))
    selected_rows = torch.cat(
        [torch.arange(group * GROUP_SIZE, (group + 1) * GROUP_SIZE) for group in selected_groups]
    )

    source_audit_tokens = min(args.source_audit_tokens, heldout.residual_in.shape[0])
    source_vs_capture_relative_l2 = None
    source_vs_capture_rms_ratio = None
    source_audit_target = None
    if source_audit_tokens > 0:
        source_audit_target = materialize_source_teacher(
            heldout.residual_in[:source_audit_tokens],
            norm_cpu,
            gate_full,
            up_full,
            down_full,
            eps,
            device,
            args.target_chunk,
        )
        captured_audit_target = heldout.residual_delta[:source_audit_tokens].float()
        source_vs_capture_relative_l2 = relative_l2(source_audit_target, captured_audit_target)
        source_rms = source_audit_target.square().mean().sqrt()
        capture_rms = captured_audit_target.square().mean().sqrt().clamp_min(1e-12)
        source_vs_capture_rms_ratio = float(source_rms / capture_rms)

    if args.teacher == "source-fp8":
        enforce_source_teacher_contract(
            source_vs_capture_relative_l2,
            args.max_source_captured_relative_l2,
            args.allow_source_mismatch,
        )
        train_target = materialize_source_teacher(
            train.residual_in, norm_cpu, gate_full, up_full, down_full, eps, device, args.target_chunk
        )
        if source_audit_tokens == heldout.residual_in.shape[0]:
            heldout_target = source_audit_target
        else:
            heldout_target = materialize_source_teacher(
                heldout.residual_in,
                norm_cpu,
                gate_full,
                up_full,
                down_full,
                eps,
                device,
                args.target_chunk,
            )
    else:
        train_target = train.residual_delta.float()
        heldout_target = heldout.residual_delta.float()

    # FP32 master parameters keep Adam moments and updates stable. Autocast in
    # ffn() still executes the large linear operations in BF16 on ROCm.
    gate = torch.nn.Parameter(gate_full.index_select(0, selected_rows).float().to(device))
    up = torch.nn.Parameter(up_full.index_select(0, selected_rows).float().to(device))
    down = torch.nn.Parameter(down_full.index_select(1, selected_rows).float().to(device))
    norm = norm_cpu.to(device)
    del source, gate_full, up_full, down_full
    if device.type == "cuda":
        torch.cuda.empty_cache()

    initial_error = evaluate(
        heldout.residual_in, heldout_target, norm, gate, up, down, eps, device, args.batch_size
    )
    optimizer = torch.optim.AdamW(
        [gate, up, down],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator().manual_seed(args.seed)
    token_count = train.residual_in.shape[0]
    losses: list[float] = []
    for step in range(args.steps):
        indices = torch.randint(token_count, (args.batch_size,), generator=generator)
        x = train.residual_in.index_select(0, indices).to(device)
        target = train_target.index_select(0, indices).to(device)
        prediction = ffn(rms_norm(x, norm, eps), gate, up, down)
        loss = F.mse_loss(prediction, target.float()) / target.float().square().mean().clamp_min(1e-12)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            print(f"step={step + 1}/{args.steps} normalized_mse={losses[-1]:.8f}", flush=True)

    final_error = evaluate(
        heldout.residual_in, heldout_target, norm, gate, up, down, eps, device, args.batch_size
    )
    improvement = 1.0 - final_error / initial_error
    args.output_dir.mkdir(parents=True)
    save_file(
        {
            "gate_proj.weight": gate.detach().cpu().to(torch.bfloat16).contiguous(),
            "up_proj.weight": up.detach().cpu().to(torch.bfloat16).contiguous(),
            "down_proj.weight": down.detach().cpu().to(torch.bfloat16).contiguous(),
        },
        args.output_dir / "student.safetensors",
    )
    metrics = {
        "version": 1,
        "kind": "gfx11_trained_ffn_v2_layer",
        "layer": args.layer,
        "teacher": args.teacher,
        "source_dir": str(args.source_dir),
        "source_shard_sha256": source_shard_sha256,
        "rms_norm_weight_bias": QWEN35_NORM_BIAS,
        "group_size": GROUP_SIZE,
        "keep_groups": args.keep_groups,
        "intermediate_size": args.keep_groups * GROUP_SIZE,
        "selection_manifest": str(args.selection_manifest),
        "selection_model_sha256": selection_manifest["model_sha256"],
        "ranking": selection_manifest["ranking"],
        "selected_groups": selected_groups,
        "source_fp8_ranking_overlap": selection_overlap,
        "train_capture": [str(path) for path in args.train_capture],
        "heldout_capture": [str(path) for path in args.heldout_capture],
        "train_tokens": int(train.residual_in.shape[0]),
        "heldout_tokens": int(heldout.residual_in.shape[0]),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "initial_heldout_relative_l2": initial_error,
        "final_heldout_relative_l2": final_error,
        "relative_l2_improvement": improvement,
        "last_normalized_mse": losses[-1] if losses else None,
        "source_audit_tokens": source_audit_tokens,
        "max_source_captured_relative_l2": args.max_source_captured_relative_l2,
        "source_mismatch_override": args.allow_source_mismatch,
        "source_vs_captured_relative_l2": source_vs_capture_relative_l2,
        "source_vs_captured_rms_ratio": source_vs_capture_rms_ratio,
        "peak_device_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
