#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""hw-gate bucket selection: changed paths -> buckets, policy hits, needs_hw.

Pure and deterministic. No git, no network, no GPU. Reads changed paths from
stdin (one per line, as `git diff --name-only BASE...HEAD` prints them) and
classifies them by prefix/glob. The tables below are the whole policy; there
is no rule engine and no per-route table.

CONTRACT
    stdin : changed paths, one per line
    stdout: JSON (also written to --json PATH when given)
        {
          "schema": "hipfire.hw-gate.select", "version": 1,
          "needs_hw": bool,                 # any bucket other than none
          "buckets": ["load", "serve", "kernel"],   # sorted, deduplicated, may be []
          "policy_paths": ["..."],          # touched paths matching POLICY (never bot-approvable)
          "surfaces": {"load": ["path", ...], "serve": [...], "kernel": [...], "policy": [...], "other": [...]}
        }
    --github-output FILE : append `needs_hw=`, `buckets=` (comma-joined), `policy=` (comma-joined) lines
    exit 0 always on well-formed input; exit 2 on usage error

BUCKET RULES (first match wins per path; a path may hit `policy` in addition)
    kernel : kernels/**, crates/rdna-compute/**, crates/hipfire-dispatch/**, crates/hip-bridge/**,
             crates/saddle-core/**
    serve  : crates/hipfire-engine/**, crates/hipfire-generate/**, crates/hipfire-daemon/src/slots.rs,
             crates/hipfire-daemon/src/serve*.rs, crates/hipfire-runtime/src/{emit_text,eos_filter,dflash,
             dflash_generic,dspark_core,spec,reset_core,triattn}.rs, crates/hipfire-arch-*/src/**/{serve,generate,spec}*.rs
    load   : crates/hipfire-loader/**, crates/hipfire-daemon/** (remaining), crates/hipfire-runtime/src/{model_load,
             hfq,loader_api,config,safetensors_source,weight_backend,multi_gpu,arch_model,arch}.rs,
             crates/hipfire-arch-*/src/**/load*.rs, crates/hipfire-arch-*/src/**/weights*.rs,
             crates/hipfire-arch-*/src/carrier.rs, crates/hipfire-config/**, crates/hipfire-registry/**,
             registry/**, Cargo.toml, Cargo.lock, crates/*/Cargo.toml
    none   : everything else (docs/**, benchmarks/**, scripts/** except hw-gate, tests/**, *.md, ...)

    `serve` and `kernel` imply `load` (the fixtures must still load through the user route).

POLICY (touching any of these => policy_paths non-empty => review.py may never greenlight)
    .github/workflows/**, .github/CODEOWNERS, scripts/hw-gate/**, scripts/leanup-thresholds.txt,
    scripts/layering.txt, scripts/ratchet-diff.sh, scripts/leanup-ratchets.sh, registry/**
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import posixpath
import sys

# -- pattern tables -----------------------------------------------------------

_KERNEL_PATTERNS: list[str] = [
    "kernels/**",
    "crates/rdna-compute/**",
    "crates/hipfire-dispatch/**",
    "crates/hip-bridge/**",
    "crates/saddle-core/**",
]

_SERVE_PATTERNS: list[str] = [
    "crates/hipfire-engine/**",
    "crates/hipfire-generate/**",
    "crates/hipfire-daemon/src/slots.rs",
    "crates/hipfire-daemon/src/serve*.rs",
    # crates/hipfire-runtime/src/{emit_text,...}.rs expanded
    "crates/hipfire-runtime/src/emit_text.rs",
    "crates/hipfire-runtime/src/eos_filter.rs",
    "crates/hipfire-runtime/src/dflash.rs",
    "crates/hipfire-runtime/src/dflash_generic.rs",
    "crates/hipfire-runtime/src/dspark_core.rs",
    "crates/hipfire-runtime/src/spec.rs",
    "crates/hipfire-runtime/src/reset_core.rs",
    "crates/hipfire-runtime/src/triattn.rs",
    # crates/hipfire-arch-*/src/**/{serve,generate,spec}*.rs expanded
    "crates/hipfire-arch-*/src/**/serve*.rs",
    "crates/hipfire-arch-*/src/**/generate*.rs",
    "crates/hipfire-arch-*/src/**/spec*.rs",
]

_LOAD_PATTERNS: list[str] = [
    "crates/hipfire-loader/**",
    "crates/hipfire-daemon/**",
    # crates/hipfire-runtime/src/{model_load,...}.rs expanded
    "crates/hipfire-runtime/src/model_load.rs",
    "crates/hipfire-runtime/src/hfq.rs",
    "crates/hipfire-runtime/src/loader_api.rs",
    "crates/hipfire-runtime/src/config.rs",
    "crates/hipfire-runtime/src/safetensors_source.rs",
    "crates/hipfire-runtime/src/weight_backend.rs",
    "crates/hipfire-runtime/src/multi_gpu.rs",
    "crates/hipfire-runtime/src/arch_model.rs",
    "crates/hipfire-runtime/src/arch.rs",
    "crates/hipfire-arch-*/src/**/load*.rs",
    "crates/hipfire-arch-*/src/**/weights*.rs",
    "crates/hipfire-arch-*/src/carrier.rs",
    "crates/hipfire-config/**",
    "crates/hipfire-registry/**",
    "registry/**",
    "Cargo.toml",
    "Cargo.lock",
    "crates/*/Cargo.toml",
]

_POLICY_PATTERNS: list[str] = [
    ".github/workflows/**",
    ".github/CODEOWNERS",
    "scripts/hw-gate/**",
    "scripts/leanup-thresholds.txt",
    "scripts/layering.txt",
    "scripts/ratchet-diff.sh",
    "scripts/leanup-ratchets.sh",
    "registry/**",
]


def _normalize(p: str) -> str:
    p = p.strip()
    if not p:
        return ""
    p = p.replace("\\", "/")
    # collapse repeated slashes via normpath
    p = posixpath.normpath(p)
    if p == ".":
        return ""
    return p


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pat) for pat in patterns)


def classify(paths: list[str]) -> dict:
    """Return the CONTRACT JSON object for `paths`. Implemented in this file."""
    buckets: set[str] = set()
    policy_paths: list[str] = []
    seen_policy: set[str] = set()
    surfaces: dict[str, list[str]] = {
        "load": [],
        "serve": [],
        "kernel": [],
        "policy": [],
        "other": [],
    }
    # track seen per surface to avoid duplicates while preserving order
    seen_surfaces: dict[str, set[str]] = {k: set() for k in surfaces}

    for raw in paths:
        p = _normalize(raw)
        if not p:
            continue

        # policy is additive, checked independently
        is_policy = _matches(p, _POLICY_PATTERNS)
        if is_policy and p not in seen_policy:
            policy_paths.append(p)
            seen_policy.add(p)
        if is_policy and p not in seen_surfaces["policy"]:
            surfaces["policy"].append(p)
            seen_surfaces["policy"].add(p)

        # bucket: first match wins
        bucket: str | None = None
        if _matches(p, _KERNEL_PATTERNS):
            bucket = "kernel"
        elif _matches(p, _SERVE_PATTERNS):
            bucket = "serve"
        elif _matches(p, _LOAD_PATTERNS):
            bucket = "load"
        else:
            bucket = "other"

        if bucket == "kernel":
            if "kernel" not in buckets:
                buckets.add("kernel")
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["kernel"]:
                surfaces["kernel"].append(p)
                seen_surfaces["kernel"].add(p)
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        elif bucket == "serve":
            if "serve" not in buckets:
                buckets.add("serve")
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["serve"]:
                surfaces["serve"].append(p)
                seen_surfaces["serve"].add(p)
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        elif bucket == "load":
            if "load" not in buckets:
                buckets.add("load")
            if p not in seen_surfaces["load"]:
                surfaces["load"].append(p)
                seen_surfaces["load"].add(p)
        else:  # other
            # only record in other if not already accounted as policy-only ? policy additive
            # but policy paths that are other should be in other? Spec says policy only -> not in other
            # To satisfy "scripts/leanup-thresholds.txt -> policy only", we put policy-only
            # paths only in policy, not other.
            if not is_policy:
                if p not in seen_surfaces["other"]:
                    surfaces["other"].append(p)
                    seen_surfaces["other"].add(p)
            # if is_policy and bucket == other, we have already recorded in policy surfaces,
            # and we do NOT record in other (policy only semantics).

    sorted_buckets = sorted(buckets)
    needs_hw = bool(sorted_buckets)

    return {
        "schema": "hipfire.hw-gate.select",
        "version": 1,
        "needs_hw": needs_hw,
        "buckets": sorted_buckets,
        "policy_paths": policy_paths,
        "surfaces": surfaces,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--json", help="also write the result here")
    ap.add_argument("--github-output", help="append needs_hw=/buckets=/policy= lines here")
    args = ap.parse_args(argv)
    paths = [line.strip() for line in sys.stdin if line.strip()]
    result = classify(paths)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"needs_hw={'true' if result['needs_hw'] else 'false'}\n")
            fh.write(f"buckets={','.join(result['buckets'])}\n")
            fh.write(f"policy={','.join(result['policy_paths'])}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
