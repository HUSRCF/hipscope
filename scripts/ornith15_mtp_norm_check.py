#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""Gate G2 — is the Ornith 1.5 MTP module's norm set actually trained?

ORNITH 1.0's DSpark drafter shipped trained matmuls beside completely
untrained RMSNorms (every weight exactly 1.0, std 0.0). That produced
tau=0.00 and cost days of engine debugging before the checkpoint itself was
identified as the defect. hipfire's math was correct to cosine 0.999873.

A trained reference measured norm mean 1.52 / std 0.14.

Exit 1 here means: drop MTP from scope and report upstream. Do not debug.
"""
import json
import sys
from pathlib import Path

import ml_dtypes  # noqa: F401 — registers bfloat16 with numpy's dtype system
import numpy as np
from safetensors import safe_open

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/nick/hf/Ornith-1.5-35B-A3B")

wm = json.loads((SRC / "model.safetensors.index.json").read_text())["weight_map"]

norm_keys = sorted(k for k in wm if k.startswith("mtp.") and "norm" in k
                   and k.endswith(".weight"))
matmul_keys = sorted(k for k in wm if k.startswith("mtp.")
                     and k.endswith(("fc.weight", "q_proj.weight", "o_proj.weight")))

if not norm_keys:
    print("FAIL: no mtp.* norm tensors found; the module layout is not what we assume")
    sys.exit(1)

handles = {}
def load(key):
    shard = wm[key]
    if shard not in handles:
        handles[shard] = safe_open(str(SRC / shard), framework="np")
    return handles[shard].get_tensor(key).astype(np.float32)

def summarize(keys, label):
    rows = []
    for k in keys:
        t = load(k)
        rows.append((k, float(t.mean()), float(t.std())))
    print(f"\n{label} ({len(rows)} tensors):")
    for k, m, s in rows[:6]:
        print(f"  {k:<62} mean={m:8.4f} std={s:8.4f}")
    if len(rows) > 6:
        print(f"  ... {len(rows) - 6} more")
    return rows

norms = summarize(norm_keys, "MTP learnable RMSNorm weights")
mms = summarize(matmul_keys, "MTP matmul weights (trained-ness control)")

# All-ones norm detection. Exactly 1.0 with zero variance is the signature.
degenerate = [(k, m, s) for k, m, s in norms if s == 0.0 and abs(m - 1.0) < 1e-6]
frac = len(degenerate) / len(norms)

print(f"\nDegenerate (exactly 1.0, std 0) norms: {len(degenerate)}/{len(norms)} "
      f"({frac:.1%})")

# Control: if the matmuls are ALSO degenerate, the export is broken wholesale
# rather than norm-specific, which is a different report.
mm_alive = [s for _, _, s in mms if s > 1e-4]
print(f"Matmuls with non-trivial std: {len(mm_alive)}/{len(mms)}")

if frac > 0.5:
    print("\nG2 FAIL — the MTP norms are untrained (ORNITH 1.0 signature).")
    if len(mm_alive) == len(mms) and mms:
        print("Matmuls ARE trained, so this is the exact 1.0 defect, not an empty export.")
    print("ACTION: drop the MTP sidecar from scope. Report upstream. Do NOT debug engine code.")
    sys.exit(1)

print("\nG2 PASS — MTP norms carry trained variance. Task 8 may proceed.")
