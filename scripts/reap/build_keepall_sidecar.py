#!/usr/bin/env python3
# Identity sidecar: keep ALL 256 experts (no pruning), ORIGINAL tid2eid.
# If the keep-map machinery is correct, running with this must reproduce the
# full-256 (no-keep-map) PPL exactly — isolates machinery bugs from REAP cost.
import struct, json, os

HUB = "/home/nick/.cache/huggingface/hub"
OUT = "/data/hipfire-models/reap_keepall_256"
def snap(m):
    base = os.path.join(HUB, m, "snapshots")
    return os.path.join(base, sorted(os.listdir(base))[0])
ORIG = snap("models--deepseek-ai--DeepSeek-V4-Flash")

_hc = {}
def header_for(root, shard):
    k = (root, shard)
    if k not in _hc:
        with open(os.path.join(root, shard), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _hc[k] = (json.loads(f.read(n)), 8 + n)
    return _hc[k]
def read_tensor(root, wm, name):
    shard = wm[name]; hdr, base = header_for(root, shard); meta = hdr[name]
    s, e = meta["data_offsets"]
    with open(os.path.join(root, shard), "rb") as f:
        f.seek(base + s); return meta["dtype"], meta["shape"], f.read(e - s)

wm = json.load(open(os.path.join(ORIG, "model.safetensors.index.json")))["weight_map"]
ORIG_EXP, NLAY, HASH = 256, 43, [0, 1, 2]
os.makedirs(OUT, exist_ok=True)
keep = [list(range(ORIG_EXP)) for _ in range(NLAY)]
json.dump({"kept_per_layer": ORIG_EXP, "num_layers": NLAY, "num_hash_layers": len(HASH),
           "original_experts": ORIG_EXP, "keep": keep},
          open(os.path.join(OUT, "keep_by_layer.json"), "w"))

DT = {"I64": (8, "<q"), "I32": (4, "<i"), "U32": (4, "<I")}
for L in HASH:
    dt, shape, data = read_tensor(ORIG, wm, f"layers.{L}.ffn.gate.tid2eid")
    esz, fmt = DT[dt]
    n = 1
    for s in shape: n *= s
    vals = [struct.unpack_from(fmt, data, i*esz)[0] for i in range(n)]
    assert 0 <= min(vals) and max(vals) < ORIG_EXP, f"L{L} range {min(vals)}..{max(vals)}"
    with open(os.path.join(OUT, f"tid2eid_l{L}.i32"), "wb") as f:
        f.write(b"".join(struct.pack("<i", v) for v in vals))
    print(f"  L{L}: orig tid2eid {dt}{shape} range [{min(vals)},{max(vals)}] -> i32 OK")
print(f"keep-all sidecar written to {OUT}")
