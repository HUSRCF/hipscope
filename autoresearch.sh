#!/usr/bin/env bash
# Autoregressive decode throughput for Qwen3.8-Flash-Next (qwen4, mq6q8-pleq8) on
# gfx1151, through the native daemon protocol (the product path), gated on the
# decode-route KLD against the BF16-source teacher.
#
# Perf: fresh daemon, load (max_seq 2048, kv q8, mtp off, graph off), 1 warmup +
# RUNS measured greedy generates of a committed 1131-token prompt, max_tokens 128.
# Metric = median decode tok/s from token-event arrival times (the daemon's own
# decode_tok_s is rounded to 0.1, too coarse for sub-1% deltas).
# Guards (non-zero exit on violation):
#   - greedy token IDs identical across runs, leading MIN_MATCH match REF_IDS;
#   - decode KLD (qwen4_kld eval --decode, KLD_CHUNKS x 255 scored tokens of
#     wikitext-2, each a single-token forward) must not exceed KLD_MAX.
set -euo pipefail
cd "$(dirname "$0")"

flock -w 3600 /tmp/hipfire-build.lock cargo build --release -q -p hipfire-daemon -p hipfire-arch-qwen4 \
    --features hipfire-arch-qwen4/lab --example qwen4_kld >/tmp/autoresearch-build.log 2>&1 \
    || { tail -40 /tmp/autoresearch-build.log; exit 1; }

exec flock -w 3600 /tmp/hipfire-gpu.lock python3 - <<'PY'
import hashlib, json, os, re, select, statistics, subprocess, sys, time

MODEL = os.path.expanduser("~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq")
KLD_REF = "/home/bjoern/hipfire-qwen4-kld/.codeinsight+research/qwen4-kld/source-teacher/bf16src-wt2-c512x32.kldref"
KLD_CHUNKS = 8
# Baseline decode-route KLD; a candidate above it fails. Lower it when a keep lowers KLD.
KLD_MAX = 0.072049  # baseline bbc6eae5e, logits sha256 988a4db1...
PROMPT_PATH = "benchmarks/prompts/glimmer_prefill_1024.txt"
PROMPT_MD5 = "0ee8f86ada3683eda452bc294ec824a9"
RUNS = int(os.environ.get("AR_RUNS", "3"))
MAX_TOKENS = 128
# Greedy ids on the baseline build (text: "The text you provided contains a repeated block of prose ...").
REF_IDS = [760, 1414, 488, 3766, 5435, 264, 11173, 2424, 314, 58655, 7854, 539, 264, 12654, 709, 7044, 13, 561, 58655, 7701, 310, 381, 264, 328, 7320, 261, 1, 466, 328, 50465, 1, 1414, 318, 51343, 7262, 364, 9640, 24277, 466, 1066, 4055, 51623, 681, 1345, 279, 6007, 2144, 369, 279, 1510, 18479, 39914, 63, 709, 13, 271, 8160, 369, 279, 26824, 5072, 11, 9971, 22405, 2243, 314, 279, 12654, 1970, 25, 271, 71093, 12305, 198, 727, 10562, 39914, 2784, 11, 292, 1590, 198, 262, 680, 11, 585, 11, 492, 283, 9774, 220, 15, 11, 220, 15, 198, 262, 1345, 585, 361, 2343, 2784, 8, 321, 492, 361, 2343, 1818, 1590, 198, 285, 413, 264, 957, 60, 2564, 292, 3681, 5491, 198, 309, 680, 1989, 2784, 957, 2387, 198, 309]
MIN_MATCH = 32  # leading tokens that must match REF_IDS

ENV = dict(os.environ, HIPFIRE_EMIT_TOKEN_IDS="1", HIPFIRE_GRAPH="0",
           HIPFIRE_AR_GRAPH="0", HIPFIRE_CASK_OFF="1", HIPFIRE_DPM_WARMUP_SECS="10")

prompt = open(PROMPT_PATH, "rb").read()
assert hashlib.md5(prompt).hexdigest() == PROMPT_MD5, "prompt bytes changed"
prompt = prompt.decode()

# ---- perf through the daemon ----
log = open("/tmp/autoresearch-daemon.log", "w")
proc = subprocess.Popen(["target/release/daemon"], env=ENV, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                        start_new_session=True)
OUT_FD = proc.stdout.fileno()
PENDING = b""

def send(msg):
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()

STALE = set()

def read_line(deadline):
    # Own line buffer over the raw fd: select() on a buffered file object misses
    # lines already pulled into Python's buffer (commit_ready stalled until the
    # daemon's 30 s commit timeout fired and aborted the turn).
    global PENDING
    while b"\n" not in PENDING:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("daemon output timeout")
        if not select.select([OUT_FD], [], [], min(left, 5.0))[0]:
            continue
        chunk = os.read(OUT_FD, 1 << 16)
        if not chunk:
            raise RuntimeError("daemon closed stdout (see /tmp/autoresearch-daemon.log)")
        PENDING += chunk
    line, PENDING = PENDING.split(b"\n", 1)
    return line.decode(errors="replace")

def read_until(stop, seconds):
    out, deadline = [], time.time() + seconds
    while True:
        line = read_line(deadline)
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "done" and ev.get("id") in STALE:
            continue  # previous generate's done: emitted only once the next command arrives
        ev["_t"] = time.perf_counter()
        out.append(ev)
        if t == "commit_ready":
            send({"type": "commit", "id": ev.get("id"), "attempt_id": ev.get("attempt_id", 1)})
            STALE.add(ev.get("id"))
        if t in stop:
            return out

def generate(gid):
    send({"type": "generate", "id": gid, "prompt": prompt, "temperature": 0.0,
          "max_tokens": MAX_TOKENS, "max_think_tokens": 1, "attempt_id": 1})
    evs = read_until({"commit_ready", "done", "error"}, 900)
    errs = [e for e in evs if e.get("type") == "error"]
    if errs:
        raise RuntimeError(f"generate error: {errs}")
    done = evs[-1]
    ids = [e.get("tok_id") for e in evs if e.get("type") == "committed"]
    toks = [e for e in evs if e.get("type") == "token"]
    text = "".join(e.get("text", "") for e in toks)
    # Steps between the first and last streamed token: decode only, no prefill.
    done["client_decode_tok_s"] = (len(toks) - 1) / (toks[-1]["_t"] - toks[0]["_t"])
    return done, ids, text

try:
    t0 = time.time()
    send({"type": "load", "model": MODEL,
          "params": {"max_seq": 2048, "kv_mode": "q8", "mtp_mode": "off"}})
    loaded = read_until({"loaded", "load_error", "error"}, 1800)
    if loaded[-1].get("type") != "loaded":
        raise RuntimeError(f"load failed: {loaded[-1]}")
    load_s = time.time() - t0
    generate("warmup")
    rows = [generate(f"r{i}") for i in range(RUNS)]
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    log.close()

dec = [d["client_decode_tok_s"] for d, _, _ in rows]
pp = [d["prefill_tokens"] / d["prefill_ms"] * 1000.0 for d, _, _ in rows]
ids = rows[-1][1]
print(f"done={ {k: v for k, v in rows[-1][0].items() if not isinstance(v, (list, dict))} }")
print(f"samples_decode={[round(x, 3) for x in dec]} daemon={[d.get('decode_tok_s') for d, _, _ in rows]}")
print(f"text={rows[-1][2]!r}")
print(f"ids={ids}")
if len(ids) < MIN_MATCH:
    print(f"FAIL: only {len(ids)} tokens generated"); sys.exit(1)
if any(r[1] != ids for r in rows):
    print("FAIL: token IDs differ between runs"); sys.exit(1)
match = len(ids)
if REF_IDS is not None:
    match = next((i for i, (a, b) in enumerate(zip(ids, REF_IDS)) if a != b), min(len(ids), len(REF_IDS)))
    if match < MIN_MATCH:
        print(f"FAIL: only {match} leading tokens match baseline"); sys.exit(1)

# ---- decode-route KLD gate ----
kld_log = "/tmp/autoresearch-kld.log"
with open(kld_log, "w") as f:
    rc = subprocess.run(["target/release/examples/qwen4_kld", "eval", "--model", MODEL,
                         "--ref", KLD_REF, "--output", "/tmp/autoresearch-kld.kldseq",
                         "--max-chunks", str(KLD_CHUNKS), "--decode"],
                        env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
tail = open(kld_log).read()
m = re.search(r"mean KLD = ([0-9.]+)\s+mean NLL = ([0-9.]+).*top1 = ([0-9.]+).*logits sha256 ([0-9a-f]+)", tail)
if rc != 0 or not m:
    print(tail[-3000:]); print("FAIL: qwen4_kld eval --decode"); sys.exit(1)
kld, nll, top1, sha = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
print(f"decode_logits_sha256={sha}")
print(f"METRIC decode_tok_s={statistics.median(dec):.3f}")
print(f"METRIC decode_kld={kld:.6f}")
print(f"METRIC decode_nll={nll:.6f}")
print(f"METRIC decode_top1={top1:.4f}")
print(f"METRIC prefill_tok_s={statistics.median(pp):.2f}")
print(f"METRIC token_match={match}")
print(f"METRIC load_s={load_s:.1f}")
if KLD_MAX is not None and kld > KLD_MAX:
    print(f"FAIL: decode KLD {kld:.6f} > baseline {KLD_MAX:.6f}"); sys.exit(1)
PY
