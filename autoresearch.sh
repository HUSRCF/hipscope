#!/usr/bin/env bash
# MTP decode throughput for Qwen3.8-Flash-Next (qwen4, mq6q8-pleq8) on gfx1151
# through the native daemon protocol.
#
# One fresh daemon, load (max_seq 2048, kv q8, graph off, mtp_mode MTP_MODE
# [on], mtp_k MTP_K [3]); per genre (committed code + prose prompts) 1 warmup
# + RUNS greedy generates, max_tokens 128. Per-genre metric = median decode
# tok/s from token-event arrival times; primary = geometric mean of the two.
# Acceptance from HIPFIRE_MTP_TRACE window events of the measured runs.
# Guards (non-zero exit): ids identical across runs, and identical to the AR
# greedy ids of the same build (greedy speculation must not change output).
set -euo pipefail
cd "$(dirname "$0")"

flock -w 3600 /tmp/hipfire-build.lock cargo build --release -q -p hipfire-daemon \
    >/tmp/autoresearch-build.log 2>&1 || { tail -40 /tmp/autoresearch-build.log; exit 1; }

if [ -f autoresearch.env ]; then
    set -a; . ./autoresearch.env; set +a
    echo "config: $(grep -v '^#' autoresearch.env | tr '\n' ' ')"
fi

exec flock -w 3600 /tmp/hipfire-gpu.lock python3 - <<'PY'
import hashlib, json, math, os, select, statistics, subprocess, sys, time

MODEL = os.path.expanduser("~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq")
PROMPTS = {
    "code": ("benchmarks/prompts/lru_cache_pep8_strict.txt", "df5dedc8040ce70ba55080c4548e6024"),
    "prose": ("benchmarks/prompts/prose_river_short.txt", "07a7880965142971dbb3cc7493f8fb94"),
}
MTP_MODE = os.environ.get("MTP_MODE", "on")
MTP_K = int(os.environ.get("MTP_K", "3"))
RUNS = int(os.environ.get("AR_RUNS", "3"))
MAX_TOKENS = 128
# AR greedy ids (mtp off) of the harness-setup build, per genre.
# Recorded by MTP_MODE=off (ar_ids.json), copied to ref_ids.json.
REF_IDS = json.load(open(".codeinsight+research/qwen4/mtp-20260927/ref_ids.json")) \
    if MTP_MODE != "off" else {}

ENV = dict(os.environ, HIPFIRE_EMIT_TOKEN_IDS="1", HIPFIRE_GRAPH="0", HIPFIRE_MTP_TRACE="1",
           HIPFIRE_AR_GRAPH="0", HIPFIRE_CASK_OFF="1", HIPFIRE_DPM_WARMUP_SECS="10")

prompts = {}
for genre, (path, md5) in PROMPTS.items():
    raw = open(path, "rb").read()
    assert hashlib.md5(raw).hexdigest() == md5, f"{path} bytes changed"
    prompts[genre] = raw.decode()

LOG_PATH = "/tmp/autoresearch-daemon.log"
log = open(LOG_PATH, "w")
proc = subprocess.Popen(["target/release/daemon"], env=ENV, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                        start_new_session=True)
OUT_FD = proc.stdout.fileno()
PENDING = b""
STALE = set()

def send(msg):
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()

def read_line(deadline):
    global PENDING
    while b"\n" not in PENDING:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("daemon output timeout")
        if not select.select([OUT_FD], [], [], min(left, 5.0))[0]:
            continue
        chunk = os.read(OUT_FD, 1 << 16)
        if not chunk:
            raise RuntimeError(f"daemon closed stdout (see {LOG_PATH})")
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
            continue
        ev["_t"] = time.perf_counter()
        out.append(ev)
        if t == "commit_ready":
            send({"type": "commit", "id": ev.get("id"), "attempt_id": ev.get("attempt_id", 1)})
            STALE.add(ev.get("id"))
        if t in stop:
            return out

def log_offset():
    log.flush()
    return os.path.getsize(LOG_PATH)

def generate(gid, prompt):
    start = log_offset()
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
    done["client_decode_tok_s"] = (len(toks) - 1) / (toks[-1]["_t"] - toks[0]["_t"])
    time.sleep(0.05)
    with open(LOG_PATH, errors="replace") as f:
        f.seek(start)
        windows = [json.loads(l.split(" ", 1)[1]) for l in f.read().splitlines()
                   if l.startswith("QWEN4_MTP_TRACE ") and '"window"' in l]
    return done, ids, text, windows

results = {}
try:
    t0 = time.time()
    send({"type": "load", "model": MODEL,
          "params": {"max_seq": 2048, "kv_mode": "q8", "mtp_mode": MTP_MODE, "mtp_k": MTP_K}})
    loaded = read_until({"loaded", "load_error", "error"}, 1800)
    if loaded[-1].get("type") != "loaded":
        raise RuntimeError(f"load failed: {loaded[-1]}")
    load_s = time.time() - t0
    for genre, prompt in prompts.items():
        generate(f"{genre}-warmup", prompt)
        results[genre] = [generate(f"{genre}-r{i}", prompt) for i in range(RUNS)]
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    log.close()

fail = False
tok_s = {}
for genre, rows in results.items():
    dec = [d["client_decode_tok_s"] for d, _, _, _ in rows]
    ids = rows[-1][1]
    windows = [w for r in rows for w in r[3]]
    committed = sum(len(w["committed"]) for w in windows)
    tpc = committed / len(windows) if windows else 1.0
    acc = sum(w["accepted"] for w in windows)
    drafted = sum(len(w["rows"]) for w in windows)
    tok_s[genre] = statistics.median(dec)
    print(f"[{genre}] samples_decode={[round(x, 3) for x in dec]} daemon={[d.get('decode_tok_s') for d, *_ in rows]}")
    print(f"[{genre}] text={rows[-1][2]!r}")
    print(f"[{genre}] ids={ids}")
    print(f"METRIC {genre}_tok_s={tok_s[genre]:.3f}")
    print(f"METRIC {genre}_tokens_per_cycle={tpc:.3f}")
    print(f"METRIC {genre}_accept_rate={acc / drafted if drafted else 0.0:.3f}")
    if len(ids) < 32:
        print(f"FAIL: {genre} only {len(ids)} tokens"); fail = True
    if any(r[1] != ids for r in rows):
        print(f"FAIL: {genre} token IDs differ between runs"); fail = True
    ref = REF_IDS.get(genre)
    if ref is not None:
        match = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), min(len(ids), len(ref)))
        print(f"METRIC {genre}_token_match={match}")
        if match < min(len(ids), len(ref)) or len(ids) != len(ref):
            print(f"FAIL: {genre} ids diverge from AR greedy at {match}"); fail = True
print(f"METRIC decode_tok_s={math.sqrt(tok_s['code'] * tok_s['prose']):.3f}")
print(f"METRIC load_s={load_s:.1f}")
if MTP_MODE == "off":
    out = {g: rows[-1][1] for g, rows in results.items()}
    os.makedirs(".codeinsight+research/qwen4/mtp-20260927", exist_ok=True)
    json.dump(out, open(".codeinsight+research/qwen4/mtp-20260927/ar_ids.json", "w"))
    print("wrote .codeinsight+research/qwen4/mtp-20260927/ar_ids.json")
sys.exit(1 if fail else 0)
PY
