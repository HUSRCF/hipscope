#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""hw-gate hardware runner: build the PR, exercise the selected buckets, write evidence.

Runs ONLY on the self-hosted runner, after a maintainer applied `hw-run`.
Everything it records is meant to be read by a human and by review.py:
decoded text verbatim, detector reports, exit codes, sha256 of every fixture,
md5 of both binaries and of every prompt. It never summarizes away evidence.

CONTRACT
    run.py --repo DIR --fixtures FILE --base SHA --head SHA --buckets load[,serve[,kernel]]
           --device N --out hw-gate.json --md hw-gate.md [--only-tag TAG]* [--skip-build]
    exit 0 : every bucket passed
    exit 1 : any fixture/route failed (evidence still written)
    exit 2 : precondition failed (fixture missing/mismatched, build failed, harness missing) — also a gate FAILURE

    hw-gate.json
        {
          "schema": "hipfire.hw-gate.evidence", "version": 1,
          "verdict": "pass" | "fail",
          "base": SHA, "head": SHA, "buckets": [...],
          "host": {"gfx": "gfx1201", "rocm": "...", "device": "3", "runner": hostname},
          "binaries": {"daemon_md5": ..., "hipfire_md5": ..., "build_seconds": float},
          "fixtures": [
            {"tag": ..., "file": ..., "sha256": ..., "sha256_ok": bool, "size_ok": bool,
             "modes": {
               "battery": {"exit": int, "seconds": float, "rows": [...], "status": "pass"|"fail", "reason": "..."},
               "chain":   {"exit": int, "seconds": float, "rows": [...], "status": "pass"|"fail", "reason": "..."}
             },
             "status": "pass" | "fail", "reason": "..."}
          ],
          "kernel": {"report": {...redline harness JSON...}, "status": ..., "reason": ...} | null,
          "logs_dir": "hw-gate-logs"
        }
    Row shape (enriched from harness --out JSON, which is a list of per-turn rows):
        {"genre":..., "finish": "stop"|"length"|None, "ctx":..., "cached":..., "gen":..., "ans_words":..., "prefill_tok_s":..., "decode_tok_s":..., "attractor":bool, "empty":bool, "runaway":bool, "recall_ok":bool, "expected_substrings":[...], "assistant_content":"...verbatim...", "prompt_md5":...}
        runaway is finish=="length"; recall_ok is every expected_substrings case-insensitively in assistant_content.

    hw-gate.md  : the same evidence rendered for a PR comment. Header table (host, binaries,
                  base..head, buckets); per-fixture turn table (mode, genre, finish, ctx, cached, gen, ans_words, prefill/decode tok/s, attractor/empty/runaway/recall flags) followed by one <details> per turn with assistant_content verbatim inside a fence from _fence(); then kernel section. review.py posts this file as the evidence comment.

BEHAVIOR
    1. Preconditions (exit 2 on any failure):
       - for every fixture in the selected buckets: file exists under models_dir, size_bytes matches,
         sha256 matches. sha256 is cached in $HIPFIRE_HOME/hw-gate-sha.json keyed by
         (realpath, size, mtime_ns, inode); a cache hit skips hashing. NEVER downgrade a mismatch
         to a warning.
       - harness scripts exist for selected buckets (scripts/serve_harness.py, scripts/redline_daemon_harness.py
         from --repo).
    2. Build: `cargo build --release` in --repo (CARGO_TARGET_DIR honoured from env). Record md5 of
       target/release/{daemon,hipfire}. --skip-build reuses existing binaries (local iteration only).
    3. Isolated home: create $HIPFIRE_HOME (env, required) with config.toml:
           [hardware]\ndevices = "<--device>"\n
       and symlink $HIPFIRE_HOME/models -> models_dir. Nothing from ~/.hipfire/config.toml leaks in.
    4. For each fixture, run the harness modes which are the union of `buckets.<bucket>.modes` across the
       selected buckets (e.g. load→battery, serve→battery+chain, kernel→battery; load+serve→battery+chain).
       Each mode is one invocation of `scripts/serve_harness.py`:
           HIP_VISIBLE_DEVICES=<device> HIPFIRE_MODELS_DIR=<models_dir> HIPFIRE_DAEMON_BIN=<bin>/daemon HIPFIRE_CLI_BIN=<bin>/hipfire \
           python3 scripts/serve_harness.py --model <models_dir>/<file> --mode <battery|chain> \
             --prompts-file <harness.battery_prompts> --max-tokens <harness.max_tokens> \
             --thinking off --thinking-effort none --max-think-tokens 0 \
             --home <HIPFIRE_HOME>/<tag>-<mode> --out <logs>/<tag>-<mode>.json --serve-log <logs>/<tag>-<mode>.serve.log
       where `--prompts-file` is only for battery, chain uses its built-in chain battery. The three thinking
       flags are all required together: --thinking off alone loses to the registry's reasoning_effort=xhigh
       for qwen3.8 (turns think and answer nothing); --thinking-effort none alone trips the harness
       preflight "max_tokens <= thinking cap"; --max-think-tokens 0 clears that. HIP_VISIBLE_DEVICES must
       be the gate device — the harness defaults it to "0" and does NOT honour a --devices flag. Do not pass
       --devices. The harness writes its own <home>/.hipfire/config.toml and symlinks models; it takes
       --home DIR (not HIPFIRE_HOME). A per-fixture subdirectory of the gate home is used so runs never
       share state. stdout+stderr are captured to <logs>/<tag>-<mode>.out for evidence but status is derived
       from the exit code and the --out JSON only: exit 0 = pass (exit 1 on attractor or preflight is fatal),
       and any row with attractor, empty, or recall miss (expected_substrings not all case-insensitively in
       assistant_content) makes that mode fail. runaway (finish=="length") is recorded and shown but not
       fatal by itself: a coherent answer cut by the cap is not a defect; a loop-to-cap sets attractor.
    5. kernel bucket: python3 scripts/redline_daemon_harness.py --model <models_dir/file> --daemon <daemon>
       <harness_args...> --out hw-gate-logs/redline.json; harness_args come from buckets.kernel.redline.harness_args.
       status=pass iff exit 0 and the report's parity fields are all true (read the harness's own summary keys; do not invent thresholds).
    6. Write JSON + MD, exit per verdict.

Nothing here may consult a model. Nothing here may skip a fixture because it is slow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


# Single seam for tests to monkeypatch
def run_cmd(argv, **kwargs):
    return subprocess.run(argv, **kwargs)


def _md5_file(path: Path) -> str | None:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _resolve_bin_dir(repo: Path) -> Path:
    cargo_target = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target:
        return Path(cargo_target) / "release"
    return Path(repo) / "target" / "release"


def verify_fixture(models_dir, fixture, cache_path) -> dict:
    """Verify single fixture file existence, size and sha256 with cache.

    Cache is JSON at cache_path keyed by "realpath:size:mtime_ns:inode" -> sha256 hex.
    Hashing is 8 MiB chunks. Never softens mismatch.
    """
    models_dir_p = Path(models_dir).expanduser() if models_dir is not None else Path(".")
    file_name = fixture.get("file", "")
    tag = fixture.get("tag", "")
    expected_sha = fixture.get("sha256", "")
    expected_size = fixture.get("size_bytes")
    file_path = models_dir_p / file_name
    result: dict = {
        "tag": tag,
        "file": file_name,
        "expected_sha256": expected_sha,
        "expected_size": expected_size,
        "path": str(file_path),
    }
    if not file_path.is_file():
        result.update(
            {
                "exists": False,
                "size_ok": False,
                "sha256_ok": False,
                "actual_size": None,
                "actual_sha256": None,
                "reason": f"missing {file_path}",
            }
        )
        return result
    try:
        st = file_path.stat()
        actual_size = st.st_size
    except Exception as e:
        result.update(
            {
                "exists": False,
                "size_ok": False,
                "sha256_ok": False,
                "actual_size": None,
                "actual_sha256": None,
                "reason": f"stat failed: {e}",
            }
        )
        return result
    size_ok = (actual_size == expected_size) if expected_size is not None else True
    # cache handling
    cache_path_p = Path(cache_path)
    cache: dict = {}
    if cache_path_p.is_file():
        try:
            cache = json.loads(cache_path_p.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}
    try:
        realpath = str(file_path.resolve())
    except Exception:
        realpath = str(file_path)
    key = f"{realpath}:{st.st_size}:{st.st_mtime_ns}:{st.st_ino}"
    cached = cache.get(key)
    if isinstance(cached, dict):
        cached_sha = cached.get("sha256")
    else:
        cached_sha = cached
    if isinstance(cached_sha, str) and cached_sha:
        actual_sha = cached_sha
    else:
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        actual_sha = h.hexdigest()
        cache[key] = actual_sha
        try:
            cache_path_p.parent.mkdir(parents=True, exist_ok=True)
            cache_path_p.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass
    sha_ok = (actual_sha == expected_sha)
    reason = ""
    if not size_ok:
        reason = f"size mismatch: expected {expected_size}, got {actual_size}"
    elif not sha_ok:
        reason = f"sha256 mismatch: expected {expected_sha}, got {actual_sha}"
    result.update(
        {
            "exists": True,
            "size_ok": bool(size_ok),
            "sha256_ok": bool(sha_ok),
            "actual_size": actual_size,
            "actual_sha256": actual_sha,
            "reason": reason,
        }
    )
    return result


# The gate measures load + coherence, not reasoning budgets: thinking models
# cannot close <think> inside a small token budget and the daemon fails that
# turn closed ("open think span at end of generation"), which would look like
# a load regression. Every fixture runs with visible reasoning off.
GATE_CONFIG_TOML = """[hardware]
devices = "{device}"

[reasoning]
mode = "off"
effort = "none"
"""


def write_isolated_home(home, device, models_dir) -> str:
    """Create isolated HIPFIRE_HOME with the pinned gate config and models symlink.

    Writes config.toml to both $HOME/config.toml and $HOME/.hipfire/config.toml
    and symlinks models at both locations. This covers both discovery paths
    (HIPFIRE_HOME direct and HOME/.hipfire fallback) and isolates from
    ~/.hipfire/config.toml. Returns the config text so it can be recorded in
    the evidence.
    """
    home_p = Path(home)
    models_p = Path(models_dir).expanduser().resolve() if models_dir else Path(".")
    home_p.mkdir(parents=True, exist_ok=True)
    sub = home_p / ".hipfire"
    sub.mkdir(parents=True, exist_ok=True)
    content = GATE_CONFIG_TOML.format(device=device)
    for cfg_path in [home_p / "config.toml", sub / "config.toml"]:
        cfg_path.write_text(content, encoding="utf-8")
    for link_path in [home_p / "models", sub / "models"]:
        try:
            if link_path.is_symlink() or link_path.exists():
                if link_path.is_symlink():
                    link_path.unlink()
                elif link_path.is_dir():
                    import shutil

                    shutil.rmtree(link_path)
                else:
                    link_path.unlink()
            link_path.symlink_to(models_p)
        except Exception:
            pass
    return content


def _get_host_gfx() -> str:
    # Try rocminfo and rocm-smi, look for gfx string
    pattern = re.compile(r"gfx\d+[a-z0-9]*", re.IGNORECASE)
    for cmd in (["rocminfo"], ["rocm-smi", "--showproductname"], ["rocm-smi"]):
        try:
            res = run_cmd(cmd, capture_output=True, text=True, timeout=2)
            out = (res.stdout or "") + (res.stderr or "")
            m = pattern.search(out)
            if m:
                return m.group(0).lower()
            if out.strip() and "gfx" in out.lower():
                # fallback return first gfx-like token
                return m.group(0) if m else out.strip().split()[0][:20]
        except Exception:
            continue
    return "unknown"


def _get_host_rocm() -> str:
    p = Path("/opt/rocm/.info/version")
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    try:
        res = run_cmd(["hipconfig", "--version"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _fence(text: str) -> str:
    """A backtick fence strictly longer than any backtick run inside `text`,
    so decoded answers that contain ``` blocks render verbatim."""
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    return "`" * max(3, longest + 1)


def _sanitize_tag(tag: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", tag)
    return sanitized or "fixture"


def _modes_for_buckets(buckets: list[str], manifest: dict) -> list[str]:
    """Union of `buckets.<bucket>.modes` across selected buckets, ordered battery then chain."""
    buckets_cfg = manifest.get("buckets", {}) if isinstance(manifest.get("buckets"), dict) else {}
    union: set[str] = set()
    for b in buckets:
        cfg = buckets_cfg.get(b, {})
        if isinstance(cfg, dict):
            modes = cfg.get("modes", [])
            if isinstance(modes, list):
                for m in modes:
                    if isinstance(m, str) and m:
                        union.add(m)
    # fallback for old manifest where no modes field but bucket implicitly exists?
    # Keep empty to avoid inventing.
    ordered: list[str] = []
    for pref in ("battery", "chain"):
        if pref in union:
            ordered.append(pref)
    for m in sorted(union):
        if m not in ordered:
            ordered.append(m)
    return ordered


def _enrich_row(raw: dict) -> dict:
    """Enrich a raw harness row into the evidence row shape, computing runaway/recall_ok."""
    if not isinstance(raw, dict):
        raw = {}
    attractor = bool(raw.get("attractor"))
    empty = bool(raw.get("empty"))
    finish = raw.get("finish")
    runaway = (finish == "length")
    expected = raw.get("expected_substrings")
    if expected is None:
        expected = []
    if not isinstance(expected, list):
        expected = [expected] if expected else []
    # filter to strings
    expected_strs: list[str] = []
    for e in expected:
        if isinstance(e, str):
            expected_strs.append(e)
        elif e is not None:
            expected_strs.append(str(e))
    assistant = raw.get("assistant_content")
    if assistant is None:
        assistant = raw.get("content", "")
    if not isinstance(assistant, str):
        assistant = str(assistant) if assistant is not None else ""
    # recall_ok case-insensitive
    if expected_strs:
        low = assistant.lower()
        recall_ok = all(es.lower() in low for es in expected_strs)
    else:
        recall_ok = True
    return {
        "genre": raw.get("genre"),
        "finish": finish,
        "ctx": raw.get("ctx"),
        "cached": raw.get("cached"),
        "gen": raw.get("gen"),
        "ans_words": raw.get("ans_words"),
        "prefill_tok_s": raw.get("prefill_tok_s"),
        "decode_tok_s": raw.get("decode_tok_s"),
        "prefill_ms": raw.get("prefill_ms"),
        "attractor": attractor,
        "empty": empty,
        "runaway": runaway,
        "recall_ok": recall_ok,
        "expected_substrings": expected_strs,
        "assistant_content": assistant,
        "prompt_md5": raw.get("prompt_md5"),
        "ans_preview": raw.get("ans_preview"),
        "reasoning_content": raw.get("reasoning_content"),
        "content": raw.get("content"),
    }


def _evaluate_mode(exit_code: int, raw_rows) -> tuple[str, str, list[dict]]:
    """Evaluate mode status from exit code and raw rows.

    Returns (status, reason, enriched_rows).
    """
    enriched: list[dict] = []
    if isinstance(raw_rows, list):
        for r in raw_rows:
            if isinstance(r, dict):
                enriched.append(_enrich_row(r))
    # status logic
    if exit_code != 0:
        return "fail", f"harness exit {exit_code}", enriched
    # check rows
    fails: list[str] = []
    for idx, row in enumerate(enriched):
        if row.get("attractor"):
            fails.append(f"row {idx} attractor")
        if row.get("empty"):
            fails.append(f"row {idx} empty")
        # runaway (finish=="length") is reported in the evidence table but is not
        # fatal on its own: a coherent long answer cut by the token cap is not a
        # defect, and a loop-to-cap is caught by the harness's `attractor` flag.
        if not row.get("recall_ok"):
            # missing substrings
            assistant = row.get("assistant_content", "")
            missing = [e for e in row.get("expected_substrings", []) if e.lower() not in assistant.lower()]
            fails.append(f"row {idx} recall miss {missing}")
    if fails:
        return "fail", "; ".join(fails), enriched
    # also if raw_rows was not a list but exit 0, still fail closed if no rows? treat as pass (empty battery not typical but not failure)
    return "pass", "", enriched


def _build_harness_argv(repo: Path, model_path: Path, mode: str, max_tokens: int, battery_prompts_path: Path | None, per_home: Path, out_path: Path, serve_log_path: Path) -> list[str]:
    """Build argv for serve_harness.py. Never includes --devices."""
    harness = Path(repo) / "scripts" / "serve_harness.py"
    argv = [
        sys.executable,
        str(harness),
        "--model",
        str(model_path),
        "--mode",
        mode,
        "--max-tokens",
        str(max_tokens),
        "--thinking",
        "off",
        "--thinking-effort",
        "none",
        "--max-think-tokens",
        "0",
        "--home",
        str(per_home),
        "--out",
        str(out_path),
        "--serve-log",
        str(serve_log_path),
    ]
    if mode == "battery" and battery_prompts_path is not None:
        argv.extend(["--prompts-file", str(battery_prompts_path)])
    return argv


def _env_for_harness(env_base: dict, device: str, models_dir: str, daemon_bin: Path, hipfire_bin: Path) -> dict:
    env = dict(env_base)
    env["HIP_VISIBLE_DEVICES"] = str(device)
    env["HIPFIRE_MODELS_DIR"] = str(models_dir)
    env["HIPFIRE_DAEMON_BIN"] = str(daemon_bin)
    env["HIPFIRE_CLI_BIN"] = str(hipfire_bin)
    return env


def _run_harness_mode(repo, fixture, env_base, logs_dir, device, mode, harness_cfg, models_dir) -> dict:
    """Run one harness mode for a fixture and return mode evidence dict."""
    repo_p = Path(repo)
    logs_dir_p = Path(logs_dir)
    logs_dir_p.mkdir(parents=True, exist_ok=True)
    tag = fixture.get("tag", "")
    safe_tag = _sanitize_tag(tag)
    file_name = fixture.get("file", "")
    model_path = Path(models_dir).expanduser() / file_name if file_name else Path(models_dir) / tag
    harness = repo_p / "scripts" / "serve_harness.py"
    # harness cfg
    battery_prompts_rel = harness_cfg.get("battery_prompts", "benchmarks/prompts/hw-gate/serve-battery.json")
    battery_prompts_path = (repo_p / battery_prompts_rel) if battery_prompts_rel else None
    max_tokens = harness_cfg.get("max_tokens", 256)
    # per-fixture home: subdirectory of gate home
    gate_home = env_base.get("HIPFIRE_HOME") or str(Path.home() / ".hipfire")
    per_home = Path(gate_home) / f"{safe_tag}-{mode}"
    # ensure no --devices leakage: we never add it
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_bin = bin_dir / "daemon"
    hipfire_bin = bin_dir / "hipfire"
    env = _env_for_harness(env_base, device, models_dir, daemon_bin, hipfire_bin)
    out_path = logs_dir_p / f"{safe_tag}-{mode}.json"
    serve_log_path = logs_dir_p / f"{safe_tag}-{mode}.serve.log"
    out_combined_path = logs_dir_p / f"{safe_tag}-{mode}.out"
    argv = _build_harness_argv(repo_p, model_path, mode, max_tokens, battery_prompts_path, per_home, out_path, serve_log_path)
    start = time.time()
    exit_code = 1
    stdout = ""
    stderr = ""
    try:
        res = run_cmd(argv, env=env, capture_output=True, text=True, timeout=900)
        exit_code = res.returncode if res.returncode is not None else 1
        stdout = res.stdout if res.stdout is not None else ""
        stderr = res.stderr if res.stderr is not None else ""
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        exit_code = 124
        stderr += "\n[timeout]"
    except FileNotFoundError as e:
        stderr = str(e)
        exit_code = 127
    except Exception as e:
        stderr = str(e)
        exit_code = 1
    seconds = time.time() - start
    # capture stdout+stderr to .out for evidence (derive status from exit+JSON only)
    try:
        combined = ""
        if stdout:
            combined += stdout
            if not combined.endswith("\n"):
                combined += "\n"
        if stderr:
            combined += stderr
        out_combined_path.write_text(combined, encoding="utf-8")
    except Exception:
        pass
    # also ensure serve log dir exists? harness writes serve-log itself
    raw_rows = None
    if out_path.is_file():
        try:
            raw_rows = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            raw_rows = None
    status, reason, enriched = _evaluate_mode(exit_code, raw_rows)
    # If exit 0 but raw_rows None -> treat as fail closed? _evaluate_mode returns pass for that case currently;
    # we want fail closed if harness succeeded but no rows written.
    if exit_code == 0 and raw_rows is None:
        # No rows but exit 0 -> still fail closed to surface missing output
        # However if harness legitimately writes empty list, that's file exists -> raw_rows = []
        # So None indicates missing or parse error.
        status = "fail"
        reason = "missing --out JSON"
        enriched = []
    # Build return dict
    return {"exit": exit_code, "seconds": float(seconds), "rows": enriched, "status": status, "reason": reason}


def _run_fixture_harness(repo, fixture, env_base, logs_dir, device, modes, harness_cfg, models_dir) -> dict:
    """Run all modes for one fixture."""
    modes_result: dict = {}
    for mode in modes:
        res = _run_harness_mode(repo, fixture, env_base, logs_dir, device, mode, harness_cfg, models_dir)
        modes_result[mode] = res
    # overall fixture status
    overall = "pass" if all(m.get("status") == "pass" for m in modes_result.values()) else ("pass" if not modes_result else "fail")
    # if no modes (should not happen) treat as pass
    reasons: list[str] = []
    for m, r in modes_result.items():
        if r.get("status") != "pass" and r.get("reason"):
            reasons.append(f"{m}: {r.get('reason')}")
        elif r.get("status") != "pass":
            reasons.append(f"{m}: fail")
    reason = "; ".join(reasons) if reasons else ""
    tag = fixture.get("tag", "")
    return {
        "tag": tag,
        "file": fixture.get("file", ""),
        "sha256": fixture.get("sha256", ""),
        "sha256_ok": True,
        "size_ok": True,
        "modes": modes_result,
        "status": overall,
        "reason": reason,
    }


def _find_file_for_tag(tag: str, manifest: dict):
    # top-level fixtures first
    for f in manifest.get("fixtures", []) if isinstance(manifest.get("fixtures"), list) else []:
        if f.get("tag") == tag:
            return f.get("file")
    # old shape fallback
    for f in manifest.get("buckets", {}).get("load", {}).get("fixtures", []) if isinstance(manifest.get("buckets"), dict) else []:
        if f.get("tag") == tag:
            return f.get("file")
    return None


def run_kernel(repo, models_dir, kernel_cfg, fixtures_manifest, env, logs_dir) -> dict:
    """Run redline daemon harness.

    kernel_cfg is buckets.kernel.redline (model_tag + harness_args).
    Flags are only real ones from redline_daemon_harness.py argparse:
      --model, --daemon, --out, plus harness_args from fixtures manifest.

    Status pass iff exit 0 and report's parity fields are all true
    (report["pass"] is the boolean summary; if missing, fail closed).
    """
    repo_p = Path(repo)
    logs_dir_p = Path(logs_dir)
    logs_dir_p.mkdir(parents=True, exist_ok=True)
    model_tag = kernel_cfg.get("model_tag", "") if isinstance(kernel_cfg, dict) else ""
    model_file = _find_file_for_tag(model_tag, fixtures_manifest) if model_tag else None
    if not model_file:
        model_file = f"{model_tag}.mq4" if model_tag else ""
    model_path = Path(models_dir).expanduser() / model_file if model_file else Path(models_dir) / (model_tag or "")
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_bin = bin_dir / "daemon"
    harness = repo_p / "scripts" / "redline_daemon_harness.py"
    redline_out = logs_dir_p / "redline.json"
    harness_args = kernel_cfg.get("harness_args", []) if isinstance(kernel_cfg, dict) else []
    harness_args = [str(a) for a in harness_args]
    cmd = [
        sys.executable,
        str(harness),
        "--model",
        str(model_path),
        "--daemon",
        str(daemon_bin),
        "--out",
        str(redline_out),
    ] + harness_args
    k_env = dict(env)
    k_env["HIPFIRE_DAEMON_BIN"] = str(daemon_bin)
    exit_code = 1
    report = None
    stderr = ""
    try:
        res = run_cmd(cmd, env=k_env, capture_output=True, text=True, timeout=900)
        exit_code = res.returncode
        stderr = (res.stderr or "") + (res.stdout or "")
        if redline_out.is_file():
            try:
                report = json.loads(redline_out.read_text(encoding="utf-8"))
            except Exception as e:
                report = {"error": f"failed to parse report: {e}", "raw": redline_out.read_text(errors="replace")[:2000]}
        else:
            report = {"error": "no report written", "stderr": stderr[-2000:]}
    except Exception as e:
        exit_code = 1
        report = {"error": str(e)}
        stderr = str(e)
    status = "fail"
    reason = ""
    if exit_code != 0:
        reason = f"harness exit {exit_code}"
        status = "fail"
    else:
        if report is None:
            reason = "no parity summary in report (fail closed)"
            status = "fail"
        elif "pass" in report:
            if report.get("pass") is True:
                status = "pass"
                reason = ""
            else:
                status = "fail"
                failures = report.get("dflash_verify_failures") or report.get("failures") or []
                if failures:
                    reason = f"parity fail: {failures}"
                else:
                    reason = "report pass is false"
        elif "bit_exact" in report or "aql_shadow" in report or "prefix_shadow" in report:
            reason = "no boolean parity summary (fail closed)"
            status = "fail"
        else:
            reason = "no boolean parity summary in report (fail closed)"
            status = "fail"
    return {"report": report, "exit": exit_code, "stderr_tail": stderr[-2000:] if stderr else "", "status": status, "reason": reason}


def render_md(evidence: dict) -> str:
    """Render hw-gate.md per CONTRACT: header table, per-fixture turn tables, details blocks, kernel."""
    lines: list[str] = []
    lines.append("# hw-gate evidence")
    lines.append("")
    host = evidence.get("host", {})
    binaries = evidence.get("binaries", {})
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| base | `{evidence.get('base','')}` |")
    lines.append(f"| head | `{evidence.get('head','')}` |")
    buckets = evidence.get("buckets", [])
    buckets_str = ",".join(buckets) if isinstance(buckets, list) else str(buckets)
    lines.append(f"| buckets | {buckets_str} |")
    lines.append(f"| host gfx | {host.get('gfx','')} |")
    lines.append(f"| host rocm | {host.get('rocm','')} |")
    lines.append(f"| device | {host.get('device','')} |")
    lines.append(f"| runner | {host.get('runner','')} |")
    lines.append(f"| daemon_md5 | {binaries.get('daemon_md5','')} |")
    lines.append(f"| hipfire_md5 | {binaries.get('hipfire_md5','')} |")
    lines.append(f"| build_seconds | {binaries.get('build_seconds','')} |")
    lines.append(f"| verdict | {evidence.get('verdict','')} |")
    lines.append(f"| logs_dir | {evidence.get('logs_dir','')} |")
    # optional error fields
    if evidence.get("precondition_error"):
        lines.append(f"| precondition_error | {evidence.get('precondition_error')} |")
    if evidence.get("build_error"):
        lines.append(f"| build_error | {str(evidence.get('build_error'))[:500]} |")
    if evidence.get("error"):
        lines.append(f"| error | {str(evidence.get('error'))[:500]} |")
    lines.append("")
    # Per-fixture sections
    lines.append("## fixtures")
    lines.append("")
    fixtures = evidence.get("fixtures", [])
    if not fixtures:
        lines.append("no fixtures")
        lines.append("")
    else:
        for fx in fixtures:
            tag = fx.get("tag", "")
            sha_ok = fx.get("sha256_ok", "")
            sha_ok_str = "✅" if sha_ok is True else ("❌" if sha_ok is False else str(sha_ok))
            size_ok = fx.get("size_ok", "")
            size_ok_str = "✅" if size_ok is True else ("❌" if size_ok is False else str(size_ok))
            status = fx.get("status", "")
            reason = (fx.get("reason", "") or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"### {tag}")
            lines.append("")
            lines.append(f"sha256_ok: {sha_ok_str} size_ok: {size_ok_str} status: {status} reason: {reason}")
            lines.append("")
            modes = fx.get("modes", {})
            if not modes:
                lines.append("no modes")
                lines.append("")
            else:
                for mode_name, mode_data in modes.items():
                    exit_code = mode_data.get("exit", "")
                    seconds = mode_data.get("seconds", "")
                    try:
                        seconds_str = f"{float(seconds):.1f}" if isinstance(seconds, (int, float)) else str(seconds)
                    except Exception:
                        seconds_str = str(seconds)
                    m_status = mode_data.get("status", "")
                    m_reason = (mode_data.get("reason", "") or "").replace("|", "\\|").replace("\n", " ")
                    lines.append(f"#### {mode_name} — exit {exit_code} seconds {seconds_str} status {m_status}")
                    if m_reason:
                        lines.append(f"reason: {m_reason}")
                    lines.append("")
                    rows = mode_data.get("rows", [])
                    if not rows:
                        lines.append("no rows")
                        lines.append("")
                    else:
                        lines.append("| mode | idx | genre | finish | ctx | cached | gen | ans_words | prefill_tok_s | decode_tok_s | attractor | empty | runaway | recall_ok |")
                        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                        for idx, row in enumerate(rows):
                            genre = row.get("genre", "")
                            finish = row.get("finish", "")
                            ctx = row.get("ctx", "")
                            cached = row.get("cached", "")
                            gen = row.get("gen", "")
                            ans_words = row.get("ans_words", "")
                            prefill = row.get("prefill_tok_s", "")
                            decode = row.get("decode_tok_s", "")
                            attractor_s = str(row.get("attractor", ""))
                            empty_s = str(row.get("empty", ""))
                            runaway_s = str(row.get("runaway", ""))
                            recall_s = str(row.get("recall_ok", ""))
                            # sanitize pipe
                            genre_s = str(genre).replace("|", "\\|") if genre is not None else ""
                            finish_s = str(finish).replace("|", "\\|") if finish is not None else ""
                            lines.append(
                                f"| {mode_name} | {idx} | {genre_s} | {finish_s} | {ctx} | {cached} | {gen} | {ans_words} | {prefill} | {decode} | {attractor_s} | {empty_s} | {runaway_s} | {recall_s} |"
                            )
                        lines.append("")
                    # details per turn verbatim
                    for idx, row in enumerate(rows):
                        assistant = row.get("assistant_content", "") or ""
                        genre = row.get("genre", "")
                        # summary includes tag mode and idx for uniqueness
                        safe_genre = str(genre) if genre else ""
                        lines.append(f"<details><summary>{tag} {mode_name} turn {idx} {safe_genre}</summary>")
                        lines.append("")
                        fence = _fence(assistant)
                        lines.append(fence)
                        lines.append(assistant)
                        lines.append(fence)
                        lines.append("")
                        lines.append("</details>")
                        lines.append("")
            lines.append("")
    # kernel
    lines.append("## kernel")
    lines.append("")
    kernel = evidence.get("kernel")
    if kernel is None:
        lines.append("not run")
        lines.append("")
    else:
        lines.append(f"status: {kernel.get('status','')}")
        if kernel.get("reason"):
            lines.append(f"reason: {kernel.get('reason')}")
        lines.append("")
        report = kernel.get("report", {})
        if isinstance(report, dict):
            if "pass" in report:
                lines.append(f"report pass: {report.get('pass')}")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--buckets", required=True, help="comma-separated")
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--only-tag", action="append", default=[])
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args(argv)

    buckets = [b.strip() for b in args.buckets.split(",") if b.strip()]
    # Validate buckets
    allowed = {"load", "serve", "kernel"}
    for b in buckets:
        if b not in allowed:
            print(f"unknown bucket {b!r}", file=sys.stderr)
            return 2

    repo_p = Path(args.repo)
    fixtures_path = Path(args.fixtures)
    out_path = Path(args.out)
    md_path = Path(args.md)
    # logs_dir sibling to out file
    if out_path.parent and str(out_path.parent) not in (".", ""):
        logs_dir = out_path.parent / "hw-gate-logs"
    else:
        logs_dir = Path.cwd() / "hw-gate-logs"
        try:
            if out_path.is_absolute():
                logs_dir = out_path.parent / "hw-gate-logs"
            else:
                logs_dir = Path(out_path.parent) / "hw-gate-logs" if str(out_path.parent) != "." else Path("hw-gate-logs")
        except Exception:
            logs_dir = Path("hw-gate-logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load fixtures
    try:
        fixtures_manifest = json.loads(fixtures_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"failed to load fixtures {fixtures_path}: {e}", file=sys.stderr)
        evidence_fail = {
            "schema": "hipfire.hw-gate.evidence",
            "version": 1,
            "verdict": "fail",
            "base": args.base,
            "head": args.head,
            "buckets": buckets,
            "host": {"gfx": "unknown", "rocm": "unknown", "device": args.device, "runner": socket.gethostname()},
            "binaries": {"daemon_md5": None, "hipfire_md5": None, "build_seconds": 0.0},
            "fixtures": [],
            "kernel": None,
            "logs_dir": "hw-gate-logs",
            "error": f"fixtures load failed: {e}",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(evidence_fail, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_md(evidence_fail), encoding="utf-8")
        return 2

    # Resolve models_dir
    models_dir_env = os.environ.get("HIPFIRE_MODELS_DIR")
    manifest_models_dir = fixtures_manifest.get("models_dir", "~/.hipfire/models")
    models_dir = Path(models_dir_env).expanduser() if models_dir_env else Path(manifest_models_dir).expanduser()
    # HIPFIRE_HOME required
    hipfire_home = os.environ.get("HIPFIRE_HOME")
    if not hipfire_home:
        print("HIPFIRE_HOME not set", file=sys.stderr)
        evidence_fail = {
            "schema": "hipfire.hw-gate.evidence",
            "version": 1,
            "verdict": "fail",
            "base": args.base,
            "head": args.head,
            "buckets": buckets,
            "host": {"gfx": _get_host_gfx(), "rocm": _get_host_rocm(), "device": args.device, "runner": socket.gethostname()},
            "binaries": {"daemon_md5": None, "hipfire_md5": None, "build_seconds": 0.0},
            "fixtures": [],
            "kernel": None,
            "logs_dir": "hw-gate-logs",
            "error": "HIPFIRE_HOME not set",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(evidence_fail, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_md(evidence_fail), encoding="utf-8")
        return 2
    hipfire_home_p = Path(hipfire_home)
    cache_path = hipfire_home_p / "hw-gate-sha.json"

    # Determine fixtures list (new shape top-level `fixtures`, fallback old shape)
    fixtures_all: list[dict] = []
    if isinstance(fixtures_manifest.get("fixtures"), list):
        fixtures_all = fixtures_manifest.get("fixtures", [])
    else:
        # fallback old shape
        fixtures_all = fixtures_manifest.get("buckets", {}).get("load", {}).get("fixtures", []) if isinstance(fixtures_manifest.get("buckets"), dict) else []
    # harness cfg
    harness_cfg: dict
    if isinstance(fixtures_manifest.get("harness"), dict):
        harness_cfg = fixtures_manifest.get("harness", {})
    else:
        serve_cfg_fallback = fixtures_manifest.get("buckets", {}).get("serve", {}) if isinstance(fixtures_manifest.get("buckets"), dict) else {}
        harness_cfg = {
            "battery_prompts": serve_cfg_fallback.get("battery_prompts", "benchmarks/prompts/hw-gate/serve-battery.json"),
            "max_tokens": serve_cfg_fallback.get("max_tokens", 256),
        }
    # ensure defaults
    if "battery_prompts" not in harness_cfg:
        harness_cfg["battery_prompts"] = "benchmarks/prompts/hw-gate/serve-battery.json"
    if "max_tokens" not in harness_cfg:
        harness_cfg["max_tokens"] = 256

    # Determine which fixtures to verify/run
    fixtures_to_verify: list[dict] = []
    for f in fixtures_all:
        if args.only_tag and f.get("tag") not in args.only_tag:
            continue
        fixtures_to_verify.append(f)

    # extra check for kernel redline model if filtered out (to keep file verification)
    extra_checks: list[dict] = []
    if "kernel" in buckets:
        kernel_bucket_cfg = fixtures_manifest.get("buckets", {}).get("kernel", {}) if isinstance(fixtures_manifest.get("buckets"), dict) else {}
        redline_cfg = kernel_bucket_cfg.get("redline", kernel_bucket_cfg) if isinstance(kernel_bucket_cfg.get("redline"), dict) else kernel_bucket_cfg
        if isinstance(redline_cfg, dict):
            model_tag = redline_cfg.get("model_tag")
            if model_tag:
                if not any(f.get("tag") == model_tag for f in fixtures_to_verify) and not any(f.get("tag") == model_tag for f in extra_checks):
                    for f in fixtures_all:
                        if f.get("tag") == model_tag:
                            extra_checks.append(f)
                            break
    all_checks = fixtures_to_verify + extra_checks

    # Modes union for harness
    modes_union = _modes_for_buckets(buckets, fixtures_manifest)
    # Preconditions: verify each fixture file
    verify_results: list[dict] = []
    precondition_failed = False
    precondition_reason = ""
    for fx in all_checks:
        vr = verify_fixture(str(models_dir), fx, str(cache_path))
        verify_results.append(vr)
        if not vr.get("size_ok") or not vr.get("sha256_ok") or not vr.get("exists"):
            precondition_failed = True
            precondition_reason = vr.get("reason", "precondition failed")

    # harness scripts existence
    if modes_union:
        if not (repo_p / "scripts" / "serve_harness.py").is_file():
            precondition_failed = True
            precondition_reason = "missing scripts/serve_harness.py"
    if "kernel" in buckets:
        if not (repo_p / "scripts" / "redline_daemon_harness.py").is_file():
            precondition_failed = True
            precondition_reason = "missing scripts/redline_daemon_harness.py"

    host_info = {"gfx": _get_host_gfx(), "rocm": _get_host_rocm(), "device": args.device, "runner": socket.gethostname()}

    if precondition_failed:
        fixtures_evidence = []
        for fx, vr in zip(all_checks, verify_results):
            fixtures_evidence.append(
                {
                    "tag": fx.get("tag", ""),
                    "file": fx.get("file", ""),
                    "sha256": fx.get("sha256", ""),
                    "sha256_ok": vr.get("sha256_ok", False),
                    "size_ok": vr.get("size_ok", False),
                    "modes": {},
                    "status": "fail",
                    "reason": vr.get("reason", precondition_reason) if vr.get("reason") else precondition_reason,
                }
            )
        # If precondition due to missing harness, fixtures_evidence may already have entries; keep reason
        if not fixtures_evidence and precondition_reason.startswith("missing"):
            # still report harness missing
            pass
        evidence = {
            "schema": "hipfire.hw-gate.evidence",
            "version": 1,
            "verdict": "fail",
            "base": args.base,
            "head": args.head,
            "buckets": buckets,
            "host": host_info,
            "binaries": {"daemon_md5": None, "hipfire_md5": None, "build_seconds": 0.0},
            "fixtures": fixtures_evidence,
            "kernel": None,
            "logs_dir": "hw-gate-logs",
            "precondition_error": precondition_reason,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_md(evidence), encoding="utf-8")
        print(f"precondition failed: {precondition_reason}", file=sys.stderr)
        return 2

    # Build
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_path = bin_dir / "daemon"
    hipfire_path = bin_dir / "hipfire"
    build_seconds = 0.0
    daemon_md5 = None
    hipfire_md5 = None
    if not args.skip_build:
        start_build = time.time()
        try:
            res = run_cmd(["cargo", "build", "--release"], cwd=str(repo_p), capture_output=True, text=True, timeout=1200)
            build_seconds = time.time() - start_build
            if res.returncode != 0:
                evidence = {
                    "schema": "hipfire.hw-gate.evidence",
                    "version": 1,
                    "verdict": "fail",
                    "base": args.base,
                    "head": args.head,
                    "buckets": buckets,
                    "host": host_info,
                    "binaries": {"daemon_md5": None, "hipfire_md5": None, "build_seconds": build_seconds},
                    "fixtures": [],
                    "kernel": None,
                    "logs_dir": "hw-gate-logs",
                    "build_error": (res.stderr or "")[-2000:],
                }
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
                md_path.write_text(render_md(evidence), encoding="utf-8")
                print(f"build failed: {res.stderr}", file=sys.stderr)
                return 2
        except Exception as e:
            build_seconds = time.time() - start_build
            evidence = {
                "schema": "hipfire.hw-gate.evidence",
                "version": 1,
                "verdict": "fail",
                "base": args.base,
                "head": args.head,
                "buckets": buckets,
                "host": host_info,
                "binaries": {"daemon_md5": None, "hipfire_md5": None, "build_seconds": build_seconds},
                "fixtures": [],
                "kernel": None,
                "logs_dir": "hw-gate-logs",
                "build_error": str(e),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            md_path.write_text(render_md(evidence), encoding="utf-8")
            print(f"build exception: {e}", file=sys.stderr)
            return 2
        daemon_md5 = _md5_file(daemon_path)
        hipfire_md5 = _md5_file(hipfire_path)
    else:
        daemon_md5 = _md5_file(daemon_path)
        hipfire_md5 = _md5_file(hipfire_path)
        build_seconds = 0.0

    # Isolated home
    try:
        host_info["config_toml"] = write_isolated_home(str(hipfire_home_p), args.device, str(models_dir))
    except Exception as e:
        print(f"write_isolated_home failed: {e}", file=sys.stderr)
        evidence = {
            "schema": "hipfire.hw-gate.evidence",
            "version": 1,
            "verdict": "fail",
            "base": args.base,
            "head": args.head,
            "buckets": buckets,
            "host": host_info,
            "binaries": {"daemon_md5": daemon_md5, "hipfire_md5": hipfire_md5, "build_seconds": build_seconds},
            "fixtures": [],
            "kernel": None,
            "logs_dir": "hw-gate-logs",
            "error": f"isolated home failed: {e}",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        md_path.write_text(render_md(evidence), encoding="utf-8")
        return 2

    # Run buckets
    env_base = dict(os.environ)
    env_base["HIPFIRE_HOME"] = str(hipfire_home_p)
    env_base["HIPFIRE_MODELS_DIR"] = str(models_dir)

    fixtures_evidence: list[dict] = []
    overall_pass = True

    if modes_union:
        for fx in fixtures_to_verify:
            vr = next((v for v in verify_results if v.get("tag") == fx.get("tag")), None)
            res = _run_fixture_harness(str(repo_p), fx, env_base, str(logs_dir), args.device, modes_union, harness_cfg, str(models_dir))
            if vr is not None:
                res["sha256_ok"] = vr.get("sha256_ok", False)
                res["size_ok"] = vr.get("size_ok", False)
                res["sha256"] = fx.get("sha256", "")
                if not vr.get("sha256_ok") or not vr.get("size_ok") or not vr.get("exists"):
                    res["status"] = "fail"
                    vr_reason = vr.get("reason", "")
                    if vr_reason:
                        res["reason"] = vr_reason + ("; " + res["reason"] if res["reason"] else "")
            fixtures_evidence.append(res)
            if res.get("status") != "pass":
                overall_pass = False
    else:
        fixtures_evidence = []

    kernel_result = None
    if "kernel" in buckets:
        kernel_bucket_cfg = fixtures_manifest.get("buckets", {}).get("kernel", {}) if isinstance(fixtures_manifest.get("buckets"), dict) else {}
        redline_cfg = kernel_bucket_cfg.get("redline", kernel_bucket_cfg) if isinstance(kernel_bucket_cfg.get("redline"), dict) else kernel_bucket_cfg
        if not isinstance(redline_cfg, dict):
            redline_cfg = {}
        kernel_result = run_kernel(str(repo_p), str(models_dir), redline_cfg, fixtures_manifest, env_base, str(logs_dir))
        if kernel_result.get("status") != "pass":
            overall_pass = False

    verdict = "pass" if overall_pass else "fail"
    evidence = {
        "schema": "hipfire.hw-gate.evidence",
        "version": 1,
        "verdict": verdict,
        "base": args.base,
        "head": args.head,
        "buckets": buckets,
        "host": host_info,
        "binaries": {"daemon_md5": daemon_md5, "hipfire_md5": hipfire_md5, "build_seconds": float(build_seconds)},
        "fixtures": fixtures_evidence,
        "kernel": kernel_result,
        "logs_dir": "hw-gate-logs",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_md(evidence), encoding="utf-8")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
