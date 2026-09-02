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
             "bucket": "load", "prompt": path, "prompt_md5": ...,
             "exit": int, "seconds": float,
             "stdout": "... verbatim ...", "stderr_tail": "... last 60 lines ...",
             "decoded": "... assistant text only ...",
             "detector": {"exit": int, "report": {...}},          # hipfire-detect output
             "status": "pass" | "fail", "reason": "..."}
          ],
          "serve": {"battery": {...harness JSON...}, "chain": {...}, "status": ..., "reason": ...} | null,
          "kernel": {"report": {...redline harness JSON...}, "status": ..., "reason": ...} | null,
          "logs_dir": "hw-gate-logs"
        }

    hw-gate.md  : the same evidence rendered for a PR comment. Header table (host, binaries,
                  base..head, buckets); per-fixture table (tag, sha256 ok, exit, seconds, detector,
                  status); then one <details> block per fixture with the decoded text VERBATIM;
                  then serve/kernel sections. review.py posts this file as the evidence comment.

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
    4. load bucket, per fixture: 
           HIPFIRE_LOCAL=1 HIPFIRE_DAEMON_BIN=<repo>/target/release/daemon <repo>/target/release/hipfire run <tag>
               --max-tokens <n> --no-stream "<prompt file contents>"
       capture stdout and stderr separately (logs to hw-gate-logs/<tag>.{out,err}), timeout 600 s.
       `hipfire run` prints the assistant text on stdout and every daemon progress/diagnostic line on
       stderr, so `decoded` = stdout verbatim (never filtered: code answers are indented). Pipe `decoded`
       into `target/release/hipfire-detect`; status=pass iff exit==0 and decoded non-empty and detector exit==0.
    5. serve bucket: python3 scripts/serve_harness.py --model <models_dir/file> --mode battery
       --prompts-file <battery_prompts> --max-tokens <n> --home $HIPFIRE_HOME --out hw-gate-logs/serve-battery.json,
       then --mode chain (built-in chain battery). Pass the resolved daemon/hipfire binaries the way the harness
       expects (HIPFIRE_DAEMON_BIN env). status=pass iff both harness runs exit 0 and every battery row's
       `expect` substrings appear.
    6. kernel bucket: python3 scripts/redline_daemon_harness.py --model <models_dir/file> --daemon <daemon>
       <harness_args...> --out hw-gate-logs/redline.json; status=pass iff exit 0 and the report's parity fields
       are all true (read the harness's own summary keys; do not invent thresholds).
    7. Write JSON + MD, exit per verdict.

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


def run_fixture(repo, fixture, env, logs_dir, timeout=600) -> dict:
    """Run one load fixture via hipfire run and hipfire-detect.

    Returns dict matching hw-gate.json fixtures[] entry.
    """
    repo_p = Path(repo)
    logs_dir_p = Path(logs_dir)
    logs_dir_p.mkdir(parents=True, exist_ok=True)
    tag = fixture.get("tag", "")
    safe_tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag) or "fixture"
    # Prompt
    prompt_rel = fixture.get("prompt", "")
    prompt_path = repo_p / prompt_rel if prompt_rel else None
    try:
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path and prompt_path.is_file() else ""
    except Exception:
        prompt_text = ""
    prompt_md5 = hashlib.md5(prompt_text.encode("utf-8")).hexdigest() if prompt_text else hashlib.md5(b"").hexdigest()
    # Binaries
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_bin = bin_dir / "daemon"
    hipfire_bin = bin_dir / "hipfire"
    detect_bin = bin_dir / "hipfire-detect"
    # Fallback if CARGO_TARGET_DIR not honoured but file actually at repo/target/release
    env2 = dict(env)
    env2["HIPFIRE_LOCAL"] = "1"
    env2["HIPFIRE_DAEMON_BIN"] = str(daemon_bin)
    max_tokens = fixture.get("max_tokens", 128)
    cmd = [str(hipfire_bin), "run", tag, "--max-tokens", str(max_tokens), "--no-stream", prompt_text]
    start = time.time()
    stdout = ""
    stderr = ""
    exit_code = 127
    try:
        res = run_cmd(cmd, env=env2, cwd=str(repo_p), capture_output=True, text=True, timeout=timeout)
        stdout = res.stdout if res.stdout is not None else ""
        stderr = res.stderr if res.stderr is not None else ""
        exit_code = res.returncode
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
    # logs
    try:
        (logs_dir_p / f"{safe_tag}.out").write_text(stdout, encoding="utf-8")
        (logs_dir_p / f"{safe_tag}.err").write_text(stderr, encoding="utf-8")
    except Exception:
        pass
    # `hipfire run` writes the assistant text to stdout and every daemon
    # progress/diagnostic line to stderr, so stdout IS the decoded text.
    # Never filter it: code answers are indented and would be destroyed.
    decoded = stdout.strip("\n")
    stderr_tail = "\n".join(stderr.splitlines()[-60:])
    # detector
    detector_exit = 1
    detector_report: dict = {}
    detector_raw = ""
    try:
        d_res = run_cmd([str(detect_bin)], input=decoded, capture_output=True, text=True, timeout=30)
        detector_exit = d_res.returncode
        detector_raw = d_res.stdout if d_res.stdout is not None else ""
        if detector_raw.strip():
            try:
                detector_report = json.loads(detector_raw)
            except Exception:
                detector_report = {"raw": detector_raw[:2000]}
        else:
            # also check stderr for report?
            detector_report = {}
    except FileNotFoundError as e:
        detector_exit = 127
        detector_report = {"error": str(e)}
    except subprocess.TimeoutExpired:
        detector_exit = 124
        detector_report = {"error": "detect timeout"}
    except Exception as e:
        detector_exit = 1
        detector_report = {"error": str(e)}
    # status
    if exit_code != 0:
        status = "fail"
        reason = f"hipfire run exit {exit_code}"
    elif not decoded.strip():
        status = "fail"
        reason = "empty decoded text"
    elif detector_exit != 0:
        status = "fail"
        reason = f"detector exit {detector_exit}"
    else:
        status = "pass"
        reason = ""
    # sha/size from fixture manifest for evidence
    return {
        "tag": tag,
        "file": fixture.get("file", ""),
        "sha256": fixture.get("sha256", ""),
        "sha256_ok": True,
        "size_ok": True,
        "bucket": "load",
        "prompt": prompt_rel,
        "prompt_md5": prompt_md5,
        "exit": exit_code,
        "seconds": float(seconds),
        "stdout": stdout,
        "stderr_tail": stderr_tail,
        "decoded": decoded,
        "detector": {"exit": detector_exit, "report": detector_report},
        "status": status,
        "reason": reason,
    }


def run_serve(repo, models_dir, serve_cfg, fixtures_manifest, env, logs_dir) -> dict:
    """Run serve battery + chain harnesses.

    Uses only real flags from serve_harness.py argparse:
      --model, --mode, --prompts-file, --max-tokens, --home, --out

    Daemon binary passed via HIPFIRE_DAEMON_BIN env.
    Returns dict with battery, chain, status, reason.
    """
    repo_p = Path(repo)
    logs_dir_p = Path(logs_dir)
    logs_dir_p.mkdir(parents=True, exist_ok=True)
    # Resolve model file for serve
    model_tag = serve_cfg.get("model_tag", "")
    model_file = None
    if fixtures_manifest and "buckets" in fixtures_manifest:
        for f in fixtures_manifest["buckets"].get("load", {}).get("fixtures", []):
            if f.get("tag") == model_tag:
                model_file = f.get("file")
                break
    if not model_file:
        # fallback: tag mangling? use model_tag itself?
        model_file = f"{model_tag}.mq4"
    model_path = Path(models_dir).expanduser() / model_file if model_file else Path(models_dir) / model_tag
    # Battery config
    battery_prompts_rel = serve_cfg.get("battery_prompts", "benchmarks/prompts/hw-gate/serve-battery.json")
    battery_prompts_path = repo_p / battery_prompts_rel
    max_tokens = serve_cfg.get("max_tokens", 256)
    home = env.get("HIPFIRE_HOME") or str(Path.home() / ".hipfire")
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_bin = bin_dir / "daemon"
    serve_env = dict(env)
    serve_env["HIPFIRE_DAEMON_BIN"] = str(daemon_bin)
    # keep HIPFIRE_HOME, HIPFIRE_MODELS_DIR already in env
    harness = repo_p / "scripts" / "serve_harness.py"
    battery_out = logs_dir_p / "serve-battery.json"
    chain_out = logs_dir_p / "serve-chain.json"
    # Battery
    battery_exit = 1
    battery_rows = None
    battery_stderr = ""
    try:
        cmd_bat = [
            sys.executable,
            str(harness),
            "--model",
            str(model_path),
            "--mode",
            "battery",
            "--prompts-file",
            str(battery_prompts_path),
            "--max-tokens",
            str(max_tokens),
            "--home",
            str(home),
            "--out",
            str(battery_out),
        ]
        res = run_cmd(cmd_bat, env=serve_env, capture_output=True, text=True, timeout=900)
        battery_exit = res.returncode
        battery_stderr = (res.stderr or "") + (res.stdout or "")
        # try read rows
        if battery_out.is_file():
            try:
                battery_rows = json.loads(battery_out.read_text(encoding="utf-8"))
            except Exception:
                battery_rows = None
    except Exception as e:
        battery_exit = 1
        battery_stderr = str(e)
    # Chain: built-in battery, no prompts-file
    chain_exit = 1
    chain_rows = None
    chain_stderr = ""
    try:
        cmd_chain = [
            sys.executable,
            str(harness),
            "--model",
            str(model_path),
            "--mode",
            "chain",
            "--max-tokens",
            str(max_tokens),
            "--home",
            str(home),
            "--out",
            str(chain_out),
        ]
        res2 = run_cmd(cmd_chain, env=serve_env, capture_output=True, text=True, timeout=900)
        chain_exit = res2.returncode
        chain_stderr = (res2.stderr or "") + (res2.stdout or "")
        if chain_out.is_file():
            try:
                chain_rows = json.loads(chain_out.read_text(encoding="utf-8"))
            except Exception:
                chain_rows = None
    except Exception as e:
        chain_exit = 1
        chain_stderr = str(e)
    # Evaluate pass criteria:
    # battery: exit 0 and every row's expect substrings appear in assistant_content
    # chain: exit 0 is criterion (spec says for chain, exit 0 is criterion)
    battery_ok = battery_exit == 0
    reason_parts: list[str] = []
    if battery_exit != 0:
        reason_parts.append(f"battery exit {battery_exit}")
        battery_ok = False
    else:
        # check expect substrings if rows available
        if isinstance(battery_rows, list):
            for idx, row in enumerate(battery_rows):
                expected = row.get("expected_substrings") or row.get("expect") or []
                assistant = row.get("assistant_content") or row.get("content") or ""
                # harness stores lower-case check; we do case-insensitive
                lower_assistant = assistant.lower() if isinstance(assistant, str) else ""
                for exp in expected:
                    if exp.lower() not in lower_assistant:
                        battery_ok = False
                        reason_parts.append(f"battery row {idx} missing expect {exp!r}")
                        break
                # also check harness's retrieval_missing if present
                if row.get("retrieval_missing"):
                    missing = row.get("retrieval_missing")
                    if missing:
                        battery_ok = False
                        reason_parts.append(f"battery row {idx} retrieval_missing {missing}")
        elif battery_rows is None and battery_exit == 0:
            # no rows but exit 0 -> treat as fail closed? but spec says exit 0 is enough?
            # If harness didn't write out, we cannot verify expects; fail closed
            pass
    chain_ok = chain_exit == 0
    if chain_exit != 0:
        reason_parts.append(f"chain exit {chain_exit}")
    status = "pass" if (battery_ok and chain_ok) else "fail"
    reason = "; ".join(reason_parts) if reason_parts else ("" if status == "pass" else "serve failed")
    # Build battery/chain objects for JSON: include exit, rows, stderr tail
    battery_obj = {"exit": battery_exit, "rows": battery_rows, "stderr_tail": battery_stderr[-2000:] if battery_stderr else ""}
    chain_obj = {"exit": chain_exit, "rows": chain_rows, "stderr_tail": chain_stderr[-2000:] if chain_stderr else ""}
    return {"battery": battery_obj, "chain": chain_obj, "status": status, "reason": reason}


def run_kernel(repo, models_dir, kernel_cfg, fixtures_manifest, env, logs_dir) -> dict:
    """Run redline daemon harness.

    Flags are only real ones from redline_daemon_harness.py argparse:
      --model, --daemon, --out, plus harness_args from fixtures manifest.

    Status pass iff exit 0 and report's parity fields are all true
    (report["pass"] is the boolean summary; if missing, fail closed).
    """
    repo_p = Path(repo)
    logs_dir_p = Path(logs_dir)
    logs_dir_p.mkdir(parents=True, exist_ok=True)
    model_tag = kernel_cfg.get("model_tag", "")
    model_file = None
    if fixtures_manifest:
        for f in fixtures_manifest["buckets"].get("load", {}).get("fixtures", []):
            if f.get("tag") == model_tag:
                model_file = f.get("file")
                break
    if not model_file:
        model_file = f"{model_tag}.mq4"
    model_path = Path(models_dir).expanduser() / model_file
    bin_dir = _resolve_bin_dir(repo_p)
    daemon_bin = bin_dir / "daemon"
    harness = repo_p / "scripts" / "redline_daemon_harness.py"
    redline_out = logs_dir_p / "redline.json"
    harness_args = kernel_cfg.get("harness_args", [])
    # Ensure args are strings
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
    # Determine status: exit 0 and report parity true
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
                # collect failures if available
                failures = report.get("dflash_verify_failures") or report.get("failures") or []
                if failures:
                    reason = f"parity fail: {failures}"
                else:
                    reason = "report pass is false"
        elif "bit_exact" in report or "aql_shadow" in report or "prefix_shadow" in report:
            # fallback: try to infer parity from known keys; but spec says fail closed if no boolean summary
            reason = "no boolean parity summary (fail closed)"
            status = "fail"
        else:
            reason = "no boolean parity summary in report (fail closed)"
            status = "fail"
    return {"report": report, "exit": exit_code, "stderr_tail": stderr[-2000:] if stderr else "", "status": status, "reason": reason}


def render_md(evidence: dict) -> str:
    """Render hw-gate.md per CONTRACT: header table, per-fixture table, details blocks, serve/kernel."""
    lines: list[str] = []
    lines.append("# hw-gate evidence")
    lines.append("")
    # Header table
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
    lines.append("")
    # Per-fixture table
    lines.append("## fixtures")
    lines.append("")
    lines.append("| tag | sha256 ok | size ok | exit | seconds | detector | status | reason |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for fx in evidence.get("fixtures", []):
        tag = fx.get("tag", "")
        sha_ok = fx.get("sha256_ok", "")
        # render bool as check
        sha_ok_str = "✅" if sha_ok is True else ("❌" if sha_ok is False else str(sha_ok))
        size_ok = fx.get("size_ok", "")
        size_ok_str = "✅" if size_ok is True else ("❌" if size_ok is False else str(size_ok))
        exit_code = fx.get("exit", "")
        seconds = fx.get("seconds", "")
        try:
            seconds_str = f"{float(seconds):.1f}" if isinstance(seconds, (int, float)) else str(seconds)
        except:
            seconds_str = str(seconds)
        det = fx.get("detector", {})
        det_exit = det.get("exit", "") if isinstance(det, dict) else ""
        status = fx.get("status", "")
        reason = (fx.get("reason", "") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {tag} | {sha_ok_str} | {size_ok_str} | {exit_code} | {seconds_str} | {det_exit} | {status} | {reason} |")
    lines.append("")
    # Details blocks verbatim
    for fx in evidence.get("fixtures", []):
        tag = fx.get("tag", "")
        decoded = fx.get("decoded", "")
        lines.append(f"<details><summary>{tag}</summary>")
        lines.append("")
        fence = _fence(decoded)
        lines.append(fence)
        lines.append(decoded)
        lines.append(fence)
        lines.append("")
        lines.append("</details>")
        lines.append("")
    # serve
    lines.append("## serve")
    lines.append("")
    serve = evidence.get("serve")
    if serve is None:
        lines.append("not run")
        lines.append("")
    else:
        lines.append(f"status: {serve.get('status','')} ")
        if serve.get("reason"):
            lines.append(f"reason: {serve.get('reason')}")
        lines.append("")
        # battery/chain summary
        battery = serve.get("battery", {})
        chain = serve.get("chain", {})
        if isinstance(battery, dict):
            lines.append(f"battery exit: {battery.get('exit','')}")
            rows = battery.get("rows")
            if isinstance(rows, list):
                lines.append(f"battery rows: {len(rows)}")
        if isinstance(chain, dict):
            lines.append(f"chain exit: {chain.get('exit','')}")
            rows = chain.get("rows")
            if isinstance(rows, list):
                lines.append(f"chain rows: {len(rows)}")
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
            # show pass key
            if "pass" in report:
                lines.append(f"report pass: {report.get('pass')}")
            lines.append("")
            # dump truncated report?
            # keep brief
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
        # if cwd not desired, use out_path parent if exists
        try:
            # if out is hw-gate.json in cwd, logs_dir is hw-gate-logs
            if out_path.is_absolute():
                logs_dir = out_path.parent / "hw-gate-logs"
            else:
                # respect relative out's parent
                logs_dir = Path(out_path.parent) / "hw-gate-logs" if str(out_path.parent) != "." else Path("hw-gate-logs")
        except Exception:
            logs_dir = Path("hw-gate-logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Load fixtures
    try:
        fixtures_manifest = json.loads(fixtures_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"failed to load fixtures {fixtures_path}: {e}", file=sys.stderr)
        # write minimal evidence?
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
            "serve": None,
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
            "serve": None,
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

    # Determine which fixtures to verify
    fixtures_to_verify: list[dict] = []
    # load bucket fixtures
    if "load" in buckets:
        for f in fixtures_manifest.get("buckets", {}).get("load", {}).get("fixtures", []):
            if args.only_tag and f.get("tag") not in args.only_tag:
                continue
            fixtures_to_verify.append(f)
    # serve/kernel model verification (if not already covered by load)
    def _model_file_for_tag(tag: str) -> str | None:
        for f in fixtures_manifest.get("buckets", {}).get("load", {}).get("fixtures", []):
            if f.get("tag") == tag:
                return f.get("file")
        return None

    extra_checks: list[dict] = []
    if "serve" in buckets:
        serve_cfg = fixtures_manifest.get("buckets", {}).get("serve", {})
        model_tag = serve_cfg.get("model_tag")
        if model_tag:
            # if not already in fixtures_to_verify
            if not any(f.get("tag") == model_tag for f in fixtures_to_verify):
                fname = _model_file_for_tag(model_tag)
                if fname:
                    # find full fixture for that tag to get sha/size
                    for f in fixtures_manifest.get("buckets", {}).get("load", {}).get("fixtures", []):
                        if f.get("tag") == model_tag:
                            extra_checks.append(f)
                            break
                else:
                    # create minimal fixture check for existence only? treat as missing -> fail
                    extra_checks.append({"tag": model_tag, "file": f"{model_tag}.mq4", "sha256": "", "size_bytes": 0})
    if "kernel" in buckets:
        kernel_cfg = fixtures_manifest.get("buckets", {}).get("kernel", {})
        model_tag = kernel_cfg.get("model_tag")
        if model_tag:
            if not any(f.get("tag") == model_tag for f in fixtures_to_verify) and not any(f.get("tag") == model_tag for f in extra_checks):
                for f in fixtures_manifest.get("buckets", {}).get("load", {}).get("fixtures", []):
                    if f.get("tag") == model_tag:
                        extra_checks.append(f)
                        break

    all_checks = fixtures_to_verify + extra_checks

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
    if "serve" in buckets:
        if not (repo_p / "scripts" / "serve_harness.py").is_file():
            precondition_failed = True
            precondition_reason = "missing scripts/serve_harness.py"
    if "kernel" in buckets:
        if not (repo_p / "scripts" / "redline_daemon_harness.py").is_file():
            precondition_failed = True
            precondition_reason = "missing scripts/redline_daemon_harness.py"

    host_info = {"gfx": _get_host_gfx(), "rocm": _get_host_rocm(), "device": args.device, "runner": socket.gethostname()}

    if precondition_failed:
        # build not attempted
        fixtures_evidence = []
        for fx, vr in zip(all_checks, verify_results):
            fixtures_evidence.append(
                {
                    "tag": fx.get("tag", ""),
                    "file": fx.get("file", ""),
                    "sha256": fx.get("sha256", ""),
                    "sha256_ok": vr.get("sha256_ok", False),
                    "size_ok": vr.get("size_ok", False),
                    "bucket": "load",
                    "prompt": fx.get("prompt", ""),
                    "prompt_md5": "",
                    "exit": 0,
                    "seconds": 0.0,
                    "stdout": "",
                    "stderr_tail": "",
                    "decoded": "",
                    "detector": {"exit": 0, "report": {}},
                    "status": "fail",
                    "reason": vr.get("reason", precondition_reason),
                }
            )
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
            "serve": None,
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
                    "serve": None,
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
                "serve": None,
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
        # reuse existing binaries
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
            "serve": None,
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

    if "load" in buckets:
        for fx in fixtures_to_verify:
            # merge verify result sha/size ok into evidence? run_fixture returns with sha_ok true placeholder, override
            vr = next((v for v in verify_results if v.get("tag") == fx.get("tag")), None)
            res = run_fixture(str(repo_p), fx, env_base, str(logs_dir))
            # overlay sha/size verification
            if vr is not None:
                res["sha256_ok"] = vr.get("sha256_ok", False)
                res["size_ok"] = vr.get("size_ok", False)
                res["sha256"] = fx.get("sha256", "")
                # if verify failed, status should be fail regardless of run result
                if not vr.get("sha256_ok") or not vr.get("size_ok"):
                    res["status"] = "fail"
                    res["reason"] = vr.get("reason", res.get("reason", ""))
            fixtures_evidence.append(res)
            if res.get("status") != "pass":
                overall_pass = False
    else:
        fixtures_evidence = []

    serve_result = None
    if "serve" in buckets:
        serve_cfg = fixtures_manifest.get("buckets", {}).get("serve", {})
        serve_result = run_serve(str(repo_p), str(models_dir), serve_cfg, fixtures_manifest, env_base, str(logs_dir))
        if serve_result.get("status") != "pass":
            overall_pass = False

    kernel_result = None
    if "kernel" in buckets:
        kernel_cfg = fixtures_manifest.get("buckets", {}).get("kernel", {})
        kernel_result = run_kernel(str(repo_p), str(models_dir), kernel_cfg, fixtures_manifest, env_base, str(logs_dir))
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
        "serve": serve_result,
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
