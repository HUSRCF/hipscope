import hashlib
import json
import sys
import tempfile
import os
from pathlib import Path
import subprocess

import pytest

# Import run module via importlib
import importlib.util

RUN_PATH = Path(__file__).resolve().parents[1] / "run.py"
spec = importlib.util.spec_from_file_location("hw_gate_run", str(RUN_PATH))
run_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_mod)

def test_verify_fixture_cache_hit_miss_mismatch(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    cache_path = tmp_path / "home" / "hw-gate-sha.json"
    # create file
    content = b"hello world " * 1000
    file_name = "test.mq4"
    fpath = models_dir / file_name
    fpath.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    size = len(content)
    fixture = {"tag": "test:tag", "file": file_name, "sha256": sha, "size_bytes": size}
    # first call miss (computes)
    res1 = run_mod.verify_fixture(str(models_dir), fixture, str(cache_path))
    assert res1["sha256_ok"] is True
    assert res1["size_ok"] is True
    assert res1["exists"] is True
    assert cache_path.is_file()
    cache1 = json.loads(cache_path.read_text())
    assert len(cache1) == 1
    # second call hit (should not recompute, same result)
    res2 = run_mod.verify_fixture(str(models_dir), fixture, str(cache_path))
    assert res2["sha256_ok"] is True
    assert res2["actual_sha256"] == sha
    cache2 = json.loads(cache_path.read_text())
    assert cache1 == cache2
    # mismatch sha
    fixture_bad_sha = {"tag": "test:tag", "file": file_name, "sha256": "0"*64, "size_bytes": size}
    res3 = run_mod.verify_fixture(str(models_dir), fixture_bad_sha, str(cache_path))
    assert res3["sha256_ok"] is False
    assert res3["size_ok"] is True
    assert "sha256 mismatch" in res3["reason"]
    # mismatch size
    fixture_bad_size = {"tag": "test:tag", "file": file_name, "sha256": sha, "size_bytes": size+1}
    res4 = run_mod.verify_fixture(str(models_dir), fixture_bad_size, str(cache_path))
    assert res4["size_ok"] is False
    assert "size mismatch" in res4["reason"]
    # missing file
    fixture_missing = {"tag": "missing:tag", "file": "nope.mq4", "sha256": sha, "size_bytes": size}
    res5 = run_mod.verify_fixture(str(models_dir), fixture_missing, str(cache_path))
    assert res5["exists"] is False
    assert res5["sha256_ok"] is False
    # cache hit/miss after file modification (mtime change)
    # modify file content with same size but different bytes
    new_content = b"x" * len(content)
    fpath.write_bytes(new_content)
    new_sha = hashlib.sha256(new_content).hexdigest()
    fixture_new = {"tag": "test:tag", "file": file_name, "sha256": new_sha, "size_bytes": size}
    res6 = run_mod.verify_fixture(str(models_dir), fixture_new, str(cache_path))
    assert res6["sha256_ok"] is True
    assert res6["actual_sha256"] == new_sha
    # cache should have grown (new key)
    cache3 = json.loads(cache_path.read_text())
    assert len(cache3) >= 2

def test_run_fixture_keeps_indented_stdout_verbatim(tmp_path, monkeypatch):
    """stdout is the decoded text; indented code must survive untouched and
    daemon progress on stderr must not leak into it."""
    code = "```python\ndef f(s):\n    seen = {}\n    for i, c in enumerate(s):\n        seen[c] = i\n    return seen\n```\n"
    stderr = "GPU dev 0: gfx1201\n  loading layer 1/2...\n[daemon-control] received commit\n"

    class R:
        def __init__(self, rc, out, err=""):
            self.returncode, self.stdout, self.stderr = rc, out, err

    def fake_run_cmd(argv, **kw):
        if argv[0].endswith("hipfire-detect"):
            assert kw["input"] == code.strip("\n")
            return R(0, '{"verdict":"ok"}')
        return R(0, code, stderr)

    monkeypatch.setattr(run_mod, "run_cmd", fake_run_cmd)
    repo = tmp_path / "repo"
    (repo / "target" / "release").mkdir(parents=True)
    for b in ("daemon", "hipfire", "hipfire-detect"):
        (repo / "target" / "release" / b).write_text("")
    prompt = repo / "p.txt"
    prompt.write_text("Write code.")
    monkeypatch.delenv("CARGO_TARGET_DIR", raising=False)
    res = run_mod.run_fixture(str(repo), {"tag": "t:x", "prompt": "p.txt", "max_tokens": 8}, {}, str(tmp_path / "logs"))
    assert res["status"] == "pass", res["reason"]
    assert res["decoded"] == code.strip("\n")
    assert "    seen = {}" in res["decoded"]
    assert "GPU dev" not in res["decoded"]
    assert "GPU dev" in res["stderr_tail"]

def test_render_md_sections(tmp_path):
    evidence = {
        "schema": "hipfire.hw-gate.evidence",
        "version": 1,
        "verdict": "pass",
        "base": "abc123",
        "head": "def456",
        "buckets": ["load", "serve"],
        "host": {"gfx": "gfx1201", "rocm": "6.2", "device": "3", "runner": "hiptrx"},
        "binaries": {"daemon_md5": "d1", "hipfire_md5": "h1", "build_seconds": 42.5},
        "fixtures": [
            {
                "tag": "qwen3.6:27b",
                "file": "qwen3.6-27b.mq4",
                "sha256": "abc",
                "sha256_ok": True,
                "size_ok": True,
                "bucket": "load",
                "prompt": "benchmarks/prompts/hw-gate/load-code.txt",
                "prompt_md5": "p1",
                "exit": 0,
                "seconds": 1.23,
                "stdout": "raw",
                "stderr_tail": "",
                "decoded": "def foo():\n    pass",
                "detector": {"exit": 0, "report": {}},
                "status": "pass",
                "reason": "",
            }
        ],
        "serve": {"battery": {"exit": 0, "rows": []}, "chain": {"exit": 0, "rows": []}, "status": "pass", "reason": ""},
        "kernel": None,
        "logs_dir": "hw-gate-logs",
    }
    md = run_mod.render_md(evidence)
    # Header table fields
    assert "| base |" in md
    assert "abc123" in md
    assert "| head |" in md
    assert "def456" in md
    assert "gfx1201" in md
    assert "hiptrx" in md
    assert "d1" in md
    # per-fixture table
    assert "| tag | sha256 ok" in md.lower() or "| tag | sha256" in md.lower() or "qwen3.6:27b" in md
    assert "qwen3.6:27b" in md
    # details block verbatim
    assert "<details><summary>qwen3.6:27b</summary>" in md
    assert "```" in md
    assert "def foo():" in md
    # serve/kernel sections
    assert "## serve" in md.lower()
    assert "## kernel" in md.lower()
    # ensure decoded inside fenced block verbatim
    assert "def foo():\n    pass" in md

def _make_repo(tmp_path, prompt_content="Write hello"):
    repo = tmp_path / "repo"
    repo.mkdir()
    # create target/release for binaries (dummy)
    bin_dir = repo / "target" / "release"
    bin_dir.mkdir(parents=True)
    (bin_dir / "daemon").write_text("dummy")
    (bin_dir / "hipfire").write_text("dummy")
    (bin_dir / "hipfire-detect").write_text("dummy")
    (bin_dir / "daemon").chmod(0o755)
    (bin_dir / "hipfire").chmod(0o755)
    (bin_dir / "hipfire-detect").chmod(0o755)
    # prompt file
    prompt_rel = "benchmarks/prompts/hw-gate/load-code.txt"
    prompt_path = repo / prompt_rel
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(prompt_content)
    # dummy scripts for precondition checks (not used by run_fixture but main checks)
    scripts = repo / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "serve_harness.py").write_text("# dummy")
    (scripts / "redline_daemon_harness.py").write_text("# dummy")
    return repo, prompt_rel

def _fake_run_factory(mode="pass"):
    # mode: pass, nonzero, empty, detector_fail
    def fake(argv, **kwargs):
        cmd_str = " ".join(argv) if isinstance(argv, list) else str(argv)
        # detect which binary
        if "hipfire-detect" in cmd_str:
            if mode == "detector_fail":
                return subprocess.CompletedProcess(argv, 1, stdout='{"error": "bad"}', stderr="")
            else:
                return subprocess.CompletedProcess(argv, 0, stdout='{"ok": true}', stderr="")
        elif "hipfire" in cmd_str and "run" in cmd_str:
            # Real `hipfire run`: assistant text on stdout, daemon progress on stderr.
            noise = "GPU dev 0: gfx1201\n  loading model\n[daemon-control] noise\n"
            if mode == "nonzero":
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr=noise + "hipfire: daemon error: x\n")
            elif mode == "empty":
                return subprocess.CompletedProcess(argv, 0, stdout="\n", stderr=noise)
            else:  # pass or detector_fail (hipfire part passes)
                return subprocess.CompletedProcess(argv, 0, stdout="Hello decoded world\nSecond line\n", stderr=noise)
        else:
            # for cargo build or other, just succeed
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    return fake

def test_run_fixture_pass(tmp_path):
    repo, prompt_rel = _make_repo(tmp_path, "prompt hello")
    fixture = {"tag": "qwen3.6:27b", "file": "qwen3.6-27b.mq4", "sha256": "abc", "size_bytes": 123, "prompt": prompt_rel, "max_tokens": 16}
    env = {"HIPFIRE_HOME": str(tmp_path / "home"), "HIPFIRE_MODELS_DIR": str(tmp_path / "models")}
    logs_dir = tmp_path / "logs"
    orig = run_mod.run_cmd
    run_mod.run_cmd = _fake_run_factory("pass")
    try:
        res = run_mod.run_fixture(str(repo), fixture, env, str(logs_dir))
    finally:
        run_mod.run_cmd = orig
    assert res["status"] == "pass"
    assert res["exit"] == 0
    assert "Hello decoded world" in res["decoded"]
    assert "GPU dev" not in res["decoded"]
    assert res["detector"]["exit"] == 0
    assert (logs_dir / "qwen3.6-27b.out").is_file()
    assert (logs_dir / "qwen3.6-27b.err").is_file()

def test_run_fixture_nonzero(tmp_path):
    repo, prompt_rel = _make_repo(tmp_path)
    fixture = {"tag": "ornith-1.5:35b-a3b", "file": "ornith.mq4", "sha256": "abc", "size_bytes": 1, "prompt": prompt_rel, "max_tokens": 8}
    env = {}
    logs_dir = tmp_path / "logs2"
    orig = run_mod.run_cmd
    run_mod.run_cmd = _fake_run_factory("nonzero")
    try:
        res = run_mod.run_fixture(str(repo), fixture, env, str(logs_dir))
    finally:
        run_mod.run_cmd = orig
    assert res["status"] == "fail"
    assert res["exit"] != 0
    assert "exit" in res["reason"].lower()

def test_run_fixture_empty_decoded(tmp_path):
    repo, prompt_rel = _make_repo(tmp_path)
    fixture = {"tag": "lfm2.5:1.2b", "file": "lfm.mq4", "sha256": "abc", "size_bytes": 1, "prompt": prompt_rel, "max_tokens": 8}
    env = {}
    logs_dir = tmp_path / "logs3"
    orig = run_mod.run_cmd
    run_mod.run_cmd = _fake_run_factory("empty")
    try:
        res = run_mod.run_fixture(str(repo), fixture, env, str(logs_dir))
    finally:
        run_mod.run_cmd = orig
    assert res["status"] == "fail"
    assert "empty" in res["reason"].lower()
    assert res["decoded"].strip() == ""

def test_run_fixture_detector_fail(tmp_path):
    repo, prompt_rel = _make_repo(tmp_path)
    fixture = {"tag": "qwen3.8:27b-mq4-xt", "file": "qwen.mq4", "sha256": "abc", "size_bytes": 1, "prompt": prompt_rel, "max_tokens": 8}
    env = {}
    logs_dir = tmp_path / "logs4"
    orig = run_mod.run_cmd
    run_mod.run_cmd = _fake_run_factory("detector_fail")
    try:
        res = run_mod.run_fixture(str(repo), fixture, env, str(logs_dir))
    finally:
        run_mod.run_cmd = orig
    assert res["status"] == "fail"
    assert "detector" in res["reason"].lower()

def test_main_exit_2_missing_fixture_no_build(tmp_path, monkeypatch):
    # Setup fixtures manifest with missing file
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    # need repo structure for binary resolution but build won't be attempted
    (repo / "scripts").mkdir()
    (repo / "scripts" / "serve_harness.py").write_text("# ok")
    (repo / "scripts" / "redline_daemon_harness.py").write_text("# ok")
    # create fixtures json with one load fixture that is missing
    fixtures_data = {
        "schema": "hipfire.hw-gate.fixtures",
        "version": 1,
        "models_dir": str(models_dir),
        "buckets": {
            "load": {
                "fixtures": [
                    {"tag": "qwen3.6:27b", "file": "missing.mq4", "sha256": "a"*64, "size_bytes": 123, "prompt": "benchmarks/prompts/hw-gate/load-code.txt", "max_tokens": 16}
                ]
            },
            "serve": {"model_tag": "qwen3.6:27b", "battery_prompts": "benchmarks/prompts/hw-gate/serve-battery.json", "max_tokens": 256},
            "kernel": {"model_tag": "qwen3.6:27b", "harness_args": ["--pm4"]}
        }
    }
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(json.dumps(fixtures_data))
    out = tmp_path / "hw-gate.json"
    md = tmp_path / "hw-gate.md"
    # also need prompt file at repo/benchmarks...
    prompt_path = repo / "benchmarks" / "prompts" / "hw-gate" / "load-code.txt"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text("hello")
    monkeypatch.setenv("HIPFIRE_HOME", str(home))
    monkeypatch.setenv("HIPFIRE_MODELS_DIR", str(models_dir))
    calls = []
    orig = run_mod.run_cmd
    def tracking_run(argv, **kwargs):
        calls.append(argv)
        # if cargo build attempted, fail test
        if argv and argv[0] == "cargo":
            pytest.fail("cargo build should not be attempted on missing fixture")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    run_mod.run_cmd = tracking_run
    try:
        rc = run_mod.main(["--repo", str(repo), "--fixtures", str(fixtures_path), "--base", "abc", "--head", "def", "--buckets", "load", "--device", "3", "--out", str(out), "--md", str(md)])
    finally:
        run_mod.run_cmd = orig
    assert rc == 2
    # ensure no cargo call
    assert all("cargo" not in " ".join(c) if isinstance(c, list) else "cargo" not in str(c) for c in calls)
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["verdict"] == "fail"


def test_render_md_fence_survives_backticks_in_decoded():
    decoded = "```python\nprint(1)\n```\nand ````four````"
    evidence = {"verdict": "pass", "base": "a", "head": "b", "buckets": ["load"], "host": {}, "binaries": {},
                "fixtures": [{"tag": "t", "sha256_ok": True, "size_ok": True, "exit": 0, "seconds": 1.0,
                              "detector": {"exit": 0}, "status": "pass", "reason": "", "decoded": decoded}],
                "serve": None, "kernel": None, "logs_dir": "x"}
    md = run_mod.render_md(evidence)
    fence = run_mod._fence(decoded)
    assert fence == "`````"
    assert f"{fence}\n{decoded}\n{fence}" in md
