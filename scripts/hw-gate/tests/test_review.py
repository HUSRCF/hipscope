# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# Load review module without installing
REVIEW_PATH = Path(__file__).parent.parent / "review.py"
spec = importlib.util.spec_from_file_location("review", REVIEW_PATH)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

REPO_ROOT = REVIEW_PATH.parent.parent.parent  # hipfire-hw-gate root? actually scripts/hw-gate is two levels below repo
# Resolve checkout for git tests
WORKTREE = Path(__file__).resolve().parents[3]  # may be repo root
FAKE_OMP = Path(__file__).parent / "fake_omp.py"
FAKE_GH = Path(__file__).parent / "fake_gh.py"


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    assert review.extract_json('{"a":1}') == {"a": 1}

def test_extract_json_fenced():
    text = 'hello\n```json\n{"phase":"prelim","x":1}\n```\nworld'
    assert review.extract_json(text) == {"phase": "prelim", "x": 1}

def test_extract_json_fenced_no_lang():
    text = '```\n{"a": 2}\n```'
    assert review.extract_json(text) == {"a": 2}

def test_extract_json_prose_around():
    text = 'Sure, here is the object: {"phase":"verdict","decision":"block"} hope that helps'
    assert review.extract_json(text) == {"phase": "verdict", "decision": "block"}

def test_extract_json_nested():
    inner = {"a": {"b": [1, 2]}, "c": 3}
    text = f"prefix {json.dumps(inner)} suffix"
    assert review.extract_json(text) == inner

def test_extract_json_last_wins():
    text = '{"first":1} some text {"second":2}'
    assert review.extract_json(text) == {"second": 2}

def test_extract_json_with_string_braces():
    text = '{"msg":"hello { not json }","ok":true}'
    assert review.extract_json(text) == {"msg": "hello { not json }", "ok": True}

def test_extract_json_none():
    assert review.extract_json("no json here") is None
    assert review.extract_json("") is None

def test_extract_json_fenced_with_prose_and_extra():
    text = textwrap.dedent("""\
        I analyzed the diff.
        ```json
        {"phase":"prelim","summary":"foo"}
        ```
        That is the result.
        """)
    assert review.extract_json(text) == {"phase": "prelim", "summary": "foo"}

def test_extract_json_balanced_ignores_unbalanced():
    # Only balanced object should be returned; trailing incomplete should be ignored
    text = '{"a":1} {"b":2'
    assert review.extract_json(text) == {"a": 1}


# ---------------------------------------------------------------------------
# apply_floor
# ---------------------------------------------------------------------------

def _base_select(buckets=None, policy_paths=None, surfaces=None):
    return {
        "buckets": buckets or ["load"],
        "policy_paths": policy_paths or [],
        "surfaces": surfaces or {},
    }

def _base_evidence(verdict="pass", kernel_status="pass", kernel_present=True):
    ev = {"verdict": verdict}
    if kernel_present:
        ev["kernel"] = {"status": kernel_status}
    else:
        ev["kernel"] = None
    return ev

def _base_verdict(decision="greenlight", gaps=None, confidence=0.9):
    return {
        "decision": decision,
        "confidence": confidence,
        "coverage": {"gaps": gaps or [], "surfaces_touched": ["load"], "surfaces_evidenced": ["load"]},
    }


def test_floor_each_block_rule_fires_alone():
    # hw_run_result != success
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "failure", [], _base_verdict())
    assert d == "block" and "hw_run_result" in r

    # evidence None
    d, r = review.apply_floor("greenlight", None, _base_select(), "success", [], _base_verdict())
    assert d == "block" and "evidence_verdict" in r

    # evidence verdict != pass
    d, r = review.apply_floor("greenlight", _base_evidence(verdict="fail"), _base_select(), "success", [], _base_verdict())
    assert d == "block" and "evidence_verdict" in r

    # kernel status fail
    d, r = review.apply_floor("greenlight", _base_evidence(kernel_status="fail"), _base_select(buckets=["kernel"]), "success", [], _base_verdict())
    assert d == "block" and "kernel_status" in r

    # kernel missing when kernel bucket
    d, r = review.apply_floor("greenlight", _base_evidence(kernel_present=False), _base_select(buckets=["kernel"]), "success", [], _base_verdict())
    assert d == "block" and "kernel_status" in r

    # model_block
    d, r = review.apply_floor("block", _base_evidence(), _base_select(), "success", [], _base_verdict(decision="block"))
    assert d == "block" and "model_block" in r


def test_floor_each_needs_human_rule_fires_alone():
    # policy_paths
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(policy_paths=["scripts/hw-gate/review.py"]), "success", [], _base_verdict())
    assert d == "needs-human" and "policy_paths" in r

    # ratchet raise
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", ["RATCHET-RAISE: foo"], _base_verdict())
    assert d == "needs-human" and "ratchet_raise" in r

    # ratchet not matching without colon
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", ["RATCHET-RAISE foo"], _base_verdict())
    assert d == "greenlight"

    # coverage gaps
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], _base_verdict(gaps=["load"]))
    assert d == "needs-human" and "coverage_gaps" in r

    # confidence
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], _base_verdict(confidence=0.79))
    assert d == "needs-human" and "confidence" in r
    # 0.8 exactly is ok
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], _base_verdict(confidence=0.8))
    assert d == "greenlight"
    # missing / non-numeric / boolean confidence never greenlights
    for bad in (None, "high", True):
        v = _base_verdict()
        if bad is None:
            del v["confidence"]
        else:
            v["confidence"] = bad
        d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], v)
        assert d == "needs-human" and "confidence" in r, bad

    # model needs-human
    d, r = review.apply_floor("needs-human", _base_evidence(), _base_select(), "success", [], _base_verdict(decision="needs-human"))
    assert d == "needs-human" and "model_needs_human" in r

    # verdict parse failed (None)
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], None)
    assert d == "needs-human" and "verdict_parse_failed" in r


def test_floor_greenlight_only_when_all_clear():
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(), "success", [], _base_verdict())
    assert d == "greenlight" and r == []


def test_floor_strictest_wins_in_combination():
    # block beats needs-human
    d, r = review.apply_floor("needs-human", _base_evidence(verdict="fail"), _base_select(policy_paths=["x"]), "failure", ["RATCHET-RAISE: hi"], _base_verdict(gaps=["x"], confidence=0.5))
    assert d == "block"
    assert "hw_run_result" in r or "evidence_verdict" in r

    # needs-human beats greenlight
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(policy_paths=["x"]), "success", [], _base_verdict())
    assert d == "needs-human"

    # model block overrides other greenlight conditions
    d, r = review.apply_floor("block", _base_evidence(), _base_select(), "success", [], _base_verdict(decision="block", confidence=0.9))
    assert d == "block"

    # multiple needs-human reasons all reported? At least the first should be there, but we check both listed
    d, r = review.apply_floor("greenlight", _base_evidence(), _base_select(policy_paths=["x"]), "success", ["RATCHET-RAISE: y"], _base_verdict(gaps=["a"], confidence=0.5))
    assert d == "needs-human"
    # Should contain policy_paths, ratchet_raise, coverage_gaps, confidence
    for expected in ["policy_paths", "ratchet_raise", "coverage_gaps", "confidence"]:
        assert expected in r, f"missing {expected} in {r}"


# ---------------------------------------------------------------------------
# helpers for e2e
# ---------------------------------------------------------------------------

def _make_repo(tmp: Path, files: dict | None = None) -> Path:
    """Init a git repo with base commit and optional head commit."""
    repo = tmp / "checkout"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    # base commit
    (repo / "base.txt").write_text("base\n")
    if files:
        for name, content in files.items():
            p = repo / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "base"], cwd=repo, check=True, capture_output=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    # head commit
    (repo / "base.txt").write_text("head\n")
    (repo / "feature.txt").write_text("new\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "feature"], cwd=repo, check=True, capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    return repo, base, head

def _write_fixture(tmp: Path, name: str, content: dict):
    p = tmp / name
    p.write_text(json.dumps(content))
    return p


# ---------------------------------------------------------------------------
# e2e main run
# ---------------------------------------------------------------------------

def _run_review(tmp: Path, verdict_decision: str = "greenlight", evidence_verdict: str = "pass", hw_run_result: str = "success",
                extra_select: dict | None = None, evidence_md_exists: bool = True,
                fake_responses: list[dict] | None = None):
    checkout, base, head = _make_repo(tmp / f"repo_{verdict_decision}_{os.getpid()}_{id(tmp)}")

    select = {
        "schema": "hipfire.hw-gate.select",
        "version": 1,
        "needs_hw": True,
        "buckets": ["load"],
        "policy_paths": [],
        "surfaces": {"load": ["crates/hipfire-loader/src/foo.rs"], "serve": [], "kernel": [], "policy": [], "other": ["feature.txt"]},
    }
    if extra_select:
        select.update(extra_select)
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))

    evidence = {
        "schema": "hipfire.hw-gate.evidence",
        "version": 1,
        "verdict": evidence_verdict,
        "base": base,
        "head": head,
        "buckets": ["load"],
        "fixtures": [],
        "kernel": None,
        "serve": None,
    }
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))

    if evidence_md_exists:
        md_path = tmp / "hw-gate.md"
        md_path.write_text("# Evidence\n\nPer-fixture table\n")
    else:
        # ensure not exists
        md_path = tmp / "hw-gate.md"
        if md_path.exists():
            md_path.unlink()

    system_prompt = tmp / "review.md"
    system_prompt.write_text("system prompt stub")

    # Prepare fake omp responses
    if fake_responses is None:
        fake_responses = [
            {"json": {"phase": "prelim", "summary": "summary", "surfaces": ["load"], "suspected_regressions": [], "extra_routes": [], "questions_for_author": []}},
            {"json": {"phase": "verdict", "decision": verdict_decision, "confidence": 0.9, "regressions": [], "coverage": {"surfaces_touched": ["load"], "surfaces_evidenced": ["load"], "gaps": []}, "eyeball": [], "rationale": "ok"}},
        ]
    responses_file = tmp / "omp_responses.json"
    responses_file.write_text(json.dumps(fake_responses))

    call_count = tmp / "omp_count"
    if call_count.exists():
        call_count.unlink()
    gh_log = tmp / "gh_log.jsonl"
    if gh_log.exists():
        gh_log.unlink()
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    omp_log = tmp / "omp_log.jsonl"
    if omp_log.exists():
        omp_log.unlink()

    out_path = tmp / "verdict.json"

    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(responses_file)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_OMP_LOG"] = str(omp_log)
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    env.pop("FAKE_OMP_GARBAGE", None)

    cmd = [
        sys.executable, str(REVIEW_PATH),
        "--repo", "o/r",
        "--pr", "1",
        "--base", base,
        "--head", head,
        "--checkout", str(checkout),
        "--evidence", str(evidence_path),
        "--select", str(select_path),
        "--hw-run-result", hw_run_result,
        "--system-prompt", str(system_prompt),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "out_path": out_path,
        "gh_log": gh_log,
        "gh_comments": gh_comments,
        "omp_log": omp_log,
        "call_count": call_count,
        "checkout": checkout,
        "base": base,
        "head": head,
    }


def test_e2e_greenlight(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp, verdict_decision="greenlight")
    assert res["returncode"] == 0, f"stderr: {res['stderr']}"
    data = json.loads(res["out_path"].read_text())
    assert data["floor"]["final_decision"] == "greenlight"
    assert data["model"]  # truthy
    # Posted markers
    comments = json.loads(res["gh_comments"].read_text())
    bodies = [c["body"] for c in comments]
    assert any("<!-- hw-gate:prelim -->" in b for b in bodies)
    assert any("<!-- hw-gate:evidence -->" in b for b in bodies)
    assert any("<!-- hw-gate:verdict -->" in b for b in bodies)
    # Review kind: greenlight should be --approve
    gh_calls = [json.loads(l)["args"] for l in res["gh_log"].read_text().splitlines() if l.strip()]
    # Find pr review call
    review_calls = [a for a in gh_calls if a[:2] == ["pr", "review"]]
    assert len(review_calls) >= 1
    last_review = review_calls[-1]
    assert "--approve" in last_review
    # hw-run removed
    assert any("labels/hw-run" in " ".join(a) and "--method" in a and "DELETE" in a for a in gh_calls)
    # labels added/removed
    assert "agent-approved" in data["posted"]["labels_added"]
    # Check omp command built correctly: read omp_log
    omp_calls = [json.loads(l)["args"] for l in res["omp_log"].read_text().splitlines() if l.strip()]
    assert len(omp_calls) >= 2
    for oc in omp_calls:
        assert "-p" in oc and "--mode" in oc and "json" in oc
        assert "--auto-approve" in oc
        assert "--tools=read,grep,glob" in oc
        assert "--max-time" in oc and "15m" in oc
        assert "--system-prompt" in oc

def test_e2e_needs_human(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp, verdict_decision="needs-human")
    assert res["returncode"] == 0, f"stderr: {res['stderr']}"
    data = json.loads(res["out_path"].read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    gh_calls = [json.loads(l)["args"] for l in res["gh_log"].read_text().splitlines() if l.strip()]
    review_calls = [a for a in gh_calls if a[:2] == ["pr", "review"]]
    assert "--comment" in review_calls[-1]
    assert "--approve" not in review_calls[-1]

def test_e2e_block(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp, verdict_decision="block")
    # Even though model says block, floor should keep block
    assert res["returncode"] == 0, f"stderr: {res['stderr']}"
    data = json.loads(res["out_path"].read_text())
    assert data["floor"]["final_decision"] == "block"
    gh_calls = [json.loads(l)["args"] for l in res["gh_log"].read_text().splitlines() if l.strip()]
    review_calls = [a for a in gh_calls if a[:2] == ["pr", "review"]]
    assert "--request-changes" in review_calls[-1]

def test_e2e_floor_overrides_greenlight_on_evidence_fail(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp, verdict_decision="greenlight", evidence_verdict="fail")
    assert res["returncode"] == 0, f"stderr: {res['stderr']}"
    data = json.loads(res["out_path"].read_text())
    assert data["floor"]["final_decision"] == "block"

def test_e2e_missing_evidence(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp, verdict_decision="greenlight", evidence_md_exists=False)
    assert res["returncode"] == 0, f"stderr: {res['stderr']}"
    comments = json.loads(res["gh_comments"].read_text())
    bodies = [c["body"] for c in comments]
    # evidence comment should mention missing
    ev_bodies = [b for b in bodies if "<!-- hw-gate:evidence -->" in b]
    assert len(ev_bodies) >= 1
    assert "No hardware evidence found" in ev_bodies[0]

def test_e2e_upsert_existing_comment(tmp_path=None):
    """Second run should PATCH instead of POST when marker already present."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    # First run
    res1 = _run_review(tmp, verdict_decision="greenlight")
    assert res1["returncode"] == 0
    # Reuse same gh_comments file but reset call count and run again with same repo
    # We need to keep comments file, but reset evidence etc? Simplify: run review again with same tmp's comments persisting
    # Use same checkout/base/head/select but new verdict decision
    # For this test, we simulate second run by reusing same gh_comments file and checking PATCH
    responses_file = tmp / "omp_responses2.json"
    responses_file.write_text(json.dumps([
        {"json": {"phase": "prelim", "summary": "s2", "surfaces": ["load"], "suspected_regressions": [], "extra_routes": [], "questions_for_author": []}},
        {"json": {"phase": "verdict", "decision": "greenlight", "confidence": 0.9, "regressions": [], "coverage": {"surfaces_touched": ["load"], "surfaces_evidenced": ["load"], "gaps": []}, "eyeball": [], "rationale": "ok2"}},
    ]))
    call_count = tmp / "omp_count"
    call_count.write_text("0")
    gh_log = tmp / "gh_log.jsonl"
    gh_log.write_text("")
    # Need to re-resolve repo paths: use same checkout as before, but we need base/head from last run
    checkout = res1["checkout"]
    base = res1["base"]
    head = res1["head"]
    select_path = tmp / "select.json"
    evidence_path = tmp / "hw-gate.json"
    system_prompt = tmp / "review.md"
    out_path = tmp / "verdict2.json"
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_RESPONSES"] = str(responses_file)
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_OMP_LOG"] = str(tmp / "omp_log2.jsonl")
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(tmp / "gh_comments.json")
    cmd = [
        sys.executable, str(REVIEW_PATH),
        "--repo", "o/r",
        "--pr", "1",
        "--base", base,
        "--head", head,
        "--checkout", str(checkout),
        "--evidence", str(evidence_path),
        "--select", str(select_path),
        "--hw-run-result", "success",
        "--system-prompt", str(system_prompt),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    gh_calls = [json.loads(l)["args"] for l in gh_log.read_text().splitlines() if l.strip()]
    # Should have seen PATCH for at least one marker
    assert any("--method" in a and "PATCH" in a for a in gh_calls)

def test_failure_path_omp_garbage_twice(tmp_path=None):
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    checkout, base, head = _make_repo(tmp / "repo_fail")
    select = {"schema": "hipfire.hw-gate.select","version":1,"needs_hw":True,"buckets":["load"],"policy_paths":[],"surfaces":{"load":[],"serve":[],"kernel":[],"policy":[],"other":[]}}
    select_path = tmp / "select.json"
    select_path.write_text(json.dumps(select))
    evidence = {"schema":"hipfire.hw-gate.evidence","version":1,"verdict":"pass","base":base,"head":head,"buckets":[],"fixtures":[],"kernel":None,"serve":None}
    evidence_path = tmp / "hw-gate.json"
    evidence_path.write_text(json.dumps(evidence))
    (tmp / "hw-gate.md").write_text("# Evidence\n")
    system_prompt = tmp / "review.md"
    system_prompt.write_text("stub")
    call_count = tmp / "omp_count"
    gh_log = tmp / "gh_log.jsonl"
    gh_comments = tmp / "gh_comments.json"
    gh_comments.write_text("[]")
    out_path = tmp / "verdict.json"
    env = os.environ.copy()
    env["HW_GATE_OMP_BIN"] = str(FAKE_OMP)
    env["HW_GATE_GH_BIN"] = str(FAKE_GH)
    env["FAKE_OMP_GARBAGE"] = "1"
    env["FAKE_OMP_CALL_COUNT"] = str(call_count)
    env["FAKE_GH_LOG"] = str(gh_log)
    env["FAKE_GH_COMMENTS"] = str(gh_comments)
    cmd = [
        sys.executable, str(REVIEW_PATH),
        "--repo", "o/r",
        "--pr", "1",
        "--base", base,
        "--head", head,
        "--checkout", str(checkout),
        "--evidence", str(evidence_path),
        "--select", str(select_path),
        "--hw-run-result", "success",
        "--system-prompt", str(system_prompt),
        "--out", str(out_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    assert result.returncode == 1, f"expected exit 1 got {result.returncode} stderr {result.stderr}"
    data = json.loads(out_path.read_text())
    assert data["floor"]["final_decision"] == "needs-human"
    comments = json.loads(gh_comments.read_text())
    bodies = [c["body"] for c in comments]
    # Should have posted needs-human verdict
    assert any("needs-human" in b or "review failed" in b for b in bodies)


# ---------------------------------------------------------------------------
# omp command line reporting
# ---------------------------------------------------------------------------

def test_omp_command_line():
    # Verify that omp_review builds expected command shape
    # We inspect via fake_omp log in an e2e run
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    res = _run_review(tmp)
    omp_calls = [json.loads(l)["args"] for l in res["omp_log"].read_text().splitlines() if l.strip()]
    # Verbatim expected prefix: omp -p --mode json --auto-approve --tools=read,grep,glob --cwd <checkout> --model <model> --system-prompt <review.md> --max-time 15m "<prompt>"
    # Check structure
    for oc in omp_calls:
        # oc is args list excluding binary (fake_omp logs sys.argv[1:])
        assert "-p" in oc
        idx = oc.index("-p")
        assert oc[idx+1] == "--mode" or oc[idx+1] == "--mode"  # actually -p --mode json
        # Provide the verbatim expected command shape for report
        print("OMP command:", " ".join(oc))
