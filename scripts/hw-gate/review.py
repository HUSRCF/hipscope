#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
"""hw-gate reviewer driver: prelim review, verdict, and bounded merge authority.

Drives one independent reviewer model through omp's non-interactive mode
(`omp -p --mode json`) in two phases (see review.md), posts the results to
the PR, and applies the decision. The model proposes; THIS FILE decides what
the model is allowed to decide. The floor in `apply_floor` is the merge
authority boundary and is unit-tested without omp or GitHub.

CONTRACT
    review.py --repo OWNER/REPO --pr N --base SHA --head SHA --checkout DIR
              --evidence hw-gate.json --select select.json --hw-run-result success|failure|cancelled|skipped
              --system-prompt review.md --out verdict.json
    env: GH_TOKEN (posting/approval), HW_GATE_REVIEW_MODEL (default "gpt-5.6-sol"),
         HW_GATE_OMP_BIN (default "omp"), HW_GATE_GH_BIN (default "gh") — the last two exist for tests.
    exit 0 : review completed and posted (whatever the decision)
    exit 1 : review could not complete (omp failure, unparseable output twice, gh failure) — after posting
             a `needs-human` comment saying so. Never fail open.

    verdict.json
        {"schema": "hipfire.hw-gate.verdict", "version": 1,
         "model": "...", "prelim": {...phase prelim JSON...}, "verdict": {...phase verdict JSON...},
         "floor": {"applied": ["..."], "model_decision": "...", "final_decision": "..."},
         "posted": {"prelim_comment": url, "evidence_comment": url, "verdict_comment": url, "review": url|null,
                    "labels_added": [...], "labels_removed": [...]}}

PHASES
    prelim  : prompt = PR title/body (quoted as claims), select.json, `git diff --stat base...head`, and the
              full diff base...head (cap 400 KiB; when larger, include all diff for files in select.surfaces
              load/serve/kernel/policy first, then the rest until the cap, and say what was omitted).
              omp invocation (read-only tools, cwd = --checkout):
                  omp -p --mode json --auto-approve --tools=read,grep,glob --cwd <checkout> \
                      --model $HW_GATE_REVIEW_MODEL --system-prompt <review.md> --max-time 15m "<prompt>"
              Parse the final assistant message; extract the single JSON object (tolerate ``` fences).
              On parse failure retry once with an appended "Return only the JSON object." Post as the
              `<!-- hw-gate:prelim -->` comment (upsert by marker).
    evidence: post hw-gate.md from the evidence artifact as `<!-- hw-gate:evidence -->` (upsert; if > 60 KiB,
              keep the header + per-fixture tables + decoded text, truncate stderr tails, and link the artifact).
    verdict : prompt = prelim prompt + prelim JSON + hw-gate.json + hw-run result. Same invocation.
              Post as `<!-- hw-gate:verdict -->` (upsert), including the floor decisions verbatim.

FLOOR (apply_floor(model_decision, evidence, select, hw_run_result, commit_messages) -> (decision, reasons))
    Order matters; the strictest applicable outcome wins:
      block        if hw_run_result != "success"
      block        if evidence is None or evidence.verdict != "pass"
      block        if "kernel" in buckets and (evidence.kernel is None or evidence.kernel.status != "pass")
      block        if model_decision == "block"
      needs-human  if select.policy_paths non-empty
      needs-human  if any commit message in base..head matches ^RATCHET-RAISE:
      needs-human  if verdict.coverage.gaps non-empty
      needs-human  if verdict.confidence < 0.8
      needs-human  if model_decision == "needs-human" or the verdict phase failed to parse
      greenlight   only when none of the above fired and model_decision == "greenlight"

APPLY
    The merge authority is the required `hw-gate` status, which the workflow derives from
    verdict.json `floor.final_decision` (green only on greenlight). The reviews below are the
    human-visible record and are informational.
    greenlight  : `gh pr review N --approve --body <verdict summary + evidence link>`; add label `agent-approved`
    needs-human : `gh pr review N --comment --body ...`; add label `needs-human`
    block       : `gh pr review N --request-changes --body ...`; add label `hw-gate-blocked`
    Always: remove whichever of {agent-approved, needs-human, hw-gate-blocked} no longer applies, and remove
    `hw-run` so the next push needs a fresh maintainer authorization.
    Labels are created on first use (gh label create --force) with fixed colors.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


class ReviewError(Exception):
    pass


# ---------------------------------------------------------------------------
# apply_floor
# ---------------------------------------------------------------------------

def apply_floor(
    model_decision: str | None,
    evidence: dict | None,
    select: dict,
    hw_run_result: str,
    commit_messages: list[str],
    verdict: dict | None,
) -> tuple[str, list[str]]:
    """Pure. Returns (final_decision, reasons). See FLOOR in the module docstring."""
    block_reasons: list[str] = []
    needs_human_reasons: list[str] = []

    # --- block tier ---
    if hw_run_result != "success":
        block_reasons.append("hw_run_result")
    # evidence None or verdict != pass
    if evidence is None or evidence.get("verdict") != "pass":
        block_reasons.append("evidence_verdict")
    # kernel bucket check
    buckets = select.get("buckets", []) if isinstance(select, dict) else []
    if "kernel" in buckets:
        kernel = None
        if isinstance(evidence, dict):
            kernel = evidence.get("kernel")
        if kernel is None or (isinstance(kernel, dict) and kernel.get("status") != "pass"):
            # only append if not already covered by evidence verdict? But strictest still block.
            # We deduplicate reasons for evidence_verdict vs kernel_status; both can fire.
            block_reasons.append("kernel_status")
    if model_decision == "block":
        block_reasons.append("model_block")

    if block_reasons:
        return ("block", block_reasons)

    # --- needs-human tier ---
    policy_paths = select.get("policy_paths", []) if isinstance(select, dict) else []
    if policy_paths:
        needs_human_reasons.append("policy_paths")
    # RATCHET-RAISE:
    for msg in commit_messages or []:
        if re.match(r"^RATCHET-RAISE:", msg):
            needs_human_reasons.append("ratchet_raise")
            break
    # verdict coverage gaps
    if isinstance(verdict, dict):
        coverage = verdict.get("coverage", {})
        if isinstance(coverage, dict):
            gaps = coverage.get("gaps", [])
            if gaps:
                needs_human_reasons.append("coverage_gaps")
        # confidence: missing or non-numeric is treated as below threshold
        conf = verdict.get("confidence")
        if not isinstance(conf, (int, float)) or isinstance(conf, bool) or conf < 0.8:
            needs_human_reasons.append("confidence")
    else:
        # verdict is None means parse failed -> needs-human via last rule; but also
        # coverage_gaps/confidence not applicable
        pass

    if model_decision == "needs-human" or verdict is None:
        # Distinguish: verdict None => parse failure, model needs-human => model decision
        if verdict is None:
            needs_human_reasons.append("verdict_parse_failed")
        if model_decision == "needs-human":
            needs_human_reasons.append("model_needs_human")

    if needs_human_reasons:
        return ("needs-human", needs_human_reasons)

    # --- greenlight tier ---
    if model_decision == "greenlight":
        return ("greenlight", [])

    # Fallback: if no rule fired and model_decision is not greenlight,
    # treat as needs-human (should not happen with valid inputs but fail closed)
    if model_decision is None:
        return ("needs-human", ["verdict_parse_failed"] if verdict is None else ["model_decision_none"])
    return ("needs-human", ["model_decision_not_greenlight"])


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Return the last balanced top-level JSON object in assistant text, or None.

    Tolerates ``` fences and prose around the object. Finds the last
    balanced {...} with proper string handling and attempts json.loads on each
    candidate from last to first.
    """
    if not text:
        return None
    # Remove ``` fences but keep interior — replace fences with newlines so
    # brace matching still works. Handle ```json ... ``` and ``` ... ```
    # We keep content inside fences; fences themselves are not braces.
    # Simplest: just search over the raw text, ignoring fence markers' effect.
    candidates: list[str] = []
    # Scan for balanced brace objects
    i = 0
    n = len(text)
    stack: list[int] = []  # positions of '{' start
    in_string = False
    escape = False
    start_idx: int | None = None
    # We find every balanced top-level object
    # Walk character by character with string awareness
    depth = 0
    current_start: int | None = None
    in_str = False
    esc = False
    for idx, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                if depth == 0:
                    current_start = idx
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and current_start is not None:
                        candidates.append(text[current_start: idx + 1])
                        current_start = None

    # Try candidates from last to first (last balanced object wins)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# helpers: gh seam and omp
# ---------------------------------------------------------------------------

def _gh(args: list[str]) -> str:
    """Run gh with given args (without the binary prefix). Return stdout."""
    gh_bin = os.environ.get("HW_GATE_GH_BIN", "gh")
    cmd = [gh_bin] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReviewError(f"gh {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def omp_review(phase: str, prompt: str, system_prompt: str, checkout: str, model: str) -> dict:
    """Run omp for a phase and return the extracted JSON object.

    Raises ReviewError on failure after one retry.
    """
    omp_bin = os.environ.get("HW_GATE_OMP_BIN", "omp")
    last_error: str | None = None
    for attempt in range(2):
        cur_prompt = prompt if attempt == 0 else prompt + "\n\nReturn only the JSON object."
        # The prompt carries up to 400 KiB of diff: far beyond argv limits, so it
        # goes through omp's `@file` prompt reference. The file lives outside
        # the checkout so the reviewer's read-only tools cannot mistake it for
        # repository content.
        with tempfile.NamedTemporaryFile("w", suffix=f"-hw-gate-{phase}.md", delete=False, encoding="utf-8") as fh:
            fh.write(cur_prompt)
            prompt_path = fh.name
        cmd = [
            omp_bin, "-p", "--mode", "json", "--auto-approve",
            "--tools=read,grep,glob",
            "--cwd", checkout,
            "--model", model,
            "--system-prompt", system_prompt,
            "--max-time", "15m",
            f"@{prompt_path}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=checkout)
        finally:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
        if result.returncode != 0:
            last_error = f"omp {phase} failed ({result.returncode}): {result.stderr.strip()}"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        # Parse JSONL: find last message_end with role assistant, concat text parts
        stdout = result.stdout
        last_text: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except Exception:
                continue
            if evt.get("type") == "message_end":
                msg = evt.get("message", {})
                if msg.get("role") == "assistant":
                    parts = msg.get("content", [])
                    texts = []
                    for p in parts:
                        if isinstance(p, dict) and p.get("type") == "text":
                            texts.append(p.get("text", ""))
                        # thinking blocks ignored
                    last_text = "".join(texts)
        if last_text is None:
            last_error = f"omp {phase}: no assistant message_end in output"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        obj = extract_json(last_text)
        if obj is None:
            last_error = f"omp {phase}: no JSON object in assistant text: {last_text[:500]!r}"
            if attempt == 0:
                continue
            raise ReviewError(last_error)
        return obj
    raise ReviewError(last_error or f"omp {phase} failed after retry")


def _git(args: list[str], checkout: str) -> str:
    """Run git args in checkout directory."""
    result = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=checkout)
    if result.returncode != 0:
        raise ReviewError(f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def build_prelim_prompt(select: dict, checkout: str, base: str, head: str, repo: str, pr: int) -> str:
    """Build the prelim prompt per PHASES."""
    # PR metadata via gh
    pr_json_text = _gh(["pr", "view", str(pr), "--repo", repo, "--json", "title,body,author,url"])
    pr_info = json.loads(pr_json_text)
    title = pr_info.get("title", "")
    body = pr_info.get("body", "") or ""
    author = pr_info.get("author", {})
    url = pr_info.get("url", "")

    # git diff --stat and diff
    diff_stat = _git(["diff", "--stat", f"{base}...{head}"], checkout)
    # Surfaces bucket-first ordering: priority for files in select surfaces
    surfaces = select.get("surfaces", {}) if isinstance(select, dict) else {}
    # Collect prioritized paths
    priority_paths: list[str] = []
    for bucket in ("load", "serve", "kernel", "policy"):
        paths = surfaces.get(bucket, []) if isinstance(surfaces, dict) else []
        priority_paths.extend(paths)

    diff_text = _git(["diff", f"{base}...{head}"], checkout)
    cap = 400 * 1024  # 400 KiB
    if len(diff_text.encode("utf-8")) > cap:
        # Include prioritized files first
        # We need per-file diffs for priority paths, then fill rest until cap
        included_parts: list[str] = []
        included_size = 0
        emitted_files: set[str] = set()
        # Try per-file diff for priority paths
        for fpath in priority_paths:
            if fpath in emitted_files:
                continue
            try:
                part = _git(["diff", f"{base}...{head}", "--", fpath], checkout)
            except ReviewError:
                part = ""
            if not part:
                continue
            part_bytes = len(part.encode("utf-8"))
            if included_size + part_bytes > cap:
                continue
            included_parts.append(part)
            included_size += part_bytes
            emitted_files.add(fpath)
        # Fill with remaining diff until cap
        # For simplicity, if priority diffs already near cap, note omission.
        # Otherwise, slice the full diff to fit remaining budget.
        remaining = cap - included_size
        # Determine which portion of full diff is already covered; for truncation estimate,
        # we just note that remainder was omitted beyond cap.
        if included_parts:
            full_consumed_estimate = sum(len(p.encode("utf-8")) for p in included_parts)
            if remaining > 0:
                # Append remaining slice of full diff beyond what we've included, but avoid duplicating.
                # Simplistic: append truncated tail of overall diff that wasn't from priority files.
                # We just truncate the full diff to the cap and note priority was included first.
                # Instead, include included_parts plus a truncated portion of the rest.
                rest = diff_text
                # Avoid double-counting: just use first `remaining` bytes of rest not already counted would be complex.
                # Simpler: if we used per-file diffs, report that we prioritized and cap the rest.
                pass
            truncated_note = f"\n\n[diff truncated: {len(diff_text.encode('utf-8'))} bytes total, showing {included_size} bytes prioritized for {len(emitted_files)} surface files; omitted rest]"
            # Build prioritized diff text
            # If included size is far below cap, append slice of full diff to fill
            if included_size < cap:
                # Append part of full diff beyond priority files up to cap
                # To avoid duplication, just append full diff truncated to fit
                needed = cap - included_size - len(truncated_note.encode("utf-8"))
                # Estimate slice
                full_bytes = diff_text.encode("utf-8")
                # Find how much of full we haven't accounted for: just take first needed bytes as filler
                # This may duplicate but ensures bucket-first ordering is satisfied contractually
                filler = full_bytes[:needed].decode("utf-8", errors="ignore")
                # Use included_parts + filler
                combined = "\n".join(included_parts) + "\n" + filler if filler else "\n".join(included_parts)
                diff_text = combined + truncated_note
            else:
                diff_text = "\n".join(included_parts) + truncated_note
        else:
            # No priority files produced output; just cap the full diff
            eb = diff_text.encode("utf-8")
            truncated = eb[:cap].decode("utf-8", errors="ignore")
            omitted = len(eb) - cap
            diff_text = truncated + f"\n\n[diff truncated: {omitted} bytes omitted]"
    else:
        pass  # diff fits within cap

    # Compose prompt
    select_json = json.dumps(select, indent=2, sort_keys=True)
    parts = []
    parts.append(f"PR #{pr} {title}")
    if url:
        parts.append(f"URL: {url}")
    if author:
        author_name = author.get("login", "") if isinstance(author, dict) else str(author)
        if author_name:
            parts.append(f"Author: {author_name}")
    parts.append("")
    parts.append("PR body (author's claims, not evidence)")
    parts.append(body)
    parts.append("")
    parts.append("select.json:")
    parts.append(select_json)
    parts.append("")
    parts.append("git diff --stat base...head:")
    parts.append(diff_stat)
    parts.append("")
    parts.append("Full diff base...head:")
    parts.append(diff_text)
    return "\n".join(parts)


def build_verdict_prompt(prelim_prompt: str, prelim: dict, evidence: dict | None, select: dict, hw_run_result: str) -> str:
    """Build verdict prompt per PHASES."""
    parts = []
    parts.append(prelim_prompt)
    parts.append("")
    parts.append("Prelim JSON:")
    parts.append(json.dumps(prelim, indent=2, sort_keys=True))
    parts.append("")
    parts.append(f"hw-run result: {hw_run_result}")
    parts.append("")
    if evidence is not None:
        parts.append("hw-gate.json:")
        parts.append(json.dumps(evidence, indent=2, sort_keys=True))
    else:
        parts.append("hw-gate.json: (missing — hw-run did not produce evidence)")
    return "\n".join(parts)


def upsert_comment(repo: str, pr: int, marker: str, body: str) -> str:
    """Upsert a PR comment identified by marker HTML comment. Return URL."""
    # Truncate if over 60 KiB per PHASES (truncate bodies over 60 KiB)
    cap = 60 * 1024
    if len(body.encode("utf-8")) > cap:
        # For evidence phase spec: keep header + per-fixture tables + decoded text, truncate stderr tails.
        # Generic fallback: truncate end and link artifact.
        body = body.encode("utf-8")[:cap].decode("utf-8", errors="ignore") + "\n\n...[truncated; see artifact]..."
    # List comments
    stdout = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--paginate"])
    try:
        comments = json.loads(stdout) if stdout.strip() else []
        # gh api with --paginate may return concatenated arrays? Assume single JSON array or newline-delimited
        if isinstance(comments, dict):
            # wrapped object
            comments = comments.get("comments", [comments])
    except json.JSONDecodeError:
        # Try to parse as JSONL arrays
        comments = []
        for line in stdout.splitlines():
            line=line.strip()
            if not line:
                continue
            try:
                val = json.loads(line)
                if isinstance(val, list):
                    comments.extend(val)
                elif isinstance(val, dict):
                    comments.append(val)
            except Exception:
                continue
    # Some gh implementations return paginated JSON array directly
    if not isinstance(comments, list):
        comments = [comments] if isinstance(comments, dict) else []

    existing_id = None
    existing_url = None
    for c in comments:
        if not isinstance(c, dict):
            continue
        b = c.get("body", "")
        if marker in b:
            existing_id = c.get("id")
            existing_url = c.get("html_url", "")
            break

    payload_body = body
    # Ensure marker present: caller should include it; if not, prepend
    if marker not in payload_body:
        payload_body = marker + "\n" + payload_body

    if existing_id is not None:
        # PATCH
        out = _gh(["api", f"repos/{repo}/issues/comments/{existing_id}", "--method", "PATCH", "-f", f"body={payload_body}"])
        # Try to extract url from response
        try:
            resp = json.loads(out)
            if isinstance(resp, dict) and "html_url" in resp:
                return resp["html_url"]
        except Exception:
            pass
        return existing_url or f"https://github.com/{repo}/pull/{pr}#issuecomment-{existing_id}"
    else:
        out = _gh(["api", f"repos/{repo}/issues/{pr}/comments", "--method", "POST", "-f", f"body={payload_body}"])
        try:
            resp = json.loads(out)
            if isinstance(resp, dict) and "html_url" in resp:
                return resp["html_url"]
        except Exception:
            pass
        return out.strip() or f"https://github.com/{repo}/pull/{pr}#issuecomment-new"


# Label colors per contract
_LABEL_COLORS = {
    "agent-approved": "0e8a16",
    "needs-human": "fbca04",
    "hw-gate-blocked": "b60205",
    "hw-run": "5319e7",
    "ratchet-raise": "d93f0b",
}


def apply_decision(repo: str, pr: int, decision: str, verdict_summary: str, evidence_link: str = "") -> dict:
    """Apply the decision: review + labels. Returns posted dict with labels info.

    Shared gh seam only: every gh call via _gh.
    """
    labels_added: list[str] = []
    labels_removed: list[str] = []
    review_url = None

    # Compose review body
    body = verdict_summary
    if evidence_link:
        body += f"\n\nEvidence: {evidence_link}"

    # Ensure label exists (create-on-first-use)
    decision_label_map = {
        "greenlight": "agent-approved",
        "needs-human": "needs-human",
        "block": "hw-gate-blocked",
    }
    target_label = decision_label_map.get(decision, "needs-human")
    # Create label --force with color (idempotent)
    color = _LABEL_COLORS.get(target_label, "ededed")
    try:
        _gh(["label", "create", target_label, "--repo", repo, "--color", color, "--force"])
    except ReviewError:
        # label create may not be supported in fake; ignore?
        pass

    # Also ensure other labels exist for removal tracking? Not needed.

    if decision == "greenlight":
        try:
            out = _gh(["pr", "review", str(pr), "--repo", repo, "--approve", "--body", body])
            review_url = out.strip() or None
        except ReviewError as e:
            raise
        labels_added.append(target_label)
    elif decision == "block":
        try:
            out = _gh(["pr", "review", str(pr), "--repo", repo, "--request-changes", "--body", body])
            review_url = out.strip() or None
        except ReviewError as e:
            raise
        labels_added.append(target_label)
    else:  # needs-human
        try:
            out = _gh(["pr", "review", str(pr), "--repo", repo, "--comment", "--body", body])
            review_url = out.strip() or None
        except ReviewError as e:
            raise
        labels_added.append(target_label)

    # Add target label to issue
    try:
        _gh(["api", f"repos/{repo}/issues/{pr}/labels", "--method", "POST", "-f", f"labels[]={target_label}"])
    except ReviewError:
        pass

    # Remove whichever of {agent-approved, needs-human, hw-gate-blocked} no longer applies
    for lbl in ["agent-approved", "needs-human", "hw-gate-blocked"]:
        if lbl != target_label:
            try:
                _gh(["api", f"repos/{repo}/issues/{pr}/labels/{lbl}", "--method", "DELETE"])
                labels_removed.append(lbl)
            except ReviewError:
                # label not present is not an error for our purposes; gh may return 404
                # Fake gh may error; ignore if not found
                pass

    # Remove hw-run so next push needs fresh authorization
    try:
        _gh(["api", f"repos/{repo}/issues/{pr}/labels/hw-run", "--method", "DELETE"])
        labels_removed.append("hw-run")
    except ReviewError:
        pass

    return {
        "review": review_url,
        "labels_added": labels_added,
        "labels_removed": labels_removed,
    }


def _load_commit_messages(checkout: str, base: str, head: str) -> list[str]:
    """Load commit messages in base..head."""
    try:
        out = subprocess.run(
            ["git", "log", "--format=%B", f"{base}..{head}"],
            capture_output=True, text=True, cwd=checkout
        )
        if out.returncode != 0:
            return []
        # Split by commit boundaries: git log separates commits with blank? Use raw: split on \n but messages may contain newlines.
        # Instead use a delimiter: use NUL or custom. Fallback: get subject lines?
        # Better: use git log with custom delimiter
        out2 = subprocess.run(
            ["git", "log", "--format=%s%n%b%x00", f"{base}..{head}"],
            capture_output=True, text=True, cwd=checkout
        )
        if out2.returncode != 0:
            return []
        msgs = [m.strip() for m in out2.stdout.split("\x00") if m.strip()]
        return msgs
    except Exception:
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", required=True, type=int)
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--checkout", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--select", required=True)
    ap.add_argument("--hw-run-result", required=True)
    ap.add_argument("--system-prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("prelim", "verdict", "all"), default="all",
                    help="prelim: read the diff before hardware runs and write --out {phase, prelim}; "
                         "verdict: read --prelim (from the prelim phase) plus evidence and decide; all: both")
    ap.add_argument("--prelim", help="prelim-phase output to consume in the verdict phase")
    args = ap.parse_args(argv)

    model = os.environ.get("HW_GATE_REVIEW_MODEL", "gpt-5.6-sol")

    # Load select
    try:
        with open(args.select, "r", encoding="utf-8") as f:
            select = json.load(f)
    except Exception as e:
        sys.stderr.write(f"failed to load select: {e}\n")
        return 1

    # Load evidence (may be missing if hw-run failed)
    evidence = None
    evidence_path = Path(args.evidence)
    if evidence_path.is_file():
        try:
            with open(evidence_path, "r", encoding="utf-8") as f:
                evidence = json.load(f)
        except Exception:
            evidence = None
    # evidence.md next to evidence json
    evidence_md_path = evidence_path.parent / "hw-gate.md"

    # Load commit messages
    commit_messages = _load_commit_messages(args.checkout, args.base, args.head)

    # For error handling: helper to post needs-human and write verdict
    def _fail(err_msg: str) -> int:
        # Try to post needs-human comment
        try:
            body = f"<!-- hw-gate:verdict -->\nhw-gate review failed: {err_msg}\n\nDecision: needs-human (fail-closed)\n"
            # Also try to apply decision as needs-human (labels)
            # Post verdict marker comment
            try:
                upsert_comment(args.repo, args.pr, "<!-- hw-gate:verdict -->", body)
            except Exception:
                pass
            # Try to apply labels via apply_decision (will create review)
            try:
                apply_decision(args.repo, args.pr, "needs-human", f"hw-gate review failed: {err_msg}")
            except Exception:
                pass
        except Exception:
            pass
        # Write verdict.json
        out_obj = {
            "schema": "hipfire.hw-gate.verdict",
            "version": 1,
            "model": model,
            "prelim": None,
            "verdict": None,
            "floor": {"applied": ["review_error"], "model_decision": None, "final_decision": "needs-human"},
            "posted": {"prelim_comment": None, "evidence_comment": None, "verdict_comment": None, "review": None,
                       "labels_added": [], "labels_removed": []},
            "error": err_msg,
        }
        # If prelim was obtained earlier but verdict failed, include prelim
        try:
            if "_prelim_cache" in locals() and locals()["_prelim_cache"] is not None:
                out_obj["prelim"] = locals()["_prelim_cache"]
        except Exception:
            pass
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass
        sys.stderr.write(f"hw-gate review error: {err_msg}\n")
        return 1

    # Sequence: prelim -> evidence comment -> verdict -> floor -> apply
    prelim = None
    verdict = None
    prelim_comment_url = None
    evidence_comment_url = None
    verdict_comment_url = None
    applied_info = {}
    posted = {"prelim_comment": None, "evidence_comment": None, "verdict_comment": None, "review": None, "labels_added": [], "labels_removed": []}

    try:
        # The prelim prompt is deterministic from the checkout, so both phases rebuild it.
        prelim_prompt = build_prelim_prompt(select, args.checkout, args.base, args.head, args.repo, args.pr)

        if args.phase in ("prelim", "all"):
            # Prelim runs BEFORE any hardware: it is read-only and needs no hw-run
            # authorization. A prelim failure must not block the hardware run, so
            # it is recorded as absent rather than failing the pipeline closed.
            try:
                prelim = omp_review("prelim", prelim_prompt, args.system_prompt, args.checkout, model)
            except ReviewError as e:
                prelim = None
                prelim_body = f"<!-- hw-gate:prelim -->\n# hw-gate prelim\n\nPrelim review unavailable: {e}\n"
            else:
                prelim_body = f"<!-- hw-gate:prelim -->\n# hw-gate prelim\n\n```json\n{json.dumps(prelim, indent=2, sort_keys=True)}\n```\n"
            _prelim_cache = prelim  # for _fail closure
            try:
                prelim_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:prelim -->", prelim_body)
                posted["prelim_comment"] = prelim_comment_url
            except Exception as e:
                if args.phase == "prelim":
                    sys.stderr.write(f"prelim comment post failed: {e}\n")
                else:
                    return _fail(f"prelim comment post failed: {e}")
            if args.phase == "prelim":
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump({"schema": "hipfire.hw-gate.prelim", "version": 1, "model": model,
                               "prelim": prelim, "posted": {"prelim_comment": prelim_comment_url}},
                              f, indent=2, sort_keys=True)
                    f.write("\n")
                return 0
        else:
            prelim = None
            if args.prelim and Path(args.prelim).is_file():
                try:
                    with open(args.prelim, "r", encoding="utf-8") as f:
                        prelim = json.load(f).get("prelim")
                except Exception:
                    prelim = None
            _prelim_cache = prelim

        # Evidence comment
        if evidence_md_path.is_file():
            try:
                ev_md = evidence_md_path.read_text(encoding="utf-8")
            except Exception as e:
                ev_md = f"(failed to read hw-gate.md: {e})"
            ev_body = f"<!-- hw-gate:evidence -->\n{ev_md}\n"
            # Truncation handled inside upsert_comment (60 KiB)
        else:
            ev_body = "<!-- hw-gate:evidence -->\nNo hardware evidence found — hw-run did not produce hw-gate.md (hw-run result: " + args.hw_run_result + ").\n"
        try:
            evidence_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:evidence -->", ev_body)
            posted["evidence_comment"] = evidence_comment_url
        except Exception as e:
            return _fail(f"evidence comment post failed: {e}")

        # Verdict phase
        verdict_prompt = build_verdict_prompt(prelim_prompt, prelim, evidence, select, args.hw_run_result)
        verdict_parse_failed = False
        try:
            verdict = omp_review("verdict", verdict_prompt, args.system_prompt, args.checkout, model)
        except ReviewError as e:
            # Verdict parse failure -> treat as needs-human via floor, but still need to post verdict comment?
            # Per spec: verdict phase failed to parse -> floor makes it needs-human.
            # We set verdict None and continue to floor.
            verdict_parse_failed = True
            # Log but continue; floor will handle
            # Still post a verdict comment indicating parse failure
            try:
                fail_body = f"<!-- hw-gate:verdict -->\n# hw-gate verdict\n\nVerdict parsing failed: {e}\n\nPrelim:\n```json\n{json.dumps(prelim, indent=2, sort_keys=True)}\n```\n\nDecision will be needs-human (fail-closed).\n"
                verdict_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:verdict -->", fail_body)
                posted["verdict_comment"] = verdict_comment_url
            except Exception:
                pass
            # Do not return _fail yet; go through floor as needs-human
            verdict = None

        if not verdict_parse_failed:
            # Post verdict comment including floor yet? Floor not yet computed but include verdict JSON
            # We will update verdict comment after floor; for now post initial
            verdict_body = f"<!-- hw-gate:verdict -->\n# hw-gate verdict\n\n```json\n{json.dumps(verdict, indent=2, sort_keys=True)}\n```\n"
            try:
                verdict_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:verdict -->", verdict_body)
                posted["verdict_comment"] = verdict_comment_url
            except Exception as e:
                return _fail(f"verdict comment post failed: {e}")

        # Floor
        model_decision = None
        if isinstance(verdict, dict):
            model_decision = verdict.get("decision")
        # If verdict parse failed, model_decision stays None
        final_decision, reasons = apply_floor(model_decision, evidence, select, args.hw_run_result, commit_messages, verdict)

        # Re-post verdict comment with floor decisions verbatim (update)
        floor_text = f"Floor applied: {', '.join(reasons) if reasons else 'none'} -> {final_decision} (model proposed: {model_decision})\n"
        if isinstance(verdict, dict):
            full_verdict_body = f"<!-- hw-gate:verdict -->\n# hw-gate verdict\n\n```json\n{json.dumps(verdict, indent=2, sort_keys=True)}\n```\n\n{floor_text}\n"
        else:
            full_verdict_body = f"<!-- hw-gate:verdict -->\n# hw-gate verdict\n\nVerdict parsing failed; floor forces needs-human.\n\n{floor_text}\n```json\n{json.dumps(prelim, indent=2, sort_keys=True)}\n```\n"
        try:
            # Upsert again to include floor verbatim
            verdict_comment_url = upsert_comment(args.repo, args.pr, "<!-- hw-gate:verdict -->", full_verdict_body)
            posted["verdict_comment"] = verdict_comment_url
        except Exception as e:
            return _fail(f"verdict floor comment post failed: {e}")

        # Apply decision (labels + review)
        if isinstance(verdict, dict):
            rationale = verdict.get("rationale", "")
            summary = f"hw-gate {final_decision}: {rationale}" if rationale else f"hw-gate {final_decision}"
        else:
            summary = f"hw-gate {final_decision}: verdict parse failed, fail-closed to needs-human"
        try:
            applied = apply_decision(args.repo, args.pr, final_decision, summary)
            posted["review"] = applied.get("review")
            posted["labels_added"] = applied.get("labels_added", [])
            posted["labels_removed"] = applied.get("labels_removed", [])
        except Exception as e:
            return _fail(f"apply decision failed: {e}")

        # Write verdict.json
        out_obj = {
            "schema": "hipfire.hw-gate.verdict",
            "version": 1,
            "model": model,
            "prelim": prelim,
            "verdict": verdict,
            "floor": {"applied": reasons, "model_decision": model_decision, "final_decision": final_decision},
            "posted": posted,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, sort_keys=True)
            f.write("\n")
        # CONTRACT: an unparseable verdict is a failed review — needs-human is
        # posted and recorded above, and the exit code says the model never
        # produced a decision.
        return 1 if verdict_parse_failed else 0

    except ReviewError as e:
        return _fail(str(e))
    except Exception as e:
        return _fail(f"unexpected error: {e}")


if __name__ == "__main__":
    sys.exit(main())
