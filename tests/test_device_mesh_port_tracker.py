#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Focused contract tests for the upstream device-mesh tracker authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check-device-mesh-port-tracker.py"
TRACKER = REPO / "docs" / "device-mesh-port-tracker.json"
INVALID_FIXTURE = REPO / "tests" / "fixtures" / "device-mesh-port-tracker.invalid.json"
INDEX = REPO / "docs" / "INDEX.md"


def _run_checker(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )


def test_canonical_tracker_satisfies_schema_and_dag():
    result = _run_checker(TRACKER)
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_fixture_covers_every_tracker_contract():
    result = _run_checker(INVALID_FIXTURE)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    for marker in (
        "duplicate obligation id",
        "duplicate obligation mapping",
        "unmapped obligations",
        "unknown dependency",
        "unknown seam gate",
        "cycle",
        "status",
        "implementation_class",
        "evidence disposition",
        "advancement",
        "completion promotion",
        "authority",
    ):
        assert marker in output, f"missing diagnostic marker {marker!r}:\n{output}"


def _write_document(document: dict, path: Path) -> Path:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _materialize_authority_evidence(document: dict) -> None:
    group = next(item for item in document["change_sets"] if item["id"] == "G0")
    group["status"] = "in_review"
    group["evidence_disposition"] = "current"
    group["upstream_base_commit"] = document["upstream"]["ref"]
    group["head_commit"] = "a" * 40
    group["merge_commit"] = None
    group["completion_evidence"] = [
        {
            "classification": "current",
            "assertion": "Current G0 authority evidence.",
            "references": ["docs/device-mesh-port-tracker.json"],
            "qualifies_for_completion": True,
        }
    ]
    gate = next(item for item in document["seam_gates"] if item["id"] == "S-AUTHORITY")
    gate["status"] = "available"
    gate["evidence_disposition"] = "current"
    gate["receipt"] = {
        "status": "complete",
        "producer_commit": group["head_commit"],
        "evidence_commit": group["head_commit"],
        "consumer_commits": {},
        "route": "python3 scripts/check-device-mesh-port-tracker.py",
        "evidence_class": "current",
        "fixture_references": [
            "docs/device-mesh-port-tracker.json",
            "tests/fixtures/device-mesh-port-tracker.invalid.json",
        ],
        "positive_probe": "pytest -q tests/test_device_mesh_port_tracker.py",
        "negative_probe": "python3 scripts/check-device-mesh-port-tracker.py tests/fixtures/device-mesh-port-tracker.invalid.json (expected non-zero)",
        "side_effect_assertions": ["No runtime or authority side effect is permitted."],
        "sole_owner": group["sole_owner"],
        "revert_identity": group["revert_identity"]["identity"],
        "durable_references": [
            "https://github.com/warpfront/hipfire/issues/666",
            "git:" + group["head_commit"],
        ],
    }


def test_domain_obligations_and_campaigns_are_canonical():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    obligations = {row["id"] for row in document["obligations"]}
    assert len(obligations) > 49
    assert not any(identifier.startswith("PR-") for identifier in obligations)
    assert [campaign["id"] for campaign in document["evidence_campaigns"]] == [
        "EC-EP",
        "EC-PP",
        "EC-TP",
        "EC-VISION",
        "EC-CLOSE",
    ]
    available = [gate["id"] for gate in document["seam_gates"] if gate["status"] in {"available", "complete"}]
    if document["change_sets"][0]["status"] == "implemented":
        assert available == []
    else:
        assert available == ["S-AUTHORITY"]


def test_group_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    document["change_sets"][5]["obligation_ids"] = ["STEP-MOE-SUBSTRATE", "STEP-002", "STEP-002R"]
    child = next(row for row in document["obligations"] if row["id"] == "STEP-002")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "rerun_required"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-group.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "G5 blocked child obligation prevents completion promotion" in output


def test_campaign_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    child = next(row for row in document["obligations"] if row["id"] == "HW-001")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "hardware_blocked"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-campaign.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "EC-EP blocked child obligation prevents completion promotion" in output


def test_final_closure_completion_rejects_one_blocked_child(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    child = next(row for row in document["obligations"] if row["id"] == "DOC-002")
    child["status"] = "blocked"
    child["evidence"]["disposition"] = "rerun_required"
    child["advancement"]["completion_rows"] = []
    result = _run_checker(_write_document(document, tmp_path / "blocked-closure.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "final closure blocked child obligation prevents completion promotion" in output


def test_campaign_dependency_namespaces_and_cycles_are_checked(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("unknown", {"EC-PP": ["EC-UNKNOWN"]}, "unknown campaign dependency"),
        ("self", {"EC-CLOSE": ["EC-CLOSE"]}, "campaign self-dependency"),
        (
            "cycle",
            {"EC-EP": ["EC-PP"], "EC-PP": ["EC-EP"]},
            "cycle in evidence-campaign DAG",
        ),
    )
    for name, updates, marker in cases:
        document = json.loads(json.dumps(base))
        for campaign_id, dependencies in updates.items():
            next(c for c in document["evidence_campaigns"] if c["id"] == campaign_id)["depends_on_campaigns"] = dependencies
        result = _run_checker(_write_document(document, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing campaign diagnostic {marker!r}:\n{output}"


def test_qualifying_evidence_requires_durable_reference(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    document["change_sets"][0]["completion_evidence"][0]["references"] = []
    result = _run_checker(_write_document(document, tmp_path / "no-evidence-ref.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "durable evidence" in output


def _satisfy_all_prerequisites(document: dict) -> None:
    base_commit = document["upstream"]["ref"]
    for obligation in document["obligations"]:
        obligation["status"] = "complete"
        obligation["evidence"]["disposition"] = "current"
        obligation["evidence"]["branch_record"] = "historical"
        obligation["evidence"]["report_refs"] = ["docs/device-mesh-port-tracker.json"]
        obligation["advancement"]["completion_rows"] = [obligation["id"]]
    for index, change_set in enumerate(document["change_sets"], start=1):
        change_set["status"] = "in_review" if change_set["id"] == "G0" else "complete"
        change_set["evidence_disposition"] = "current"
        change_set["upstream_base_commit"] = base_commit
        change_set["head_commit"] = "a" * 40 if change_set["id"] == "G0" else f"{index:040x}"
        change_set["merge_commit"] = None if change_set["id"] == "G0" else f"{index + 100:040x}"
        change_set["completion_evidence"] = [
            {
                "classification": "current",
                "assertion": "Current grouped evidence packet.",
                "references": ["docs/device-mesh-port-tracker.json"],
                "qualifies_for_completion": True,
            }
        ]
    for index, campaign in enumerate(document["evidence_campaigns"], start=201):
        campaign["status"] = "complete"
        campaign["evidence_disposition"] = "current"
        campaign["upstream_base_commit"] = base_commit
        campaign["head_commit"] = f"{index:040x}"
        campaign["merge_commit"] = f"{index + 100:040x}"
        campaign["sole_owner"] = f"{campaign['id']} evidence owner"
        campaign["revert_identity"] = {
            "identity": f"{campaign['id']}:campaign-revert",
            "strategy": "revert-entire-evidence-campaign",
            "scope": "Revert this evidence campaign as one unit.",
        }
        campaign["completion_evidence"] = [
            {
                "classification": "current",
                "assertion": "Current campaign evidence packet.",
                "references": ["docs/device-mesh-port-tracker.json"],
                "qualifies_for_completion": True,
            }
        ]
    closure = document["final_closure_packet"]
    closure["status"] = "complete"
    closure["evidence_disposition"] = "current"
    closure["upstream_base_commit"] = base_commit
    closure["head_commit"] = f"{len(document['evidence_campaigns']) + 300:040x}"
    closure["merge_commit"] = f"{len(document['evidence_campaigns']) + 400:040x}"
    closure["sole_owner"] = "FCP-00 final closure owner"
    closure["revert_identity"] = {
        "identity": "FCP-00:single-final-closure-revert",
        "strategy": "revert-entire-final-closure",
        "scope": "Revert FCP-00 as one unit.",
    }
    closure["completion_evidence"] = [
        {
            "classification": "current",
            "assertion": "Current final closure packet.",
            "references": ["docs/device-mesh-port-tracker.json"],
            "qualifies_for_completion": True,
        }
    ]
    groups = {group["id"]: group for group in document["change_sets"]}
    campaigns = {campaign["id"]: campaign for campaign in document["evidence_campaigns"]}
    owners = {**groups, **campaigns, closure["id"]: closure}

    def owner_commit(owner: dict) -> str:
        return owner["merge_commit"] or owner["head_commit"]

    for gate in document["seam_gates"]:
        gate["status"] = "available"
        gate["evidence_disposition"] = "current"
        producer = owners[gate["producer"]]
        consumer_commits = {
            consumer: owner_commit(owners[consumer])
            for consumer in gate["consumers"]
            if owners[consumer]["status"] in {"complete", "in_review"}
        }
        gate["receipt"] = {
            "status": "complete",
            "producer_commit": owner_commit(producer),
            "evidence_commit": owner_commit(producer),
            "consumer_commits": consumer_commits,
            "route": "Current executable seam route with pinned fixture.",
            "evidence_class": "current",
            "fixture_references": ["docs/device-mesh-port-tracker.json"],
            "positive_probe": "Current positive seam probe command.",
            "negative_probe": "Current fail-closed seam probe command.",
            "side_effect_assertions": ["No duplicate owner or hidden side effect is permitted."],
            "sole_owner": producer["sole_owner"],
            "revert_identity": producer["revert_identity"]["identity"],
            "durable_references": ["docs/device-mesh-port-tracker.json"],
        }


def test_domain_row_contracts_are_not_copied_placeholders():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in document["obligations"]}
    assert "mtp_k" in rows["COR-001"]["scope"]
    assert "ModelMeta" in rows["COR-001"]["acceptance"]
    assert "LoadedModel" in rows["COR-004"]["acceptance"]
    assert "cross-request" in rows["COR-004"]["acceptance"]
    assert "transactional" in rows["COR-005"]["scope"]
    assert "fault injection" in rows["COR-005"]["acceptance"]
    assert "Qwen35" in rows["GEN-001"]["scope"]
    assert "DeltaNet" in rows["GEN-001"]["acceptance"]
    assert "standard-attention" not in rows["GEN-001"]["scope"]
    assert "standard-attention" in rows["AXIS-001"]["scope"]
    assert "on-disk" in rows["SPEC-003"]["acceptance"]
    assert "rollback" in rows["SPEC-003"]["acceptance"]
    assert "PP+MTP" in rows["SPEC-004"]["scope"]
    assert "compressed .mtp" in rows["SPEC-004"]["scope"]
    assert "64 MiB" in rows["SPEC-004"]["acceptance"]
    assert rows["COR-001"]["legacy_status"] == "complete"
    assert rows["SPEC-003"]["legacy_status"] == "deferred"
    assert rows["COR-002"]["depends_on"] == ["COR-004"]
    assert rows["GEN-001"]["depends_on"] == [
        "COR-002",
        "STEP-001",
        "STEP-002",
        "STEP-003",
        "STEP-005-QWEN35",
        "SPEC-001",
    ]
    assert rows["SPEC-003"]["depends_on"] == ["COR-001"]
    assert rows["SPEC-004"]["depends_on"] == ["GEN-001", "SPEC-002", "SPEC-003"]


def test_group_merge_wait_declaration_is_exact(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("missing", lambda group: group.update(merge_waits_on=[]), "G5 merge_waits_on must equal the approved map"),
        (
            "disagree",
            lambda group: group["parallel_lane"].update(merge_waits_on=[]),
            "G5 parallel_lane.merge_waits_on must match top-level merge_waits_on",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        group = next(item for item in document["change_sets"] if item["id"] == "G5")
        mutation(group)
        result = _run_checker(_write_document(document, tmp_path / f"merge-wait-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing merge-wait diagnostic {marker!r}:\n{output}"


def test_change_set_identities_and_current_seam_receipts_are_required(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _materialize_authority_evidence(document)
    group = next(item for item in document["change_sets"] if item["id"] == "G0")
    assert "upstream_base_commit" in group
    assert "head_commit" in group
    assert "merge_commit" in group
    gate = next(item for item in document["seam_gates"] if item["id"] == "S-AUTHORITY")
    gate["receipt"] = None
    result = _run_checker(_write_document(document, tmp_path / "missing-receipt.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "S-AUTHORITY current/available seam requires a complete receipt" in output


def test_current_seam_receipt_fields_and_group_identity_fail_closed(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "missing-group-base",
            lambda document: next(group for group in document["change_sets"] if group["id"] == "G0").pop("upstream_base_commit"),
            "G0 missing upstream_base_commit",
        ),
        (
            "missing-receipt-field",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].pop("durable_references"),
            "S-AUTHORITY receipt durable_references must be non-empty",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _materialize_authority_evidence(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing identity/receipt diagnostic {marker!r}:\n{output}"


def test_only_authority_seam_is_available_before_port_work():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    available = [gate["id"] for gate in document["seam_gates"] if gate["status"] in {"available", "complete"}]
    if document["change_sets"][0]["status"] == "implemented":
        assert available == []
    else:
        assert available == ["S-AUTHORITY"]


def test_commit_identity_and_receipt_mutations_fail_closed(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "issue-head",
            lambda document: next(group for group in document["change_sets"] if group["id"] == "G0").update(
                head_commit="https://github.com/warpfront/hipfire/issues/666#g0"
            ),
            "G0 in_review status requires a 40-hex head_commit",
        ),
        (
            "null-producer",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                producer_commit=None
            ),
            "S-AUTHORITY receipt requires a 40-hex producer_commit",
        ),
        (
            "base-producer",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                producer_commit=document["upstream"]["ref"]
            ),
            "S-AUTHORITY receipt producer_commit must not equal upstream_base_commit",
        ),
        (
            "placeholder-route",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                route="recorded by the owning change set"
            ),
            "S-AUTHORITY receipt route must be concrete",
        ),
        (
            "owner-mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                sole_owner="wrong owner"
            ),
            "S-AUTHORITY receipt sole_owner does not match G0 owner",
        ),
        (
            "revert-mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                revert_identity="wrong revert"
            ),
            "S-AUTHORITY receipt revert_identity does not match G0 identity",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _materialize_authority_evidence(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"identity-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing identity/receipt diagnostic {marker!r}:\n{output}"


def test_promoted_consumer_receipt_requires_matching_commit(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        (
            "missing",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                consumer_commits={}
            ),
            "S-AUTHORITY receipt consumer commit required for G1",
        ),
        (
            "mismatch",
            lambda document: next(gate for gate in document["seam_gates"] if gate["id"] == "S-AUTHORITY")["receipt"].update(
                consumer_commits={"G1": "a" * 40, "G4": "b" * 40}
            ),
            "S-AUTHORITY receipt consumer commit does not match G1 identity",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        mutation(document)
        result = _run_checker(_write_document(document, tmp_path / f"consumer-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing consumer diagnostic {marker!r}:\n{output}"


def test_campaign_and_fcp_identity_schema_and_owner_receipts(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    for owner in [*document["evidence_campaigns"], document["final_closure_packet"]]:
        assert all(field in owner for field in ("upstream_base_commit", "head_commit", "merge_commit"))
        assert "sole_owner" in owner
        assert "revert_identity" in owner
    _satisfy_all_prerequisites(document)
    cases = (
        (
            "campaign-owner",
            "S-HARDWARE-EP",
            lambda receipt: receipt.update(sole_owner="wrong"),
            "S-HARDWARE-EP receipt sole_owner does not match EC-EP owner",
        ),
        (
            "close-revert",
            "S-CLOSE",
            lambda receipt: receipt.update(revert_identity="wrong"),
            "S-CLOSE receipt revert_identity does not match EC-CLOSE identity",
        ),
    )
    for name, gate_id, mutation, marker in cases:
        mutated = json.loads(json.dumps(document))
        gate = next(item for item in mutated["seam_gates"] if item["id"] == gate_id)
        mutation(gate["receipt"])
        result = _run_checker(_write_document(mutated, tmp_path / f"{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing owner receipt diagnostic {marker!r}:\n{output}"


def test_ready_consumer_is_omitted_but_complete_consumer_is_keyed(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    groups = {group["id"]: group for group in document["change_sets"]}
    groups["G1"]["status"] = "ready"
    for group in document["change_sets"]:
        if group["id"] not in {"G0", "G1", "G4"}:
            group["status"] = "blocked"
            group["evidence_disposition"] = "rerun_required"
            group["completion_evidence"] = []
    for campaign in document["evidence_campaigns"]:
        campaign["status"] = "blocked"
        campaign["evidence_disposition"] = "hardware_blocked"
        campaign["completion_evidence"] = []
    closure = document["final_closure_packet"]
    closure["status"] = "blocked"
    closure["evidence_disposition"] = "rerun_required"
    closure["completion_evidence"] = []
    for gate in document["seam_gates"]:
        if gate["id"] == "S-AUTHORITY":
            gate["status"] = "available"
            gate["evidence_disposition"] = "current"
            gate["receipt"]["consumer_commits"] = {"G4": groups["G4"]["merge_commit"]}
        else:
            gate["status"] = "proposed"
            gate["evidence_disposition"] = "rerun_required"
            gate["receipt"] = None
    result = _run_checker(_write_document(document, tmp_path / "compact-consumers.json"))
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert document["seam_gates"][0]["receipt"]["consumer_commits"] == {"G4": groups["G4"]["merge_commit"]}


def test_final_closure_requires_current_qualifying_evidence(tmp_path: Path):
    base = json.loads(TRACKER.read_text(encoding="utf-8"))
    cases = (
        ("empty", lambda closure: closure.update(completion_evidence=[]), "final closure completion promotion requires qualifying current evidence"),
        (
            "nonqualifying",
            lambda closure: closure.update(
                completion_evidence=[
                    {
                        "classification": "current",
                        "assertion": "Not qualifying.",
                        "references": ["docs/device-mesh-port-tracker.json"],
                        "qualifies_for_completion": False,
                    }
                ]
            ),
            "final closure completion promotion requires qualifying current evidence",
        ),
        (
            "negative",
            lambda closure: closure["negative_evidence"][0].update(qualifies_for_completion=True),
            "final closure negative evidence cannot qualify for completion",
        ),
    )
    for name, mutation, marker in cases:
        document = json.loads(json.dumps(base))
        _satisfy_all_prerequisites(document)
        mutation(document["final_closure_packet"])
        result = _run_checker(_write_document(document, tmp_path / f"closure-{name}.json"))
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert marker in output, f"missing closure diagnostic {marker!r}:\n{output}"


def test_final_closure_required_seams_are_reciprocal(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    next(g for g in document["seam_gates"] if g["id"] == "S-DENSE-AXIS")["consumers"].remove("FCP-00")
    result = _run_checker(_write_document(document, tmp_path / "fcp-seam.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "FCP-00 consumed seam gate S-DENSE-AXIS omits this consumer" in output


def test_legacy_pr_provenance_requires_set_coverage(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    for obligation in document["obligations"]:
        obligation["legacy_pr_ids"] = [
            identifier for identifier in obligation["legacy_pr_ids"] if identifier != "PR-34M"
        ]
    result = _run_checker(_write_document(document, tmp_path / "legacy-missing.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "missing legacy PR provenance coverage" in output


def test_g5_merge_wait_blocks_promotion(tmp_path: Path):
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    _satisfy_all_prerequisites(document)
    next(group for group in document["change_sets"] if group["id"] == "G3")["status"] = "blocked"
    result = _run_checker(_write_document(document, tmp_path / "g5-merge-wait.json"))
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "G5 merge wait prevents completion promotion" in output


def test_grouped_tracker_maps_each_obligation_once():
    document = json.loads(TRACKER.read_text(encoding="utf-8"))
    obligations = {row["id"] for row in document["obligations"]}
    change_set_mapped = [
        obligation_id
        for change_set in document["change_sets"]
        for obligation_id in change_set["obligation_ids"]
    ]
    campaign_mapped = [
        obligation_id
        for campaign in document["evidence_campaigns"]
        for obligation_id in campaign["obligation_ids"]
    ]
    final_mapped = document["final_closure_packet"]["obligation_ids"]
    mapped = change_set_mapped + campaign_mapped + final_mapped
    assert len(document["change_sets"]) == 16
    assert len(document["evidence_campaigns"]) == 5
    assert len(mapped) == len(obligations) == 68
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == obligations


def test_docs_index_links_tracker_without_replacing_authorities():
    body = INDEX.read_text(encoding="utf-8")
    assert "[`docs/device-mesh-port-tracker.json`](device-mesh-port-tracker.json)" in body
    assert "[`docs/VALIDATION.md`](VALIDATION.md)" in body
    assert "[`docs/admissions.yml`](admissions.yml)" in body
