#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# hipfire — see LICENSE and NOTICE in the project root.
"""`arch_id_for` must recognise the ornith1.5 family.

The daily registry workflow is fail-closed: an unknown tag family makes
`arch_id_for` return None, which aborts the ENTIRE run and writes nothing —
every other model's entry included, not just the new one. So a missing family
is not a cosmetic omission, it is an outage of the published registry.

Ornith 1.5 is a Qwen3.5-family VL finetune: the 35B-A3B is qwen3_5_moe
(arch 6), the 9B is dense qwen3_5 (arch 5). Keyed on "a3b" exactly like the
qwen3.5 family it derives from.
"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "registry_gen", Path(__file__).parent / "registry_gen.py"
)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)


def test_ornith15_a3b_is_arch6():
    entry = {"file": "ornith-1.5-35b-a3b.mq4"}
    assert rg.arch_id_for("ornith1.5:35b-a3b", entry) == 6


def test_ornith15_dense_is_arch5():
    # The 9B is dense qwen3_5. Not shipped by this PR, but the mapping must not
    # silently hand it arch 6 if someone adds it later.
    entry = {"file": "ornith-1.5-9b.mq4"}
    assert rg.arch_id_for("ornith1.5:9b", entry) == 5


def test_unknown_family_still_fails_closed():
    # The fail-closed contract itself is worth pinning: if this ever starts
    # returning a default instead of None, an unmapped model would ship with a
    # wrong arch_id rather than stopping the run.
    assert rg.arch_id_for("notamodel:1b", {"file": "x.mq4"}) is None


def test_mq4_extension_is_a_known_quant():
    assert rg.quant_for("ornith-1.5-35b-a3b.mq4") == "mq4"
