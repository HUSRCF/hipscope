#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = Path(__file__).with_name("train_layer.py")
SPEC = importlib.util.spec_from_file_location("train_layer", MODULE_PATH)
TRAIN = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = TRAIN
SPEC.loader.exec_module(TRAIN)


class TrainLayerTests(unittest.TestCase):
    def test_blockwise_dequant(self):
        weight = torch.ones((129, 257), dtype=torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = TRAIN.dequant_fp8_blockwise(weight, scale)
        self.assertEqual(result.shape, weight.shape)
        self.assertEqual(float(result[0, 0]), 1.0)
        self.assertEqual(float(result[0, 128]), 2.0)
        self.assertEqual(float(result[128, 256]), 6.0)

    def test_group_ranking_tie_break(self):
        gate = torch.ones((512, 2))
        up = torch.ones((512, 2))
        down = torch.ones((2, 512))
        self.assertEqual(TRAIN.rank_groups(gate, up, down), [0, 1])
        gate[256:] *= 2
        self.assertEqual(TRAIN.rank_groups(gate, up, down), [1, 0])

    def test_qwen35_rms_norm_adds_weight_bias(self):
        x = torch.tensor([[3.0, 4.0]])
        weight = torch.zeros(2)
        result = TRAIN.rms_norm(x, weight, 0.0).float()
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True))
        torch.testing.assert_close(result, expected, rtol=5e-3, atol=5e-3)

    def test_capture_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.arange(12, dtype=np.float16).reshape(3, 4)
            y = x + np.float16(2)
            x.tofile(root / "in.f16")
            y.tofile(root / "out.f16")
            (root / "tensor_manifest.json").write_text(
                '{"version":1,"layer":2,"hidden_dim":4,"dtype":"f16-le",'
                '"chunks":[{"tokens":3,"residual_in_file":"in.f16",'
                '"residual_out_file":"out.f16"}]}'
            )
            capture = TRAIN.read_capture(root, 2, 2)
            self.assertEqual(tuple(capture.residual_in.shape), (2, 4))
            torch.testing.assert_close(
                capture.residual_delta, torch.full((2, 4), 2.0, dtype=torch.float16)
            )

    def test_capture_v2_reads_direct_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x = np.full((2, 4), 1024.0, dtype=np.float16)
            delta = np.full((2, 4), 0.25, dtype=np.float16)
            x.tofile(root / "in.f16")
            delta.tofile(root / "delta.f16")
            (root / "tensor_manifest.json").write_text(
                '{"version":2,"layer":2,"hidden_dim":4,"dtype":"f16-le",'
                '"chunks":[{"tokens":2,"residual_in_file":"in.f16",'
                '"ffn_delta_file":"delta.f16"}]}'
            )
            capture = TRAIN.read_capture(root, 2, 2)
            torch.testing.assert_close(capture.residual_in, torch.from_numpy(x))
            torch.testing.assert_close(capture.residual_delta, torch.from_numpy(delta))

    def test_capture_rejects_zero_token_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tensor_manifest.json").write_text(
                '{"version":2,"layer":2,"hidden_dim":4,"dtype":"f16-le",'
                '"chunks":[{"tokens":0,"residual_in_file":"in.f16",'
                '"ffn_delta_file":"delta.f16"}]}'
            )
            with self.assertRaisesRegex(ValueError, "non-positive token count"):
                TRAIN.read_capture(root, 2, 2)

    def test_capture_tensor_path_cannot_escape_root(self):
        root = Path("capture")
        self.assertEqual(TRAIN.capture_tensor_path(root, "chunk.f16"), root / "chunk.f16")
        for name in (".", "..", "../chunk.f16", "nested/chunk.f16", "/tmp/chunk.f16", ""):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "simple relative name"):
                    TRAIN.capture_tensor_path(root, name)

    def test_selection_requires_matching_capture_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = root / "capture"
            capture.mkdir()
            (capture / "run_manifest.json").write_text(
                json.dumps({"model_sha256": "production-model"})
            )
            selection = root / "selection.json"
            selection.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "kind": "hipfire_dense_ffn_group_selection",
                        "model_sha256": "production-model",
                        "group_size": 256,
                        "keep_groups": 2,
                        "layers": [{"layer": 3, "groups": [1, 4]}],
                    }
                )
            )
            groups, _ = TRAIN.load_selection(selection, 3, 2, [capture])
            self.assertEqual(groups, [1, 4])

            (capture / "run_manifest.json").write_text(
                json.dumps({"model_sha256": "different-model"})
            )
            with self.assertRaisesRegex(ValueError, "model SHA-256 mismatch"):
                TRAIN.load_selection(selection, 3, 2, [capture])

    def test_source_teacher_contract_fails_closed(self):
        TRAIN.enforce_source_teacher_contract(0.2, 0.25, False)
        TRAIN.enforce_source_teacher_contract(2.4, 0.25, True)
        with self.assertRaisesRegex(ValueError, "non-empty source audit"):
            TRAIN.enforce_source_teacher_contract(None, 0.25, False)
        with self.assertRaisesRegex(ValueError, "disagrees with the captured"):
            TRAIN.enforce_source_teacher_contract(2.4, 0.25, False)

    def test_capture_splits_reject_duplicates_and_overlap(self):
        train = Path("train")
        heldout = Path("heldout")
        TRAIN.validate_capture_splits([train], [heldout])
        with self.assertRaisesRegex(ValueError, "duplicate train"):
            TRAIN.validate_capture_splits([train, train], [heldout])
        with self.assertRaisesRegex(ValueError, "duplicate held-out"):
            TRAIN.validate_capture_splits([train], [heldout, heldout])
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            TRAIN.validate_capture_splits([train], [train])


if __name__ == "__main__":
    unittest.main()
