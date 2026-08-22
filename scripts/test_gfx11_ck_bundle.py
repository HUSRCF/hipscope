#!/usr/bin/env python3
"""No-GPU contract tests for the gfx1100 CK bundle tools."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGER = ROOT / "scripts" / "package-gfx11-ck-bundle.sh"
INSTALLER = ROOT / "scripts" / "install-gfx11-ck-bundle.sh"


def run(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(script), *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


class Gfx11CkBundleToolsTest(unittest.TestCase):
    def test_help_is_available_without_gpu_or_rocm(self) -> None:
        for script in (PACKAGER, INSTALLER):
            with self.subTest(script=script.name):
                result = run(script, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Usage:", result.stdout)

    def test_installer_requires_exactly_one_source(self) -> None:
        neither = run(INSTALLER)
        self.assertEqual(neither.returncode, 2)
        self.assertIn("exactly one", neither.stderr)

        both = run(INSTALLER, "--bundle", "/tmp/a", "--url", "https://invalid/a")
        self.assertEqual(both.returncode, 2)
        self.assertIn("exactly one", both.stderr)

    def test_remote_install_requires_archive_checksum(self) -> None:
        result = run(INSTALLER, "--url", "https://invalid.example/bundle.tar.gz")
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --sha256", result.stderr)

    def test_packager_rejects_path_like_version_before_artifact_access(self) -> None:
        result = run(PACKAGER, "--version", "../../escape", "--allow-dirty")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsafe bundle version", result.stderr)


if __name__ == "__main__":
    unittest.main()
