#!/usr/bin/env python3
"""Step 1.5 — tokenize the slice with both stacks, byte-compare token streams.

If both produce identical streams: continue with the GGUF anchor track.
If not: fall through to bridge work or drop the GGUF anchor entirely (see
plan §"Tokenizer alignment + bridge investigation").

Usage:
  python3 tokenizer_parity.py --gguf <path-to-bf16.gguf> [--slice <path>]

Status: STUB — full impl lands when the slice is generated and a hipfire
tokenizer entry point is identified. For now this script just sketches the
intended check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def hf_tokenize(slice_path: Path) -> list[int]:
    """Tokenize via HF tokenizers lib using the upstream Qwen tokenizer.

    TODO: pin a tokenizer.json (e.g. from HF Hub Qwen/Qwen3.5-9B-Instruct)
    or use whatever hipfire's loader uses. Currently a stub.
    """
    raise NotImplementedError(
        "hf_tokenize stub — wire to HF tokenizers + a pinned tokenizer.json"
    )


def llama_cpp_tokenize(gguf_path: Path, slice_path: Path) -> list[int]:
    """Tokenize via llama.cpp's bundled tokenizer in the GGUF.

    Invokes `llama-tokenize` (from the pinned llama.cpp commit) and parses
    its output. Currently a stub.
    """
    raise NotImplementedError(
        "llama_cpp_tokenize stub — wire to `llama-tokenize -m <gguf> -f <slice> --ids`"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf", required=True, help="path to BF16 GGUF for tokenizer extraction")
    ap.add_argument("--slice", default="../slice/wikitext2-1024s-2048ctx.txt")
    args = ap.parse_args()

    slice_path = Path(args.slice)
    if not slice_path.exists():
        print(f"slice not found: {slice_path}", file=sys.stderr)
        print("Run ../slice/make_slice.sh first.", file=sys.stderr)
        return 2

    print(f"tokenizer_parity: comparing tokenizations of {slice_path}", file=sys.stderr)
    print("  STUB: full implementation pending Step 1 follow-up commit.", file=sys.stderr)
    print("  Next: implement hf_tokenize() + llama_cpp_tokenize() above.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
