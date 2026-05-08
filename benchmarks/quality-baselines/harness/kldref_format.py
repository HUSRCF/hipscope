"""Hipfire-internal KLD reference format — Python reader.

Format spec (rev-3.1, see docs/plans/issue-113-quant-quality-eval.md
§"Hipfire-derived top-K format"):

    Header (32 bytes):
      bytes  0-7   magic "HFKLDR\\0\\0"  (8 ASCII chars, null-padded)
      bytes  8-11  version              (uint32, currently 1)
      bytes 12-15  n_ctx                (uint32)
      bytes 16-19  n_vocab              (uint32)
      bytes 20-23  n_chunk              (uint32)
      bytes 24-25  top_k                (uint16)
      bytes 26-27  flags                (uint16, currently 0)
      bytes 28-31  reserved             (uint32, zero)

    Tokens:
      n_ctx × n_chunk × uint32 token IDs

    Per-chunk × per-scored-token (n_ctx − 1 − n_ctx/2 tokens per chunk):
      fp32 max_logit
      fp32 log_sum_exp
      uint32 top_indices[top_k]
      fp32   top_logits[top_k]
      fp32   sum_exp_residual

Reconstruction at consumer:
  log_prob[i] = logit[i] - max_logit - log_sum_exp
  where logit[i] is top_logits[j] if i == top_indices[j] for some j.
  For i not in top-K, the bulk mass is bounded by sum_exp_residual.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


MAGIC = b"HFKLDR\x00\x00"
VERSION = 1


@dataclass
class KldRefHeader:
    version: int
    n_ctx: int
    n_vocab: int
    n_chunk: int
    top_k: int
    flags: int

    @property
    def per_token_bytes(self) -> int:
        # 2*4 (max_logit, log_sum_exp) + top_k*4 (indices) + top_k*4 (logits) + 4 (residual)
        return 12 + self.top_k * 8

    @property
    def scored_per_chunk(self) -> int:
        return self.n_ctx - 1 - self.n_ctx // 2

    @property
    def total_scored_tokens(self) -> int:
        return self.scored_per_chunk * self.n_chunk


@dataclass
class TokenBlock:
    """Per-token reference distribution (top-K + residual)."""
    max_logit: float
    log_sum_exp: float
    top_indices: list[int]   # len == top_k
    top_logits: list[float]  # len == top_k, in original logit space (not log-prob)
    sum_exp_residual: float


def read_header(f) -> KldRefHeader:
    raw = f.read(32)
    if len(raw) != 32:
        raise ValueError(f"short read on header: got {len(raw)}, want 32")
    magic = raw[:8]
    if magic != MAGIC:
        raise ValueError(f"bad magic: got {magic!r}, want {MAGIC!r}")
    version, n_ctx, n_vocab, n_chunk = struct.unpack("<IIII", raw[8:24])
    top_k, flags = struct.unpack("<HH", raw[24:28])
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, this reader supports {VERSION}")
    return KldRefHeader(
        version=version, n_ctx=n_ctx, n_vocab=n_vocab,
        n_chunk=n_chunk, top_k=top_k, flags=flags,
    )


def read_tokens(f, header: KldRefHeader) -> list[int]:
    n = header.n_ctx * header.n_chunk
    raw = f.read(n * 4)
    if len(raw) != n * 4:
        raise ValueError(f"short read on tokens: got {len(raw)}, want {n*4}")
    return list(struct.unpack(f"<{n}I", raw))


def read_block(f, header: KldRefHeader) -> TokenBlock:
    raw = f.read(header.per_token_bytes)
    if len(raw) != header.per_token_bytes:
        raise ValueError(f"short read on block: got {len(raw)}, want {header.per_token_bytes}")
    off = 0
    max_logit, log_sum_exp = struct.unpack_from("<ff", raw, off); off += 8
    top_indices = list(struct.unpack_from(f"<{header.top_k}I", raw, off))
    off += header.top_k * 4
    top_logits = list(struct.unpack_from(f"<{header.top_k}f", raw, off))
    off += header.top_k * 4
    (sum_exp_residual,) = struct.unpack_from("<f", raw, off); off += 4
    return TokenBlock(
        max_logit=max_logit, log_sum_exp=log_sum_exp,
        top_indices=top_indices, top_logits=top_logits,
        sum_exp_residual=sum_exp_residual,
    )


def iter_blocks(f, header: KldRefHeader) -> Iterator[TokenBlock]:
    for _ in range(header.total_scored_tokens):
        yield read_block(f, header)


def open_ref(path: str | Path) -> tuple[KldRefHeader, list[int], Iterator[TokenBlock]]:
    """Open a hipfire KLD reference file. Returns (header, tokens, block_iter).

    The block_iter is consumed in order (one pass) — keep the file open while
    iterating; the function does not buffer all blocks in memory.
    """
    f = open(path, "rb")
    header = read_header(f)
    tokens = read_tokens(f, header)
    return header, tokens, iter_blocks(f, header)


# ---------- Per-sequence-KLD result format (small sidecar) ----------

# After eval_hipfire.rs / eval_gguf.sh run a candidate against a ref, they
# emit a small "per-sequence-KLD" file that kld_reduce.py aggregates.
#
# Layout (very simple, designed to be human-inspectable too):
#   bytes  0-7   magic "HFKSEQ\0\0"
#   bytes  8-11  version (uint32, 1)
#   bytes 12-15  n_chunk (uint32)
#   bytes 16-19  reserved (uint32, zero)
#   bytes 20-?   n_chunk × {fp64 mean_kld_seq, fp64 mean_p99_seq}
#                  (16 B per sequence)
# Total size: 20 + n_chunk * 16

SEQKLD_MAGIC = b"HFKSEQ\x00\x00"
SEQKLD_VERSION = 1


def write_per_seq_kld(path: str | Path, mean_kld_per_seq: list[float], p99_kld_per_seq: list[float]) -> None:
    if len(mean_kld_per_seq) != len(p99_kld_per_seq):
        raise ValueError("mean and p99 sequences must have same length")
    n_chunk = len(mean_kld_per_seq)
    with open(path, "wb") as f:
        f.write(SEQKLD_MAGIC)
        f.write(struct.pack("<III", SEQKLD_VERSION, n_chunk, 0))
        for m, p in zip(mean_kld_per_seq, p99_kld_per_seq):
            f.write(struct.pack("<dd", m, p))


def read_per_seq_kld(path: str | Path) -> tuple[list[float], list[float]]:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != SEQKLD_MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        version, n_chunk, _reserved = struct.unpack("<III", f.read(12))
        if version != SEQKLD_VERSION:
            raise ValueError(f"unsupported version {version}")
        means, p99s = [], []
        for _ in range(n_chunk):
            m, p = struct.unpack("<dd", f.read(16))
            means.append(m)
            p99s.append(p)
        return means, p99s
