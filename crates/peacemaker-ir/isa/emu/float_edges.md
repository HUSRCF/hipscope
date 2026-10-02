# Measured floating edges

`gfx1151-edges-v1.tar.zst` is a lossless hardware capture, not a general numerical-model certificate. It contains the gfx1151 scalar/packed/VOPD manifest and its files, f16/bf16 WMMA records, native code objects and executable fingerprints, producer sources, reconstruction tools, and the own-run model reports. There is no measured gfx1201 table in this archive.

Extract with `zstd -dc gfx1151-edges-v1.tar.zst | tar -xf -`. The original manifests retain absolute producer paths; bundled readers relocate absent paths by basename beside the manifest.

## Scalar, packed and VOPD

`capture-gfx1151.manifest.json` describes 99 ordinary table specs, their typed operand sets, exact native instructions and modifiers, operand axis order, descriptor FP modes, seeds and binary fingerprints. All 26,303,051 ordinary tuples per state were measured in all four VCC/SCC initial states. Literal and SGPR operands are wave-uniform; any unused lanes are not counted.

A table records one or two raw destination words and captured per-lane VCC/SCC bits. Wave32 VCC masks are reconstructed from the manifest's lane/axis tiling; these fields are not independent whole-wave VCC masks. `fold` tables retain a baseline and lossless exceptions for the other states; `full` tables retain every state. Compressed file hashes refer to decompressed bytes, with exact lengths recorded in the manifest. `edge_recon.py MANIFEST verify` checks the 313 ordinary/pair files; `spec NAME --idx i,j,k [--state S]` reconstructs raw operands and outputs. The two packed-extra files have their own raw SHA256/length fields and were separately checked.

VOPD captures use native paired instructions. The full pair contexts are checked against independent native-half tables, retaining any exceptions. The measured fmac/fmac context count is 274,877,906,944; mul/add-literal and its forced-literal twin each check 67,108,864. All three report zero pair-context and half-reference exceptions. This is a context-independence measurement, not comparison against a CPU arithmetic model.

## WMMA

For each f16/bf16 operation, `.frag.bin` contains 66 fragments of 256 little-endian u16 words in row-major **physical packed K** order, and `.cset.bin` contains 34 raw FP32 accumulator values. The complete Cartesian corpus contains 592,416 tuples including four flag initial states. Tuple order is `((ia*66+ib)*34+ic)*4+flags`.

Each 16-byte `.rec.bin` record is four little-endian u32 words `{meta,word,vcc_lo,vcc_hi}`. `meta.bit0` is captured SCC; `meta.bit1` says all 256 D words are equal to `word`. Otherwise `word` indexes a 1024-byte `.blk.bin` block of all 256 raw FP32 outputs in row-major order. No physical destination word is omitted. On gfx1151 D register j/lane l maps to row `2*j+(l>>4)`, column `l&15`.

`edge_wmma_read.py verify META --data-dir DIR` validates completeness, all record references, sizes, counters and fingerprints. `lookup META --tuple N --data-dir DIR` reconstructs the inputs, all outputs and flags. Native descriptor round modes are zero, both denorm modes are three, Wave32 is enabled, IEEE/DX10 are enabled on gfx1151, and measured MODE is `0x000003f0`. VCC and SCC are unchanged in every captured WMMA tuple.

Own-run production `mma.rs` comparisons checked **151,658,496 destination words per operation**, with **zero mismatches** for both f16 and bf16. This proves equality on the captured Cartesian corpus only, not the required randomized or full-domain gates.

## Qualification limits

The scalar F32 set does **not** contain the proposed DIV_FMAS scaled-rounding discriminator words `0x14800000` and `0x88800000`, or the `2^80` overflow discriminator `0x67800000`. Do not enable generic VCC=1 DIV_FMAS rounding semantics from this capture alone. Ordinary ABS/CLAMP/OMOD variants absent from the audited census are not claimed measured. Full 2^32 SFU comparisons, the randomized WMMA gates, gfx1201 measurements, and the production FP8 WMMA value model remain unqualified. Preliminary source-derived models must not be described as hardware-certified.
