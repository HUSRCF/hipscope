# Direct gate + MQ-rotate bridge A/B

W7900 / gfx1100, ROCm 7.14, Qwen3.6-27B MQ4, asym3 K + Q8 V cache. This experiment extended the optional quantized CK sidecar with a direct postprocess that consumed the FP16 attention output, applied `sigmoid(gate)`, performed the MagnumQuant FWHT rotation, and wrote the final FP32 projection input. The baseline used the same binary and sidecar attention kernel but retained the ordinary FP16-to-FP32 conversion followed by the runtime gate/rotate path.

PP512/TG16 produced identical token IDs in both modes. In the five-run PP8192 comparison, the direct sidecar bridge was slightly slower: median `11961.4 ms` / `684.9 tok/s` versus `11936.4 ms` / `686.3 tok/s` (`-0.20%`). Each corresponding direct-bridge run was slower, so the result is not a favorable ordering artifact. The optional postprocess ABI and production routing were rejected and removed.

The result narrows the next useful boundary: removing one intermediate FP32 pass is not enough. A future bridge must let the output projection consume CK output without a separately launched sigmoid/FWHT postprocess, or fuse a materially larger block.
