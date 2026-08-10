# Fused output-gate bridge A/B

W7900 / gfx1100, ROCm 7.14. The candidate fused CK's FP16-to-FP32 conversion with Qwen's `sigmoid(gate) * output` pass. The operator rows are medians of three fresh benchmark invocations; the final tensors were bit-identical. The end-to-end row compares two binaries from the same source/build window, each using five PP8192 repeats after DPM warmup.

The component path saved 1.2%-2.7% for K=2048, but was neutral for K=8192. Qwen3.6-27B PP8192 was also neutral (`680.4` versus `679.9 tok/s`). The production ABI/runtime change was therefore rejected; a larger fusion boundary is required to create measurable model-level value.
