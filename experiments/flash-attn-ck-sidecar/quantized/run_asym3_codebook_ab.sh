#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
RUNTIME_LD_LIBRARY_PATH="${HIP_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
GPU_ID="${GPU_ID:-0}"
TRIALS="${TRIALS:-5}"
SLEEP_SECS="${SLEEP_SECS:-3}"
Q_ROWS="${Q_ROWS:-128 512}"
CONTEXTS="${CONTEXTS:-8192}"
CK_BM=64
CK_BN=32
CK_OUTPUT_F32=0
CODEBOOK_STORAGE="${CODEBOOK_STORAGE:-lds}"
RESULT_DIR="${RESULT_DIR:-${ROOT}/results/asym3_codebook_ab_$(date +%Y%m%d_%H%M%S)}"

if [[ "${CODEBOOK_STORAGE}" != "constant" && "${CODEBOOK_STORAGE}" != "lds" ]]; then
    echo "CODEBOOK_STORAGE must be constant or lds" >&2
    exit 2
fi
read -r -a query_row_values <<<"${Q_ROWS}"
read -r -a context_values <<<"${CONTEXTS}"
if (( ${#query_row_values[@]} == 0 || ${#context_values[@]} == 0 )); then
    echo "Q_ROWS and CONTEXTS must contain at least one integer" >&2
    exit 2
fi

mkdir -p "${RESULT_DIR}/bin"
printf 'trial\torder\tmode\tquery_rows\tseqlen_k\tck_total_ms\tnative_ms\tmax_abs\n' \
    >"${RESULT_DIR}/results.tsv"
{
    echo "date=$(date -Is)"
    echo "git_head=$(git -C "${ROOT}" rev-parse HEAD)"
    echo "gpu_arch=${GPU_ARCH}"
    echo "gpu_id=${GPU_ID}"
    echo "trials=${TRIALS}"
    echo "q_rows=${Q_ROWS}"
    echo "contexts=${CONTEXTS}"
    echo "rocm_path=${ROCM_PATH}"
    echo "hip_root=${HIP_ROOT}"
    echo "runtime_ld_library_path=${RUNTIME_LD_LIBRARY_PATH}"
    echo "ck_bm=${CK_BM}"
    echo "ck_bn=${CK_BN}"
    echo "ck_output_f32=${CK_OUTPUT_F32}"
    echo "baseline_asym3_codebook=0"
    echo "candidate_codebook_storage=${CODEBOOK_STORAGE}"
    "${ROCM_PATH}/bin/hipcc" --version | head -1
} >"${RESULT_DIR}/meta.txt"

build_one() {
    local mode="$1" constant_codebook=0 lds_codebook=0
    if [[ "${mode}" == "candidate" ]]; then
        if [[ "${CODEBOOK_STORAGE}" == "constant" ]]; then
            constant_codebook=1
        else
            lds_codebook=1
        fi
    fi
    env BUILD_ONLY=1 ROCM_PATH="${ROCM_PATH}" GPU_ARCH="${GPU_ARCH}" \
        CK_BM="${CK_BM}" CK_BN="${CK_BN}" CK_OUTPUT_F32="${CK_OUTPUT_F32}" \
        ASYM3_CODEBOOK="${constant_codebook}" ASYM3_LDS_CODEBOOK="${lds_codebook}" \
        BIN="${RESULT_DIR}/bin/${mode}" \
        bash "${ROOT}/run_quantized_ck_pipeline_smoke.sh" \
        >"${RESULT_DIR}/build_${mode}.log" 2>&1
}

run_one() {
    local trial="$1" order="$2" mode="$3"
    local log="${RESULT_DIR}/trial_${trial}_${order}_${mode}.log"
    env HIP_VISIBLE_DEVICES="${GPU_ID}" \
        LD_LIBRARY_PATH="${RUNTIME_LD_LIBRARY_PATH}" \
        "${RESULT_DIR}/bin/${mode}" --native-ab-qrows "${query_rows}" "${context_values[@]}" \
        >"${log}" 2>&1
    awk -v trial="${trial}" -v order="${order}" -v mode="${mode}" \
        -v q="${query_rows}" '
        /^case=native-ab/ {
            seq=""; total=""; native=""; maxabs="";
            for (i=1; i<=NF; ++i) {
                split($i, kv, "=");
                if (kv[1] == "seqlen_k") seq=kv[2];
                else if (kv[1] == "ck_total_ms") total=kv[2];
                else if (kv[1] == "native_ms") native=kv[2];
                else if (kv[1] == "max_abs") maxabs=kv[2];
            }
            print trial, order, mode, q, seq, total, native, maxabs;
        }
        ' OFS='\t' "${log}" | tee -a "${RESULT_DIR}/results.tsv"
}

build_one baseline
build_one candidate
sha256sum "${RESULT_DIR}/bin/baseline" "${RESULT_DIR}/bin/candidate" \
    >>"${RESULT_DIR}/meta.txt"

for trial in $(seq 1 "${TRIALS}"); do
    if (( trial % 2 == 1 )); then
        modes=(baseline candidate)
    else
        modes=(candidate baseline)
    fi
    for query_rows in "${query_row_values[@]}"; do
        order=0
        for mode in "${modes[@]}"; do
            order=$((order + 1))
            run_one "${trial}" "${order}" "${mode}"
            sleep "${SLEEP_SECS}"
        done
    done
done

python3 - "${RESULT_DIR}/results.tsv" "${TRIALS}" "${Q_ROWS}" "${CONTEXTS}" <<'PY'
import csv
import statistics
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
trials = int(sys.argv[2])
expected_q = [int(value) for value in sys.argv[3].split()]
expected_contexts = [int(value) for value in sys.argv[4].split()]
expected_rows = trials * len(expected_q) * len(expected_contexts) * 2
if len(rows) != expected_rows:
    raise SystemExit(f"incomplete result table: got {len(rows)}, expected {expected_rows}")
for q in expected_q:
    for context in expected_contexts:
        values = {}
        for mode in ("baseline", "candidate"):
            selected = [
                row
                for row in rows
                if int(row["query_rows"]) == q
                and int(row["seqlen_k"]) == context
                and row["mode"] == mode
            ]
            if len(selected) != trials:
                raise SystemExit(
                    f"q={q} context={context} mode={mode}: "
                    f"got {len(selected)} rows, expected {trials}"
                )
            if any(float(row["max_abs"]) > 0.03 for row in selected):
                raise SystemExit(
                    f"q={q} context={context} mode={mode}: "
                    "correctness threshold exceeded"
                )
            values[mode] = [float(row["ck_total_ms"]) for row in selected]
            print(
                f"q={q} context={context} {mode}: "
                f"median={statistics.median(values[mode]):.6f} ms raw={values[mode]}"
            )
        ratio = statistics.median(values["baseline"]) / statistics.median(values["candidate"])
        print(
            f"q={q} context={context} baseline_over_candidate="
            f"{ratio:.4f}x ({(ratio - 1) * 100:+.2f}%)"
        )
PY

echo "results=${RESULT_DIR}"
