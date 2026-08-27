#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIPFIRE_BIN="${HIPFIRE_BIN:-${ROOT}/target/release/hipfire}"
DAEMON_BIN="${DAEMON_BIN:-${ROOT}/target/release/daemon}"
DENSE_SIDECAR="${DENSE_SIDECAR:-${ROOT}/experiments/flash-attn-ck-sidecar/build/libhipfire_flash_attn_ck.so}"
QUANTIZED_SIDECAR="${QUANTIZED_SIDECAR:-${ROOT}/experiments/flash-attn-ck-sidecar/quantized/build/libhipfire_flash_attn_ck_quantized_staged.so}"
KERNEL_CACHE="${KERNEL_CACHE:-${ROOT}/.hipfire_kernels/gfx1100}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dist}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
VERSION="${VERSION:-$(git -C "${ROOT}" rev-parse --short=12 HEAD)}"
ALLOW_DIRTY=0

usage() {
    cat <<'EOF'
Usage: package-gfx11-ck-bundle.sh [options]

Package a prebuilt hipfire CLI and its relocatable CK sidecars.

Options:
  --hipfire PATH       CK-enabled hipfire binary
  --daemon PATH        CK-enabled daemon binary
  --dense PATH         Dense FlashAttention CK sidecar
  --quantized PATH     Staged quantized CK sidecar
  --kernel-cache PATH  Precompiled gfx1100 kernel directory
  --output-dir PATH    Destination directory (default: dist)
  --version NAME       Bundle version (default: current git commit)
  --gpu-arch ARCH      Target architecture (default: gfx1100)
  --rocm-path PATH     ROCm root used for provenance
  --allow-dirty        Permit a development bundle from a dirty worktree
  --help               Show this help
EOF
}

while (($#)); do
    case "$1" in
        --hipfire) HIPFIRE_BIN="$2"; shift 2 ;;
        --daemon) DAEMON_BIN="$2"; shift 2 ;;
        --dense) DENSE_SIDECAR="$2"; shift 2 ;;
        --quantized) QUANTIZED_SIDECAR="$2"; shift 2 ;;
        --kernel-cache) KERNEL_CACHE="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --version) VERSION="$2"; shift 2 ;;
        --gpu-arch) GPU_ARCH="$2"; shift 2 ;;
        --rocm-path) ROCM_PATH="$2"; shift 2 ;;
        --allow-dirty) ALLOW_DIRTY=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ "${GPU_ARCH}" == "gfx1100" ]] || {
    echo "this preview bundle contains gfx1100 code objects only, got ${GPU_ARCH}" >&2
    exit 2
}
[[ "${VERSION}" =~ ^[A-Za-z0-9._+-]+$ ]] || {
    echo "unsafe bundle version: ${VERSION}" >&2
    exit 2
}
if [[ "${ALLOW_DIRTY}" -eq 0 && -n "$(git -C "${ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    echo "refusing to package a dirty worktree; commit first or use --allow-dirty for local testing" >&2
    exit 2
fi

for command in git readelf sha256sum tar; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done
[[ -x "${HIPFIRE_BIN}" ]] || { echo "missing executable: ${HIPFIRE_BIN}" >&2; exit 2; }
[[ -x "${DAEMON_BIN}" ]] || { echo "missing executable: ${DAEMON_BIN}" >&2; exit 2; }
[[ -f "${DENSE_SIDECAR}" ]] || { echo "missing dense sidecar: ${DENSE_SIDECAR}" >&2; exit 2; }
[[ -f "${QUANTIZED_SIDECAR}" ]] || { echo "missing quantized sidecar: ${QUANTIZED_SIDECAR}" >&2; exit 2; }
[[ -d "${KERNEL_CACHE}" ]] || { echo "missing gfx1100 kernel cache: ${KERNEL_CACHE}" >&2; exit 2; }
HSACO_COUNT="$(find "${KERNEL_CACHE}" -maxdepth 1 -type f -name '*.hsaco' | wc -l)"
HASH_COUNT="$(find "${KERNEL_CACHE}" -maxdepth 1 -type f -name '*.hash' | wc -l)"
if [[ "${HSACO_COUNT}" -eq 0 || "${HSACO_COUNT}" -ne "${HASH_COUNT}" ]]; then
    echo "invalid kernel cache: hsaco=${HSACO_COUNT}, hash=${HASH_COUNT}" >&2
    exit 2
fi
for binary in "${HIPFIRE_BIN}" "${DAEMON_BIN}"; do
    grep -aFq 'HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB' "${binary}" || {
        echo "binary does not contain the CK loader marker: ${binary}" >&2
        echo "rebuild it with the flash-attn-ck Cargo feature" >&2
        exit 2
    }
done

grep -Fq 'libhipfire_flash_attn_ck.so' < <(readelf -d "${QUANTIZED_SIDECAR}") || {
    echo "quantized sidecar does not depend on libhipfire_flash_attn_ck.so" >&2
    exit 2
}
grep -Fq '$ORIGIN' < <(readelf -d "${QUANTIZED_SIDECAR}") || {
    echo "quantized sidecar is not relocatable: missing \$ORIGIN RUNPATH" >&2
    exit 2
}

GIT_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
ROCM_VERSION="unknown"
if [[ -x "${ROCM_PATH}/bin/hipconfig" ]]; then
    ROCM_VERSION="$("${ROCM_PATH}/bin/hipconfig" --version | sed -n '1p' | tr -cs 'A-Za-z0-9._+-' '_')"
fi
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "${ROOT}" show -s --format=%ct HEAD)}"
BUNDLE_NAME="hipfire-gfx11-ck-${VERSION}"

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
STAGING="$(mktemp -d "${OUTPUT_DIR}/.${BUNDLE_NAME}.XXXXXX")"
cleanup() { rm -rf "${STAGING}"; }
trap cleanup EXIT

BUNDLE_ROOT="${STAGING}/${BUNDLE_NAME}"
mkdir -p "${BUNDLE_ROOT}/bin/kernels/compiled/gfx1100" "${BUNDLE_ROOT}/lib"
install -m 0755 "${HIPFIRE_BIN}" "${BUNDLE_ROOT}/bin/hipfire"
install -m 0755 "${DAEMON_BIN}" "${BUNDLE_ROOT}/bin/daemon"
install -m 0755 "${DENSE_SIDECAR}" "${BUNDLE_ROOT}/lib/libhipfire_flash_attn_ck.so"
install -m 0755 "${QUANTIZED_SIDECAR}" \
    "${BUNDLE_ROOT}/lib/libhipfire_flash_attn_ck_quantized_staged.so"
cp -a "${KERNEL_CACHE}/." "${BUNDLE_ROOT}/bin/kernels/compiled/gfx1100/"

cat >"${BUNDLE_ROOT}/bin/hipfire-gfx11" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
while [[ -L "${SOURCE}" ]]; do
    DIR="$(cd -P "$(dirname "${SOURCE}")" && pwd)"
    SOURCE="$(readlink "${SOURCE}")"
    [[ "${SOURCE}" = /* ]] || SOURCE="${DIR}/${SOURCE}"
done
ROOT="$(cd -P "$(dirname "${SOURCE}")/.." && pwd)"

if [[ "${HIPFIRE_GFX11_BUNDLE_SKIP_CHECK:-0}" != "1" ]]; then
    supported_gpu=0
    if command -v rocminfo >/dev/null && \
        grep -Eq 'Name:[[:space:]]+gfx1100' < <(rocminfo 2>/dev/null); then
        supported_gpu=1
    elif command -v rocm-smi >/dev/null && \
        grep -Eq 'GFX Version:[[:space:]]+gfx1100' < <(rocm-smi --showproductname 2>/dev/null); then
        supported_gpu=1
    fi
    [[ "${supported_gpu}" -eq 1 ]] || {
        echo "hipfire-gfx11: this bundle requires a gfx1100 GPU" >&2
        exit 2
    }
fi

export LD_LIBRARY_PATH="${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HIPFIRE_DAEMON_BIN="${ROOT}/bin/daemon"
export HIPFIRE_FLASH_ATTN_CK_QUANTIZED_LIB="${ROOT}/lib/libhipfire_flash_attn_ck_quantized_staged.so"
export HIPFIRE_KV_MODE="${HIPFIRE_KV_MODE:-asym3}"
export HIPFIRE_GRAPH="${HIPFIRE_GRAPH:-0}"
export HIPFIRE_PREFILL_MAX_BATCH="${HIPFIRE_PREFILL_MAX_BATCH:-2048}"
export HIPFIRE_FLASH_PARTIALS_BATCH="${HIPFIRE_FLASH_PARTIALS_BATCH:-64}"
export HIPFIRE_QKVZA_SPLIT_TAIL="${HIPFIRE_QKVZA_SPLIT_TAIL:-1}"
export HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64="${HIPFIRE_RDNA3_HFQ4_GATE_UP_X256Y64:-1}"
export HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64="${HIPFIRE_RDNA3_HFQ4_RESIDUAL_X256Y64:-1}"
export HIPFIRE_RDNA3_HFQ4_AUX_X256Y64="${HIPFIRE_RDNA3_HFQ4_AUX_X256Y64:-1}"
export HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE="${HIPFIRE_RDNA3_HFQ4_PERM_NIBBLE:-1}"
export HIPFIRE_RDNA3_Q8_GROUP128="${HIPFIRE_RDNA3_Q8_GROUP128:-1}"
export HIPFIRE_RDNA3_Q8_GROUP128_ROW2="${HIPFIRE_RDNA3_Q8_GROUP128_ROW2:-1}"
export HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT="${HIPFIRE_RDNA3_Q8_GROUP128_QUAD_ROW_WEIGHT:-1}"
export HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128="${HIPFIRE_RDNA3_FUSED_SWIGLU_Q8_GROUP128:-1}"
export HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE="${HIPFIRE_RDNA3_FFN_F16_INTERMEDIATE:-1}"
export HIPFIRE_RDNA3_GDN_CONV_TOKEN_PARALLEL="${HIPFIRE_RDNA3_GDN_CONV_TOKEN_PARALLEL:-1}"
export HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW="${HIPFIRE_RDNA3_Q8_GROUP256_SERIAL_ROW:-1}"
export HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP="${HIPFIRE_RDNA3_Q8_GROUP256_GATE_UP:-0}"

exec "${ROOT}/bin/hipfire" "$@"
EOF
chmod 0755 "${BUNDLE_ROOT}/bin/hipfire-gfx11"

cat >"${BUNDLE_ROOT}/manifest.env" <<EOF
BUNDLE_FORMAT_VERSION=1
BUNDLE_VERSION=${VERSION}
GIT_COMMIT=${GIT_COMMIT}
GPU_ARCH=${GPU_ARCH}
ROCM_VERSION=${ROCM_VERSION}
DENSE_ABI_VERSION=1
QUANTIZED_ABI_VERSION=1
KERNEL_CACHE_ABI=3
KERNEL_HSACO_COUNT=${HSACO_COUNT}
EOF

(
    cd "${BUNDLE_ROOT}"
    mapfile -d '' FILES < <(
        find bin lib -type f -print0
        printf '%s\0' manifest.env
    )
    printf '%s\0' "${FILES[@]}" | sort -z | xargs -0 sha256sum > SHA256SUMS
)

ARCHIVE="${OUTPUT_DIR}/${BUNDLE_NAME}.tar.gz"
tar \
    --sort=name \
    --mtime="@${SOURCE_DATE_EPOCH}" \
    --owner=0 --group=0 --numeric-owner \
    -C "${STAGING}" \
    -czf "${ARCHIVE}" \
    "${BUNDLE_NAME}"
(
    cd "${OUTPUT_DIR}"
    sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256"
)

echo "bundle=${ARCHIVE}"
echo "checksum=${ARCHIVE}.sha256"
