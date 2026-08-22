#!/usr/bin/env bash
set -euo pipefail

PREFIX="${HIPFIRE_GFX11_PREFIX:-${HOME}/.local/opt/hipfire-gfx11-ck}"
BIN_DIR="${HIPFIRE_GFX11_BIN_DIR:-${HOME}/.local/bin}"
BUNDLE=""
URL=""
EXPECTED_SHA256=""
SKIP_GPU_CHECK=0

usage() {
    cat <<'EOF'
Usage: install-gfx11-ck-bundle.sh (--bundle PATH | --url URL) [options]

Install a prebuilt gfx11 CK bundle without a source checkout or CK build.

Options:
  --bundle PATH       Local .tar.gz bundle
  --url URL           Download a .tar.gz bundle with curl
  --sha256 HEX        Required archive checksum for --url
  --prefix PATH       Versioned install root
  --bin-dir PATH      Directory for the hipfire-gfx11 symlink
  --skip-gpu-check    Permit installation when gfx1100/1101/1102 is not visible
  --help              Show this help
EOF
}

while (($#)); do
    case "$1" in
        --bundle) BUNDLE="$2"; shift 2 ;;
        --url) URL="$2"; shift 2 ;;
        --sha256) EXPECTED_SHA256="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --bin-dir) BIN_DIR="$2"; shift 2 ;;
        --skip-gpu-check) SKIP_GPU_CHECK=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -n "${BUNDLE}" && -n "${URL}" ]] || [[ -z "${BUNDLE}" && -z "${URL}" ]]; then
    echo "select exactly one of --bundle or --url" >&2
    exit 2
fi
if [[ -n "${URL}" && ! "${EXPECTED_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    echo "--url requires --sha256 with 64 hexadecimal characters" >&2
    exit 2
fi

for command in sha256sum tar readelf ldd; do
    command -v "${command}" >/dev/null || { echo "missing command: ${command}" >&2; exit 2; }
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/hipfire-gfx11-install.XXXXXX")"
STAGED_DESTINATION=""
cleanup() {
    rm -rf "${WORK}"
    if [[ -n "${STAGED_DESTINATION}" && -e "${STAGED_DESTINATION}" ]]; then
        rm -rf "${STAGED_DESTINATION}"
    fi
}
trap cleanup EXIT

if [[ -n "${URL}" ]]; then
    command -v curl >/dev/null || { echo "curl is required for --url" >&2; exit 2; }
    BUNDLE="${WORK}/bundle.tar.gz"
    curl --fail --location --retry 3 --output "${BUNDLE}" "${URL}"
else
    BUNDLE="$(cd "$(dirname "${BUNDLE}")" && pwd)/$(basename "${BUNDLE}")"
fi
[[ -f "${BUNDLE}" ]] || { echo "bundle not found: ${BUNDLE}" >&2; exit 2; }

if [[ -n "${EXPECTED_SHA256}" ]]; then
    ACTUAL_SHA256="$(sha256sum "${BUNDLE}" | awk '{print $1}')"
    [[ "${ACTUAL_SHA256}" == "${EXPECTED_SHA256,,}" ]] || {
        echo "bundle SHA256 mismatch: expected ${EXPECTED_SHA256,,}, found ${ACTUAL_SHA256}" >&2
        exit 2
    }
fi

while IFS= read -r member; do
    case "${member}" in
        /*|../*|*/../*|*/..) echo "unsafe archive member: ${member}" >&2; exit 2 ;;
    esac
done < <(tar -tzf "${BUNDLE}")

tar -xzf "${BUNDLE}" -C "${WORK}"
mapfile -t ROOTS < <(find "${WORK}" -mindepth 1 -maxdepth 1 -type d -name 'hipfire-gfx11-ck-*' | sort)
[[ "${#ROOTS[@]}" -eq 1 ]] || { echo "bundle must contain exactly one hipfire-gfx11-ck-* root" >&2; exit 2; }
SOURCE_ROOT="${ROOTS[0]}"

for path in bin/hipfire bin/daemon bin/hipfire-gfx11 lib/libhipfire_flash_attn_ck.so \
    lib/libhipfire_flash_attn_ck_quantized_staged.so manifest.env SHA256SUMS; do
    [[ -f "${SOURCE_ROOT}/${path}" ]] || { echo "bundle is missing ${path}" >&2; exit 2; }
done
KERNEL_DIR="${SOURCE_ROOT}/bin/kernels/compiled/gfx1100"
[[ -d "${KERNEL_DIR}" ]] || { echo "bundle is missing the gfx1100 kernel cache" >&2; exit 2; }
HSACO_COUNT="$(find "${KERNEL_DIR}" -maxdepth 1 -type f -name '*.hsaco' | wc -l)"
HASH_COUNT="$(find "${KERNEL_DIR}" -maxdepth 1 -type f -name '*.hash' | wc -l)"
if [[ "${HSACO_COUNT}" -eq 0 || "${HSACO_COUNT}" -ne "${HASH_COUNT}" ]]; then
    echo "invalid bundled kernel cache: hsaco=${HSACO_COUNT}, hash=${HASH_COUNT}" >&2
    exit 2
fi
(
    cd "${SOURCE_ROOT}"
    sha256sum --check --strict --quiet SHA256SUMS
)

manifest_value() {
    local key="$1"
    sed -n "s/^${key}=//p" "${SOURCE_ROOT}/manifest.env" | tail -n1
}
[[ "$(manifest_value BUNDLE_FORMAT_VERSION)" == "1" ]] || { echo "unsupported bundle format" >&2; exit 2; }
[[ "$(manifest_value KERNEL_CACHE_ABI)" == "3" ]] || { echo "unsupported kernel cache ABI" >&2; exit 2; }
[[ "${HSACO_COUNT}" == "$(manifest_value KERNEL_HSACO_COUNT)" ]] || {
    echo "kernel cache count does not match the bundle manifest" >&2
    exit 2
}
VERSION="$(manifest_value BUNDLE_VERSION)"
GPU_ARCH="$(manifest_value GPU_ARCH)"
[[ "${VERSION}" =~ ^[A-Za-z0-9._+-]+$ ]] || { echo "unsafe bundle version: ${VERSION}" >&2; exit 2; }
[[ "${GPU_ARCH}" == "gfx1100" ]] || { echo "unsupported bundle GPU arch: ${GPU_ARCH}" >&2; exit 2; }

grep -Fq '$ORIGIN' < <(readelf -d "${SOURCE_ROOT}/lib/libhipfire_flash_attn_ck_quantized_staged.so") || {
    echo "quantized sidecar is not relocatable" >&2
    exit 2
}
if grep -q 'not found' < <(ldd "${SOURCE_ROOT}/lib/libhipfire_flash_attn_ck_quantized_staged.so"); then
    ldd "${SOURCE_ROOT}/lib/libhipfire_flash_attn_ck_quantized_staged.so" >&2
    echo "bundle has unresolved dynamic dependencies" >&2
    exit 2
fi

if [[ "${SKIP_GPU_CHECK}" -eq 0 ]]; then
    supported_gpu=0
    if command -v rocminfo >/dev/null && \
        grep -Eq 'Name:[[:space:]]+gfx1100' < <(rocminfo 2>/dev/null); then
        supported_gpu=1
    elif command -v rocm-smi >/dev/null && \
        grep -Eq 'GFX Version:[[:space:]]+gfx1100' < <(rocm-smi --showproductname 2>/dev/null); then
        supported_gpu=1
    fi
    [[ "${supported_gpu}" -eq 1 ]] || {
        echo "no supported gfx1100 GPU is visible" >&2
        exit 2
    }
fi

mkdir -p "${PREFIX}/releases" "${BIN_DIR}"
PREFIX="$(cd "${PREFIX}" && pwd)"
BIN_DIR="$(cd "${BIN_DIR}" && pwd)"
DESTINATION="${PREFIX}/releases/${VERSION}"
if [[ -e "${DESTINATION}" ]]; then
    echo "release already exists: ${DESTINATION}" >&2
    exit 2
fi
if [[ -e "${PREFIX}/current" && ! -L "${PREFIX}/current" ]]; then
    echo "refusing to replace non-symlink ${PREFIX}/current" >&2
    exit 2
fi

STAGED_DESTINATION="${PREFIX}/releases/.${VERSION}.installing.$$"
cp -a "${SOURCE_ROOT}" "${STAGED_DESTINATION}"
mv "${STAGED_DESTINATION}" "${DESTINATION}"
STAGED_DESTINATION=""
ln -sfn "releases/${VERSION}" "${PREFIX}/current"
ln -sfn "${PREFIX}/current/bin/hipfire-gfx11" "${BIN_DIR}/hipfire-gfx11"

echo "installed=${DESTINATION}"
echo "launcher=${BIN_DIR}/hipfire-gfx11"
echo "run: ${BIN_DIR}/hipfire-gfx11 --help"
