#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_ROOT="${FLASH_ATTN_ROOT:?set FLASH_ATTN_ROOT to a flash-attention checkout containing composable_kernel}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_ARCH="${GPU_ARCH:-gfx1100}"
HEAD_DIMS="${HEAD_DIMS:-64,128,256}"
MAX_JOBS="${MAX_JOBS:-8}"
OUT="${OUT:-${ROOT}/build/libhipfire_flash_attn_ck.so}"
BUILD_DIR="$(dirname "${OUT}")"
HIP_ROOT="$("${ROCM_PATH}/bin/hipconfig" --path)"
HIP_LIB_DIR="${HIP_ROOT}/lib"
EXTERNAL_CK_ROOT="${FLASH_ATTN_ROOT}/csrc/composable_kernel"
REQUIRED_CK_REV="13f6d635653bd5ffbfcac8577f1ef09590c23d78"
FA4_GFX11_D256_REV="be194c0792e79ae26f71bf507e51b4d9136cf22c"
CK_USE_FA4_GFX11_D256_RECIPE="${CK_USE_FA4_GFX11_D256_RECIPE:-0}"
RECIPE_PATCH="${ROOT}/gfx11_ck_recipe.patch"
REQUIRED_RECIPE_SHA256="b43ea8d12e14cef04518225acaa69b63e62991ba4a83efcd596fc108105ac765"
CK_ROOT="${BUILD_DIR}/ck-source"

case "${GPU_ARCH}" in
    gfx11*)
        CK_TARGET="gfx11"
        APPLY_GFX11_RECIPE=1
        ;;
    gfx12*)
        CK_TARGET="gfx12"
        APPLY_GFX11_RECIPE=0
        ;;
    *)
        echo "unsupported GPU_ARCH ${GPU_ARCH}; expected gfx11* or gfx12*" >&2
        exit 2
        ;;
esac

case "${CK_TARGET}:${HEAD_DIMS}" in
    gfx11:64) EXPECTED_GENERATED_SOURCES=9 ;;
    gfx11:128 | gfx11:256) EXPECTED_GENERATED_SOURCES=5 ;;
    gfx11:64,128) EXPECTED_GENERATED_SOURCES=13 ;;
    gfx11:64,128,256) EXPECTED_GENERATED_SOURCES=17 ;;
    gfx12:64 | gfx12:128 | gfx12:256) EXPECTED_GENERATED_SOURCES=5 ;;
    gfx12:64,128) EXPECTED_GENERATED_SOURCES=9 ;;
    gfx12:64,128,256) EXPECTED_GENERATED_SOURCES=13 ;;
    *)
        echo "unsupported HEAD_DIMS ${HEAD_DIMS}; expected 64, 128, 256, 64,128, or 64,128,256" >&2
        exit 2
        ;;
esac

if [[ "${CK_USE_FA4_GFX11_D256_RECIPE}" == "1" ]]; then
    if [[ "${CK_TARGET}:${HEAD_DIMS}" != "gfx11:256" ]]; then
        echo "FA4 gfx11 recipe currently supports HEAD_DIMS=256 only" >&2
        exit 2
    fi
    CK_SOURCE_REV="${FA4_GFX11_D256_REV}"
    CK_GIT_ROOT="${CK_GIT_ROOT:-${FLASH_ATTN_ROOT}}"
    CK_ARCHIVE_SUBTREE="${CK_ARCHIVE_SUBTREE:-csrc/composable_kernel}"
    APPLY_GFX11_RECIPE=0
    EXPECTED_GENERATED_SOURCES=17
else
    CK_SOURCE_REV="${REQUIRED_CK_REV}"
    CK_GIT_ROOT="${EXTERNAL_CK_ROOT}"
    CK_ARCHIVE_SUBTREE=""
fi

if [[ ! -f "${HIP_LIB_DIR}/libamdhip64.so" ]]; then
    echo "missing HIP runtime under ${HIP_LIB_DIR}" >&2
    exit 2
fi

if [[ ! -d "${EXTERNAL_CK_ROOT}" ]]; then
    echo "missing composable_kernel source under ${EXTERNAL_CK_ROOT}" >&2
    exit 2
fi
if ! git -C "${CK_GIT_ROOT}" cat-file -e "${CK_SOURCE_REV}^{commit}"; then
    echo "composable_kernel checkout does not contain requested revision ${CK_SOURCE_REV}" >&2
    exit 2
fi
if (( APPLY_GFX11_RECIPE )); then
    if [[ ! -f "${RECIPE_PATCH}" ]]; then
        echo "missing bundled gfx11 CK recipe ${RECIPE_PATCH}" >&2
        exit 2
    fi
    RECIPE_SHA256="$(sha256sum "${RECIPE_PATCH}" | cut -d' ' -f1)"
    if [[ "${RECIPE_SHA256}" != "${REQUIRED_RECIPE_SHA256}" ]]; then
        echo "gfx11 CK recipe SHA256 mismatch: ${RECIPE_SHA256}" >&2
        exit 2
    fi
fi

reset_dir() {
    local directory="$1"
    mkdir -p "${directory}"
    find "${directory}" -mindepth 1 -delete
}

reset_dir "${BUILD_DIR}/generated"
reset_dir "${BUILD_DIR}/objects"
reset_dir "${CK_ROOT}"

if [[ -n "${CK_ARCHIVE_SUBTREE}" ]]; then
    git -C "${CK_GIT_ROOT}" archive "${CK_SOURCE_REV}:${CK_ARCHIVE_SUBTREE}" | tar -x -C "${CK_ROOT}"
else
    git -C "${CK_GIT_ROOT}" archive "${CK_SOURCE_REV}" | tar -x -C "${CK_ROOT}"
fi
if (( APPLY_GFX11_RECIPE )); then
    patch -d "${CK_ROOT}" -p1 < "${RECIPE_PATCH}"
fi

FMHA_DIR="${CK_ROOT}/example/ck_tile/01_fmha"
GENERATOR="${FMHA_DIR}/generate.py"
if [[ ! -f "${FMHA_DIR}/fmha_fwd.hpp" || ! -f "${GENERATOR}" ]]; then
    echo "missing CK FMHA source under ${FMHA_DIR}" >&2
    exit 2
fi

LIST="${BUILD_DIR}/sources.list"
FILTER="*d*_fp16_batch*_nlogits_nbias_*nlse_ndropout_nskip_nqscale_ntrload*"

python3 "${GENERATOR}" \
    --targets "${CK_TARGET}" \
    --api fwd \
    --receipt 2 \
    --optdim "${HEAD_DIMS}" \
    --filter "${FILTER}" \
    --list_blobs "${LIST}"

if [[ "$(wc -l < "${LIST}")" -ne "${EXPECTED_GENERATED_SOURCES}" ]]; then
    echo "expected ${EXPECTED_GENERATED_SOURCES} ${CK_TARGET} FP16/D${HEAD_DIMS} generated sources" >&2
    echo "the supplied FlashAttention tree does not carry the validated CK recipe" >&2
    exit 2
fi

python3 "${GENERATOR}" \
    --targets "${CK_TARGET}" \
    --api fwd \
    --receipt 2 \
    --optdim "${HEAD_DIMS}" \
    --filter "${FILTER}" \
    --output_dir "${BUILD_DIR}/generated"

COMMON_FLAGS=(
    -std=c++20
    -O3
    -fPIC
    --offload-arch="${GPU_ARCH}"
    -DCK_TILE_FMHA_FWD_FAST_EXP2=1
    -fgpu-flush-denormals-to-zero
    -DCK_ENABLE_BF16
    -DCK_ENABLE_FP16
    -DCK_ENABLE_FP32
    -DCK_ENABLE_FP64
    -DCK_ENABLE_INT8
    -D__HIP_PLATFORM_HCC__=1
    -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3
    -Wno-pass-failed
    -mllvm --lsr-drop-solution=1
    -fno-offload-uniform-block
    -mllvm -enable-post-misched=0
    -mllvm -amdgpu-early-inline-all=true
    -mllvm -amdgpu-function-calls=false
    -I"${ROOT}"
    -I"${FMHA_DIR}"
    -I"${CK_ROOT}/include"
    -I"${CK_ROOT}/library/include"
)

mapfile -t GENERATED_SOURCES < <(find "${BUILD_DIR}/generated" -maxdepth 2 -type f -name 'fmha_fwd*.cpp' | sort)
if [[ "${#GENERATED_SOURCES[@]}" -ne "${EXPECTED_GENERATED_SOURCES}" ]]; then
    echo "generated ${#GENERATED_SOURCES[@]} sources, expected ${EXPECTED_GENERATED_SOURCES}" >&2
    exit 2
fi

compile_one() {
    local source="$1"
    local stem
    stem="$(basename "${source}" .cpp)"
    "${ROCM_PATH}/bin/hipcc" "${COMMON_FLAGS[@]}" -x hip -c "${source}" \
        -o "${BUILD_DIR}/objects/${stem}.o"
}
export -f compile_one

for source in "${GENERATED_SOURCES[@]}" "${ROOT}/hipfire_flash_attn_ck.cpp"; do
    compile_one "${source}" &
    while (( "$(jobs -pr | wc -l)" >= MAX_JOBS )); do
        wait -n
    done
done
wait

"${ROCM_PATH}/bin/hipcc" \
    -shared \
    -Wl,-z,defs \
    -Wl,-soname,libhipfire_flash_attn_ck.so \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    "${BUILD_DIR}"/objects/*.o \
    -lamdhip64 \
    -o "${OUT}"

"${ROCM_PATH}/bin/hipcc" \
    -std=c++20 \
    -O2 \
    -I"${ROOT}" \
    "${ROOT}/smoke_raw_abi.cpp" \
    -L"${BUILD_DIR}" \
    -Wl,-rpath,'$ORIGIN' \
    -Wl,-rpath,"${HIP_LIB_DIR}" \
    -lhipfire_flash_attn_ck \
    -o "${BUILD_DIR}/smoke_raw_abi"

file "${OUT}"
du -h "${OUT}"
echo "${OUT}"
