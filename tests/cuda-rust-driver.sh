#!/usr/bin/env bash
set -euo pipefail

project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream=${1:?Usage: bash tests/cuda-rust-driver.sh AIMDO_V0_6_0_SOURCE [RUST_STATICLIB]}
archive=${2:-"$project/build/rust-cuda/release/libaimdo_vbar.a"}
compiler=${CC:-cc}
fixture="$project/tests/fixtures/hybrid"

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'The CUDA driver fixtures require Linux x86_64.' >&2
    exit 1
fi

temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT
mkdir "$temp/drivers" "$temp/no-bootstrap"

fake_driver() {
    local output=$1 driver_id=$2
    shift 2
    "$compiler" -std=gnu11 -O2 -shared -fPIC -I"$fixture" \
        -Wl,-soname,"$(basename "$output")" -DDRIVER_ID="$driver_id" "$@" \
        "$project/tests/cuda-rust-driver-fake.c" -o "$output"
}

fake_driver "$temp/drivers/libcuda.so.1" 1
fake_driver "$temp/drivers/libcuda.so" 2
fake_driver "$temp/no-bootstrap/libcuda.so.1" 3 -DOMIT_GET_PROC_ADDRESS

build_and_run() {
    local name=$1 source=$2
    shift 2
    "$compiler" -std=gnu11 -O2 -DAIMDO_CUDA -I"$source" \
        "$project/tests/cuda-rust-driver.c" "$upstream/src-posix/module-load.c" \
        "$@" -Wl,--gc-sections -pthread -ldl -lm -lrt -lutil -o "$temp/$name"
    if ! env -u LD_PRELOAD LD_LIBRARY_PATH="$temp/drivers" \
            "$temp/$name" "$temp/no-bootstrap/libcuda.so.1" > "$temp/$name.trace"; then
        tail -n 30 "$temp/$name.trace" >&2
        return 1
    fi
}

build_and_run upstream "$upstream/src" "$upstream/src-cuda/dispatch.c" "$upstream/src/control.c"
build_and_run rust "$fixture" -DAIMDO_RUST_CUDA \
    "$fixture/rust-dispatch-bridge.c" "$fixture/rust-control-bridge.c" "$archive"
diff -u "$temp/upstream.trace" "$temp/rust.trace"
printf 'CUDA driver and device context traces match upstream (%s observations).\n' \
    "$(wc -l < "$temp/rust.trace" | tr -d ' ')"
timeout 30 env -u LD_PRELOAD LD_LIBRARY_PATH="$temp/drivers" "$temp/rust" --check-publication
