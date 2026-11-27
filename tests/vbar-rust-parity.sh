#!/usr/bin/env bash
set -euo pipefail

project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream=${1:?Usage: bash tests/vbar-rust-parity.sh AIMDO_V0_6_0_SOURCE [RUST_STATICLIB]}
library=${2:-"$project/rust/target/release/libaimdo_vbar.a"}
temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT

build_and_run() {
    local source=$1 name=$2
    shift 2
    cp -- "$source" "$temp/model-vbar.c"
    "${CC:-cc}" -std=c11 -O2 -I"$temp" "$project/tests/vbar-rust-parity.c" "$@" -o "$temp/$name"
    if ! "$temp/$name" > "$temp/$name.trace"; then
        tail -n 30 "$temp/$name.trace" >&2
        return 1
    fi
}

cp -- "$project/tests/vbar-rust-parity-plat.h" "$temp/plat.h"
build_and_run "$upstream/src/model-vbar.c" upstream
build_and_run "$project/tests/fixtures/hybrid/model-vbar.c" rust "$library"
diff -u "$temp/upstream.trace" "$temp/rust.trace"
printf 'VBAR operation traces match upstream (%s events and observations).\n' "$(wc -l < "$temp/rust.trace" | tr -d ' ')"
