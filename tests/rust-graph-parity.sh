#!/bin/bash
set -euo pipefail

project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
upstream=${1:?Usage: bash tests/rust-graph-parity.sh AIMDO_V0_6_0_SOURCE}
compiler=${CC:-cc}
temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT

case $(uname -s) in
    Linux) native=(-pthread -ldl -lm -lrt -lutil) ;;
    Darwin) native=(-framework Security -framework CoreFoundation) ;;
    *) echo 'The simulated CUDA driver test requires Linux or macOS.' >&2; exit 1 ;;
esac

cp -- "$project/tests/rust-graph-parity-plat.h" "$temp/plat.h"
reference=(malloc-graph.c malloc-rogue.c vmm-ref.c)
for source in "${reference[@]}"; do
    cp -- "$upstream/src/$source" "$temp/$source"
done
reference=("${reference[@]/#/$temp/}")

rustc --edition 2021 --crate-type staticlib -C panic=abort -C opt-level=3 -D warnings \
    "$project/tests/rust-graph-parity.rs" -o "$temp/librust_graph.a"

probe="$project/tests/rust-graph-parity.c"
"$compiler" -std=c11 -O2 -I"$temp" -I"$upstream/src" "$probe" \
    "${reference[@]}" -o "$temp/upstream"
"$compiler" -std=c11 -O2 -DRUST_GRAPH -I"$temp" -I"$upstream/src" "$probe" \
    "$temp/librust_graph.a" "${native[@]}" -o "$temp/rust"

"$temp/upstream" > "$temp/upstream.trace"
"$temp/rust" > "$temp/rust.trace"
diff -u "$temp/upstream.trace" "$temp/rust.trace"
printf 'Allocation graph traces match AIMDO v0.6.0 (%s observations).\n' \
    "$(wc -l < "$temp/rust.trace" | tr -d ' ')"
