#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
wheel=${1:?Usage: bash tests/rust-wheel-smoke.sh WHEEL}
python=${AIMDO_TEST_PYTHON:-python3}

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo 'The wheel smoke test requires Linux x86_64.' >&2
    exit 1
fi

temp=$(mktemp -d)
trap 'rm -rf -- "$temp"' EXIT

"${CC:-cc}" -std=gnu11 -O0 -shared -fPIC -I"$root/tests/fixtures/hybrid" \
    -Wl,-soname,libcuda.so.1 "$root/tests/rust-wheel-smoke-driver.c" -o "$temp/libcuda.so.1"
"$python" -m pip install --disable-pip-version-check --quiet --no-deps --no-index --target "$temp/package" "$wheel"
ulimit -c 0
timeout 60 env -u LD_PRELOAD LD_LIBRARY_PATH="$temp" PYTHONPATH="$temp/package" \
    PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "$python" -X faulthandler "$root/tests/rust-wheel-smoke.py"
