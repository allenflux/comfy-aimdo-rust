#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${AIMDO_BUILD_PYTHON:-python3}

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "Build the 5090 wheel in a Linux x86_64 environment." >&2
    exit 1
fi

cd "$ROOT_DIR"
bash scripts/build-linux-aimdo.sh
mkdir -p dist
wheel_dir=$(mktemp -d)
trap 'rm -rf -- "$wheel_dir"' EXIT
package_dir="$wheel_dir/source"
mkdir -p "$package_dir/comfy_aimdo"
cp pyproject.toml setup.py README.md LICENSE "$package_dir/"
cp -R licenses "$package_dir/"
for module in comfy_aimdo/*.py; do
    if [ "${module##*/}" != _version.py ]; then
        cp "$module" "$package_dir/comfy_aimdo/"
    fi
done
cp comfy_aimdo/_native.abi3.so "$package_dir/comfy_aimdo/"
"$PYTHON_BIN" -m pip wheel --no-deps --no-build-isolation --wheel-dir "$wheel_dir" "$package_dir"
auditwheel repair --plat manylinux2014_x86_64 --exclude libcuda.so.1 \
    --wheel-dir dist "$wheel_dir"/comfy_aimdo_rust-*-cp310-abi3-linux_x86_64.whl

mkdir -p "$package_dir/rust" "$package_dir/vendor"
cp MANIFEST.in build-linux-docker "$package_dir/"
cp -R scripts docker "$package_dir/"
cp rust/Cargo.toml rust/Cargo.lock rust/MAP.md "$package_dir/rust/"
cp -R rust/src "$package_dir/rust/"
cp build/funchook-1.1.3.tar.gz "$package_dir/vendor/"
(cd "$package_dir" && "$PYTHON_BIN" setup.py -q sdist --dist-dir "$ROOT_DIR/dist")
