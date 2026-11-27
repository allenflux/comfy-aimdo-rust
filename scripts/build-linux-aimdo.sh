#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
NATIVE_MODULE="$ROOT_DIR/comfy_aimdo/_native.abi3.so"
FUNCHOOK_VERSION=1.1.3
FUNCHOOK_SRC="$BUILD_DIR/funchook-$FUNCHOOK_VERSION"
FUNCHOOK_TARBALL="$BUILD_DIR/funchook-$FUNCHOOK_VERSION.tar.gz"
FUNCHOOK_SHA256=d49d682f5d2386c173b6e006967ecfee37c45fd1eef3e9d4232992ccdaca4ef5

ARCH=$(uname -m)

if [ "$ARCH" != "x86_64" ] || [ "$(uname -s)" != "Linux" ]; then
    echo "The Rust CUDA build requires Linux x86_64." >&2
    exit 1
fi
FUNCHOOK_BUILD_DIR="$BUILD_DIR/funchook-$FUNCHOOK_VERSION-distorm"

mkdir -p "$BUILD_DIR"
if [ ! -f "$FUNCHOOK_TARBALL" ]; then
    if [ -f "$ROOT_DIR/vendor/${FUNCHOOK_TARBALL##*/}" ]; then
        cp "$ROOT_DIR/vendor/${FUNCHOOK_TARBALL##*/}" "$FUNCHOOK_TARBALL"
    else
        URL="https://github.com/kubo/funchook/releases/download/v$FUNCHOOK_VERSION/funchook-$FUNCHOOK_VERSION.tar.gz"
        curl -fL "$URL" -o "$FUNCHOOK_TARBALL.tmp"
        mv "$FUNCHOOK_TARBALL.tmp" "$FUNCHOOK_TARBALL"
    fi
fi
printf '%s  %s\n' "$FUNCHOOK_SHA256" "$FUNCHOOK_TARBALL" | sha256sum --check --status

if [ ! -f "$FUNCHOOK_SRC/CMakeLists.txt" ]; then
    tar -xzf "$FUNCHOOK_TARBALL" -C "$BUILD_DIR"
fi

if [ ! -f "$FUNCHOOK_BUILD_DIR/libfunchook.a" ] || [ ! -f "$FUNCHOOK_BUILD_DIR/libdistorm.a" ]; then
    if ! command -v cmake >/dev/null 2>&1; then
        echo "cmake is required to build funchook" >&2
        exit 1
    fi

    mkdir -p "$FUNCHOOK_BUILD_DIR"

    cmake -S "$FUNCHOOK_SRC" -B "$FUNCHOOK_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DFUNCHOOK_BUILD_SHARED=OFF \
        -DFUNCHOOK_BUILD_STATIC=ON \
        -DFUNCHOOK_BUILD_TESTS=OFF \
        -DFUNCHOOK_DISASM=distorm \
        -DFUNCHOOK_INSTALL=OFF

    cmake --build "$FUNCHOOK_BUILD_DIR" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
fi

export CARGO_HOME="${AIMDO_CARGO_CACHE:-$BUILD_DIR/cargo-home}"
PYO3_PYTHON="${AIMDO_BUILD_PYTHON:-python3}" \
    cargo rustc --locked --release --manifest-path "$ROOT_DIR/rust/Cargo.toml" \
        --target-dir "$BUILD_DIR/rust-native" --features rust-backend -- -C relocation-model=pic

cc -shared -o "$NATIVE_MODULE" -pthread \
    -Wl,--undefined=PyInit__native,--version-script,"$ROOT_DIR/scripts/python-exports.map" \
    "$BUILD_DIR/rust-native/release/libaimdo_vbar.a" \
    "$FUNCHOOK_BUILD_DIR/libfunchook.a" "$FUNCHOOK_BUILD_DIR/libdistorm.a" \
    -ldl -lm -lrt -lutil
