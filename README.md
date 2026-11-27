# comfy-aimdo-rust

Rust CUDA backend for the ComfyUI AI Model Dynamic Offloader. It is based on
`comfy-aimdo` v0.6.0 (`68088de9dce9f5e32231fccf230938769c5562da`). The wheel
targets Linux x86_64 with NVIDIA CUDA and CPython 3.10 or newer.

The distribution installs the existing `comfy_aimdo` import package. A private
Python extension contains the Rust backend, and the build links the third-party
Funchook library for CUDA allocation hooks. The release was tested with a
simulated CUDA driver. Physical 5090 validation is pending. The source archive
includes the migration and validation notes in `rust/MAP.md`.

## Install in a ComfyUI image

Replace `comfy-aimdo==0.6.0` in the image's Python dependencies with
`comfy-aimdo-rust==0.1.0`. If the image already contains the original
distribution, uninstall it before installing this one. Both distributions
provide the same `comfy_aimdo` Python modules.

## Build for the Linux x86_64 5090 Pod

From this directory, build with Docker:

```sh
bash build-linux-docker
```

Install the `comfy_aimdo_rust-0.1.0-*.whl` from `dist/` with the interpreter
that launches ComfyUI.

```sh
python -m pip install --no-index --no-deps /wheels/comfy_aimdo_rust-0.1.0-*.whl
```

After launching the image, check for `DynamicVRAM support detected and enabled`.
Use `python -m pip show comfy-aimdo-rust` to verify the installed version. On a GPU test environment, run
`tests/vbar-pageout-cuda.py` with the installed package and repeat the
Qwen → Wan → Qwen workflow to exercise both faulting and reclamation.

Run the end-to-end wheel check with a simulated CUDA driver on Linux x86_64.

```sh
bash tests/rust-wheel-smoke.sh dist/comfy_aimdo_rust-0.1.0-*.whl
```

To compare the VBAR and CUDA context units with the v0.6.0 checkout, build the
Rust fixture libraries and run the differential tests.

```sh
cargo build --release --locked --manifest-path rust/Cargo.toml
bash tests/vbar-rust-parity.sh /path/to/upstream-v0.6.0
cargo rustc --release --locked --manifest-path rust/Cargo.toml --features cuda-control -- -C relocation-model=pic
bash tests/cuda-rust-driver.sh /path/to/upstream-v0.6.0 rust/target/release/libaimdo_vbar.a
```

## Upstream architecture

This project is a pytorch VRAM allocator that implements on-demand offloading of model weights when the primary pytorch VRAM allocator comes under pressure.

## How it works:

* The pytorch application creates a Virtual Base Address Register (**VBAR**) for a model. Creating a VBAR doesn't cost any VRAM, only GPU virtual address space (which is pretty much free).
* The pytorch application allocates tensors for model weights within the VBAR. These tensors are initially un-allocated and will segfault if touched.
* The pytorch application faults in the tensors using the `fault()` API at the time the tensor is needed. This is where VRAM actually gets allocated.

##### If the `fault()` is successful (sufficient VRAM for this tensor):
1.  **If the fault() resultant signature is changed or unknown:**
    * The application uses `tensor::_copy()` to populate the weight data on the GPU.
    * The application saves the returned signature against this weight for future comparison
2.  The layer uses the weight tensor.
3.  The application calls `unpin()` on the tensor to allow it to be freed under pressure later if needed.

##### If the `fault()` is unsuccessful (offloaded weight):
1.  The application allocates a temporary regular GPU tensor.
2.  Uses `_copy` to populate weight data on the GPU.
3.  The layer uses the temporary as the weight.
4.  Pytorch garbage collects the temp when the layer is finished.

see examples/example.py

---

## Priorities:

* The most recent VBARs are the highest priority and lower addresses in the VBAR take priority over higher addresses.
* Applications should order their tensor allocations in the VBAR in load-priority order with the lowest addresses for the highest priority weights.
* Calling `fault()` on a weight that is higher priority than other weights will cause those lower priority weights to get freed to make space.
* Having a weight evicted sets that VBAR's watermark to that weight's level. Any weights in the same VBAR above the watermark automatically fail the `fault()` API. This avoids constantly faulting in all weights each model iteration while allowing the application to just blindly call `fault()` every layer and check the results. There is no need for the application to manage any VRAM quotas or watermarks.
* Existing VBARs can be pushed to top priority with the `prioritize()` API. This allows use of an already loaded or partially model (e.g. using the same model twice in a complex workflow). Using `prioritize` resets the offload watermark of that model to no offloading, giving its weights priority over any other currently loaded models.

---

## Backend:

* VBAR allocation is done with `cuMemAddressReserve()`, faulting with `cuMemCreate()` and `cuMemMap()` and all frees done with appropriate converse APIs.
* For consistency with VBAR memory management, main pytorch allocator plugin is also implemented with `cuMemAddressReserve` -> `cuMemCreate` -> `cuMemMap`. This also behaves a lot better on Windows systems with System Memory fallback.

On AMD, the equivalent HIP APIs (`hipMemAddressReserve` -> `hipMemCreate` -> `hipMemMap`, and their converse calls) are used throughout via the same flow.

## Caveats:

* There is no real way for this allocator to tell the difference between high usage and bad fragmentation in the pytorch caching allocator. As we always return success to the pytorch caching allocator it experiences no pressure while weights are being offloaded which means it can run in an extremely fragmented mode. The assumption is model weight access patterns are reasonably regular over blocks or iterations and it finds a good set of sizes to cache. What you should generally do though, is completely flush the pytorch caching allocator before each new model run, which avoids completely un-used reservations from taking priority over the next models weights.
