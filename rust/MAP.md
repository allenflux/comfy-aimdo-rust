# AIMDO Rust migration

The Linux x86_64 CUDA build compiles the project's implementation in Rust.
Third-party Funchook is statically linked. The `comfy-aimdo-rust` distribution
provides the `comfy_aimdo` Python modules through a private extension whose only
exported ELF symbol is `PyInit__native`.

| Unit | State | Validation |
| --- | --- | --- |
| VBAR allocation, faulting, pins, and eviction | `port.rs`, `vbar.rs` | v0.6.0 differential and full wheel tests with a simulated driver |
| CUDA dispatch and device context | `cuda.rs`, `context.rs`, `platform.rs` | Linux simulated-driver parity and concurrent initialization tests |
| POSIX services, HostBuffer, and model mmap | `thread.rs`, `hostbuf.rs`, `mmap.rs`, `xfer_file.rs` | Full wheel file mapping, host buffering, and file transfer tests |
| Allocation hooks, VRAMBuffer, and VMM references | `alloc_hooks.rs`, `vrambuf.rs`, `vmm_ref.rs` | Funchook interception and simulated allocation pressure |
| Allocation graph and replay | `graph.rs`, `rogue.rs` | Branching, alias, replay, and failure recovery differential tests |
| Private Python extension and wheel | `python.rs`, `comfy_aimdo/_bindings.py` | Linux wheel installation and full native smoke test on CPython 3.10 and 3.13 |

Physical GPU validation is pending. Validate PyTorch pressure, CUDA Graph
replay, cancellation, and Qwen → Wan → Qwen on a 5090 before replacing the
current deployment wheel.
