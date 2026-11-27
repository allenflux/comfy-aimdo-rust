# comfy-aimdo-rust

`comfy-aimdo-rust` 为 ComfyUI 的 DynamicVRAM 提供 Rust CUDA 后端。它基于 Comfy Org 的
`comfy-aimdo` v0.6.0，将模型权重的显存驻留和回收、CUDA 分配跟踪及分配图重放移植到 Rust，
保留 ComfyUI 使用的 `comfy_aimdo` Python 接口。[上游 v0.6.0 源码](https://github.com/Comfy-Org/comfy-aimdo/tree/68088de9dce9f5e32231fccf230938769c5562da)

当前 wheel 面向 Linux x86_64、NVIDIA CUDA 和 CPython 3.10 及以上，目标部署环境为 RTX 5090。
Python 接口由私有原生扩展提供，CUDA 分配钩子静态链接第三方 Funchook。

这版面向同一张 GPU 上长驻的 ComfyUI 进程，例如在 Qwen 图像工作流和 Wan 视频工作流之间切换。
模型及工作流仍由 ComfyUI 加载和执行，Rust 后端负责动态权重页的调入、驻留和回收。

## 安装

PyPI 发行包名为 `comfy-aimdo-rust`，版本为 `0.1.1`，Python 导入名为 `comfy_aimdo`。
Rust 版和原版共用导入路径，切换前先卸载当前发行包。所有命令都使用启动 ComfyUI 的同一个
Python 解释器。安装后重启 ComfyUI 进程以加载新扩展。[PyPI 0.1.1](https://pypi.org/project/comfy-aimdo-rust/0.1.1/)

```sh
python -m pip uninstall -y comfy-aimdo
python -m pip install --no-deps 'comfy-aimdo-rust==0.1.1'
python -m pip show comfy-aimdo-rust
```

也可以使用下载或本地构建的 wheel。以下命令假定 wheel 已复制到 `/wheels/`。

```sh
python -m pip install --no-index --no-deps /wheels/comfy_aimdo_rust-0.1.1-*.whl
```

## Docker 与 Kubernetes 部署

在镜像的 Python 依赖清单中将 `comfy-aimdo==0.6.0` 替换为 `comfy-aimdo-rust==0.1.1`
或对应的本地 wheel。基础镜像已预装原版时，在构建镜像时先卸载它。入口脚本如果会重新安装
`requirements.txt`，也要更新该文件中的依赖项。

更新 StatefulSet 使用的镜像，等待新 Pod 启动。容器重建会丢弃可写层中的手动安装。
如果在现有 Pod 中临时验证，且 ComfyUI 由 supervisor 管理，使用 supervisor 重启现有
worker，让它加载新扩展。[Kubernetes 容器存储](https://kubernetes.io/docs/tasks/configure-pod-container/configure-volume-storage/)

镜像部署可通过切回原镜像回滚。在同一个 Python 环境里回滚时，先卸载 Rust 发行包，再装回原版并重启。

```sh
python -m pip uninstall -y comfy-aimdo-rust
python -m pip install --no-deps 'comfy-aimdo==0.6.0'
```

## 验证加载与回收

给现有 ComfyUI 启动命令加上 `--verbose DEBUG`，在 Pod 里先确认运行进程加载了 Rust 扩展。

```sh
grep -Hm1 -F 'comfy_aimdo/_native.abi3.so' /proc/[0-9]*/maps 2>/dev/null
```

用同一进程依次运行 Qwen → Wan → Qwen。最后一次 Qwen 更换采样 seed，确保重新执行采样。

- `DynamicVRAM support detected and enabled` 表示 DynamicVRAM 初始化成功。
- `prepared for dynamic VRAM loading` 表示该模型走动态加载。`Staged` 表示为权重准备的空间。
- 支持 `AIMDO free:` 日志的 ComfyUI 版本中，`vbar_mb > 0` 表示这次调用确实释放了 VBAR 页。
  也可以对照同一进程、同一设备上同一个 VBAR 地址的 `Actual Resident VRAM` 日志，确认驻留量下降。

已在模拟 CUDA 驱动上运行 wheel 安装和功能检查，并与上游 v0.6.0 对照验证。
物理 RTX 5090 上的模型切换和显存回收仍待验证。可在空闲的测试 GPU 上运行下面的写回测试，
再用实际工作流检查模型切换、显存峰值和输出结果。

```sh
python tests/vbar-pageout-cuda.py
```

## 从源码构建与测试

从 Git 仓库的根目录运行构建和测试命令。Docker 构建 Linux x86_64 wheel 和源码归档，产物在 `dist/`。
[项目仓库](https://github.com/allenflux/comfy-aimdo-rust)

```sh
bash build-linux-docker
```

在 Linux x86_64 上用模拟 CUDA 驱动测试 wheel。

```sh
bash tests/rust-wheel-smoke.sh dist/comfy_aimdo_rust-0.1.1-*.whl
```

在 Linux x86_64 上使用本地的上游 v0.6.0 源码目录做差分测试。

```sh
cargo build --release --locked --manifest-path rust/Cargo.toml
bash tests/vbar-rust-parity.sh /path/to/upstream-v0.6.0
cargo rustc --release --locked --manifest-path rust/Cargo.toml --features cuda-control -- -C relocation-model=pic
bash tests/cuda-rust-driver.sh /path/to/upstream-v0.6.0 rust/target/release/libaimdo_vbar.a
bash tests/rust-graph-parity.sh /path/to/upstream-v0.6.0
```

## 工作原理

模型先为 VBAR 预留一段 GPU 虚拟地址。前向计算需要权重时，`fault()` 才为相应页
分配并映射物理显存。页签名用于判断权重是否需要重新拷贝，使用完成后通过 `unpin()` 允许回收。
如果 `fault()` 因显存压力返回 `None`，上层用临时 PyTorch 张量完成当前计算。

显存压力下按 VBAR 优先级和水位回收未锁定的页。优先级可随模型切换更新，水位用于避免同一权重反复换入换出。
VBAR 通过 CUDA 驱动的 `cuMemAddressReserve`、`cuMemCreate`、`cuMemMap` 等接口管理映射。
[运行示例](examples/example.py)

| 模块 | 职责 |
| --- | --- |
| [VBAR](rust/src/vbar.rs)、[分页](rust/src/port.rs) | 权重按需加载、锁页、写回和回收 |
| [CUDA](rust/src/cuda.rs)、[设备上下文](rust/src/context.rs) | 驱动分发、显存压力和设备状态 |
| [分配钩子](rust/src/alloc_hooks.rs)、[分配图](rust/src/graph.rs) | CUDA 分配跟踪、分配图录制和重放 |
| [HostBuffer](rust/src/hostbuf.rs)、[模型映射](rust/src/mmap.rs) | 主机缓冲区、模型文件映射和数据传输 |
| [Python 扩展](rust/src/python.rs) | `comfy_aimdo` 接口的原生入口 |

迁移范围和验证记录见 [Rust 迁移记录](rust/MAP.md)。

## 许可证与致谢

项目沿用上游 GPL-3.0-only 许可证。感谢 Comfy Org、rattus 和 `comfy-aimdo` 的贡献者。
发行包同时附带静态链接依赖的许可证。[项目许可证](LICENSE) · [Funchook](licenses/funchook.txt) · [diStorm](licenses/distorm.txt) · [上游仓库](https://github.com/Comfy-Org/comfy-aimdo)
