import ctypes
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

from comfy_aimdo import control


MIB = 1024**2
PAGE = 32 * MIB
CAPACITY = 32 * 1024**3
DEVICE = 2
STREAM = 0x1234


def driver_function(driver, name, flag, *argtypes):
    pointer = ctypes.c_void_p()
    assert driver.cuGetProcAddress(name.encode(), ctypes.byref(pointer), 12060, flag, None) == 0
    assert pointer.value
    return ctypes.CFUNCTYPE(ctypes.c_int, *argtypes)(pointer.value)


def allocate(function, size, *args):
    pointer = ctypes.c_uint64()
    assert function(ctypes.byref(pointer), size, *args) == 0
    assert pointer.value
    return pointer.value


def hook_functions(driver):
    alloc = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
    return (
        (driver_function(driver, "cuMemAlloc", 1, *alloc),
         driver_function(driver, "cuMemFree", 1, ctypes.c_uint64), ()),
        (driver_function(driver, "cuMemAllocAsync", 1, *alloc, ctypes.c_void_p),
         driver_function(driver, "cuMemFreeAsync", 1, ctypes.c_uint64, ctypes.c_void_p), (STREAM,)),
        (driver_function(driver, "cuMemAllocAsync", 2, *alloc, ctypes.c_void_p),
         driver_function(driver, "cuMemFreeAsync", 2, ctypes.c_uint64, ctypes.c_void_p), (STREAM,)),
    )


def check_hooks(driver, hooks):
    for (alloc, free, args), size, accounted in zip(hooks, (17, 2049, MIB + 1), (1024, 2049, 2 * MIB)):
        calls = driver.aimdo_smoke_raw_calls()
        pointer = allocate(alloc, size, *args)
        assert control.get_total_vram_usage() == accounted
        assert free(pointer, *args) == 0
        assert control.get_total_vram_usage() == 0
        assert driver.aimdo_smoke_raw_calls() == calls + 2
    print("Six CUDA hook targets are installed and account allocations through the driver trampolines.")


def check_vbar(hook):
    from comfy_aimdo.model_vbar import ModelVBAR, vbar_signature_compare, vbars_analyze

    alloc, free, args = hook
    writeback = []
    size = 2 * PAGE - 3
    model = ModelVBAR(size, DEVICE, lambda offset, count: writeback.append((offset, count)))
    address = model.base_addr
    signature = model.fault(address, size)
    assert signature is not None and model.get_residency() == [3, 3]
    model.unpin(address, size)
    assert model.get_residency() == [1, 1]
    assert vbars_analyze(DEVICE) == model.loaded_size() == 2 * PAGE

    original_headroom = control.get_simple_vram_headroom()
    control.set_simple_vram_headroom(CAPACITY)
    pointer = allocate(alloc, 17, *args)
    assert model.get_residency() == [0, 0] and model.loaded_size() == 0
    assert writeback == [(PAGE, PAGE - 3), (0, PAGE)]
    assert model.fault(address, size) is None
    assert free(pointer, *args) == 0
    control.set_simple_vram_headroom(original_headroom)

    model.prioritize()
    restored = model.fault(address, size)
    assert restored is not None and not vbar_signature_compare(signature, restored)
    model.unpin(address, size)
    assert model.free_memory(PAGE) == PAGE and model.get_residency() == [1, 0]
    assert writeback[-1] == (PAGE, PAGE - 3)
    calls = len(writeback)
    model.close()
    assert len(writeback) == calls
    assert control.get_total_vram_usage() == 0
    print("VBAR pages fault, stay pinned, page out under hooked allocation pressure, and refault.")


def check_vram_buffer():
    from comfy_aimdo.vram_buffer import VRAMBuffer

    buffer = VRAMBuffer(20 * MIB, DEVICE)
    assert control.get_total_vram_usage() == 0
    assert buffer.get(16 * MIB - 1, 1) == (buffer, buffer.base_addr + 1, 16 * MIB - 1)
    assert control.get_total_vram_usage() == 16 * MIB
    buffer.get(17 * MIB)
    assert control.get_total_vram_usage() == 20 * MIB
    del buffer
    assert control.get_total_vram_usage() == 0
    print("VRAMBuffer grows across chunks and frees all driver mappings.")


def check_graph(driver, hook):
    alloc, free, args = hook

    def run():
        first = allocate(alloc, 8 * MIB, *args)
        second = allocate(alloc, 16 * MIB, *args)
        assert free(first, *args) == free(second, *args) == 0
        combined = allocate(alloc, 24 * MIB, *args)
        assert combined == first
        assert free(combined, *args) == 0
        return first, second, combined

    calls = driver.aimdo_smoke_raw_calls()
    stream = SimpleNamespace(cuda_stream=STREAM, device=SimpleNamespace(index=DEVICE))
    graph = control.record(stream, assert_graph_breaks=True)
    recorded = run()
    assert (graph.peak_used, graph.virtual_bytes, graph.physical_bytes) == (24 * MIB,) * 3
    assert graph.pop() is False
    graph.push()
    assert run() == recorded
    assert graph.pop() is False
    assert driver.aimdo_smoke_raw_calls() == calls
    del graph
    assert control.get_total_vram_usage() == 0
    print("CUDA allocation graph captures and replays the same addresses without raw allocations.")


def check_host_buffers():
    from comfy_aimdo.host_buffer import HostBuffer, write_host_to_file
    from comfy_aimdo.model_mmap import ModelMMAP

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "mapping.bin"
        payload = bytes(range(32))
        path.write_bytes(payload)
        model = ModelMMAP(path)
        file_handle = model.get_file_handle()
        assert os.pread(file_handle, len(payload), 0) == payload
        assert ctypes.string_at(model.get(), len(payload)) == payload and model.bounce()

        buffer = HostBuffer(0, max_grow_size=4096, mark_cold=False)
        address = buffer.extend(4096, register=False)
        buffer.read_file_slice(file_handle, 7, 11)
        assert ctypes.string_at(address, 11) == payload[7:18]
        buffer.register(0, 4096)
        buffer.unregister(0)
        with path.open("r+b") as file:
            write_host_to_file(file, 0, 11, address)
        assert path.read_bytes()[:11] == payload[7:18]
        buffer.truncate(0, do_unregister=False)
        del buffer, model
    print("HostBuffer reads, registers, writes, and truncates; model mmap keeps a valid file handle.")


def main():
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuGetProcAddress.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                        ctypes.c_uint64, ctypes.c_void_p]
    driver.cuGetProcAddress.restype = ctypes.c_int
    for measurement in (driver.aimdo_smoke_raw_calls, driver.aimdo_smoke_live_resources):
        measurement.argtypes = []
        measurement.restype = ctypes.c_uint

    assert control.init(implementation="cuda")
    assert control.init_devices([DEVICE])
    control.set_log_warning()
    hooks = hook_functions(driver)
    check_hooks(driver, hooks)
    check_vbar(hooks[1])
    check_vram_buffer()
    check_graph(driver, hooks[1])
    check_host_buffers()
    control.deinit()
    assert driver.aimdo_smoke_live_resources() == 0

    alloc, free, args = hooks[1]
    pointer = allocate(alloc, 17, *args)
    assert free(pointer, *args) == 0
    assert driver.aimdo_smoke_live_resources() == 0
    try:
        build = f"comfy-aimdo-rust {version('comfy-aimdo-rust')}"
    except PackageNotFoundError:
        build = f"comfy-aimdo {version('comfy-aimdo')}"
    print(f"{build} passed wheel smoke, cleanup and hooked function restoration.")


if __name__ == "__main__":
    main()
