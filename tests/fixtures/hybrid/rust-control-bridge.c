#include "plat.h"
#include "aimdo-time.h"

_Static_assert(sizeof(size_t) == 8 && sizeof(ssize_t) == 8 && sizeof(void *) == 8,
               "Rust CUDA control requires a 64-bit target");
_Static_assert(sizeof(bool) == 1 && sizeof(int) == 4, "Rust CUDA control argument layout");
_Static_assert(sizeof(HostbufFileReaderSlot) == 32 &&
               offsetof(HostbufFileReaderSlot, buffer) == 0 &&
               offsetof(HostbufFileReaderSlot, offset) == 8 &&
               offsetof(HostbufFileReaderSlot, stream) == 16 &&
               offsetof(HostbufFileReaderSlot, event) == 24, "Rust file reader slot layout");
_Static_assert(sizeof(AimdoContext) == 41184 &&
               offsetof(AimdoContext, _device_id) == 0 &&
               offsetof(AimdoContext, _vram_capacity) == 8 &&
               offsetof(AimdoContext, _integrated_ram_headroom) == 16 &&
               offsetof(AimdoContext, _extra_vram_headroom) == 24 &&
               offsetof(AimdoContext, _malloc_async_clamp) == 32 &&
               offsetof(AimdoContext, _total_vram_usage) == 40 &&
               offsetof(AimdoContext, _total_vram_last_check) == 48 &&
               offsetof(AimdoContext, _deficit_sync) == 56 &&
               offsetof(AimdoContext, _control_timestamp_last_check) == 64 &&
               offsetof(AimdoContext, _highest_priority) == 72 &&
               offsetof(AimdoContext, _lowest_priority) == 80 &&
               offsetof(AimdoContext, _vbars_dirty) == 88 &&
               offsetof(AimdoContext, _allocations_dirty) == 89 &&
               offsetof(AimdoContext, _integrated_device) == 90 &&
               offsetof(AimdoContext, _vmm_table) == 96 &&
               offsetof(AimdoContext, _size_table) == 32864 &&
               offsetof(AimdoContext, _size_table_lock) == 41056 &&
               offsetof(AimdoContext, _rogues) == 41064 &&
               offsetof(AimdoContext, _rogue_candidates) == 41072 &&
               offsetof(AimdoContext, _hostbuf_file_reader_slots) == 41080 &&
               offsetof(AimdoContext, _hostbuf_file_reader_active) == 41176,
               "Rust CUDA context layout");

void aimdo_rs_set_simple_vram_headroom(int64_t bytes);
AimdoContext *aimdo_rs_get_devctx(int device);
bool aimdo_rs_set_devctx_for_device(int device);
bool aimdo_rs_set_devctx_for_current_cuda_device(void);
bool aimdo_rs_plat_init(void);
void aimdo_rs_plat_cleanup(void);
bool aimdo_rs_cuda_budget_deficit(const char **method);
void aimdo_rs_analyze(AimdoContext *context);
uint64_t aimdo_rs_get_total_vram_usage(AimdoContext *context);
void aimdo_rs_cleanup(void);
bool aimdo_rs_init(const int *device_ids, const uint64_t *extra_headrooms, size_t count);

#define CONTROL_INTERNAL __attribute__((visibility("hidden")))

CONTROL_INTERNAL uint64_t aimdo_control_c_tick(void) {
    return GET_TICK();
}

CONTROL_INTERNAL bool aimdo_control_c_current_device(int *device) {
    return check_cu_impl(cuCtxGetDevice(device), "cuCtxGetDevice(&device)", VVERBOSE, DEBUG);
}

CONTROL_INTERNAL bool aimdo_control_c_device_get(int *device, int ordinal) {
    return check_cu_impl(cuDeviceGet(device, ordinal), "cuDeviceGet(&dev, cuda_device_ids[i])",
                         VVERBOSE, DEBUG);
}

CONTROL_INTERNAL bool aimdo_control_c_device_memory(int device, uint64_t *capacity) {
    size_t bytes = 0;
    bool success = check_cu_impl(cuDeviceTotalMem(&bytes, device), "cuDeviceTotalMem(&vram_capacity, dev)",
                                 VVERBOSE, DEBUG);
    *capacity = bytes;
    return success;
}

CONTROL_INTERNAL bool aimdo_control_c_device_integrated(int dev) {
    int integrated = 0;
    return CHECK_CU(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, dev)) &&
           integrated;
}

CONTROL_INTERNAL bool aimdo_control_c_mem_info(size_t *free_bytes, size_t *total_bytes, const char *label) {
    return check_cu_impl(cuMemGetInfo(free_bytes, total_bytes), label, VVERBOSE, DEBUG);
}

CONTROL_INTERNAL void aimdo_control_c_log_integrated_headroom(size_t headroom) {
    log(INFO, "comfy-aimdo integrated Linux GPU RAM headroom: %zu MB\n", headroom / M);
}

CONTROL_INTERNAL void aimdo_control_c_log_device(int dev, size_t capacity) {
    char dev_name[256];
    if (!CHECK_CU(cuDeviceGetName(dev_name, sizeof(dev_name), dev))) {
        sprintf(dev_name, "<unknown>");
    }
    log(INFO, "comfy-aimdo inited for GPU: %s (VRAM: %zu MB)\n", dev_name, capacity / M);
}

CONTROL_INTERNAL void aimdo_control_c_log_poll(bool integrated, size_t available,
                                              size_t total_or_headroom, ssize_t deficit,
                                              size_t recorded, const char *method) {
    const char *caller = "cuda_budget_deficit";
    if (integrated) {
        log(DEBUG,
            "%s: MemAvailable poll available=%zu MB headroom=%zu MB deficit_sync=%zd MB recorded=%zu MB\n",
            caller, available / M, total_or_headroom / M, deficit / (ssize_t)M, recorded / M);
    } else {
        log(DEBUG,
            "%s: device memory poll free=%zu MB total=%zu MB deficit_sync=%zd MB recorded=%zu MB\n",
            caller, available / M, total_or_headroom / M, deficit / (ssize_t)M, recorded / M);
    }
    log(DEBUG, "%s: prevailing method %s\n", caller, method);
}

CONTROL_INTERNAL void aimdo_control_c_log_analysis_header(void) {
    log(DEBUG, "--- VRAM Stats ---\n");
}

CONTROL_INTERNAL void aimdo_control_c_log_analysis(size_t recorded, size_t free_bytes, size_t total) {
    log(DEBUG, "  Aimdo Recorded Usage:  %7zu MB\n", recorded / M);
    log(DEBUG, "  Device: %7zu MB / %7zu MB Free\n", free_bytes / M, total / M);
}

SHARED_EXPORT
void set_simple_vram_headroom(int64_t bytes) {
    aimdo_rs_set_simple_vram_headroom(bytes);
}

SHARED_EXPORT
int64_t get_simple_vram_headroom(void) {
    return aimdo_rs_get_simple_vram_headroom();
}

SHARED_EXPORT
void set_nvml_pressure(bool enabled) {
    (void)enabled;
}

SHARED_EXPORT
void *get_devctx(int device) {
    return aimdo_rs_get_devctx(device);
}

bool set_devctx_for_device(int device) {
    return aimdo_rs_set_devctx_for_device(device);
}

bool set_devctx_for_current_cuda_device(void) {
    return aimdo_rs_set_devctx_for_current_cuda_device();
}

SHARED_EXPORT
bool plat_init(void) {
    return aimdo_rs_plat_init();
}

SHARED_EXPORT
void plat_cleanup(void) {
    aimdo_rs_plat_cleanup();
}

bool cuda_budget_deficit(const char **method) {
    return aimdo_rs_cuda_budget_deficit(method);
}

SHARED_EXPORT
void aimdo_analyze(void *context) {
    aimdo_rs_analyze(context);
}

SHARED_EXPORT
uint64_t get_total_vram_usage(void *context) {
    return aimdo_rs_get_total_vram_usage(context);
}

SHARED_EXPORT
void cleanup(void) {
    aimdo_rs_cleanup();
}

SHARED_EXPORT
bool init(const int *device_ids, const uint64_t *extra_headrooms, size_t count) {
    return aimdo_rs_init(device_ids, extra_headrooms, count);
}
