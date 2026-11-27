#include "cuda-rust-driver-api.h"
#include "plat.h"
#include "aimdo-time.h"
#include "xfer-file.h"

#include <dlfcn.h>
#include <pthread.h>

extern bool plat_init(void);
extern void plat_cleanup(void);
extern bool init(const int *device_ids, const uint64_t *headrooms, size_t count);
extern void cleanup(void);
extern void *get_devctx(int device_id);
extern uint64_t get_total_vram_usage(void *devctx);

#define LEGACY CU_GET_PROC_ADDRESS_LEGACY_STREAM
#define PTSZ CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM

static const DriverQuery expected_queries[] = {
    {"cuInit", 12060, LEGACY},
    {"cuGetErrorString", 12060, LEGACY},
    {"cuCtxGetDevice", 12060, LEGACY},
    {"cuCtxSynchronize", 12060, LEGACY},
    {"cuDeviceGet", 12060, LEGACY},
    {"cuDeviceGetAttribute", 12060, LEGACY},
    {"cuDeviceTotalMem", 12060, LEGACY},
    {"cuDeviceGetName", 12060, LEGACY},
    {"cuDeviceGetUuid", 12060, LEGACY},
    {"cuMemGetInfo", 12060, LEGACY},
    {"cuMemAlloc", 12060, LEGACY},
    {"cuMemFree", 12060, LEGACY},
    {"cuMemAllocAsync", 12060, LEGACY},
    {"cuMemAllocAsync", 12060, PTSZ},
    {"cuMemFreeAsync", 12060, LEGACY},
    {"cuMemFreeAsync", 12060, PTSZ},
    {"cuMemAllocHost", 12060, LEGACY},
    {"cuMemFreeHost", 12060, LEGACY},
    {"cuMemHostRegister", 12060, LEGACY},
    {"cuMemHostUnregister", 12060, LEGACY},
    {"cuMemAddressReserve", 12060, LEGACY},
    {"cuMemAddressFree", 12060, LEGACY},
    {"cuMemCreate", 12060, LEGACY},
    {"cuMemMap", 12060, LEGACY},
    {"cuMemSetAccess", 12060, LEGACY},
    {"cuMemUnmap", 12060, LEGACY},
    {"cuMemRelease", 12060, LEGACY},
    {"cuMemcpyHtoDAsync", 12060, LEGACY},
    {"cuMemcpyDtoHAsync", 12060, LEGACY},
    {"cuEventCreate", 12060, LEGACY},
    {"cuEventDestroy", 12060, LEGACY},
    {"cuEventRecord", 12060, LEGACY},
    {"cuEventSynchronize", 12060, LEGACY},
};

/* These fixtures do not install CUDA function hooks or allocate GPU memory. */
int log_level = __NONE__;
uint64_t log_shot_counter = 1;
static bool xfer_ok = true;
static bool hooks_ok = true;
static unsigned active_allocators;

void aimdo_log(int level, const char *file, int line, const char *format, ...) {
    (void)level;
    (void)file;
    (void)line;
    (void)format;
}

void log_reset_shots(void) { log_shot_counter++; }

bool xfer_file_init(void) {
    puts("file init");
    return xfer_ok;
}

void xfer_file_cleanup(void) { puts("file cleanup"); }

bool aimdo_setup_hooks(void) {
    puts("hooks init");
    assert(g_cuda.p_cuInit);
    return hooks_ok;
}

void aimdo_teardown_hooks(void) {
    puts("hooks cleanup");
    assert(g_cuda.p_cuInit);
}

bool allocations_init(void) {
    printf("allocations init %d\n", g_devctx->_device_id);
    active_allocators++;
    return true;
}

void allocations_cleanup(void) {
    assert(active_allocators);
    printf("allocations cleanup %d\n", g_devctx->_device_id);
    active_allocators--;
}

void hostbuf_file_reader_cleanup(void) {
    printf("hostbuf cleanup %d\n", g_devctx->_device_id);
}

uint64_t vbars_analyze(void *device, bool only_dirty) {
    (void)device;
    (void)only_dirty;
    return 0;
}

void allocations_analyze(bool only_dirty) { (void)only_dirty; }

static void check_empty_dispatch(void) {
    AimdoCudaDispatch empty = {0};
    assert(memcmp(&g_cuda, &empty, sizeof(empty)) == 0);
}

static DriverState *driver_state(void *driver) {
    GetDriverState get_state = (GetDriverState)dlsym(driver, "aimdo_test_cuda_state");
    assert(get_state);
    return get_state();
}

static void close_driver(const char *path, void *driver) {
    assert(dlclose(driver) == 0);
    assert(!dlopen(path, RTLD_NOLOAD | RTLD_LAZY));
}

static void check_queries(DriverState *state, size_t count) {
    assert(state->query_count == count);
    for (size_t i = 0; i < count; i++) {
        const DriverQuery *actual = &state->queries[i];
        const DriverQuery *expected = &expected_queries[i];
        assert(strcmp(actual->name, expected->name) == 0);
        assert(actual->version == expected->version);
        assert(actual->flags == expected->flags);
        printf("driver %d query %s %d %llu\n", state->id, actual->name, actual->version, actual->flags);
    }
    state->query_count = 0;
}

static void check_async_entries(void) {
    CUdeviceptr pointer = 0;
    assert(g_cuda.p_cuGetProcAddress && g_cuda.p_cuInit && !g_cuda.p_cuDeviceGetLuid);
    assert(g_cuda.p_cuMemAllocAsync(&pointer, 1, NULL) == CUDA_SUCCESS && pointer == 102);
    assert(g_cuda.p_cuMemAllocAsync_ptsz(&pointer, 1, NULL) == CUDA_SUCCESS && pointer == 203);
    assert(g_cuda.p_cuMemFreeAsync(1, NULL) == 304);
    assert(g_cuda.p_cuMemFreeAsync_ptsz(1, NULL) == 405);
}

static void check_resolver(const char *no_bootstrap_path) {
    puts("resolver");
    assert(!aimdo_cuda_runtime_init());
    check_empty_dispatch();

    void *no_bootstrap = dlopen(no_bootstrap_path, RTLD_NOW | RTLD_LOCAL);
    assert(no_bootstrap);
    assert(!aimdo_cuda_runtime_init());
    check_empty_dispatch();
    close_driver(no_bootstrap_path, no_bootstrap);

    void *fallback = dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
    assert(fallback);
    DriverState *fallback_state = driver_state(fallback);
    assert(aimdo_cuda_runtime_init());
    check_queries(fallback_state, ARRAY_SIZE(expected_queries));
    check_async_entries();
    assert(fallback_state->init_count == 1);
    aimdo_cuda_runtime_cleanup();
    check_empty_dispatch();

    void *preferred = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    assert(preferred);
    DriverState *state = driver_state(preferred);
    state->failing_query = 13;
    assert(!aimdo_cuda_runtime_init());
    check_queries(state, 14);
    check_empty_dispatch();
    assert(!state->init_count && !fallback_state->query_count);

    state->query_error = 999;
    assert(!aimdo_cuda_runtime_init());
    check_queries(state, 14);
    check_empty_dispatch();
    assert(!state->init_count);

    state->failing_query = -1;
    state->init_error = 999;
    assert(!aimdo_cuda_runtime_init());
    check_queries(state, ARRAY_SIZE(expected_queries));
    check_empty_dispatch();
    assert(state->init_count == 1);

    state->init_error = CUDA_SUCCESS;
    assert(aimdo_cuda_runtime_init());
    assert(aimdo_cuda_runtime_init());
    check_queries(state, ARRAY_SIZE(expected_queries));
    check_async_entries();
    assert(state->init_count == 2);
    assert(fallback_state->init_count == 1 && !fallback_state->query_count);
    aimdo_cuda_runtime_cleanup();
    aimdo_cuda_runtime_cleanup();
    check_empty_dispatch();
    close_driver("libcuda.so.1", preferred);
    close_driver("libcuda.so", fallback);
}

static void *check_thread_context(void *expected) {
    assert(!g_devctx);
    assert(set_devctx_for_current_cuda_device());
    assert(g_devctx == expected);
    return NULL;
}

static void check_control(void) {
    puts("control");
    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    assert(driver);
    DriverState *state = driver_state(driver);

    xfer_ok = false;
    assert(!plat_init());
    check_queries(state, ARRAY_SIZE(expected_queries));
    check_empty_dispatch();
    xfer_ok = true;

    hooks_ok = false;
    assert(!plat_init());
    check_queries(state, ARRAY_SIZE(expected_queries));
    check_empty_dispatch();
    hooks_ok = true;

    assert(plat_init());
    check_queries(state, ARRAY_SIZE(expected_queries));
    int device_ids[] = {2, 7, 8};
    uint64_t headrooms[] = {M, 2 * M, 3 * M};
    state->failing_total_device = 7;
    assert(!init(device_ids, headrooms, 2));
    assert(!g_devctx && !get_devctx(2) && !get_devctx(7) && !active_allocators);

    state->failing_total_device = -1;
    assert(init(device_ids, headrooms, ARRAY_SIZE(device_ids)));
    assert(!g_devctx && active_allocators == 3);
    assert(!init(device_ids, headrooms, 1));
    AimdoContext *discrete = get_devctx(2);
    AimdoContext *integrated = get_devctx(7);
    AimdoContext *small_integrated = get_devctx(8);
    assert(discrete && integrated && small_integrated && !get_devctx(-1));
    assert(discrete->_device_id == 2 && discrete->_vram_capacity == 32ULL * G);
    assert(!discrete->_integrated_device && !discrete->_integrated_ram_headroom);
    assert(discrete->_extra_vram_headroom == M && discrete->_malloc_async_clamp == UINT64_MAX);
    assert(discrete->_hostbuf_file_reader_active == -1);
    assert(integrated->_vram_capacity == 128ULL * G && integrated->_integrated_device);
    assert(integrated->_integrated_ram_headroom == 8ULL * G);
    assert(small_integrated->_integrated_ram_headroom == 2ULL * G);

    state->current_device = 7;
    assert(set_devctx_for_device(2) && g_devctx == discrete);
    pthread_t thread;
    assert(pthread_create(&thread, NULL, check_thread_context, integrated) == 0);
    assert(pthread_join(thread, NULL) == 0);
    assert(g_devctx == discrete);

    const char *method = "unset";
    discrete->_total_vram_usage = 42;
    assert(cuda_budget_deficit(&method));
    assert(state->memory_query_count == 1 && strcmp(method, "cuMemGetInfo") == 0);
    assert(discrete->_total_vram_last_check == 42);
    assert(discrete->_deficit_sync == (ssize_t)VRAM_HEADROOM - 18LL * G);
    discrete->_control_timestamp_last_check = GET_TICK();
    method = "unset";
    assert(cuda_budget_deficit(&method));
    assert(state->memory_query_count == 1 && strcmp(method, "unset") == 0);

    state->current_device_error = 999;
    assert(!set_devctx_for_current_cuda_device() && !g_devctx);
    assert(!set_devctx_for_device(-1) && !g_devctx);
    assert(get_total_vram_usage(discrete) == 42);
    cleanup();
    assert(!g_devctx && !get_devctx(2) && !active_allocators);
    plat_cleanup();
    check_empty_dispatch();
    close_driver("libcuda.so.1", driver);
}

#ifdef AIMDO_RUST_CUDA
static pthread_mutex_t publication_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t publication_changed = PTHREAD_COND_INITIALIZER;
static bool initializing_second_device;
static bool release_second_device;
static bool initialization_finished;

static void pause_second_device(CUdevice device) {
    if (device != 7) {
        return;
    }
    assert(pthread_mutex_lock(&publication_mutex) == 0);
    initializing_second_device = true;
    assert(pthread_cond_broadcast(&publication_changed) == 0);
    while (!release_second_device) {
        assert(pthread_cond_wait(&publication_changed, &publication_mutex) == 0);
    }
    assert(pthread_mutex_unlock(&publication_mutex) == 0);
}

static void *initialize_devices(void *unused) {
    (void)unused;
    int devices[] = {2, 7};
    uint64_t headrooms[] = {M, 2 * M};
    assert(init(devices, headrooms, ARRAY_SIZE(devices)));
    assert(!g_devctx);
    assert(pthread_mutex_lock(&publication_mutex) == 0);
    initialization_finished = true;
    assert(pthread_cond_broadcast(&publication_changed) == 0);
    assert(pthread_mutex_unlock(&publication_mutex) == 0);
    return NULL;
}

static void check_context_publication(void) {
    void *driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    assert(driver && plat_init());
    DriverState *state = driver_state(driver);
    state->current_device = 7;
    state->before_total_memory = pause_second_device;

    pthread_t worker;
    assert(pthread_create(&worker, NULL, initialize_devices, NULL) == 0);
    assert(pthread_mutex_lock(&publication_mutex) == 0);
    while (!initializing_second_device && !initialization_finished) {
        assert(pthread_cond_wait(&publication_changed, &publication_mutex) == 0);
    }
    assert(initializing_second_device && !initialization_finished);
    assert(!get_devctx(2) && !get_devctx(7));
    assert(!set_devctx_for_device(2) && !g_devctx);
    assert(!set_devctx_for_current_cuda_device() && !g_devctx);
    release_second_device = true;
    assert(pthread_cond_broadcast(&publication_changed) == 0);
    assert(pthread_mutex_unlock(&publication_mutex) == 0);
    assert(pthread_join(worker, NULL) == 0);

    AimdoContext *discrete = get_devctx(2);
    AimdoContext *integrated = get_devctx(7);
    assert(discrete && discrete->_vram_capacity == 32ULL * G && discrete->_extra_vram_headroom == M);
    assert(integrated && integrated->_vram_capacity == 128ULL * G && integrated->_extra_vram_headroom == 2 * M);
    assert(integrated->_integrated_device && integrated->_integrated_ram_headroom == 8ULL * G);
    assert(set_devctx_for_current_cuda_device() && g_devctx == integrated);
    cleanup();
    assert(!g_devctx && !get_devctx(2) && !get_devctx(7));
    plat_cleanup();
    close_driver("libcuda.so.1", driver);
    puts("CUDA contexts become visible only after all devices initialize.");
}
#endif

int main(int argc, char **argv) {
    assert(argc == 2);
#ifdef AIMDO_RUST_CUDA
    if (strcmp(argv[1], "--check-publication") == 0) {
        check_context_publication();
        return 0;
    }
#endif
    check_resolver(argv[1]);
    check_control();
    return 0;
}
