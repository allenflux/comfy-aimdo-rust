#include <inttypes.h>
#include <stdio.h>

#include "model-vbar.c"

#define PAGE ((size_t)VBAR_PAGE_SIZE)

enum {
    FAULT_SUCCESS = 0,
    FAULT_OOM = 1,
    FAULT_ERROR = 2,
};

typedef struct Mapping {
    CUdeviceptr address;
    CUmemGenericAllocationHandle handle;
    AimdoContext *owner;
} Mapping;

static Mapping mappings[32];
static CUdeviceptr next_address = 1ULL << 32;
static CUmemGenericAllocationHandle next_handle = 1;
static unsigned sync_calls;
static unsigned fail_sync_call;
static bool sync_paused;
static bool writeback_ok = true;
static CUresult map_results[2];
static unsigned map_result_count;
static unsigned map_result_index;

static bool malloc_graph_sync_paused(void) { return sync_paused; }

static ssize_t budget_deficit(size_t bytes) {
    return (ssize_t)(total_vram_usage + bytes) + simple_vram_headroom - (ssize_t)vram_capacity;
}

static CUresult cuCtxSynchronize(void) {
    bool fails = ++sync_calls == fail_sync_call;
    printf("sync %d\n", fails);
    return fails ? CUDA_ERROR_UNKNOWN : CUDA_SUCCESS;
}

static CUresult cuMemAddressReserve(CUdeviceptr *address, size_t size, size_t alignment,
                                   CUdeviceptr hint, unsigned long long flags) {
    assert(!alignment && !hint && !flags && size > 0);
    *address = next_address;
    next_address += size;
    printf("reserve %llu %zu\n", *address, size);
    return CUDA_SUCCESS;
}

static CUresult cuMemAddressFree(CUdeviceptr address, size_t size) {
    for (size_t i = 0; i < sizeof(mappings) / sizeof(mappings[0]); i++) {
        assert(!mappings[i].address || mappings[i].address < address || mappings[i].address >= address + size);
    }
    printf("address-free %llu %zu\n", address, size);
    return CUDA_SUCCESS;
}

static CUresult cuMemUnmap(CUdeviceptr address, size_t size) {
    assert(size == PAGE);
    for (size_t i = 0; i < sizeof(mappings) / sizeof(mappings[0]); i++) {
        if (mappings[i].address == address) {
            assert(mappings[i].owner == context);
            mappings[i].address = 0;
            printf("unmap %llu %zu\n", address, size);
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static void unmap_workaround(CUdeviceptr address, size_t size) {
    printf("unmap-workaround %llu %zu\n", address, size);
}

static CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    for (size_t i = 0; i < sizeof(mappings) / sizeof(mappings[0]); i++) {
        if (mappings[i].handle == handle) {
            assert(!mappings[i].address && mappings[i].owner == context);
            mappings[i] = (Mapping){0};
            printf("release %llu\n", handle);
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static CUresult three_stooges(CUdeviceptr address, size_t size, int device,
                             CUmemGenericAllocationHandle *handle) {
    assert(size == PAGE);
    CUresult result = CUDA_SUCCESS;
    if (map_result_index < map_result_count) {
        result = map_results[map_result_index++];
    }
    printf("map %llu %zu %d %d\n", address, size, device, result);
    if (result != CUDA_SUCCESS) {
        return result;
    }
    for (size_t i = 0; i < sizeof(mappings) / sizeof(mappings[0]); i++) {
        assert(!mappings[i].address || mappings[i].address != address);
    }
    for (size_t i = 0; i < sizeof(mappings) / sizeof(mappings[0]); i++) {
        if (!mappings[i].handle) {
            *handle = next_handle++;
            mappings[i] = (Mapping){address, *handle, context};
            total_vram_usage += size;
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static bool writeback(uint64_t offset, uint64_t size) {
    printf("writeback %" PRIu64 " %" PRIu64 " %d\n", offset, size, writeback_ok);
    return writeback_ok;
}

static void observe(AimdoContext *device, ModelVBAR *model) {
    printf("state %zu %zu %zu %" PRIu64 " %d:", model->watermark, model->watermark_limit,
           model->resident_count, device->usage, device->dirty);
    for (size_t p = 0; p < model->nr_pages; p++) {
        ResidentPage *page = &model->residency_map[p];
        printf(" %llu/%" PRIu32 "/%zu", page->handle, page->pin_count, page->serial);
    }
    puts("");
    assert(vbar_loaded_size(device, model) == model->resident_count * PAGE);
}

static void fault(AimdoContext *device, ModelVBAR *model, size_t offset, size_t size,
                  int expected, uint32_t first_signature, uint32_t second_signature) {
    uint32_t signature[3] = {0};
    int result = vbar_fault(device, model, offset, size, signature);
    printf("fault result %d: %" PRIu32 " %" PRIu32 " %" PRIu32 "\n", result,
           signature[0], signature[1], signature[2]);
    assert(result == expected);
    assert(signature[0] == first_signature && signature[1] == second_signature);
    observe(device, model);
}

static void destroy(AimdoContext *device) {
    assert(!device->usage);
    assert(device->highest->lower == device->lowest && device->lowest->higher == device->highest);
    free(device->highest);
}

static void residency(void) {
    puts("residency");
    AimdoContext device = {.capacity = 8 * PAGE};
    ModelVBAR *model = vbar_allocate(&device, 3 * PAGE, 0, NULL);
    assert(model);
    fault(&device, model, PAGE - 1, 2, FAULT_SUCCESS, 1, 1);
    fault(&device, model, PAGE - 1, 2, FAULT_SUCCESS, 1, 1);
    assert(vbar_unpin(&device, model, PAGE - 1, 2));
    assert(vbar_free_memory(&device, model, PAGE) == 0);
    assert(model->watermark == 0 && model->resident_count == 2);
    fault(&device, model, PAGE - 1, 2, FAULT_OOM, 0, 0);
    assert(vbar_unpin(&device, model, PAGE - 1, 2));
    assert(!device.usage);
    observe(&device, model);

    vbar_prioritize(&device, model, 0);
    fault(&device, model, 0, PAGE, FAULT_SUCCESS, 2, 0);
    assert(vbar_unpin(&device, model, 0, PAGE));
    vbar_set_watermark_limit(&device, model, PAGE);
    assert(vbar_free_memory(&device, model, PAGE) == 0);
    assert(model->watermark == 1 && model->resident_count == 1);
    observe(&device, model);
    assert(vbar_set_watermark(&device, model, 0));
    vbar_free(&device, model);
    destroy(&device);
}

static void pressure(void) {
    puts("pressure");
    AimdoContext device = {.capacity = 5 * PAGE};
    AimdoContext other_device = {.capacity = 2 * PAGE};
    ModelVBAR *low = vbar_allocate(&device, 2 * PAGE, 0, NULL);
    ModelVBAR *high = vbar_allocate(&device, 3 * PAGE, 0, NULL);
    ModelVBAR *other = vbar_allocate(&other_device, 2 * PAGE, 1, NULL);
    assert(low && high && other);
    fault(&device, low, 0, 2 * PAGE, FAULT_SUCCESS, 1, 1);
    assert(vbar_unpin(&device, low, 0, 2 * PAGE));
    fault(&other_device, other, 0, PAGE, FAULT_SUCCESS, 1, 0);
    fault(&device, high, 0, 2 * PAGE, FAULT_SUCCESS, 1, 1);
    assert(low->watermark == 1 && low->resident_count == 1);
    observe(&device, low);

    sync_paused = true;
    unsigned before = sync_calls;
    assert(vbars_free(PAGE));
    assert(!low->resident_count && sync_calls == before);
    assert(vbars_free(PAGE));
    assert(high->watermark == 0 && high->resident_count == 2 && sync_calls == before);
    sync_paused = false;

    observe(&device, high);
    assert(vbar_unpin(&device, high, 0, 2 * PAGE));
    assert(!device.usage && other_device.usage == PAGE);
    vbar_free(&device, low);
    vbar_free(&device, high);
    vbar_free(&other_device, other);
    destroy(&device);
    destroy(&other_device);
}

static void allocation_failures(void) {
    puts("allocation failures");
    AimdoContext device = {.capacity = 6 * PAGE};
    ModelVBAR *low = vbar_allocate(&device, 2 * PAGE, 0, NULL);
    ModelVBAR *high = vbar_allocate(&device, 3 * PAGE, 0, NULL);
    assert(low && high);
    fault(&device, low, 0, 2 * PAGE, FAULT_SUCCESS, 1, 1);
    assert(vbar_unpin(&device, low, 0, 2 * PAGE));

    map_results[0] = CUDA_ERROR_OUT_OF_MEMORY;
    map_result_count = 1;
    map_result_index = 0;
    fault(&device, high, 0, PAGE, FAULT_SUCCESS, 1, 0);
    assert(low->resident_count == 1);

    map_results[0] = CUDA_SUCCESS;
    map_results[1] = CUDA_ERROR_UNKNOWN;
    map_result_count = 2;
    map_result_index = 0;
    fault(&device, high, PAGE, 2 * PAGE, FAULT_ERROR, 1, 0);
    assert(high->resident_count == 2);
    assert(!high->residency_map[1].pin_count);

    map_results[0] = CUDA_ERROR_OUT_OF_MEMORY;
    map_results[1] = CUDA_ERROR_OUT_OF_MEMORY;
    map_result_index = 0;
    fault(&device, high, 2 * PAGE, PAGE, FAULT_ERROR, 0, 0);
    assert(!low->resident_count && high->resident_count == 2);

    vbar_free(&device, low);
    vbar_free(&device, high);
    destroy(&device);
}

static void writeback_failures(void) {
    puts("writeback failures");
    AimdoContext device = {.capacity = 4 * PAGE};
    ModelVBAR *model = vbar_allocate(&device, PAGE + 17, 0, writeback);
    assert(model);
    fault(&device, model, 0, PAGE + 17, FAULT_SUCCESS, 1, 1);
    assert(vbar_set_watermark(&device, model, 0));
    writeback_ok = false;
    assert(!vbar_unpin(&device, model, 0, PAGE + 17));
    assert(!model->residency_map[0].pin_count && !model->residency_map[1].pin_count);
    assert(model->resident_count == 2);
    observe(&device, model);

    writeback_ok = true;
    assert(vbar_unpin(&device, model, 0, PAGE + 17));
    assert(!model->resident_count);
    vbar_prioritize(&device, model, 0);
    fault(&device, model, PAGE, 17, FAULT_SUCCESS, 2, 0);
    assert(vbar_unpin(&device, model, PAGE, 17));

    fail_sync_call = sync_calls + 2;
    assert(vbar_free_memory(&device, model, PAGE) == -1);
    assert(model->watermark == 2 && model->resident_count == 1);
    writeback_ok = false;
    assert(!vbars_free(PAGE));
    assert(model->watermark == 2 && model->resident_count == 1);
    observe(&device, model);
    vbar_free(&device, model);
    destroy(&device);
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    residency();
    pressure();
    allocation_failures();
    writeback_failures();
    return 0;
}
