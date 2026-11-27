#include "gpu_abi.h"

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MIB (1ULL << 20)
#define CAPACITY (32ULL << 30)
#define TABLE_SIZE 128
#define HOOK_TARGET __attribute__((noinline, patchable_function_entry(16, 0)))

typedef struct Mapping {
    CUdeviceptr address;
    size_t size;
    CUmemGenericAllocationHandle handle;
} Mapping;

static Mapping reservations[TABLE_SIZE];
static Mapping allocations[TABLE_SIZE];
static Mapping mappings[TABLE_SIZE];
static size_t handles[TABLE_SIZE];
static CUdeviceptr next_address = 1ULL << 34;
static unsigned raw_calls;
static unsigned host_registrations;
static unsigned event_count;

static void add(Mapping *table, Mapping value) {
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        assert(!table[i].size || value.address + value.size <= table[i].address ||
               table[i].address + table[i].size <= value.address);
    }
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        if (!table[i].size) {
            table[i] = value;
            return;
        }
    }
    abort();
}

static size_t remove_mapping(Mapping *table, CUdeviceptr address) {
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        if (table[i].size && table[i].address == address) {
            size_t size = table[i].size;
            table[i] = (Mapping){0};
            return size;
        }
    }
    abort();
}

static CUdeviceptr address_for(size_t size, size_t alignment) {
    if (!alignment) {
        alignment = 2 * MIB;
    }
    CUdeviceptr address = (next_address + alignment - 1) & ~(alignment - 1);
    next_address = address + size;
    return address;
}

static CUresult raw_alloc(CUdeviceptr *pointer, size_t size) {
    raw_calls++;
    assert(size > 0);
    *pointer = address_for(size, 0);
    add(allocations, (Mapping){.address = *pointer, .size = size});
    return CUDA_SUCCESS;
}

static CUresult raw_free(CUdeviceptr pointer) {
    raw_calls++;
    if (pointer) {
        remove_mapping(allocations, pointer);
    }
    return CUDA_SUCCESS;
}

HOOK_TARGET static CUresult alloc_sync(CUdeviceptr *pointer, size_t size) {
    return raw_alloc(pointer, size);
}

HOOK_TARGET static CUresult free_sync(CUdeviceptr pointer) {
    return raw_free(pointer);
}

HOOK_TARGET static CUresult alloc_async(CUdeviceptr *pointer, size_t size, CUstream stream) {
    (void)stream;
    return raw_alloc(pointer, size);
}

HOOK_TARGET static CUresult free_async(CUdeviceptr pointer, CUstream stream) {
    (void)stream;
    return raw_free(pointer);
}

HOOK_TARGET static CUresult alloc_async_ptsz(CUdeviceptr *pointer, size_t size, CUstream stream) {
    (void)stream;
    return raw_alloc(pointer, size);
}

HOOK_TARGET static CUresult free_async_ptsz(CUdeviceptr pointer, CUstream stream) {
    (void)stream;
    return raw_free(pointer);
}

static CUresult reserve(CUdeviceptr *pointer, size_t size, size_t alignment, CUdeviceptr address,
                        unsigned long long flags) {
    assert(size && !address && !flags);
    *pointer = address_for(size, alignment);
    add(reservations, (Mapping){.address = *pointer, .size = size});
    return CUDA_SUCCESS;
}

static CUresult free_address(CUdeviceptr pointer, size_t size) {
    assert(remove_mapping(reservations, pointer) == size);
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        assert(!mappings[i].size || mappings[i].address < pointer || mappings[i].address >= pointer + size);
    }
    return CUDA_SUCCESS;
}

static CUresult create(CUmemGenericAllocationHandle *handle, size_t size,
                       const CUmemAllocationProp *properties, unsigned long long flags) {
    assert(size && !flags && properties->type == CU_MEM_ALLOCATION_TYPE_PINNED);
    assert(properties->location.type == CU_MEM_LOCATION_TYPE_DEVICE && properties->location.id == 2);
    for (size_t i = 1; i < TABLE_SIZE; i++) {
        if (!handles[i]) {
            handles[i] = size;
            *handle = i;
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static CUresult map(CUdeviceptr pointer, size_t size, size_t offset,
                    CUmemGenericAllocationHandle handle, unsigned long long flags) {
    assert(size && !flags && handle < TABLE_SIZE && offset + size <= handles[handle]);
    bool reserved = false;
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        if (reservations[i].size && reservations[i].address <= pointer &&
            pointer + size <= reservations[i].address + reservations[i].size) {
            reserved = true;
            break;
        }
    }
    assert(reserved);
    add(mappings, (Mapping){pointer, size, handle});
    return CUDA_SUCCESS;
}

static CUresult access(CUdeviceptr pointer, size_t size, const CUmemAccessDesc *descriptors,
                       size_t count) {
    assert(pointer && size && count == 1);
    assert(descriptors->location.type == CU_MEM_LOCATION_TYPE_DEVICE && descriptors->location.id == 2);
    assert(descriptors->flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    return CUDA_SUCCESS;
}

static CUresult unmap(CUdeviceptr pointer, size_t size) {
    size_t unmapped = 0;
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        CUdeviceptr address = mappings[i].address;
        if (mappings[i].size && address >= pointer && address < pointer + size) {
            assert(address + mappings[i].size <= pointer + size);
            unmapped += mappings[i].size;
            mappings[i] = (Mapping){0};
        }
    }
    assert(unmapped == size);
    return CUDA_SUCCESS;
}

static CUresult release(CUmemGenericAllocationHandle handle) {
    assert(handle < TABLE_SIZE && handles[handle]);
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        assert(mappings[i].handle != handle);
    }
    handles[handle] = 0;
    return CUDA_SUCCESS;
}

static CUresult init(unsigned flags) {
    assert(!flags);
    return CUDA_SUCCESS;
}

static CUresult error_string(CUresult error, const char **message) {
    assert(error != CUDA_SUCCESS);
    *message = "simulated CUDA error";
    return CUDA_SUCCESS;
}

static CUresult current_device(CUdevice *device) {
    *device = 2;
    return CUDA_SUCCESS;
}

static CUresult device_get(CUdevice *device, int ordinal) {
    assert(ordinal == 2);
    return current_device(device);
}

static CUresult device_attribute(int *value, CUdevice_attribute attribute, CUdevice device) {
    assert(attribute == CU_DEVICE_ATTRIBUTE_INTEGRATED && device == 2);
    *value = 0;
    return CUDA_SUCCESS;
}

static CUresult device_total(size_t *size, CUdevice device) {
    assert(device == 2);
    *size = CAPACITY;
    return CUDA_SUCCESS;
}

static CUresult device_name(char *name, int length, CUdevice device) {
    assert(device == 2);
    snprintf(name, (size_t)length, "AIMDO smoke device");
    return CUDA_SUCCESS;
}

static CUresult memory_info(size_t *free, size_t *total) {
    size_t usage = 0;
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        usage += allocations[i].size + handles[i];
    }
    assert(usage <= CAPACITY);
    *total = CAPACITY;
    *free = CAPACITY - usage;
    return CUDA_SUCCESS;
}

static CUresult register_host(void *pointer, size_t size, unsigned flags) {
    assert(pointer && size && flags == 0);
    host_registrations++;
    return CUDA_SUCCESS;
}

static CUresult unregister_host(void *pointer) {
    assert(pointer && host_registrations);
    host_registrations--;
    return CUDA_SUCCESS;
}

static CUresult event_create(CUevent *event, unsigned flags) {
    assert(flags == CU_EVENT_DISABLE_TIMING);
    *event = malloc(1);
    assert(*event);
    event_count++;
    return CUDA_SUCCESS;
}

static CUresult event_destroy(CUevent event) {
    assert(event && event_count);
    event_count--;
    free(event);
    return CUDA_SUCCESS;
}

static CUresult event_record(CUevent event, CUstream stream) {
    (void)stream;
    assert(event);
    return CUDA_SUCCESS;
}

static CUresult event_sync(CUevent event) {
    assert(event);
    return CUDA_SUCCESS;
}

static CUresult sync_context(void) { return CUDA_SUCCESS; }

static CUresult unused(void) {
    abort();
}

CUresult cuGetProcAddress(const char *name, void **pointer, int version, cuuint64_t flags,
                          CUdriverProcAddressQueryResult *status) {
    assert(version == 12060 && status == NULL);
    assert(flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM || flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM);
    bool ptsz = flags == CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM;
    if (strcmp(name, "cuMemAllocAsync") == 0) {
        *pointer = ptsz ? (void *)alloc_async_ptsz : (void *)alloc_async;
        return CUDA_SUCCESS;
    }
    if (strcmp(name, "cuMemFreeAsync") == 0) {
        *pointer = ptsz ? (void *)free_async_ptsz : (void *)free_async;
        return CUDA_SUCCESS;
    }
    assert(!ptsz);
    const struct { const char *name; void *function; } functions[] = {
        {"cuInit", init}, {"cuGetErrorString", error_string}, {"cuCtxGetDevice", current_device},
        {"cuCtxSynchronize", sync_context}, {"cuDeviceGet", device_get},
        {"cuDeviceGetAttribute", device_attribute}, {"cuDeviceTotalMem", device_total},
        {"cuDeviceGetName", device_name}, {"cuMemGetInfo", memory_info},
        {"cuMemAlloc", alloc_sync}, {"cuMemFree", free_sync}, {"cuMemAddressReserve", reserve},
        {"cuMemAddressFree", free_address}, {"cuMemCreate", create}, {"cuMemMap", map},
        {"cuMemSetAccess", access}, {"cuMemUnmap", unmap}, {"cuMemRelease", release},
        {"cuMemHostRegister", register_host}, {"cuMemHostUnregister", unregister_host},
        {"cuEventCreate", event_create}, {"cuEventDestroy", event_destroy},
        {"cuEventRecord", event_record}, {"cuEventSynchronize", event_sync},
        {"cuDeviceGetUuid", unused}, {"cuMemAllocHost", unused}, {"cuMemFreeHost", unused},
        {"cuMemcpyHtoDAsync", unused}, {"cuMemcpyDtoHAsync", unused},
    };
    for (size_t i = 0; i < sizeof(functions) / sizeof(functions[0]); i++) {
        if (strcmp(functions[i].name, name) == 0) {
            *pointer = functions[i].function;
            return CUDA_SUCCESS;
        }
    }
    *pointer = NULL;
    return CUDA_ERROR_UNKNOWN;
}

unsigned aimdo_smoke_raw_calls(void) { return raw_calls; }

unsigned aimdo_smoke_live_resources(void) {
    unsigned live = host_registrations + event_count;
    for (size_t i = 0; i < TABLE_SIZE; i++) {
        live += !!reservations[i].size + !!allocations[i].size + !!mappings[i].size + !!handles[i];
    }
    return live;
}
