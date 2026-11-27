#include "cuda-rust-driver-api.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static DriverState state = {
    .id = DRIVER_ID,
    .failing_query = -1,
    .failing_total_device = -1,
};

DriverState *aimdo_test_cuda_state(void) { return &state; }

#ifndef OMIT_GET_PROC_ADDRESS

#define GIB (1ULL << 30)

static CUresult fake_init(unsigned int flags) {
    assert(flags == 0);
    state.init_count++;
    return state.init_error;
}

static CUresult fake_error_string(CUresult error, const char **name) {
    assert(error != CUDA_SUCCESS);
    *name = "simulated CUDA error";
    return CUDA_SUCCESS;
}

static CUresult fake_current_device(CUdevice *device) {
    *device = state.current_device;
    return state.current_device_error;
}

static CUresult fake_device_get(CUdevice *device, int ordinal) {
    *device = ordinal;
    return CUDA_SUCCESS;
}

static CUresult fake_device_attribute(int *value, CUdevice_attribute attribute, CUdevice device) {
    assert(attribute == CU_DEVICE_ATTRIBUTE_INTEGRATED);
    *value = device != 2;
    return CUDA_SUCCESS;
}

static CUresult fake_total_memory(size_t *size, CUdevice device) {
    if (state.before_total_memory) {
        state.before_total_memory(device);
    }
    if (device == state.failing_total_device) {
        return 999;
    }
    *size = device == 2 ? 32 * GIB : device == 7 ? 128 * GIB : 16 * GIB;
    return CUDA_SUCCESS;
}

static CUresult fake_device_name(char *name, int size, CUdevice device) {
    snprintf(name, (size_t)size, "test GPU %d", device);
    return CUDA_SUCCESS;
}

static CUresult fake_memory_info(size_t *free, size_t *total) {
    state.memory_query_count++;
    *free = 18 * GIB;
    *total = 32 * GIB;
    return CUDA_SUCCESS;
}

static CUresult fake_alloc_legacy(CUdeviceptr *pointer, size_t size, CUstream stream) {
    *pointer = size + (stream != NULL) + 101;
    return CUDA_SUCCESS;
}

static CUresult fake_alloc_ptsz(CUdeviceptr *pointer, size_t size, CUstream stream) {
    *pointer = size + (stream != NULL) + 202;
    return CUDA_SUCCESS;
}

static CUresult fake_free_legacy(CUdeviceptr pointer, CUstream stream) {
    return (CUresult)(pointer + (stream != NULL) + 303);
}

static CUresult fake_free_ptsz(CUdeviceptr pointer, CUstream stream) {
    return (CUresult)(pointer + (stream != NULL) + 404);
}

static CUresult pass(void) { return CUDA_SUCCESS; }

static void *resolve(const char *symbol, cuuint64_t flags) {
    if (strcmp(symbol, "cuInit") == 0) return (void *)fake_init;
    if (strcmp(symbol, "cuGetErrorString") == 0) return (void *)fake_error_string;
    if (strcmp(symbol, "cuCtxGetDevice") == 0) return (void *)fake_current_device;
    if (strcmp(symbol, "cuDeviceGet") == 0) return (void *)fake_device_get;
    if (strcmp(symbol, "cuDeviceGetAttribute") == 0) return (void *)fake_device_attribute;
    if (strcmp(symbol, "cuDeviceTotalMem") == 0) return (void *)fake_total_memory;
    if (strcmp(symbol, "cuDeviceGetName") == 0) return (void *)fake_device_name;
    if (strcmp(symbol, "cuMemGetInfo") == 0) return (void *)fake_memory_info;
    if (strcmp(symbol, "cuMemAllocAsync") == 0) {
        return flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? (void *)fake_alloc_legacy : (void *)fake_alloc_ptsz;
    }
    if (strcmp(symbol, "cuMemFreeAsync") == 0) {
        return flags == CU_GET_PROC_ADDRESS_LEGACY_STREAM ? (void *)fake_free_legacy : (void *)fake_free_ptsz;
    }
    return (void *)pass;
}

CUresult cuGetProcAddress(const char *symbol, void **function, int version, cuuint64_t flags,
                          CUdriverProcAddressQueryResult *status) {
    /* Aimdo asks for CUDA's oldest common ABI using the null status convention. */
    assert(status == NULL);
    size_t index = state.query_count++;
    assert(index < sizeof(state.queries) / sizeof(state.queries[0]));
    state.queries[index] = (DriverQuery){symbol, version, flags};
    *function = resolve(symbol, flags);
    if (index == (size_t)state.failing_query) {
        if (!state.query_error) {
            *function = NULL;
        }
        return state.query_error;
    }
    return CUDA_SUCCESS;
}

#endif
