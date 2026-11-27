#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>

#define SHARED_EXPORT
#define CHECK_CU(result) ((result) == CUDA_SUCCESS)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define log(...) ((void)0);
#define log_reset_shots() ((void)0)

typedef unsigned long long CUdeviceptr;
typedef unsigned long long CUmemGenericAllocationHandle;
typedef int CUresult;

enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_UNKNOWN = 999,
};

typedef struct ModelVBAR ModelVBAR;

typedef struct AimdoContext {
    uint64_t capacity;
    uint64_t usage;
    uint64_t clamp;
    ModelVBAR *highest;
    ModelVBAR *lowest;
    bool dirty;
} AimdoContext;

static AimdoContext *context;
static int64_t simple_vram_headroom;

#define vram_capacity (context->capacity)
#define total_vram_usage (context->usage)
#define malloc_async_clamp (context->clamp)
#define highest_priority_p (context->highest)
#define lowest_priority_p (context->lowest)
#define highest_priority (*highest_priority_p)
#define lowest_priority (*lowest_priority_p)
#define vbars_dirty (context->dirty)

static inline void set_devctx(AimdoContext *devctx) { context = devctx; }

static bool malloc_graph_sync_paused(void);
static ssize_t budget_deficit(size_t bytes);
static CUresult cuCtxSynchronize(void);
static CUresult cuMemAddressReserve(CUdeviceptr *address, size_t size, size_t alignment,
                                   CUdeviceptr hint, unsigned long long flags);
static CUresult cuMemAddressFree(CUdeviceptr address, size_t size);
static CUresult cuMemUnmap(CUdeviceptr address, size_t size);
static void unmap_workaround(CUdeviceptr address, size_t size);
static CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
static CUresult three_stooges(CUdeviceptr address, size_t size, int device,
                             CUmemGenericAllocationHandle *handle);
