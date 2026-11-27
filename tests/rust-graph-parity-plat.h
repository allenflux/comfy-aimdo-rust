#pragma once

#include "gpu_dispatch.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define M (1024 * 1024)
#define SHARED_EXPORT
#define ALIGN_UP(a, b) (((a) + (b) - 1) & ~((b) - 1))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define cuMemAddressReserve g_cuda.p_cuMemAddressReserve
#define cuMemAddressFree g_cuda.p_cuMemAddressFree
#define cuMemCreate g_cuda.p_cuMemCreate
#define cuMemMap g_cuda.p_cuMemMap
#define cuMemSetAccess g_cuda.p_cuMemSetAccess
#define cuMemUnmap g_cuda.p_cuMemUnmap
#define cuMemRelease g_cuda.p_cuMemRelease
#define cuCtxSynchronize g_cuda.p_cuCtxSynchronize
#define unmap_workaround(address, size)

typedef struct Context {
    int _device_id;
    uint64_t usage;
    void *rogue_list;
    void *candidate_list;
} Context;

extern _Thread_local Context *g_devctx;

#define total_vram_usage (g_devctx->usage)
#define rogues (g_devctx->rogue_list)
#define global_rogue_candidates (g_devctx->candidate_list)

void allocations_lock(void);
void allocations_unlock(void);
bool vbars_free(ssize_t size);

static inline void set_devctx(void *context) {
    g_devctx = context;
}

static inline ssize_t budget_deficit(size_t size) {
    return (ssize_t)size - 1024 * M;
}
