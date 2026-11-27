#include "plat.h"

#define VBAR_PAGE_SIZE (32 << 20)

#define VBAR_GET_PAGE_NR(x) ((x) / VBAR_PAGE_SIZE)
#define VBAR_GET_PAGE_NR_UP(x) VBAR_GET_PAGE_NR((x) + VBAR_PAGE_SIZE - 1)

typedef struct ResidentPage {
    CUmemGenericAllocationHandle handle;
    uint32_t pin_count;
    size_t serial;
} ResidentPage;

typedef struct ModelVBAR {
    CUdeviceptr vbar;
    size_t nr_pages;
    size_t watermark;
    size_t watermark_limit;

    int device;

    void *higher;
    void *lower;

    size_t resident_count;
    uint64_t pageout_size;
    bool (*pageout_callback)(uint64_t offset, uint64_t size);

    ResidentPage residency_map[1]; /* Must be last! */
} ModelVBAR;

/* The Rust page manager uses the same layout as these C allocations. */
_Static_assert(sizeof(size_t) == 8 && sizeof(ssize_t) == 8 && sizeof(void *) == 8,
               "Rust VBAR requires a 64-bit target");
_Static_assert(sizeof(bool) == 1 && sizeof(int) == 4, "Rust VBAR argument layout");
_Static_assert(VBAR_PAGE_SIZE == 33554432, "Rust VBAR page size");
_Static_assert(CUDA_SUCCESS == 0 && CUDA_ERROR_OUT_OF_MEMORY == 2, "Rust VBAR CUDA results");
_Static_assert(sizeof(CUdeviceptr) == 8 && sizeof(CUmemGenericAllocationHandle) == 8,
               "Rust VBAR CUDA handle layout");
_Static_assert(offsetof(ResidentPage, handle) == 0 &&
               offsetof(ResidentPage, pin_count) == 8 &&
               offsetof(ResidentPage, serial) == 16 && sizeof(ResidentPage) == 24,
               "Rust VBAR resident page layout");
_Static_assert(offsetof(ModelVBAR, vbar) == 0 &&
               offsetof(ModelVBAR, nr_pages) == 8 &&
               offsetof(ModelVBAR, watermark) == 16 &&
               offsetof(ModelVBAR, watermark_limit) == 24 &&
               offsetof(ModelVBAR, device) == 32 &&
               offsetof(ModelVBAR, higher) == 40 &&
               offsetof(ModelVBAR, lower) == 48 &&
               offsetof(ModelVBAR, resident_count) == 56 &&
               offsetof(ModelVBAR, pageout_size) == 64 &&
               offsetof(ModelVBAR, pageout_callback) == 72 &&
               offsetof(ModelVBAR, residency_map) == 80,
               "Rust VBAR state layout");

#if defined(__GNUC__) && !defined(_WIN32)
#define VBAR_INTERNAL __attribute__((visibility("hidden")))
#else
#define VBAR_INTERNAL
#endif

VBAR_INTERNAL int aimdo_rs_vbar_modify_page(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin);
VBAR_INTERNAL bool aimdo_rs_vbars_free(ssize_t size);
VBAR_INTERNAL int aimdo_rs_vbar_fault(ModelVBAR *mv, uint64_t offset, uint64_t size, uint32_t *signature);
VBAR_INTERNAL bool aimdo_rs_vbar_unpin(ModelVBAR *mv, uint64_t offset, uint64_t size);
VBAR_INTERNAL int64_t aimdo_rs_vbar_free_memory(ModelVBAR *mv, uint64_t size);

static inline void one_time_setup() {
    if (!highest_priority_p) {
        highest_priority_p = calloc(2, sizeof(*highest_priority_p));
        if (!highest_priority_p) {
            log(CRITICAL, "Host OOM\n");
            return;
        }
        lowest_priority_p = highest_priority_p + 1;
    }
    if (!highest_priority.lower) {
        assert(!lowest_priority.higher);
        highest_priority.lower = &lowest_priority;
        lowest_priority.higher = &highest_priority;
    }
}

VBAR_INTERNAL void aimdo_vbar_c_prepare_pressure(void) {
    one_time_setup();
    vbars_dirty = true;
}

VBAR_INTERNAL ModelVBAR *aimdo_vbar_c_first(void) {
    return lowest_priority.higher;
}

VBAR_INTERNAL ModelVBAR *aimdo_vbar_c_end(void) {
    return &highest_priority;
}

VBAR_INTERNAL bool aimdo_vbar_c_sync_paused(void) {
    return malloc_graph_sync_paused();
}

VBAR_INTERNAL bool aimdo_vbar_c_sync(void) {
    return CHECK_CU(cuCtxSynchronize());
}

VBAR_INTERNAL ssize_t aimdo_vbar_c_deficit(size_t size) {
    return budget_deficit(size);
}

VBAR_INTERNAL ssize_t aimdo_vbar_c_surplus(size_t size) {
    return (ssize_t)vram_capacity - ((ssize_t)(total_vram_usage + size) + (ssize_t)simple_vram_headroom);
}

VBAR_INTERNAL int aimdo_vbar_c_map(uint64_t addr, int device, CUmemGenericAllocationHandle *handle) {
    return three_stooges(addr, VBAR_PAGE_SIZE, device, handle);
}

VBAR_INTERNAL void aimdo_vbar_c_unmap(uint64_t addr, uint64_t handle) {
    CHECK_CU(cuMemUnmap(addr, VBAR_PAGE_SIZE));
    unmap_workaround(addr, VBAR_PAGE_SIZE);
    CHECK_CU(cuMemRelease(handle));
    total_vram_usage -= VBAR_PAGE_SIZE;
}

VBAR_INTERNAL void aimdo_vbar_c_log(int event, size_t page) {
    switch (event) {
    case 0:
        log(VVERBOSE, "VBAR Allocation is above watermark\n");
        break;
    case 1:
        log(DEBUG, "VBAR allocation cancelled due to allocation-check watermark reduction\n");
        break;
    case 2:
        log(VERBOSE, "VBAR needs to allocate VRAM for page %d\n", (int)page);
        break;
    case 3:
        log(AIMDO_LOG_ERROR, "VRAM Allocation failed (non OOM)\n");
        break;
    case 4:
        log(DEBUG, "VBAR allocator attempt exceeds available VRAM ...\n");
        break;
    case 5:
        log(DEBUG, "VBAR allocation cancelled due to backup-free watermark reduction\n");
        break;
    case 6:
        log(AIMDO_LOG_ERROR, "VRAM Allocation failed\n");
        break;
    }
}

SHARED_EXPORT
uint64_t vbars_analyze(void *devctx, bool only_dirty) {
    size_t calculated_total_vram = 0;

    set_devctx((AimdoContext *)devctx);

    one_time_setup();
    if (only_dirty && !vbars_dirty) {
        return 0;
    }
    vbars_dirty = false;
    log(DEBUG, "---------------- VBAR Usage ---------------\n")

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        size_t actual_resident_count = 0;

        for (size_t p = 0; p < i->nr_pages; p++) {
            ResidentPage *rp = &i->residency_map[p];

            if (rp->handle) {
                actual_resident_count++;

                if (p >= i->watermark) {
                    log(WARNING, "VBAR %p: Resident page %zu is ABOVE watermark %zu\n",
                        (void*)i, p, i->watermark);
                }

                if (rp->pin_count) {
                    log(WARNING, "VBAR %p: Page %zu pin_count=%u\n", (void*)i, p, rp->pin_count);
                }
            }
        }

        if (actual_resident_count != i->resident_count) {
            log(WARNING, "VBAR %p: resident_count sync error! Struct: %zu, Actual: %zu\n",
                (void*)i, i->resident_count, actual_resident_count);
        }

        calculated_total_vram += (actual_resident_count * VBAR_PAGE_SIZE);

        log(DEBUG, "VBAR %p: Actual Resident VRAM = %zu MB\n",
            (void*)i, (actual_resident_count * VBAR_PAGE_SIZE) / M);
    }

    log(DEBUG, "Total VRAM for VBARs: %zu MB\n", calculated_total_vram / M);
    return (uint64_t)calculated_total_vram;
}

/* -1: writeback failed, 0: retained, 1: freed. */
static inline int mod1(ModelVBAR *mv, size_t page_nr, bool do_free, bool do_unpin) {
    return aimdo_rs_vbar_modify_page(mv, page_nr, do_free, do_unpin);
}

bool vbars_free(ssize_t size) {
    return aimdo_rs_vbars_free(size);
}

static inline void remove_vbar(ModelVBAR *mv) {
    ((ModelVBAR *)mv->lower)->higher = mv->higher;
    ((ModelVBAR *)mv->higher)->lower = mv->lower;
}

static inline void insert_vbar(ModelVBAR *mv) {
    mv->lower = highest_priority.lower;
    ((ModelVBAR *)highest_priority.lower)->higher = mv;
    mv->higher = &highest_priority;
    highest_priority.lower = mv;
}

static inline void insert_vbar_last(ModelVBAR *mv) {
    mv->higher = lowest_priority.higher;
    ((ModelVBAR *)lowest_priority.higher)->lower = mv;
    mv->lower = &lowest_priority;
    lowest_priority.higher = mv;
}

/* The optional callback completes writeback before returning. No VBAR reentry. */
SHARED_EXPORT
void *vbar_allocate(void *devctx, uint64_t size, int device,
                    bool (*pageout_callback)(uint64_t, uint64_t)) {
    ModelVBAR *mv;

    set_devctx((AimdoContext *)devctx);

    one_time_setup();
    log_reset_shots();
    log(DEBUG, "%s (start): size=%zuM, device=%d\n", __func__, size / M, device);
    vbars_dirty = true;

    size_t nr_pages = VBAR_GET_PAGE_NR_UP(size);
    size_t nr_pages_max = VBAR_GET_PAGE_NR(vram_capacity);
    if (nr_pages_max < nr_pages) {
        nr_pages = nr_pages_max;
    }

    if (!(mv = calloc(1, sizeof(*mv) + nr_pages * sizeof(mv->residency_map[0])))) {
        log(CRITICAL, "Host OOM\n");
        return NULL;
    }

    mv->pageout_size = size;
    mv->pageout_callback = pageout_callback;
    size = (uint64_t)nr_pages * VBAR_PAGE_SIZE;

    /* FIXME: Do I care about alignment? Does Cuda just look after itself? */
    if (!CHECK_CU(cuMemAddressReserve(&mv->vbar, size, 0, 0, 0))) {
        log(AIMDO_LOG_ERROR, "Could not reseve Virtual Address space for VBAR\n");
        free(mv);
        return NULL;
    }

    mv->device = device;
    mv->nr_pages = mv->watermark = nr_pages;

    insert_vbar(mv);

    log(DEBUG, "%s (return): vbar=%p\n", __func__, (void *)mv);
    return mv;
}

SHARED_EXPORT
void vbar_set_watermark_limit(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    mv->watermark_limit = VBAR_GET_PAGE_NR_UP(size);
}

SHARED_EXPORT
bool vbar_set_watermark(void *devctx, void *vbar, uint64_t size) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t watermark = VBAR_GET_PAGE_NR_UP(size);

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: size=%zu\n", __func__, size);
    vbars_dirty = true;

    if (watermark > mv->nr_pages) {
        watermark = mv->nr_pages;
    }

    if (watermark < mv->watermark) {
        CHECK_CU(cuCtxSynchronize());
        for (size_t page_nr = watermark; page_nr < mv->watermark; page_nr++) {
            if (mod1(mv, page_nr, true, false) < 0) {
                return false;
            }
        }
    }

    mv->watermark = watermark;
    return true;
}

SHARED_EXPORT
void vbars_reset_watermark_limits(void *devctx) {
    set_devctx((AimdoContext *)devctx);
    one_time_setup();
    log(VERBOSE, "%s\n", __func__);

    for (ModelVBAR *i = lowest_priority.higher; i && i != &highest_priority; i = i->higher) {
        i->watermark_limit = 0;
    }
}

SHARED_EXPORT
void vbar_prioritize(void *devctx, void *vbar, uint64_t clamp) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);
    malloc_async_clamp = clamp;

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar(mv);

    mv->watermark = mv->nr_pages;
}

SHARED_EXPORT
void vbar_deprioritize(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    log_reset_shots();

    remove_vbar(mv);
    insert_vbar_last(mv);
}

SHARED_EXPORT
uint64_t vbar_get(void *devctx, void *vbar) {
    set_devctx((AimdoContext *)devctx);
    log(DEBUG, "%s vbar=%p\n", __func__, vbar);
    return (uint64_t)((ModelVBAR *)vbar)->vbar;
}

#define VBAR_FAULT_SUCCESS           0

SHARED_EXPORT
int vbar_fault(void *devctx, void *vbar, uint64_t offset, uint64_t size, uint32_t *signature) {
    set_devctx((AimdoContext *)devctx);

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbars_dirty = true;

    int ret = aimdo_rs_vbar_fault(vbar, offset, size, signature);
    if (ret == VBAR_FAULT_SUCCESS) {
        log(VVERBOSE, "%s (return) %d\n", __func__, ret);
    }
    return ret;
}

SHARED_EXPORT
bool vbar_unpin(void *devctx, void *vbar, uint64_t offset, uint64_t size) {
    set_devctx((AimdoContext *)devctx);

    log(VVERBOSE, "%s (start): offset=%lldk, size=%lldk\n", __func__, (ull)(offset / K), (ull)(size / K));
    vbars_dirty = true;
    return aimdo_rs_vbar_unpin(vbar, offset, size);
}

SHARED_EXPORT
void vbar_free(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s: vbar=%p\n", __func__, vbar);
    vbars_dirty = true;

    CHECK_CU(cuCtxSynchronize());

    mv->pageout_callback = NULL; /* Destruction discards rather than pages out. */
    for (uint64_t page_nr = 0; page_nr < mv->nr_pages; page_nr++) {
        mod1(mv, page_nr, true, true);
    }
    remove_vbar(mv);
    CHECK_CU(cuMemAddressFree(mv->vbar, (size_t)mv->nr_pages * VBAR_PAGE_SIZE));
    CHECK_CU(cuCtxSynchronize());
    free(mv);
}

SHARED_EXPORT
size_t vbar_loaded_size(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;

    set_devctx((AimdoContext *)devctx);

    return mv->resident_count * VBAR_PAGE_SIZE;
}

SHARED_EXPORT
size_t vbar_get_nr_pages(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    set_devctx((AimdoContext *)devctx);
    return mv->nr_pages;
}

SHARED_EXPORT
size_t vbar_get_watermark(void *devctx, void *vbar) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    set_devctx((AimdoContext *)devctx);
    return mv->watermark;
}

SHARED_EXPORT
void vbar_get_residency(void *devctx, void *vbar, uint8_t *out, size_t max_pages) {
    ModelVBAR *mv = (ModelVBAR *)vbar;
    size_t n = mv->nr_pages < max_pages ? mv->nr_pages : max_pages;

    set_devctx((AimdoContext *)devctx);
    for (size_t i = 0; i < n; i++) {
        ResidentPage *rp = &mv->residency_map[i];
        /* bit 0: resident, bit 1: pinned */
        out[i] = (rp->handle ? 1 : 0) | (rp->pin_count ? 2 : 0);
    }
}

SHARED_EXPORT
int64_t vbar_free_memory(void *devctx, void *vbar, uint64_t size) {
    set_devctx((AimdoContext *)devctx);

    log(DEBUG, "%s (start): size=%lldk\n", __func__, (ull)size);
    vbars_dirty = true;
    return aimdo_rs_vbar_free_memory(vbar, size);
}
