#include "plat.h"

#include <assert.h>
#include <inttypes.h>

#define PAGE (8 * M)
#define MAX_MAPPINGS 128

bool malloc_graph_sync_paused(void);
bool malloc_graph_alloc(CUdeviceptr *ptr, size_t bytes, CUstream stream, int *status);
bool malloc_graph_free(CUdeviceptr ptr, CUstream stream, int *status);
void *malloc_graph_create(void *context, CUstream stream, bool assert_breaks);
bool malloc_graph_pause(void *graph, bool pause, bool sync);
bool malloc_graph_set_stream(void *graph, CUstream stream);
bool malloc_graph_push(void *graph, const char *name);
int malloc_graph_pop(void *graph);
bool malloc_graph_abort(void *graph);
uint64_t malloc_graph_stat(void *graph, int stat);
void malloc_graph_destroy(void *graph);
bool free_rogue(CUdeviceptr ptr, int *status);

_Thread_local Context *g_devctx;
#ifndef RUST_GRAPH
AimdoCudaDispatch g_cuda;
#endif

void *aimdo_find_loaded_module(const char *const *names, size_t count) {
    abort();
}

void aimdo_cuda_c_log(int event, const char *name, int code, int status) {
    abort();
}

static int held;

void allocations_lock(void) {
    assert(held++ == 0);
}

void allocations_unlock(void) {
    assert(held-- == 1);
}

bool vbars_free(ssize_t size) {
    printf("pressure %zd\n", size);
    return true;
}

static CUdeviceptr cursor = 1ULL << 36;
static CUmemGenericAllocationHandle next_handle = 1;
static struct Mapping {
    CUdeviceptr address;
    CUmemGenericAllocationHandle handle;
} mapped[MAX_MAPPINGS];
static bool live[MAX_MAPPINGS];
static unsigned map_calls;
static int fail_creates;

static CUresult fake_reserve(CUdeviceptr *address, size_t bytes, size_t align,
                             CUdeviceptr request, unsigned long long flags) {
    assert(!request && !flags);
    *address = cursor;
    cursor += bytes;
    printf("reserve %llu %zu %zu\n", *address, bytes, align);
    return CUDA_SUCCESS;
}

static CUresult fake_addr_free(CUdeviceptr address, size_t bytes) {
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        assert(mapped[i].address < address || mapped[i].address >= address + bytes);
    }
    printf("addr-free %llu %zu\n", address, bytes);
    return CUDA_SUCCESS;
}

static CUresult fake_create(CUmemGenericAllocationHandle *handle, size_t bytes,
                            const CUmemAllocationProp *prop, unsigned long long flags) {
    assert(!flags && prop->type == CU_MEM_ALLOCATION_TYPE_PINNED &&
           prop->location.type == CU_MEM_LOCATION_TYPE_DEVICE && prop->location.id == 0);
    if (fail_creates) {
        fail_creates--;
        puts("oom");
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    *handle = next_handle++;
    assert(*handle < MAX_MAPPINGS);
    live[*handle] = true;
    printf("create %llu %zu\n", *handle, bytes);
    return CUDA_SUCCESS;
}

static CUresult fake_map(CUdeviceptr ptr, size_t bytes, size_t offset,
                         CUmemGenericAllocationHandle handle, unsigned long long flags) {
    assert(bytes == PAGE && !offset && !flags && live[handle]);
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        assert(mapped[i].address != ptr);
    }
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        if (!mapped[i].address) {
            mapped[i] = (struct Mapping){ptr, handle};
            map_calls++;
            printf("map %llu %llu\n", ptr, handle);
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static CUresult fake_access(CUdeviceptr ptr, size_t bytes,
                            const CUmemAccessDesc *desc, size_t count) {
    assert(bytes == PAGE && count == 1 && desc->location.type == CU_MEM_LOCATION_TYPE_DEVICE &&
           desc->location.id == 0 && desc->flags == CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        if (mapped[i].address == ptr) {
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static CUresult fake_unmap(CUdeviceptr ptr, size_t bytes) {
    assert(bytes == PAGE);
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        if (mapped[i].address == ptr) {
            mapped[i].address = 0;
            printf("unmap %llu\n", ptr);
            return CUDA_SUCCESS;
        }
    }
    abort();
}

static CUresult fake_release(CUmemGenericAllocationHandle handle) {
    for (size_t i = 0; i < MAX_MAPPINGS; i++) {
        assert(!mapped[i].address || mapped[i].handle != handle);
    }
    assert(live[handle]);
    live[handle] = false;
    printf("release %llu\n", handle);
    return CUDA_SUCCESS;
}

static CUresult fake_sync(void) {
    puts("sync");
    return CUDA_SUCCESS;
}

static CUstream stream = (CUstream)1;

static CUdeviceptr allocate(size_t bytes) {
    CUdeviceptr ptr = 0;
    int status = -1;
    assert(malloc_graph_alloc(&ptr, bytes, stream, &status) && status == CUDA_SUCCESS && ptr);
    printf("alloc %zu %llu\n", bytes, ptr);
    return ptr;
}

static void free_graph(CUdeviceptr ptr) {
    int status = -1;
    assert(malloc_graph_free(ptr, stream, &status) && status == CUDA_SUCCESS);
    printf("free %llu\n", ptr);
}

static void stats(void *graph) {
    printf("stats %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",
           malloc_graph_stat(graph, 0), malloc_graph_stat(graph, 1),
           malloc_graph_stat(graph, 2), malloc_graph_stat(graph, 3));
}

static void free_rogue_checked(CUdeviceptr ptr) {
    int status = -1;
    assert(free_rogue(ptr, &status) && status == CUDA_SUCCESS);
}

static void assert_empty(const Context *context) {
    assert(!context->usage && !context->rogue_list && !context->candidate_list && !held);
}

int main(void) {
    g_cuda = (AimdoCudaDispatch){
        .p_cuMemAddressReserve = fake_reserve, .p_cuMemAddressFree = fake_addr_free,
        .p_cuMemCreate = fake_create, .p_cuMemMap = fake_map, .p_cuMemSetAccess = fake_access,
        .p_cuMemUnmap = fake_unmap, .p_cuMemRelease = fake_release,
        .p_cuCtxSynchronize = fake_sync,
    };

    Context context = {0};
    void *graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    assert(malloc_graph_pause(graph, true, true) && malloc_graph_sync_paused());
    int status = -1;
    CUdeviceptr untouched = 77;
    assert(!malloc_graph_alloc(&untouched, 4096, stream, &status) && untouched == 77 && status == -1);
    assert(malloc_graph_pause(graph, false, true) && !malloc_graph_sync_paused());
    assert(malloc_graph_set_stream(graph, (CUstream)2));
    assert(!malloc_graph_alloc(&untouched, 4096, stream, &status));
    assert(malloc_graph_set_stream(graph, stream));

    CUdeviceptr small = allocate(5000);
    CUdeviceptr large = allocate(PAGE + 1);
    assert(malloc_graph_push(graph, "nested"));
    CUdeviceptr nested = allocate(PAGE);
    free_graph(nested);
    assert(malloc_graph_pop(graph) == 1);
    free_graph(large);
    free_graph(small);
    unsigned before = map_calls;
    assert(malloc_graph_pop(graph) == 1);
    stats(graph);
    for (int i = 0; i < 2; i++) {
        assert(malloc_graph_push(graph, NULL));
        assert(allocate(5000) == small);
        assert(allocate(PAGE + 1) == large);
        assert(malloc_graph_push(graph, "nested"));
        assert(allocate(PAGE) == nested);
        free_graph(nested);
        assert(malloc_graph_pop(graph) == 1);
        free_graph(large);
        free_graph(small);
        assert(malloc_graph_pop(graph) == 1);
    }
    assert(map_calls == before);
    assert(malloc_graph_push(graph, NULL));
    assert(allocate(5000) == small);
    CUdeviceptr alternate = allocate(2 * PAGE + 1);
    free_graph(alternate);
    free_graph(small);
    assert(malloc_graph_pop(graph) == 2);
    stats(graph);
    malloc_graph_destroy(graph);
    assert_empty(&context);

    graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    large = allocate(PAGE);
    small = allocate(8999);
    assert(malloc_graph_pop(graph) == 1);
    stats(graph);
    free_rogue_checked(large);
    assert(malloc_graph_push(graph, NULL));
    assert(allocate(PAGE) == large);
    free_graph(large);
    assert(malloc_graph_pop(graph) == 2);
    malloc_graph_destroy(graph);
    free_rogue_checked(small);
    assert_empty(&context);

    graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    small = allocate(3601);
    large = allocate(PAGE);
    assert(malloc_graph_abort(graph) && malloc_graph_abort(graph));
    stats(graph);
    malloc_graph_destroy(graph);
    free_rogue_checked(small);
    free_rogue_checked(large);
    assert_empty(&context);

    graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    fail_creates = 1;
    CUdeviceptr a = allocate(2 * PAGE);
    CUdeviceptr b = allocate(PAGE);
    free_graph(a);
    CUdeviceptr c = allocate(3 * PAGE);
    free_graph(b);
    free_graph(c);
    assert(malloc_graph_stat(graph, 1) == 6 * PAGE);
    CUdeviceptr d = allocate(4 * PAGE);
    assert(malloc_graph_stat(graph, 1) == 6 * PAGE && malloc_graph_stat(graph, 2) == 4 * PAGE);
    free_graph(d);
    assert(malloc_graph_pop(graph) == 1);
    stats(graph);
    malloc_graph_destroy(graph);
    assert_empty(&context);

    graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    assert(malloc_graph_push(graph, "inner-small"));
    small = allocate(2048);
    assert(malloc_graph_pop(graph) == 1);
    free_rogue_checked(small); /* The candidate has not been handed off yet. */
    assert(malloc_graph_pop(graph) == 0 && malloc_graph_abort(graph));
    stats(graph);
    malloc_graph_destroy(graph);
    assert_empty(&context);

    graph = malloc_graph_create(&context, stream, false);
    assert(graph);
    fail_creates = 2;
    CUdeviceptr failed = 7;
    assert(malloc_graph_alloc(&failed, PAGE, stream, &status) && !failed && status == CUDA_ERROR_OUT_OF_MEMORY);
    assert(malloc_graph_pop(graph) == 0 && malloc_graph_abort(graph));
    stats(graph);
    malloc_graph_destroy(graph);
    assert_empty(&context);
}
