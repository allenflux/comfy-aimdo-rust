#pragma once

#include "gpu_dispatch.h"

typedef struct DriverQuery {
    const char *name;
    int version;
    cuuint64_t flags;
} DriverQuery;

typedef struct DriverState {
    int id;
    size_t query_count;
    DriverQuery queries[40];
    unsigned init_count;
    unsigned memory_query_count;
    int failing_query;
    CUresult query_error;
    CUresult init_error;
    CUdevice current_device;
    CUdevice failing_total_device;
    CUresult current_device_error;
    void (*before_total_memory)(CUdevice device);
} DriverState;

typedef DriverState *(*GetDriverState)(void);
