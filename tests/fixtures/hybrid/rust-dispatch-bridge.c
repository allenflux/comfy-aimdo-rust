#include "plat.h"

__attribute__((visibility("hidden")))
void aimdo_cuda_c_log(int event, const char *name) {
    switch (event) {
    case 0:
        log(AIMDO_LOG_ERROR, "aimdo_cuda_runtime_init: failed to resolve cuGetProcAddress\n");
        break;
    case 1:
        log(AIMDO_LOG_ERROR, "aimdo_cuda_runtime_init: failed to resolve required CUDA symbol %s\n", name);
        break;
    case 2:
        log(AIMDO_LOG_ERROR, "aimdo_cuda_runtime_init: cuInit failed\n");
        break;
    }
}
