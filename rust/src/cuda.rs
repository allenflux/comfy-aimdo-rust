use core::ffi::{c_char, c_int, c_uint, c_void, CStr};
use core::mem::transmute;
use core::ptr::{addr_of, addr_of_mut, null_mut};

pub(crate) type CuResult = c_int;
pub(crate) type CuDevice = c_int;
pub(crate) type CuDevicePtr = u64;
pub(crate) type CuStream = *mut c_void;
pub(crate) type CuEvent = *mut c_void;
pub(crate) type CuMemAllocationHandle = u64;

pub(crate) const CUDA_SUCCESS: CuResult = 0;

// 固定 v0.6.0 请求的 Driver API ABI，避免新驱动返回与绑定签名不一致的版本。
const CUDA_ABI_VERSION: c_int = 12060;
// 异步分配和释放要分别解析传统默认流和线程默认流两个入口。
const LEGACY_STREAM: u64 = 1;
const PER_THREAD_STREAM: u64 = 2;

const MISSING_RESOLVER: c_int = 0;
const MISSING_SYMBOL: c_int = 1;
const INIT_FAILED: c_int = 2;

#[repr(C)]
pub(crate) struct CuUuid {
    pub(crate) bytes: [c_char; 16],
}

#[repr(C)]
pub(crate) struct CuMemLocation {
    pub(crate) kind: c_uint,
    pub(crate) id: c_int,
}

#[repr(C)]
pub(crate) struct CuMemAllocationFlags {
    pub(crate) compression_type: u8,
    pub(crate) gpu_direct_rdma_capable: u8,
    pub(crate) usage: u16,
    pub(crate) reserved: [u8; 4],
}

#[repr(C)]
pub(crate) struct CuMemAllocationProp {
    pub(crate) kind: c_uint,
    pub(crate) requested_handle_types: c_uint,
    pub(crate) location: CuMemLocation,
    pub(crate) win32_handle_metadata: *mut c_void,
    pub(crate) alloc_flags: CuMemAllocationFlags,
}

#[repr(C)]
pub(crate) struct CuMemAccessDesc {
    pub(crate) location: CuMemLocation,
    pub(crate) flags: c_uint,
}

type CuInit = unsafe extern "C" fn(c_uint) -> CuResult;
type CuGetProcAddress =
    unsafe extern "C" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_uint) -> CuResult;
type CuDeviceGetLuid = unsafe extern "C" fn(*mut c_char, *mut c_uint, CuDevice) -> CuResult;

// 字段顺序与 v0.6.0 的 AimdoCudaDispatch 一致，便于复用 ABI 对照测试。
// 解析器和 Windows 专用的 LUID 槽位不参与下表的批量解析。
macro_rules! define_dispatch {
    ($($field:ident: ($($argument:ty),*) => $symbol:literal, $flags:ident;)+) => {
        #[repr(C)]
        pub(crate) struct AimdoCudaDispatch {
            pub(crate) init: Option<CuInit>,
            pub(crate) get_proc_address: Option<CuGetProcAddress>,
            $(pub(crate) $field: Option<unsafe extern "C" fn($($argument),*) -> CuResult>,)+
            pub(crate) device_get_luid: Option<CuDeviceGetLuid>,
        }

        impl AimdoCudaDispatch {
            const EMPTY: Self = Self {
                init: None,
                get_proc_address: None,
                $($field: None,)+
                device_get_luid: None,
            };

            unsafe fn resolve(dispatch: *mut Self, resolver: CuGetProcAddress) -> Option<CuInit> {
                // SAFETY: Initialization is serialized and each name and ABI below is the
                // corresponding CUDA Driver API signature at CUDA_ABI_VERSION.
                unsafe {
                    let resolved = resolve_symbol(resolver, c"cuInit", LEGACY_STREAM);
                    if resolved.is_null() {
                        return None;
                    }
                    let init = transmute::<*mut c_void, CuInit>(resolved);
                    (*dispatch).init = Some(init);
                    $(
                        let resolved = resolve_symbol(resolver, $symbol, $flags);
                        if resolved.is_null() {
                            return None;
                        }
                        (*dispatch).$field = Some(transmute::<
                            *mut c_void,
                            unsafe extern "C" fn($($argument),*) -> CuResult,
                        >(resolved));
                    )+
                    Some(init)
                }
            }
        }
    };
}

define_dispatch! {
    get_error_string: (CuResult, *mut *const c_char) => c"cuGetErrorString", LEGACY_STREAM;
    ctx_get_device: (*mut CuDevice) => c"cuCtxGetDevice", LEGACY_STREAM;
    ctx_synchronize: () => c"cuCtxSynchronize", LEGACY_STREAM;
    device_get: (*mut CuDevice, c_int) => c"cuDeviceGet", LEGACY_STREAM;
    device_get_attribute: (*mut c_int, c_uint, CuDevice) => c"cuDeviceGetAttribute", LEGACY_STREAM;
    device_total_mem: (*mut usize, CuDevice) => c"cuDeviceTotalMem", LEGACY_STREAM;
    device_get_name: (*mut c_char, c_int, CuDevice) => c"cuDeviceGetName", LEGACY_STREAM;
    device_get_uuid: (*mut CuUuid, CuDevice) => c"cuDeviceGetUuid", LEGACY_STREAM;
    mem_get_info: (*mut usize, *mut usize) => c"cuMemGetInfo", LEGACY_STREAM;
    mem_alloc_v2: (*mut CuDevicePtr, usize) => c"cuMemAlloc", LEGACY_STREAM;
    mem_free_v2: (CuDevicePtr) => c"cuMemFree", LEGACY_STREAM;
    mem_alloc_async: (*mut CuDevicePtr, usize, CuStream) => c"cuMemAllocAsync", LEGACY_STREAM;
    mem_alloc_async_ptsz: (*mut CuDevicePtr, usize, CuStream) => c"cuMemAllocAsync", PER_THREAD_STREAM;
    mem_free_async: (CuDevicePtr, CuStream) => c"cuMemFreeAsync", LEGACY_STREAM;
    mem_free_async_ptsz: (CuDevicePtr, CuStream) => c"cuMemFreeAsync", PER_THREAD_STREAM;
    mem_alloc_host: (*mut *mut c_void, usize) => c"cuMemAllocHost", LEGACY_STREAM;
    mem_free_host: (*mut c_void) => c"cuMemFreeHost", LEGACY_STREAM;
    mem_host_register: (*mut c_void, usize, c_uint) => c"cuMemHostRegister", LEGACY_STREAM;
    mem_host_unregister: (*mut c_void) => c"cuMemHostUnregister", LEGACY_STREAM;
    mem_address_reserve: (*mut CuDevicePtr, usize, usize, CuDevicePtr, u64) => c"cuMemAddressReserve", LEGACY_STREAM;
    mem_address_free: (CuDevicePtr, usize) => c"cuMemAddressFree", LEGACY_STREAM;
    mem_create: (*mut CuMemAllocationHandle, usize, *const CuMemAllocationProp, u64) => c"cuMemCreate", LEGACY_STREAM;
    mem_map: (CuDevicePtr, usize, usize, CuMemAllocationHandle, u64) => c"cuMemMap", LEGACY_STREAM;
    mem_set_access: (CuDevicePtr, usize, *const CuMemAccessDesc, usize) => c"cuMemSetAccess", LEGACY_STREAM;
    mem_unmap: (CuDevicePtr, usize) => c"cuMemUnmap", LEGACY_STREAM;
    mem_release: (CuMemAllocationHandle) => c"cuMemRelease", LEGACY_STREAM;
    memcpy_htod_async: (CuDevicePtr, *const c_void, usize, CuStream) => c"cuMemcpyHtoDAsync", LEGACY_STREAM;
    memcpy_dtoh_async: (*mut c_void, CuDevicePtr, usize, CuStream) => c"cuMemcpyDtoHAsync", LEGACY_STREAM;
    event_create: (*mut CuEvent, c_uint) => c"cuEventCreate", LEGACY_STREAM;
    event_destroy: (CuEvent) => c"cuEventDestroy", LEGACY_STREAM;
    event_record: (CuEvent, CuStream) => c"cuEventRecord", LEGACY_STREAM;
    event_synchronize: (CuEvent) => c"cuEventSynchronize", LEGACY_STREAM;
}

const _: () = {
    assert!(core::mem::size_of::<AimdoCudaDispatch>() == 35 * 8);
    assert!(core::mem::size_of::<CuMemAllocationProp>() == 32);
    assert!(core::mem::size_of::<CuMemAccessDesc>() == 12);
};

#[unsafe(no_mangle)]
pub(crate) static mut g_cuda: AimdoCudaDispatch = AimdoCudaDispatch::EMPTY;
static mut CUDA_MODULE: *mut c_void = null_mut();

unsafe extern "C" {
    fn aimdo_find_loaded_module(libraries: *const *const c_char, count: usize) -> *mut c_void;
    fn aimdo_cuda_c_log(event: c_int, name: *const c_char);
    fn dlsym(module: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(module: *mut c_void) -> c_int;
}

unsafe fn resolve_symbol(resolver: CuGetProcAddress, symbol: &CStr, flags: u64) -> *mut c_void {
    // SAFETY: The caller supplies a resolver from the loaded driver; symbol is NUL-terminated.
    unsafe {
        let mut resolved = null_mut();
        let result = resolver(
            symbol.as_ptr(),
            &mut resolved,
            CUDA_ABI_VERSION,
            flags,
            null_mut(),
        );
        if result != CUDA_SUCCESS || resolved.is_null() {
            aimdo_cuda_c_log(MISSING_SYMBOL, symbol.as_ptr());
            return null_mut();
        }
        resolved
    }
}

/// # Safety
/// Platform startup serializes this call with dispatch use and runtime teardown.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_cuda_runtime_init() -> bool {
    // SAFETY: The caller guarantees exclusive initialization of the global dispatch table.
    unsafe {
        let dispatch = addr_of_mut!(g_cuda);
        if (*dispatch).init.is_some() {
            return true;
        }

        let libraries = [c"libcuda.so.1".as_ptr(), c"libcuda.so".as_ptr()];
        let module = aimdo_find_loaded_module(libraries.as_ptr(), libraries.len());
        addr_of_mut!(CUDA_MODULE).write(module);
        if module.is_null() {
            return false;
        }

        let resolver_name = c"cuGetProcAddress";
        let symbol = dlsym(module, resolver_name.as_ptr());
        if symbol.is_null() {
            aimdo_cuda_c_log(MISSING_RESOLVER, resolver_name.as_ptr());
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        let resolver = transmute::<*mut c_void, CuGetProcAddress>(symbol);
        // 先解析到本地表，只有整个表可用且驱动初始化成功才发布。
        let mut next = AimdoCudaDispatch::EMPTY;
        next.get_proc_address = Some(resolver);
        let Some(init) = AimdoCudaDispatch::resolve(&mut next, resolver) else {
            aimdo_cuda_runtime_cleanup();
            return false;
        };
        let result = init(0);
        if result == CUDA_SUCCESS {
            dispatch.write(next);
            return true;
        }
        aimdo_cuda_c_log(INIT_FAILED, c"cuInit".as_ptr());
        aimdo_cuda_runtime_cleanup();
        false
    }
}

/// # Safety
/// Hooks and dispatch users must stop using the driver before platform teardown.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_cuda_runtime_cleanup() {
    // SAFETY: Teardown has exclusive access; the module handle came from the loader above.
    unsafe {
        addr_of_mut!(g_cuda).write(AimdoCudaDispatch::EMPTY);
        let module = addr_of!(CUDA_MODULE).read();
        if !module.is_null() {
            dlclose(module);
            addr_of_mut!(CUDA_MODULE).write(null_mut());
        }
    }
}
