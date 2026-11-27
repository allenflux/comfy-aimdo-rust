use crate::context::{
    aimdo_rs_cuda_budget_deficit, aimdo_rs_get_current_devctx, aimdo_rs_get_simple_vram_headroom,
};
use crate::cuda::{
    CuMemAccessDesc, CuMemAllocationFlags, CuMemAllocationProp, CuMemLocation, CUDA_SUCCESS,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::fmt;
use std::io::Write;
use std::panic::Location;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::Mutex;

pub(crate) const MIB: usize = 1024 * 1024;
pub(crate) const CRITICAL: i32 = 1;
pub(crate) const ERROR: i32 = 2;
pub(crate) const WARNING: i32 = 3;
pub(crate) const INFO: i32 = 4;
pub(crate) const DEBUG: i32 = 5;
pub(crate) const VERBOSE: i32 = 6;
pub(crate) const VVERBOSE: i32 = 7;

macro_rules! cuda {
    ($name:ident $(, $arg:expr)* $(,)?) => {{
        match (*core::ptr::addr_of!($crate::cuda::g_cuda)).$name {
            Some(function) => function($($arg),*),
            None => 3, // CUDA_ERROR_NOT_INITIALIZED
        }
    }};
}
pub(crate) use cuda;

type LogCallback = unsafe extern "C" fn(c_int, *const c_char);
static LOG_CALLBACK: Mutex<Option<LogCallback>> = Mutex::new(None);
static LOG_LEVEL: AtomicI32 = AtomicI32::new(0);
static LOG_EPOCH: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static LOG_SHOTS: RefCell<HashMap<&'static Location<'static>, u64>> = RefCell::new(HashMap::new());
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn set_log_callback(callback: Option<LogCallback>) {
    *LOG_CALLBACK
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = callback;
}

macro_rules! log_level_setters {
    ($($name:ident: $level:expr),+ $(,)?) => {$ (
        #[unsafe(no_mangle)]
        pub(crate) extern "C" fn $name() {
            LOG_LEVEL.store($level, Ordering::Relaxed);
        }
    )+};
}

log_level_setters! {
    set_log_level_none: -1,
    set_log_level_critical: CRITICAL,
    set_log_level_error: ERROR,
    set_log_level_warning: WARNING,
    set_log_level_info: INFO,
    set_log_level_debug: DEBUG,
    set_log_level_verbose: VERBOSE,
    set_log_level_vverbose: VVERBOSE,
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn log_reset_shots() {
    LOG_EPOCH.fetch_add(1, Ordering::Relaxed);
}

fn log_enabled(level: i32) -> bool {
    let configured = LOG_LEVEL.load(Ordering::Relaxed);
    (configured == 0 || configured >= level) && LOG_EPOCH.load(Ordering::Relaxed) != 0
}

fn emit_log(level: i32, args: fmt::Arguments<'_>, caller: &Location<'_>) {
    let level_name = [
        "ALL", "CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "VERBOSE", "VVERBOSE",
    ]
    .get(level as usize)
    .copied()
    .unwrap_or("UNKNOWN");
    let mut message = format!(
        "aimdo: {}:{}:{level_name}:{args}",
        caller.file(),
        caller.line()
    );
    if message.len() > 2047 {
        let mut end = 2047;
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        message.truncate(end);
    }
    let callback = *LOG_CALLBACK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    if let Some(callback) = callback {
        if let Ok(message) = CString::new(message) {
            // SAFETY: The Python binding keeps the callback alive until platform teardown.
            unsafe { callback(level, message.as_ptr()) };
        }
    } else {
        let _ = std::io::stderr().write_all(message.as_bytes());
    }
}

#[track_caller]
pub(crate) fn log(level: i32, args: fmt::Arguments<'_>) {
    if log_enabled(level) {
        emit_log(level, args, Location::caller());
    }
}

#[track_caller]
pub(crate) fn log_shot(level: i32, args: fmt::Arguments<'_>) {
    if !log_enabled(level) {
        return;
    }
    let epoch = LOG_EPOCH.load(Ordering::Relaxed);
    let caller = Location::caller();
    let fresh = LOG_SHOTS.try_with(|shots| shots.borrow_mut().insert(caller, epoch) != Some(epoch));
    if matches!(fresh, Ok(true)) {
        emit_log(level, args, caller);
    }
}

#[track_caller]
pub(crate) unsafe fn check_cuda(result: i32, label: &str) -> bool {
    // SAFETY: The caller guarantees a live driver dispatch table when it is available.
    unsafe { check_cuda_at(result, label, VVERBOSE, DEBUG) }
}

#[track_caller]
pub(crate) unsafe fn check_cuda_at(
    result: i32,
    label: &str,
    oom_level: i32,
    error_level: i32,
) -> bool {
    if result == CUDA_SUCCESS {
        return true;
    }
    let level = if result == 2 { oom_level } else { error_level };
    if log_enabled(level) {
        let mut text = core::ptr::null();
        // SAFETY: CUDA returns a valid, NUL-terminated driver-owned error string on success.
        let description = if unsafe { cuda!(get_error_string, result, &mut text) } == CUDA_SUCCESS
            && !text.is_null()
        {
            unsafe { CStr::from_ptr(text) }
        } else {
            c"<FATAL - CANNOT PARSE CUDA ERROR CODE>"
        };
        log(
            level,
            format_args!(
                "CUDA API FAILED ({result}): {label}: {}\n",
                description.to_string_lossy()
            ),
        );
    }
    false
}

pub(crate) unsafe fn budget_deficit(bytes: usize) -> isize {
    let mut source = c"unknown".as_ptr();
    // SAFETY: Allocation callers select their device context before checking its budget.
    unsafe {
        aimdo_rs_cuda_budget_deficit(&mut source);
        let ctx = aimdo_rs_get_current_devctx();
        let used = (*ctx).total_vram_usage;
        let simple = (used.wrapping_add(bytes as u64) as isize)
            .wrapping_add(aimdo_rs_get_simple_vram_headroom() as isize)
            .wrapping_sub((*ctx).vram_capacity as isize);
        let delta = (*ctx)
            .deficit_sync
            .wrapping_add(used as isize)
            .wrapping_sub((*ctx).total_vram_last_check as isize)
            .wrapping_add(bytes as isize);
        let deficit = simple
            .max(delta)
            .wrapping_add((*ctx).extra_vram_headroom as isize);
        if deficit > 0 && log_enabled(DEBUG) {
            let method = if simple > delta {
                c"simple"
            } else {
                CStr::from_ptr(source)
            };
            log(DEBUG, format_args!("budget_deficit: Prevailing Method: {} Deficit: {} Extra Headroom: {} Alloc Size {}\n",
                method.to_string_lossy(), deficit as usize / MIB, (*ctx).extra_vram_headroom as usize / MIB, bytes / MIB));
        }
        deficit
    }
}

pub(crate) unsafe fn map_pages(address: u64, bytes: usize, device: i32, handle: *mut u64) -> i32 {
    let mut allocation = 0;
    let location = CuMemLocation {
        kind: 1,
        id: device,
    };
    let properties = CuMemAllocationProp {
        kind: 1,
        requested_handle_types: 0,
        location,
        win32_handle_metadata: null_mut(),
        alloc_flags: CuMemAllocationFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
            reserved: [0; 4],
        },
    };
    let access = CuMemAccessDesc {
        location: CuMemLocation {
            kind: 1,
            id: device,
        },
        flags: 3,
    };
    // SAFETY: The caller selects a context and reserves the requested range before mapping.
    unsafe {
        let result = cuda!(mem_create, &mut allocation, bytes, &properties, 0);
        if !check_cuda_at(result, "cuMemCreate", VVERBOSE, ERROR) {
            return result;
        }
        let result = cuda!(mem_map, address, bytes, 0, allocation, 0);
        if !check_cuda_at(result, "cuMemMap", VVERBOSE, ERROR) {
            check_cuda_at(
                cuda!(mem_release, allocation),
                "cuMemRelease",
                VVERBOSE,
                ERROR,
            );
            return result;
        }
        let result = cuda!(mem_set_access, address, bytes, &access, 1);
        if !check_cuda_at(result, "cuMemSetAccess", VVERBOSE, ERROR) {
            check_cuda_at(
                cuda!(mem_unmap, address, bytes),
                "cuMemUnmap",
                VVERBOSE,
                ERROR,
            );
            check_cuda_at(
                cuda!(mem_release, allocation),
                "cuMemRelease",
                VVERBOSE,
                ERROR,
            );
            return result;
        }
        let context = aimdo_rs_get_current_devctx();
        (*context).total_vram_usage = (*context).total_vram_usage.wrapping_add(bytes as u64);
        *handle = allocation;
        CUDA_SUCCESS
    }
}

struct LoadedModule<'a> {
    wanted: &'a CStr,
    path: Option<CString>,
}

unsafe extern "C" fn match_loaded_module(
    info: *mut libc::dl_phdr_info,
    _: usize,
    data: *mut c_void,
) -> c_int {
    // SAFETY: The loader owns info during the callback and data refers to our stack search.
    unsafe {
        let name = (*info).dlpi_name;
        if name.is_null() {
            return 0;
        }
        let path = CStr::from_ptr(name);
        let basename = path.to_bytes().rsplit(|byte| *byte == b'/').next();
        let search = &mut *data.cast::<LoadedModule<'_>>();
        if basename == Some(search.wanted.to_bytes()) {
            search.path = Some(path.to_owned());
            return 1;
        }
        0
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_find_loaded_module(
    libraries: *const *const c_char,
    count: usize,
) -> *mut c_void {
    let flags = libc::RTLD_LAZY | libc::RTLD_LOCAL | libc::RTLD_NOLOAD;
    // SAFETY: The caller passes count valid C string pointers. NOLOAD prevents implicitly
    // creating a second driver instance and successful handles are retained by the caller.
    unsafe {
        for &name in std::slice::from_raw_parts(libraries, count) {
            let mut search = LoadedModule {
                wanted: CStr::from_ptr(name),
                path: None,
            };
            libc::dl_iterate_phdr(
                Some(match_loaded_module),
                (&mut search as *mut LoadedModule<'_>).cast(),
            );
            if let Some(path) = search.path {
                let module = libc::dlopen(path.as_ptr(), flags);
                if !module.is_null() {
                    return module;
                }
            }
            let module = libc::dlopen(name, flags);
            if !module.is_null() {
                return module;
            }
        }
    }
    log(
        ERROR,
        format_args!("aimdo_find_loaded_module: failed to find an already-loaded module\n"),
    );
    null_mut()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_cuda_c_log(event: i32, name: *const c_char) {
    match event {
        0 => log(
            ERROR,
            format_args!("aimdo_cuda_runtime_init: failed to resolve cuGetProcAddress\n"),
        ),
        1 => {
            // SAFETY: The resolver reports the NUL-terminated name it asked CUDA to resolve.
            let name = unsafe { CStr::from_ptr(name) };
            log(
                ERROR,
                format_args!(
                    "aimdo_cuda_runtime_init: failed to resolve required CUDA symbol {}\n",
                    name.to_string_lossy()
                ),
            );
        }
        2 => log(
            ERROR,
            format_args!("aimdo_cuda_runtime_init: cuInit failed\n"),
        ),
        _ => {}
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_control_c_tick() -> u64 {
    let mut time = libc::timeval {
        tv_sec: 0,
        tv_usec: 0,
    };
    // SAFETY: The buffer is a writable Linux timeval and no timezone output is needed.
    unsafe { libc::gettimeofday(&mut time, null_mut()) };
    (time.tv_sec as u64)
        .wrapping_mul(1000)
        .wrapping_add(time.tv_usec as u64 / 1000)
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_current_device(device: *mut i32) -> bool {
    // SAFETY: The caller supplies the writable CUDA device output pointer.
    unsafe { check_cuda(cuda!(ctx_get_device, device), "cuCtxGetDevice(&device)") }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_device_get(device: *mut i32, ordinal: i32) -> bool {
    // SAFETY: The caller supplies the writable CUDA device output pointer.
    unsafe {
        check_cuda(
            cuda!(device_get, device, ordinal),
            "cuDeviceGet(&dev, cuda_device_ids[i])",
        )
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_device_memory(
    device: i32,
    capacity: *mut u64,
) -> bool {
    let mut bytes = 0;
    // SAFETY: The device lookup supplies a valid context and writable capacity pointer.
    unsafe {
        let ok = check_cuda(
            cuda!(device_total_mem, &mut bytes, device),
            "cuDeviceTotalMem(&vram_capacity, dev)",
        );
        *capacity = bytes as u64;
        ok
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_device_integrated(device: i32) -> bool {
    let mut integrated = 0;
    // SAFETY: CUDA writes the integer flag to stack storage. Linux CUDA uses attribute 18.
    (unsafe {
        check_cuda(
            cuda!(device_get_attribute, &mut integrated, 18, device),
            "cuDeviceGetAttribute(INTEGRATED)",
        )
    }) && integrated != 0
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_mem_info(
    free: *mut usize,
    total: *mut usize,
    label: *const c_char,
) -> bool {
    // SAFETY: The caller supplies CUDA output buffers and a NUL-terminated diagnostic label.
    unsafe {
        check_cuda(
            cuda!(mem_get_info, free, total),
            &CStr::from_ptr(label).to_string_lossy(),
        )
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_control_c_log_integrated_headroom(headroom: usize) {
    log(
        INFO,
        format_args!(
            "comfy-aimdo integrated Linux GPU RAM headroom: {} MB\n",
            headroom / MIB
        ),
    );
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_log_device(device: i32, capacity: usize) {
    let mut name = [0; 256];
    // SAFETY: CUDA will write at most the supplied length. Read up to the first NUL.
    let name = if unsafe {
        check_cuda(
            cuda!(
                device_get_name,
                name.as_mut_ptr(),
                name.len() as i32,
                device
            ),
            "cuDeviceGetName",
        )
    } {
        name[name.len() - 1] = 0;
        unsafe { CStr::from_ptr(name.as_ptr()) }
    } else {
        c"<unknown>"
    };
    log(
        INFO,
        format_args!(
            "comfy-aimdo inited for GPU: {} (VRAM: {} MB)\n",
            name.to_string_lossy(),
            capacity / MIB
        ),
    );
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_control_c_log_poll(
    integrated: bool,
    available: usize,
    total_or_headroom: usize,
    deficit: isize,
    recorded: usize,
    method: *const c_char,
) {
    if log_enabled(DEBUG) {
        let (label, field, total) = if integrated {
            ("MemAvailable", "available", "headroom")
        } else {
            ("device memory", "free", "total")
        };
        log(DEBUG, format_args!("cuda_budget_deficit: {label} poll {field}={} MB {total}={} MB deficit_sync={} MB recorded={} MB\n",
            available / MIB, total_or_headroom / MIB, deficit / MIB as isize, recorded / MIB));
        // SAFETY: The caller supplies a constant, NUL-terminated method string.
        let method = unsafe { CStr::from_ptr(method) };
        log(
            DEBUG,
            format_args!(
                "cuda_budget_deficit: prevailing method {}\n",
                method.to_string_lossy()
            ),
        );
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_control_c_log_analysis_header() {
    log(DEBUG, format_args!("--- VRAM Stats ---\n"));
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_control_c_log_analysis(recorded: usize, free: usize, total: usize) {
    log(
        DEBUG,
        format_args!("  Aimdo Recorded Usage:  {:7} MB\n", recorded / MIB),
    );
    log(
        DEBUG,
        format_args!(
            "  Device: {:7} MB / {:7} MB Free\n",
            free / MIB,
            total / MIB
        ),
    );
}

pub(crate) extern "C" fn set_nvml_pressure(_: bool) {}
