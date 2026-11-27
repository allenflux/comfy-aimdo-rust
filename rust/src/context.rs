use std::cell::Cell;
use std::ffi::{c_char, c_void};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem::{offset_of, size_of};
use std::ptr::{addr_of_mut, null_mut};
use std::sync::atomic::{AtomicI64, AtomicPtr, Ordering};
use std::sync::Mutex;

const MIB: usize = 1024 * 1024;
const VRAM_HEADROOM: usize = 256 * MIB;
const INTEGRATED_SIMPLE_ONLY_DEFICIT: isize = -(1 << 60);

#[repr(C)]
pub(crate) struct HostbufFileReaderSlot {
    pub(crate) buffer: *mut u8,
    pub(crate) offset: u64,
    pub(crate) stream: *mut c_void,
    pub(crate) event: *mut c_void,
}

#[repr(C)]
pub(crate) struct AimdoContext {
    pub(crate) device_id: i32,
    pub(crate) vram_capacity: u64,
    pub(crate) integrated_ram_headroom: u64,
    pub(crate) extra_vram_headroom: u64,
    pub(crate) malloc_async_clamp: u64,
    pub(crate) total_vram_usage: u64,
    pub(crate) total_vram_last_check: u64,
    pub(crate) deficit_sync: isize,
    pub(crate) control_timestamp_last_check: u64,
    pub(crate) highest_priority: *mut c_void,
    pub(crate) lowest_priority: *mut c_void,
    pub(crate) vbars_dirty: bool,
    pub(crate) allocations_dirty: bool,
    pub(crate) integrated_device: bool,
    pub(crate) vmm_table: [*mut c_void; 4096],
    pub(crate) size_table: [*mut c_void; 1024],
    pub(crate) size_table_lock: *mut c_void,
    pub(crate) rogues: *mut c_void,
    pub(crate) rogue_candidates: *mut c_void,
    pub(crate) hostbuf_file_reader_slots: [HostbufFileReaderSlot; 3],
    pub(crate) hostbuf_file_reader_active: i32,
}

const _: () = {
    assert!(size_of::<HostbufFileReaderSlot>() == 32);
    assert!(offset_of!(HostbufFileReaderSlot, buffer) == 0);
    assert!(offset_of!(HostbufFileReaderSlot, offset) == 8);
    assert!(offset_of!(HostbufFileReaderSlot, stream) == 16);
    assert!(offset_of!(HostbufFileReaderSlot, event) == 24);
    assert!(size_of::<AimdoContext>() == 41184);
    assert!(offset_of!(AimdoContext, device_id) == 0);
    assert!(offset_of!(AimdoContext, vram_capacity) == 8);
    assert!(offset_of!(AimdoContext, integrated_ram_headroom) == 16);
    assert!(offset_of!(AimdoContext, extra_vram_headroom) == 24);
    assert!(offset_of!(AimdoContext, malloc_async_clamp) == 32);
    assert!(offset_of!(AimdoContext, total_vram_usage) == 40);
    assert!(offset_of!(AimdoContext, total_vram_last_check) == 48);
    assert!(offset_of!(AimdoContext, deficit_sync) == 56);
    assert!(offset_of!(AimdoContext, control_timestamp_last_check) == 64);
    assert!(offset_of!(AimdoContext, highest_priority) == 72);
    assert!(offset_of!(AimdoContext, lowest_priority) == 80);
    assert!(offset_of!(AimdoContext, vbars_dirty) == 88);
    assert!(offset_of!(AimdoContext, allocations_dirty) == 89);
    assert!(offset_of!(AimdoContext, integrated_device) == 90);
    assert!(offset_of!(AimdoContext, vmm_table) == 96);
    assert!(offset_of!(AimdoContext, size_table) == 32864);
    assert!(offset_of!(AimdoContext, size_table_lock) == 41056);
    assert!(offset_of!(AimdoContext, rogues) == 41064);
    assert!(offset_of!(AimdoContext, rogue_candidates) == 41072);
    assert!(offset_of!(AimdoContext, hostbuf_file_reader_slots) == 41080);
    assert!(offset_of!(AimdoContext, hostbuf_file_reader_active) == 41176);
};

#[repr(C)]
struct DeviceList {
    count: usize,
    contexts: [AimdoContext; 0],
}

static DEVICES: AtomicPtr<DeviceList> = AtomicPtr::new(null_mut());
static DEVICE_LIFECYCLE: Mutex<()> = Mutex::new(());
static SIMPLE_VRAM_HEADROOM: AtomicI64 = AtomicI64::new(VRAM_HEADROOM as i64);
thread_local! {
    // PyTorch 的钩子可能从不同线程进入，按当前线程绑定对应的 CUDA 设备上下文。
    static CURRENT_DEVICE: Cell<*mut AimdoContext> = const { Cell::new(null_mut()) };
}

unsafe extern "C" {
    fn calloc(count: usize, size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
    fn aimdo_cuda_runtime_init() -> bool;
    fn aimdo_cuda_runtime_cleanup();
    fn aimdo_setup_hooks() -> bool;
    fn aimdo_teardown_hooks();
    fn xfer_file_init() -> bool;
    fn xfer_file_cleanup();
    fn allocations_init() -> bool;
    fn allocations_cleanup();
    fn hostbuf_file_reader_cleanup();
    fn log_reset_shots();
    fn vbars_analyze(context: *mut AimdoContext, only_dirty: bool) -> u64;
    fn allocations_analyze(only_dirty: bool);

    fn aimdo_control_c_tick() -> u64;
    fn aimdo_control_c_current_device(device: *mut i32) -> bool;
    fn aimdo_control_c_device_get(device: *mut i32, ordinal: i32) -> bool;
    fn aimdo_control_c_device_memory(device: i32, capacity: *mut u64) -> bool;
    fn aimdo_control_c_device_integrated(device: i32) -> bool;
    fn aimdo_control_c_mem_info(free: *mut usize, total: *mut usize, label: *const c_char) -> bool;
    fn aimdo_control_c_log_integrated_headroom(headroom: usize);
    fn aimdo_control_c_log_device(device: i32, capacity: usize);
    fn aimdo_control_c_log_poll(
        integrated: bool,
        available: usize,
        total_or_headroom: usize,
        deficit: isize,
        recorded: usize,
        method: *const c_char,
    );
    fn aimdo_control_c_log_analysis_header();
    fn aimdo_control_c_log_analysis(recorded: usize, free: usize, total: usize);
}

unsafe fn context_at(list: *mut DeviceList, index: usize) -> *mut AimdoContext {
    // SAFETY: The caller ensures index is within the array allocated after the list header.
    unsafe {
        addr_of_mut!((*list).contexts)
            .cast::<AimdoContext>()
            .add(index)
    }
}

fn integrated_headroom(total: u64) -> u64 {
    (total / 16).clamp(2 * 1024 * MIB as u64, 8 * 1024 * MIB as u64)
}

fn mem_available() -> Option<usize> {
    let mut file = BufReader::new(File::open("/proc/meminfo").ok()?);
    let mut line = String::new();
    while file.read_line(&mut line).ok()? != 0 {
        if let Some(kib) = line
            .strip_prefix("MemAvailable:")
            .and_then(|value| value.split_ascii_whitespace().next())
            .and_then(|value| value.parse::<usize>().ok())
        {
            return Some(kib.wrapping_mul(1024));
        }
        line.clear();
    }
    None
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_get_current_devctx() -> *mut AimdoContext {
    CURRENT_DEVICE.get()
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_set_devctx(context: *mut AimdoContext) {
    CURRENT_DEVICE.set(context);
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_get_simple_vram_headroom() -> i64 {
    SIMPLE_VRAM_HEADROOM.load(Ordering::Relaxed)
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_set_simple_vram_headroom(bytes: i64) {
    SIMPLE_VRAM_HEADROOM.store(bytes, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_get_devctx(device: i32) -> *mut AimdoContext {
    let list = DEVICES.load(Ordering::Acquire);
    if list.is_null() {
        return null_mut();
    }
    // SAFETY: Only initialized lists are published. The caller keeps device work quiescent
    // during cleanup, including any hook already using a returned raw context pointer.
    unsafe {
        for index in 0..(*list).count {
            let context = context_at(list, index);
            if (*context).device_id == device {
                return context;
            }
        }
    }
    null_mut()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_set_devctx_for_device(device: i32) -> bool {
    // SAFETY: This call has the same context lifetime requirement as the lookup it wraps.
    let context = unsafe { aimdo_rs_get_devctx(device) };
    aimdo_rs_set_devctx(context);
    !context.is_null()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_set_devctx_for_current_cuda_device() -> bool {
    let mut device = 0;
    // SAFETY: The CUDA hooks are active only while the driver dispatch is initialized.
    if !unsafe { aimdo_control_c_current_device(&mut device) } {
        aimdo_rs_set_devctx(null_mut());
        return false;
    }
    // SAFETY: The CUDA hook caller owns the current device's context lifetime.
    unsafe { aimdo_rs_set_devctx_for_device(device) }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_plat_init() -> bool {
    // SAFETY: Platform setup is serialized by the caller, and runs before device initialization.
    unsafe {
        log_reset_shots();
        if !aimdo_cuda_runtime_init() {
            return false;
        }
        if !xfer_file_init() {
            aimdo_cuda_runtime_cleanup();
            return false;
        }
        if aimdo_setup_hooks() {
            return true;
        }
        xfer_file_cleanup();
        aimdo_cuda_runtime_cleanup();
        false
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn aimdo_rs_plat_cleanup() {
    // SAFETY: The caller has finished using the device contexts and installed hooks.
    unsafe {
        xfer_file_cleanup();
        aimdo_teardown_hooks();
        aimdo_cuda_runtime_cleanup();
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_cuda_budget_deficit(method: *mut *const c_char) -> bool {
    // SAFETY: The caller selects a live context and provides storage for the method name.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        let now = aimdo_control_c_tick();
        if now.wrapping_sub((*context).control_timestamp_last_check) < 2000 {
            return true;
        }
        (*context).control_timestamp_last_check = now;
        (*context).total_vram_last_check = (*context).total_vram_usage;

        if (*context).integrated_device {
            let Some(available) = mem_available() else {
                (*context).deficit_sync = INTEGRATED_SIMPLE_ONLY_DEFICIT;
                return false;
            };
            let headroom = (*context).integrated_ram_headroom as usize;
            let deficit = (headroom as isize).wrapping_sub(available as isize);
            (*context).deficit_sync = deficit;
            *method = c"/proc/meminfo (integrated RAM)".as_ptr();
            aimdo_control_c_log_poll(
                true,
                available,
                headroom,
                deficit,
                (*context).total_vram_usage as usize,
                *method,
            );
            return true;
        }

        let mut free_bytes = 0;
        let mut total_bytes = 0;
        if !aimdo_control_c_mem_info(
            &mut free_bytes,
            &mut total_bytes,
            c"cuMemGetInfo(&free_vram, &total_vram)".as_ptr(),
        ) {
            return false;
        }
        let deficit = (VRAM_HEADROOM as isize).wrapping_sub(free_bytes as isize);
        (*context).deficit_sync = deficit;
        *method = c"cuMemGetInfo".as_ptr();
        aimdo_control_c_log_poll(
            false,
            free_bytes,
            total_bytes,
            deficit,
            (*context).total_vram_usage as usize,
            *method,
        );
        true
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_analyze(context: *mut AimdoContext) {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller provides a live context after platform initialization.
    unsafe {
        let mut free_bytes = 0;
        let mut total_bytes = 0;
        aimdo_control_c_log_analysis_header();
        aimdo_control_c_mem_info(
            &mut free_bytes,
            &mut total_bytes,
            c"cuMemGetInfo(&free_bytes, &total_bytes)".as_ptr(),
        );
        aimdo_control_c_log_analysis(
            (*context).total_vram_usage as usize,
            free_bytes,
            total_bytes,
        );
        vbars_analyze(context, true);
        allocations_analyze(true);
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_get_total_vram_usage(context: *mut AimdoContext) -> u64 {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller provides a live context after successful device initialization.
    unsafe { (*context).total_vram_usage }
}

unsafe fn cleanup_device_list(list: *mut DeviceList) {
    // SAFETY: The caller owns this unpublished list. Cleanup functions accept the zeroed
    // slots that follow a failed initialization, as well as fully initialized entries.
    unsafe {
        for index in 0..(*list).count {
            let context = context_at(list, index);
            aimdo_rs_set_devctx(context);
            hostbuf_file_reader_cleanup();
            allocations_cleanup();
            free((*context).highest_priority);
        }
        aimdo_rs_set_devctx(null_mut());
        free(list.cast());
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_cleanup() {
    let Ok(_lifecycle) = DEVICE_LIFECYCLE.try_lock() else {
        return;
    };
    // 先阻止新的查找。已取得裸指针的调用需要在进入清理前退出。
    let list = DEVICES.swap(null_mut(), Ordering::AcqRel);
    if list.is_null() {
        aimdo_rs_set_devctx(null_mut());
    } else {
        // SAFETY: The caller has stopped all users of previously returned context pointers.
        unsafe { cleanup_device_list(list) };
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn aimdo_rs_init(
    device_ids: *const i32,
    extra_headrooms: *const u64,
    count: usize,
) -> bool {
    let Ok(_lifecycle) = DEVICE_LIFECYCLE.try_lock() else {
        return false;
    };
    if !DEVICES.load(Ordering::Acquire).is_null() {
        return false;
    }
    let Some(bytes) = count
        .checked_mul(size_of::<AimdoContext>())
        .and_then(|contexts| contexts.checked_add(size_of::<DeviceList>()))
    else {
        return false;
    };

    // SAFETY: C supplies count elements in both arrays. calloc initializes the header and
    // context entries, and their base remains stable until cleanup on the host lifecycle.
    unsafe {
        let list = calloc(1, bytes).cast::<DeviceList>();
        if list.is_null() {
            return false;
        }
        (*list).count = count;
        for index in 0..count {
            let context = context_at(list, index);
            let ordinal = *device_ids.add(index);
            (*context).device_id = ordinal;
            (*context).extra_vram_headroom = *extra_headrooms.add(index);
            (*context).malloc_async_clamp = u64::MAX;
            (*context).hostbuf_file_reader_active = -1;
            aimdo_rs_set_devctx(context);

            let mut device = 0;
            if !allocations_init()
                || !aimdo_control_c_device_get(&mut device, ordinal)
                || !aimdo_control_c_device_memory(device, addr_of_mut!((*context).vram_capacity))
            {
                cleanup_device_list(list);
                return false;
            }
            (*context).integrated_device = aimdo_control_c_device_integrated(device);
            let capacity = (*context).vram_capacity;
            if (*context).integrated_device {
                let headroom = integrated_headroom(capacity);
                (*context).integrated_ram_headroom = headroom;
                aimdo_control_c_log_integrated_headroom(headroom as usize);
            }
            aimdo_control_c_log_device(device, capacity as usize);
        }
        // 所有设备的分配表和锁初始化成功后，再让 CUDA 分配钩子看到设备列表。
        DEVICES.store(list, Ordering::Release);
    }
    aimdo_rs_set_devctx(null_mut());
    true
}
