use core::ffi::{c_char, c_void, CStr};
use core::mem::size_of;
use core::ptr::{addr_of, addr_of_mut, null, null_mut};

use crate::context::{
    aimdo_rs_get_current_devctx, aimdo_rs_set_devctx_for_current_cuda_device,
    aimdo_rs_set_devctx_for_device, AimdoContext,
};
use crate::cuda::{g_cuda, CuStream};
use crate::platform::{
    budget_deficit, check_cuda, log, log_shot, DEBUG, ERROR, MIB, VERBOSE, VVERBOSE,
};
use crate::vbar::vbars_free;
use crate::vrambuf::{
    align_up, vrambuf_create, vrambuf_destroy, vrambuf_grow, VramBuffer, CUDA_PAGE,
};

const CUDA_ERROR_UNKNOWN: i32 = 999;
const CUDA_MALLOC_HEADROOM: usize = 128 * MIB;
const KIB: usize = 1024;

type Alloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type Free = unsafe extern "C" fn(u64) -> i32;
type AllocAsync = unsafe extern "C" fn(*mut u64, usize, CuStream) -> i32;
type FreeAsync = unsafe extern "C" fn(u64, CuStream) -> i32;

unsafe extern "C" {
    fn free_rogue(ptr: u64, result: *mut i32) -> bool;
    fn malloc_graph_alloc(ptr: *mut u64, size: usize, stream: CuStream, result: *mut i32) -> bool;
    fn malloc_graph_free(ptr: u64, stream: CuStream, result: *mut i32) -> bool;
    fn funchook_create() -> *mut c_void;
    fn funchook_prepare(
        hook: *mut c_void,
        target: *mut *mut c_void,
        replacement: *mut c_void,
    ) -> i32;
    fn funchook_install(hook: *mut c_void, flags: i32) -> i32;
    fn funchook_uninstall(hook: *mut c_void, flags: i32) -> i32;
    fn funchook_destroy(hook: *mut c_void) -> i32;
    fn funchook_error_message(hook: *const c_void) -> *const c_char;
}

struct SizeEntry {
    ptr: u64,
    size: usize,
    next: *mut SizeEntry,
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn allocations_init() -> bool {
    // SAFETY: The allocation mutex is initialized before this context is published.
    unsafe {
        let mutex =
            libc::malloc(size_of::<libc::pthread_mutex_t>()).cast::<libc::pthread_mutex_t>();
        if mutex.is_null() || libc::pthread_mutex_init(mutex, null()) != 0 {
            libc::free(mutex.cast());
            return false;
        }
        (*aimdo_rs_get_current_devctx()).size_table_lock = mutex.cast();
        true
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn allocations_cleanup() {
    // SAFETY: Context cleanup owns the table and the mutex has no remaining users.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        let mutex = (*context).size_table_lock.cast::<libc::pthread_mutex_t>();
        if !mutex.is_null() {
            libc::pthread_mutex_destroy(mutex);
            libc::free(mutex.cast());
            (*context).size_table_lock = null_mut();
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn allocations_lock() {
    // SAFETY: The selected context has completed allocations_init.
    unsafe { libc::pthread_mutex_lock((*aimdo_rs_get_current_devctx()).size_table_lock.cast()) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn allocations_unlock() {
    // SAFETY: This thread holds the selected context's mutex.
    unsafe { libc::pthread_mutex_unlock((*aimdo_rs_get_current_devctx()).size_table_lock.cast()) };
}

fn accounted_size(size: usize) -> usize {
    if size <= KIB {
        KIB
    } else if size > CUDA_PAGE / 2 {
        align_up(size, CUDA_PAGE)
    } else {
        size
    }
}

unsafe fn size_bucket(context: *mut AimdoContext, ptr: u64) -> *mut *mut SizeEntry {
    let hash = ((ptr >> 10) ^ (ptr >> 21)) as usize % 1024;
    // SAFETY: The bounded hash indexes the context's size table.
    unsafe {
        addr_of_mut!((*context).size_table)
            .cast::<*mut SizeEntry>()
            .add(hash)
    }
}

unsafe fn account_alloc(ptr: u64, size: usize) {
    // SAFETY: Registered CUDA allocations have a live per-device size table and lock.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        allocations_lock();
        (*context).total_vram_usage = (*context)
            .total_vram_usage
            .wrapping_add(accounted_size(size) as u64);
        let entry = libc::malloc(size_of::<SizeEntry>()).cast::<SizeEntry>();
        if !entry.is_null() {
            let bucket = size_bucket(context, ptr);
            entry.write(SizeEntry {
                ptr,
                size,
                next: *bucket,
            });
            *bucket = entry;
        }
        allocations_unlock();
    }
}

unsafe fn account_free(ptr: u64, stream: CuStream) {
    // SAFETY: This free is for the selected device; the lock protects the entry chain.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        allocations_lock();
        let mut previous = size_bucket(context, ptr);
        while !(*previous).is_null() {
            let entry = *previous;
            if (*entry).ptr == ptr {
                *previous = (*entry).next;
                log(
                    VVERBOSE,
                    format_args!(
                        "Freed: ptr={ptr:#x}, size={}k, stream={stream:p}\n",
                        (*entry).size / KIB
                    ),
                );
                (*context).total_vram_usage = (*context)
                    .total_vram_usage
                    .wrapping_sub(accounted_size((*entry).size) as u64);
                allocations_unlock();
                libc::free(entry.cast());
                return;
            }
            previous = addr_of_mut!((*entry).next);
        }
        allocations_unlock();
        log(
            DEBUG,
            format_args!("account_free: could not account free at {ptr:#x}\n"),
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_cuda_malloc(
    ptr: *mut u64,
    size: usize,
    original: Option<Alloc>,
) -> i32 {
    let Some(original) = original else { return 1 };
    if ptr.is_null() {
        return 1;
    }
    // SAFETY: The installed hook receives CUDA's valid arguments and trampoline.
    unsafe {
        if !aimdo_rs_set_devctx_for_current_cuda_device() {
            return original(ptr, size);
        }
        *ptr = 0;
        let bytes = size.wrapping_add(CUDA_MALLOC_HEADROOM);
        if !vbars_free(budget_deficit(bytes)) {
            return CUDA_ERROR_UNKNOWN;
        }
        let mut allocated = 0;
        let first = original(&mut allocated, size);
        if !check_cuda(first, "true_cuMemAlloc_v2(&dptr, size)") {
            if !vbars_free(bytes as isize) {
                return CUDA_ERROR_UNKNOWN;
            }
            let status = original(&mut allocated, size);
            if !check_cuda(status, "status") {
                return status;
            }
        }
        *ptr = allocated;
        account_alloc(allocated, size);
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_cuda_free(ptr: u64, original: Option<Free>) -> i32 {
    if ptr == 0 {
        return 0;
    }
    let Some(original) = original else { return 1 };
    // SAFETY: The installed hook receives CUDA's valid arguments and trampoline.
    unsafe {
        if !aimdo_rs_set_devctx_for_current_cuda_device() {
            return original(ptr);
        }
        let mut status = 0;
        if free_rogue(ptr, &mut status) {
            return status;
        }
        status = original(ptr);
        if check_cuda(status, "status") {
            account_free(ptr, null_mut());
        }
        status
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_cuda_malloc_async(
    ptr: *mut u64,
    size: usize,
    stream: CuStream,
    original: Option<AllocAsync>,
) -> i32 {
    log(
        VVERBOSE,
        format_args!(
            "aimdo_cuda_malloc_async (start) size={}k stream={stream:p}\n",
            size / KIB
        ),
    );
    let Some(original) = original else { return 1 };
    if ptr.is_null() {
        return 1;
    }
    // SAFETY: The installed hook receives CUDA's valid arguments and trampoline.
    unsafe {
        if !aimdo_rs_set_devctx_for_current_cuda_device() {
            return original(ptr, size, stream);
        }
        let mut status = 0;
        if malloc_graph_alloc(ptr, size, stream, &mut status) {
            return status;
        }
        *ptr = 0;
        let clamp = (*aimdo_rs_get_current_devctx()).malloc_async_clamp as usize;
        if !vbars_free(budget_deficit(size.min(clamp))) {
            return CUDA_ERROR_UNKNOWN;
        }
        let mut allocated = 0;
        status = original(&mut allocated, size, stream);
        if !check_cuda(status, "status") {
            if !vbars_free(size as isize) {
                return CUDA_ERROR_UNKNOWN;
            }
            status = original(&mut allocated, size, stream);
            if !check_cuda(status, "status") {
                return status;
            }
        }
        *ptr = allocated;
        account_alloc(allocated, size);
        log(
            VVERBOSE,
            format_args!("aimdo_cuda_malloc_async (return): ptr={allocated:#x}\n"),
        );
        0
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_cuda_free_async(
    ptr: u64,
    stream: CuStream,
    original: Option<FreeAsync>,
) -> i32 {
    log(
        VVERBOSE,
        format_args!("aimdo_cuda_free_async (start) ptr={ptr:#x}\n"),
    );
    if ptr == 0 {
        return 0;
    }
    let Some(original) = original else { return 1 };
    // SAFETY: The installed hook receives CUDA's valid arguments and trampoline.
    unsafe {
        if !aimdo_rs_set_devctx_for_current_cuda_device() {
            return original(ptr, stream);
        }
        let mut status = 0;
        if free_rogue(ptr, &mut status) || malloc_graph_free(ptr, stream, &mut status) {
            return status;
        }
        status = original(ptr, stream);
        if check_cuda(status, "status") {
            account_free(ptr, stream);
        }
        status
    }
}

fn vmm_hash(ptr: u64) -> usize {
    (ptr as usize >> 21) % 4096
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn allocations_analyze(only_dirty: bool) {
    // SAFETY: The caller selected the live context whose buffers are being inspected.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        if only_dirty && !(*context).allocations_dirty {
            return;
        }
        (*context).allocations_dirty = false;
        log(DEBUG, format_args!("--- Allocation Analysis Start ---\n"));
        let mut total = 0;
        let mut count = 0;
        for bucket in 0..4096 {
            let mut buffer = (*context).vmm_table[bucket].cast::<VramBuffer>();
            while !buffer.is_null() {
                log(
                    DEBUG,
                    format_args!(
                        "  [Bucket {bucket:4}] Ptr: {:#x} | Size: {:7}k\n",
                        (*buffer).base_ptr,
                        (*buffer).allocated / KIB
                    ),
                );
                total += (*buffer).allocated;
                count += 1;
                buffer = (*buffer).next;
            }
        }
        log(
            DEBUG,
            format_args!(
                "{count} Active Allocations for a total of {:7} MB\n",
                total / MIB
            ),
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn alloc_fn(size: usize, device: i32, _stream: CuStream) -> *mut c_void {
    log(
        VERBOSE,
        format_args!("alloc_fn (start): size={}k, device={device}\n", size / KIB),
    );
    // SAFETY: The registered device owns the reservation and the allocator table entry.
    unsafe {
        if !aimdo_rs_set_devctx_for_device(device) {
            return null_mut();
        }
        let buffer = vrambuf_create(device, size);
        if buffer.is_null() {
            return null_mut();
        }
        if !vrambuf_grow(buffer, size) {
            vrambuf_destroy(buffer);
            return null_mut();
        }
        let context = aimdo_rs_get_current_devctx();
        let ptr = (*buffer).base_ptr;
        let bucket = vmm_hash(ptr);
        (*buffer).next = (*context).vmm_table[bucket].cast();
        (*context).vmm_table[bucket] = buffer.cast();
        (*context).allocations_dirty = true;
        log(VERBOSE, format_args!("alloc_fn (return): ptr={ptr:#x}\n"));
        ptr as *mut c_void
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn free_fn(ptr: *mut c_void, size: usize, device: i32, stream: CuStream) {
    log_shot(DEBUG, format_args!("Pytorch is freeing VRAM ...\n"));
    log(
        VERBOSE,
        format_args!(
            "free_fn (start) ptr={ptr:p} size={}k, device={device}\n",
            size / KIB
        ),
    );
    // SAFETY: The registered device owns the VMM table and any matching reservation.
    unsafe {
        if ptr.is_null() || !aimdo_rs_set_devctx_for_device(device) {
            return;
        }
        let context = aimdo_rs_get_current_devctx();
        let bucket = vmm_hash(ptr as u64);
        let mut previous = addr_of_mut!((*context).vmm_table[bucket]).cast::<*mut VramBuffer>();
        while !(*previous).is_null() {
            let buffer = *previous;
            if (*buffer).base_ptr == ptr as u64 && (*buffer).device == device {
                *previous = (*buffer).next;
                (*context).allocations_dirty = true;
                vrambuf_destroy(buffer);
                log(
                    VERBOSE,
                    format_args!(
                        "Freed: ptr={ptr:p}, size={}k, stream={stream:p}\n",
                        size / KIB
                    ),
                );
                return;
            }
            previous = addr_of_mut!((*buffer).next);
        }
        log(ERROR, format_args!("free_fn could not find VRAM@{ptr:p}\n"));
    }
}

static mut HOOKS: *mut c_void = null_mut();
static mut TRUE_ALLOC: Option<Alloc> = None;
static mut TRUE_FREE: Option<Free> = None;
static mut TRUE_ALLOC_ASYNC: Option<AllocAsync> = None;
static mut TRUE_ALLOC_ASYNC_PTSZ: Option<AllocAsync> = None;
static mut TRUE_FREE_ASYNC: Option<FreeAsync> = None;
static mut TRUE_FREE_ASYNC_PTSZ: Option<FreeAsync> = None;

unsafe extern "C" fn hook_alloc(ptr: *mut u64, size: usize) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_malloc(ptr, size, TRUE_ALLOC) }
}

unsafe extern "C" fn hook_free(ptr: u64) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_free(ptr, TRUE_FREE) }
}

unsafe extern "C" fn hook_alloc_async(ptr: *mut u64, size: usize, stream: CuStream) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_malloc_async(ptr, size, stream, TRUE_ALLOC_ASYNC) }
}

unsafe extern "C" fn hook_alloc_async_ptsz(ptr: *mut u64, size: usize, stream: CuStream) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_malloc_async(ptr, size, stream, TRUE_ALLOC_ASYNC_PTSZ) }
}

unsafe extern "C" fn hook_free_async(ptr: u64, stream: CuStream) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_free_async(ptr, stream, TRUE_FREE_ASYNC) }
}

unsafe extern "C" fn hook_free_async_ptsz(ptr: u64, stream: CuStream) -> i32 {
    // SAFETY: Hooks are installed only after their trampoline is prepared.
    unsafe { aimdo_cuda_free_async(ptr, stream, TRUE_FREE_ASYNC_PTSZ) }
}

struct HookEntry {
    trampoline: *mut *mut c_void,
    target: *mut c_void,
    replacement: *mut c_void,
    name: &'static str,
}

unsafe fn hook_error(hook: *mut c_void, operation: core::fmt::Arguments<'_>, result: i32) {
    // SAFETY: The hook library owns a NUL-terminated detail string for a live handle.
    unsafe {
        let detail = funchook_error_message(hook);
        if detail.is_null() {
            log(
                ERROR,
                format_args!("{operation} failed: {result} <unknown funchook error>\n"),
            );
        } else {
            log(
                ERROR,
                format_args!(
                    "{operation} failed: {result} {}\n",
                    CStr::from_ptr(detail).to_string_lossy()
                ),
            );
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_setup_hooks() -> bool {
    // SAFETY: CUDA dispatch and hook startup are serialized by platform initialization.
    unsafe {
        let dispatch = addr_of!(g_cuda);
        if (*dispatch).mem_alloc_v2.is_none() {
            log(
                ERROR,
                format_args!("aimdo_setup_hooks: CUDA hook targets are not resolved\n"),
            );
            return false;
        }
        let hook = funchook_create();
        if hook.is_null() {
            log(
                ERROR,
                format_args!("aimdo_setup_hooks: funchook_create failed\n"),
            );
            return false;
        }
        HOOKS = hook;
        macro_rules! entry {
            ($storage:ident, $field:ident, $replacement:ident, $name:literal) => {
                HookEntry {
                    trampoline: addr_of_mut!($storage).cast(),
                    target: (*dispatch).$field.map_or(null_mut(), |f| f as *mut c_void),
                    replacement: $replacement as *mut c_void,
                    name: $name,
                }
            };
        }
        let entries = [
            entry!(TRUE_ALLOC, mem_alloc_v2, hook_alloc, "cuMemAlloc_v2"),
            entry!(TRUE_FREE, mem_free_v2, hook_free, "cuMemFree_v2"),
            entry!(
                TRUE_ALLOC_ASYNC,
                mem_alloc_async,
                hook_alloc_async,
                "cuMemAllocAsync"
            ),
            entry!(
                TRUE_ALLOC_ASYNC_PTSZ,
                mem_alloc_async_ptsz,
                hook_alloc_async_ptsz,
                "cuMemAllocAsync_ptsz"
            ),
            entry!(
                TRUE_FREE_ASYNC,
                mem_free_async,
                hook_free_async,
                "cuMemFreeAsync"
            ),
            entry!(
                TRUE_FREE_ASYNC_PTSZ,
                mem_free_async_ptsz,
                hook_free_async_ptsz,
                "cuMemFreeAsync_ptsz"
            ),
        ];
        for entry in entries {
            if entry.target.is_null() {
                log(
                    ERROR,
                    format_args!("aimdo_setup_hooks: failed to resolve {}\n", entry.name),
                );
                aimdo_teardown_hooks();
                return false;
            }
            // funchook replaces this pointer with a callable trampoline in the same ABI.
            *entry.trampoline = entry.target;
            let result = funchook_prepare(hook, entry.trampoline, entry.replacement);
            if result != 0 {
                hook_error(
                    hook,
                    format_args!("aimdo_setup_hooks: funchook_prepare({})", entry.name),
                    result,
                );
                aimdo_teardown_hooks();
                return false;
            }
        }
        let result = funchook_install(hook, 0);
        if result != 0 {
            hook_error(
                hook,
                format_args!("aimdo_setup_hooks: funchook_install"),
                result,
            );
            aimdo_teardown_hooks();
            return false;
        }
        log(
            DEBUG,
            format_args!("aimdo_setup_hooks: hooks successfully installed\n"),
        );
        true
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_teardown_hooks() {
    // SAFETY: Platform teardown excludes concurrent setup and ends remaining hook use.
    unsafe {
        let hook = HOOKS;
        if hook.is_null() {
            return;
        }
        let mut result = funchook_uninstall(hook, 0);
        if result == 0 || result == 10 {
            // FUNCHOOK_ERROR_NOT_INSTALLED
            result = funchook_destroy(hook);
        }
        if result != 0 {
            hook_error(
                hook,
                format_args!("aimdo_teardown_hooks: funchook teardown"),
                result,
            );
        }
        HOOKS = null_mut();
    }
}
