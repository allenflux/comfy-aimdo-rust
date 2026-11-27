#![cfg_attr(not(any(test, feature = "cuda-control")), no_std)]

#[cfg(all(
    feature = "cuda-control",
    not(all(target_os = "linux", target_arch = "x86_64"))
))]
compile_error!("The Rust CUDA control path targets Linux x86_64.");

#[cfg(all(feature = "cuda-control", target_os = "linux", target_arch = "x86_64"))]
mod context;
#[cfg(all(feature = "cuda-control", target_os = "linux", target_arch = "x86_64"))]
mod cuda;

#[cfg(feature = "rust-backend")]
mod alloc_hooks;
#[cfg(feature = "rust-backend")]
mod graph;
#[cfg(feature = "rust-backend")]
mod hostbuf;
#[cfg(feature = "rust-backend")]
mod mmap;
#[cfg(feature = "rust-backend")]
mod platform;
#[cfg(feature = "rust-backend")]
mod python;
#[cfg(feature = "rust-backend")]
mod rogue;
#[cfg(feature = "rust-backend")]
mod thread;
#[cfg(feature = "rust-backend")]
mod vbar;
#[cfg(feature = "rust-backend")]
mod vmm_ref;
#[cfg(feature = "rust-backend")]
mod vrambuf;
#[cfg(feature = "rust-backend")]
mod xfer_file;

use core::ptr::addr_of_mut;

pub(crate) const PAGE_BYTES: u64 = 32 << 20;
const CUDA_SUCCESS: i32 = 0;
const CUDA_ERROR_OUT_OF_MEMORY: i32 = 2;
const FAULT_SUCCESS: i32 = 0;
const FAULT_OOM: i32 = 1;
const FAULT_ERROR: i32 = 2;

#[repr(C)]
pub(crate) struct ResidentPage {
    pub(crate) handle: u64,
    pub(crate) pin_count: u32,
    pub(crate) serial: usize,
}

#[repr(C)]
pub struct ModelVBAR {
    pub(crate) vbar: u64,
    pub(crate) nr_pages: usize,
    pub(crate) watermark: usize,
    pub(crate) watermark_limit: usize,
    pub(crate) device: i32,
    pub(crate) higher: *mut ModelVBAR,
    pub(crate) lower: *mut ModelVBAR,
    pub(crate) resident_count: usize,
    pub(crate) pageout_size: u64,
    pub(crate) pageout_callback: Option<unsafe extern "C" fn(u64, u64) -> bool>,
    pub(crate) residency_map: [ResidentPage; 0],
}

#[repr(i32)]
enum FaultEvent {
    AboveWatermark,
    AllocationCheckWatermark,
    AllocatingPage,
    NonOomFailure,
    BackupFree,
    BackupWatermark,
    AllocationRetryFailure,
}

unsafe extern "C" {
    fn aimdo_vbar_c_prepare_pressure();
    fn aimdo_vbar_c_first() -> *mut ModelVBAR;
    fn aimdo_vbar_c_end() -> *mut ModelVBAR;
    fn aimdo_vbar_c_sync_paused() -> bool;
    fn aimdo_vbar_c_sync() -> bool;
    fn aimdo_vbar_c_deficit(bytes: usize) -> isize;
    fn aimdo_vbar_c_surplus(bytes: usize) -> isize;
    fn aimdo_vbar_c_map(address: u64, device: i32, handle: *mut u64) -> i32;
    fn aimdo_vbar_c_unmap(address: u64, handle: u64);
    fn aimdo_vbar_c_log(event: i32, page: usize);

    #[cfg(not(any(test, feature = "cuda-control")))]
    fn abort() -> !;
}

#[cfg(not(any(test, feature = "cuda-control")))]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo<'_>) -> ! {
    // SAFETY: abort does not unwind across the C ABI.
    unsafe { abort() }
}

pub(crate) fn pages_for_bytes(bytes: u64) -> usize {
    (bytes.wrapping_add(PAGE_BYTES - 1) / PAGE_BYTES) as usize
}

pub(crate) unsafe fn page_at(vbar: *mut ModelVBAR, page: usize) -> *mut ResidentPage {
    // SAFETY: The C allocation includes nr_pages entries after the header.
    unsafe {
        addr_of_mut!((*vbar).residency_map)
            .cast::<ResidentPage>()
            .add(page)
    }
}

/// # Safety
/// `vbar` and `page` must refer to a live C allocation on the current AIMDO device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_rs_vbar_modify_page(
    vbar: *mut ModelVBAR,
    page: usize,
    do_free: bool,
    do_unpin: bool,
) -> i32 {
    // SAFETY: C owns the object and device context for this call. Raw pointers keep any C
    // driver or writeback callback from overlapping a Rust reference to the object.
    unsafe {
        let entry = page_at(vbar, page);
        let free_page = do_free && (*entry).handle != 0 && (do_unpin || (*entry).pin_count == 0);
        if free_page {
            let offset = page as u64 * PAGE_BYTES;
            if let Some(writeback) = (*vbar).pageout_callback {
                let size = PAGE_BYTES.min((*vbar).pageout_size - offset);
                if !aimdo_vbar_c_sync() || !writeback(offset, size) {
                    return -1;
                }
            }
            aimdo_vbar_c_unmap((*vbar).vbar + offset, (*entry).handle);
            (*entry).handle = 0;
            (*vbar).resident_count -= 1;
        }
        if do_unpin {
            (*entry).pin_count = 0;
        }
        i32::from(free_page)
    }
}

/// # Safety
/// The caller must have selected the live AIMDO device context for this thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_rs_vbars_free(size: isize) -> bool {
    // SAFETY: The selected context owns the linked list and its allocations for this call.
    unsafe {
        let sync = !aimdo_vbar_c_sync_paused();
        aimdo_vbar_c_prepare_pressure();
        if size <= 0 {
            return true;
        }

        let mut pages_needed = pages_for_bytes(size as u64);
        let mut touched_watermark = false;
        let mut candidate = aimdo_vbar_c_first();
        let end = aimdo_vbar_c_end();
        while pages_needed > 0 && candidate != end {
            while pages_needed > 0 && (*candidate).watermark > (*candidate).watermark_limit {
                if !touched_watermark {
                    // Graph callers pin all pages they need before pausing synchronization.
                    if sync {
                        aimdo_vbar_c_sync();
                    }
                    touched_watermark = true;
                }
                let freed =
                    aimdo_rs_vbar_modify_page(candidate, (*candidate).watermark - 1, true, false);
                if freed < 0 {
                    return false;
                }
                pages_needed -= freed as usize;
                (*candidate).watermark -= 1;
            }
            candidate = (*candidate).higher;
        }
        if touched_watermark && sync {
            aimdo_vbar_c_sync();
        }
        true
    }
}

unsafe fn next_absent_page(vbar: *mut ModelVBAR, mut cursor: usize) -> usize {
    // SAFETY: The watermark is at most the number of C-allocated page entries.
    unsafe {
        while cursor < (*vbar).watermark && (*page_at(vbar, cursor)).handle != 0 {
            cursor += 1;
        }
    }
    cursor
}

unsafe fn spend_surplus(
    vbar: *mut ModelVBAR,
    target: usize,
    mut cursor: usize,
    surplus: &mut isize,
) -> usize {
    // SAFETY: The caller supplies the live VBAR while walking its residency map.
    unsafe {
        while *surplus >= PAGE_BYTES as isize && cursor < target && cursor < (*vbar).watermark {
            *surplus -= PAGE_BYTES as isize;
            cursor = next_absent_page(vbar, cursor + 1);
        }
    }
    cursor
}

unsafe fn free_for_vbar(vbar: *mut ModelVBAR, target: usize, mut surplus: isize) -> bool {
    // SAFETY: Faulting runs with the same live VBAR and context as the pressure eviction path.
    unsafe {
        let mut cursor = spend_surplus(vbar, target, next_absent_page(vbar, 0), &mut surplus);
        let mut synced = false;
        let mut candidate = aimdo_vbar_c_first();
        let end = aimdo_vbar_c_end();
        while ((cursor < target && cursor < (*vbar).watermark) || surplus < 0) && candidate != end {
            while ((cursor < target && cursor < (*vbar).watermark) || surplus < 0)
                && (*candidate).watermark > (*candidate).watermark_limit
            {
                let page = (*candidate).watermark - 1;
                let entry = page_at(candidate, page);
                if !synced && (*entry).handle != 0 && (*entry).pin_count == 0 {
                    aimdo_vbar_c_sync();
                    synced = true;
                }
                let freed = aimdo_rs_vbar_modify_page(candidate, page, true, false);
                if freed < 0 {
                    return false;
                }
                if freed != 0 {
                    surplus += PAGE_BYTES as isize;
                    cursor = spend_surplus(vbar, target, cursor, &mut surplus);
                }
                (*candidate).watermark -= 1;
            }
            candidate = (*candidate).higher;
        }
        if synced {
            aimdo_vbar_c_sync();
        }
        true
    }
}

/// # Safety
/// `vbar` must be live on the selected AIMDO device. `signature` must have room for
/// one `u32` per page covered by the range that can be faulted.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_rs_vbar_fault(
    vbar: *mut ModelVBAR,
    offset: u64,
    size: u64,
    signature: *mut u32,
) -> i32 {
    // SAFETY: The caller supplies the VBAR, selected device, and writable signature storage.
    unsafe {
        let first_page = (offset / PAGE_BYTES) as usize;
        let page_end = pages_for_bytes(offset.wrapping_add(size));
        if !aimdo_rs_vbars_free(aimdo_vbar_c_deficit(0)) {
            return FAULT_ERROR;
        }
        if page_end > (*vbar).watermark {
            aimdo_vbar_c_log(FaultEvent::AboveWatermark as i32, 0);
            return FAULT_OOM;
        }

        let mut checked_miss = false;
        for page in first_page..page_end {
            let entry = page_at(vbar, page);
            if (*entry).handle == 0 {
                if !checked_miss {
                    let pending_bytes = (page_end - page) * PAGE_BYTES as usize;
                    if !free_for_vbar(vbar, page_end, aimdo_vbar_c_surplus(pending_bytes)) {
                        return FAULT_ERROR;
                    }
                    checked_miss = true;
                    if page_end > (*vbar).watermark {
                        aimdo_vbar_c_log(FaultEvent::AllocationCheckWatermark as i32, 0);
                        return FAULT_OOM;
                    }
                }

                aimdo_vbar_c_log(FaultEvent::AllocatingPage as i32, page);
                let address = (*vbar).vbar + page as u64 * PAGE_BYTES;
                let error = if aimdo_vbar_c_deficit(PAGE_BYTES as usize) > 0 {
                    CUDA_ERROR_OUT_OF_MEMORY
                } else {
                    aimdo_vbar_c_map(address, (*vbar).device, addr_of_mut!((*entry).handle))
                };
                if error != CUDA_SUCCESS {
                    if error != CUDA_ERROR_OUT_OF_MEMORY {
                        aimdo_vbar_c_log(FaultEvent::NonOomFailure as i32, 0);
                        return FAULT_ERROR;
                    }
                    aimdo_vbar_c_log(FaultEvent::BackupFree as i32, 0);
                    if !aimdo_rs_vbars_free(PAGE_BYTES as isize) {
                        return FAULT_ERROR;
                    }
                    if page_end > (*vbar).watermark {
                        aimdo_vbar_c_log(FaultEvent::BackupWatermark as i32, 0);
                        return FAULT_OOM;
                    }
                    if aimdo_vbar_c_map(address, (*vbar).device, addr_of_mut!((*entry).handle))
                        != CUDA_SUCCESS
                    {
                        aimdo_vbar_c_log(FaultEvent::AllocationRetryFailure as i32, 0);
                        return FAULT_ERROR;
                    }
                }
                (*entry).serial = (*entry).serial.wrapping_add(1);
                (*vbar).resident_count += 1;
            }
            *signature.add(page - first_page) = (*entry).serial as u32;
        }

        for page in first_page..page_end {
            let entry = page_at(vbar, page);
            (*entry).pin_count = (*entry).pin_count.wrapping_add(1);
        }
        FAULT_SUCCESS
    }
}

/// # Safety
/// `vbar` must be a live allocation on the selected AIMDO device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_rs_vbar_unpin(vbar: *mut ModelVBAR, offset: u64, size: u64) -> bool {
    // SAFETY: The caller supplies the live VBAR and the range is bounded by its page count.
    unsafe {
        let first_page = (offset / PAGE_BYTES) as usize;
        let page_end = pages_for_bytes(offset.wrapping_add(size));
        if page_end > (*vbar).watermark {
            aimdo_vbar_c_sync();
        }

        let resident_end = page_end.min((*vbar).nr_pages);
        for page in first_page..resident_end {
            let entry = page_at(vbar, page);
            (*entry).pin_count = (*entry).pin_count.saturating_sub(1);
        }
        // Drop every pin before writeback, since a callback can interrupt eviction.
        for page in first_page..resident_end {
            if aimdo_rs_vbar_modify_page(vbar, page, page >= (*vbar).watermark, false) < 0 {
                return false;
            }
        }

        if page_end > (*vbar).watermark {
            aimdo_vbar_c_sync();
        }
        true
    }
}

/// # Safety
/// `vbar` must be a live allocation on the selected AIMDO device.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_rs_vbar_free_memory(vbar: *mut ModelVBAR, size: u64) -> i64 {
    // SAFETY: The caller supplies the live VBAR and selected context; its watermark bounds access.
    unsafe {
        let mut pages_to_free = pages_for_bytes(size);
        let mut pages_freed = 0;
        aimdo_vbar_c_sync();
        while pages_to_free > 0 && (*vbar).watermark > (*vbar).watermark_limit {
            let freed = aimdo_rs_vbar_modify_page(vbar, (*vbar).watermark - 1, true, false);
            if freed < 0 {
                return -1;
            }
            pages_to_free -= freed as usize;
            pages_freed += i64::from(freed);
            (*vbar).watermark -= 1;
        }
        aimdo_vbar_c_sync();
        pages_freed * PAGE_BYTES as i64
    }
}
