use core::mem::size_of;
use core::ptr::{addr_of_mut, null_mut};

use crate::context::{
    aimdo_rs_get_current_devctx, aimdo_rs_get_simple_vram_headroom, aimdo_rs_set_devctx,
    AimdoContext,
};
use crate::platform::{
    budget_deficit, check_cuda, cuda, log, log_reset_shots, map_pages, CRITICAL, DEBUG, ERROR, MIB,
    VERBOSE, VVERBOSE, WARNING,
};
use crate::{
    aimdo_rs_vbar_fault, aimdo_rs_vbar_free_memory, aimdo_rs_vbar_modify_page, aimdo_rs_vbar_unpin,
    aimdo_rs_vbars_free, page_at, pages_for_bytes, ModelVBAR, ResidentPage, PAGE_BYTES,
};

unsafe extern "C" {
    fn malloc_graph_sync_paused() -> bool;
}

const PAGE: usize = PAGE_BYTES as usize;

// 每张卡维护自己的 VBAR 优先级。两个哨兵只负责链表边界，不占权重显存。
unsafe fn priorities() -> (*mut ModelVBAR, *mut ModelVBAR) {
    // SAFETY: The caller selected a live context and owns its VBAR list accesses.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        if (*context).highest_priority.is_null() {
            let high = libc::calloc(2, size_of::<ModelVBAR>()).cast::<ModelVBAR>();
            if high.is_null() {
                log(CRITICAL, format_args!("Host OOM\n"));
                return (null_mut(), null_mut());
            }
            let low = high.add(1);
            (*high).lower = low;
            (*low).higher = high;
            (*context).highest_priority = high.cast();
            (*context).lowest_priority = low.cast();
        }
        (
            (*context).highest_priority.cast(),
            (*context).lowest_priority.cast(),
        )
    }
}

unsafe fn remove(vbar: *mut ModelVBAR) {
    // SAFETY: The caller owns a linked VBAR with real or sentinel neighbors.
    unsafe {
        (*(*vbar).lower).higher = (*vbar).higher;
        (*(*vbar).higher).lower = (*vbar).lower;
    }
}

unsafe fn insert(vbar: *mut ModelVBAR, lower: *mut ModelVBAR, higher: *mut ModelVBAR) {
    // SAFETY: The list is live and lower and higher are adjacent on entry.
    unsafe {
        (*vbar).lower = lower;
        (*vbar).higher = higher;
        (*lower).higher = vbar;
        (*higher).lower = vbar;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_prepare_pressure() {
    // SAFETY: The allocator hook selected the device before entering the pressure path.
    unsafe {
        priorities();
        (*aimdo_rs_get_current_devctx()).vbars_dirty = true;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_first() -> *mut ModelVBAR {
    // SAFETY: The current device was initialized by the pressure caller.
    unsafe {
        let low = (*aimdo_rs_get_current_devctx())
            .lowest_priority
            .cast::<ModelVBAR>();
        if low.is_null() {
            null_mut()
        } else {
            (*low).higher
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_end() -> *mut ModelVBAR {
    // SAFETY: The caller selected a live device context.
    unsafe { (*aimdo_rs_get_current_devctx()).highest_priority.cast() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_sync_paused() -> bool {
    // SAFETY: The graph API uses the current thread's graph state.
    unsafe { malloc_graph_sync_paused() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_sync() -> bool {
    // SAFETY: The native caller initialized the CUDA dispatch.
    unsafe { check_cuda(cuda!(ctx_synchronize), "cuCtxSynchronize()") }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_deficit(bytes: usize) -> isize {
    // SAFETY: The caller selected a live device context.
    unsafe { budget_deficit(bytes) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_surplus(bytes: usize) -> isize {
    // SAFETY: The caller selected a live device context.
    unsafe {
        let context = aimdo_rs_get_current_devctx();
        let usage = (*context).total_vram_usage.wrapping_add(bytes as u64) as isize;
        ((*context).vram_capacity as isize)
            .wrapping_sub(usage.wrapping_add(aimdo_rs_get_simple_vram_headroom() as isize))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_map(address: u64, device: i32, handle: *mut u64) -> i32 {
    // SAFETY: The fault caller supplies a reserved address and writable handle storage.
    unsafe { map_pages(address, PAGE, device, handle) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_unmap(address: u64, handle: u64) {
    // SAFETY: The allocator tracked the mapping and no consumer pin remains.
    unsafe {
        check_cuda(
            cuda!(mem_unmap, address, PAGE),
            "cuMemUnmap(vaddr, VBAR_PAGE_SIZE)",
        );
        check_cuda(cuda!(mem_release, handle), "cuMemRelease(rp->handle)");
        let context = aimdo_rs_get_current_devctx();
        (*context).total_vram_usage = (*context).total_vram_usage.wrapping_sub(PAGE_BYTES);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn aimdo_vbar_c_log(event: i32, page: usize) {
    match event {
        0 => log(
            VVERBOSE,
            format_args!("VBAR Allocation is above watermark\n"),
        ),
        1 => log(
            DEBUG,
            format_args!("VBAR allocation cancelled due to allocation-check watermark reduction\n"),
        ),
        2 => log(
            VERBOSE,
            format_args!("VBAR needs to allocate VRAM for page {page}\n"),
        ),
        3 => log(ERROR, format_args!("VRAM Allocation failed (non OOM)\n")),
        4 => log(
            DEBUG,
            format_args!("VBAR allocator attempt exceeds available VRAM ...\n"),
        ),
        5 => log(
            DEBUG,
            format_args!("VBAR allocation cancelled due to backup-free watermark reduction\n"),
        ),
        6 => log(ERROR, format_args!("VRAM Allocation failed\n")),
        _ => {}
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbars_free(bytes: isize) -> bool {
    // SAFETY: The caller selected a live device context before evicting pages.
    unsafe { aimdo_rs_vbars_free(bytes) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbars_analyze(context: *mut AimdoContext, only_dirty: bool) -> u64 {
    // SAFETY: The caller supplies a live context and the list stays valid during analysis.
    unsafe {
        aimdo_rs_set_devctx(context);
        let (high, low) = priorities();
        if high.is_null() || (only_dirty && !(*context).vbars_dirty) {
            return 0;
        }
        (*context).vbars_dirty = false;
        log(
            DEBUG,
            format_args!("---------------- VBAR Usage ---------------\n"),
        );
        let mut total = 0;
        let mut model = (*low).higher;
        while !model.is_null() && model != high {
            let mut resident = 0;
            for page in 0..(*model).nr_pages {
                let entry = page_at(model, page);
                if (*entry).handle != 0 {
                    resident += 1;
                    if page >= (*model).watermark {
                        log(
                            WARNING,
                            format_args!(
                                "VBAR {model:p}: Resident page {page} is ABOVE watermark {}\n",
                                (*model).watermark
                            ),
                        );
                    }
                    if (*entry).pin_count != 0 {
                        log(
                            WARNING,
                            format_args!(
                                "VBAR {model:p}: Page {page} pin_count={}\n",
                                (*entry).pin_count
                            ),
                        );
                    }
                }
            }
            if resident != (*model).resident_count {
                log(WARNING, format_args!("VBAR {model:p}: resident_count sync error! Struct: {}, Actual: {resident}\n", (*model).resident_count));
            }
            let bytes = resident * PAGE;
            total += bytes;
            log(
                DEBUG,
                format_args!(
                    "VBAR {model:p}: Actual Resident VRAM = {} MB\n",
                    bytes / MIB
                ),
            );
            model = (*model).higher;
        }
        log(
            DEBUG,
            format_args!("Total VRAM for VBARs: {} MB\n", total / MIB),
        );
        total as u64
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_allocate(
    context: *mut AimdoContext,
    size: u64,
    device: i32,
    pageout: Option<unsafe extern "C" fn(u64, u64) -> bool>,
) -> *mut ModelVBAR {
    // SAFETY: The C ABI supplies the context and a callback valid for the reservation lifetime.
    unsafe {
        aimdo_rs_set_devctx(context);
        let (high, _) = priorities();
        if high.is_null() {
            return null_mut();
        }
        log_reset_shots();
        log(
            DEBUG,
            format_args!(
                "vbar_allocate (start): size={}M, device={device}\n",
                size / MIB as u64
            ),
        );
        (*context).vbars_dirty = true;
        let pages = pages_for_bytes(size).min(((*context).vram_capacity / PAGE_BYTES) as usize);
        let bytes = size_of::<ModelVBAR>() + pages * size_of::<ResidentPage>();
        let model = libc::calloc(1, bytes).cast::<ModelVBAR>();
        if model.is_null() {
            log(CRITICAL, format_args!("Host OOM\n"));
            return null_mut();
        }
        // 这里只预留虚拟地址，权重页的物理显存在 fault 时按需映射。
        if !check_cuda(
            cuda!(
                mem_address_reserve,
                addr_of_mut!((*model).vbar),
                pages * PAGE,
                0,
                0,
                0
            ),
            "cuMemAddressReserve(&mv->vbar, size, 0, 0, 0)",
        ) {
            log(
                ERROR,
                format_args!("Could not reserve Virtual Address space for VBAR\n"),
            );
            libc::free(model.cast());
            return null_mut();
        }
        (*model).pageout_size = size;
        (*model).pageout_callback = pageout;
        (*model).device = device;
        (*model).nr_pages = pages;
        (*model).watermark = pages;
        insert(model, (*high).lower, high);
        log(
            DEBUG,
            format_args!("vbar_allocate (return): vbar={model:p}\n"),
        );
        model
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_set_watermark_limit(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    size: u64,
) {
    // SAFETY: The caller owns a live model reservation in the supplied context.
    unsafe {
        aimdo_rs_set_devctx(context);
        log(
            DEBUG,
            format_args!("vbar_set_watermark_limit: size={size}\n"),
        );
        (*model).watermark_limit = pages_for_bytes(size);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_set_watermark(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    size: u64,
) -> bool {
    // SAFETY: The caller owns a live model reservation in the supplied context.
    unsafe {
        aimdo_rs_set_devctx(context);
        log(DEBUG, format_args!("vbar_set_watermark: size={size}\n"));
        (*context).vbars_dirty = true;
        let watermark = pages_for_bytes(size).min((*model).nr_pages);
        if watermark < (*model).watermark {
            aimdo_vbar_c_sync();
            for page in watermark..(*model).watermark {
                if aimdo_rs_vbar_modify_page(model, page, true, false) < 0 {
                    return false;
                }
            }
        }
        (*model).watermark = watermark;
        true
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbars_reset_watermark_limits(context: *mut AimdoContext) {
    // SAFETY: The caller supplies the device whose VBAR list is being traversed.
    unsafe {
        aimdo_rs_set_devctx(context);
        let (high, low) = priorities();
        log(VERBOSE, format_args!("vbars_reset_watermark_limits\n"));
        if high.is_null() {
            return;
        }
        let mut model = (*low).higher;
        while !model.is_null() && model != high {
            (*model).watermark_limit = 0;
            model = (*model).higher;
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_prioritize(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    clamp: u64,
) {
    // SAFETY: The caller owns a linked VBAR in this device's list.
    unsafe {
        aimdo_rs_set_devctx(context);
        (*context).malloc_async_clamp = clamp;
        log(DEBUG, format_args!("vbar_prioritize vbar={model:p}\n"));
        (*context).vbars_dirty = true;
        log_reset_shots();
        let high = (*context).highest_priority.cast::<ModelVBAR>();
        remove(model);
        insert(model, (*high).lower, high);
        (*model).watermark = (*model).nr_pages;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_deprioritize(context: *mut AimdoContext, model: *mut ModelVBAR) {
    // SAFETY: The caller owns a linked VBAR in this device's list.
    unsafe {
        aimdo_rs_set_devctx(context);
        log(DEBUG, format_args!("vbar_deprioritize vbar={model:p}\n"));
        (*context).vbars_dirty = true;
        log_reset_shots();
        let low = (*context).lowest_priority.cast::<ModelVBAR>();
        remove(model);
        insert(model, low, (*low).higher);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_get(context: *mut AimdoContext, model: *mut ModelVBAR) -> u64 {
    aimdo_rs_set_devctx(context);
    log(DEBUG, format_args!("vbar_get vbar={model:p}\n"));
    // SAFETY: The caller supplies a live VBAR.
    unsafe { (*model).vbar }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_fault(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    offset: u64,
    size: u64,
    signature: *mut u32,
) -> i32 {
    aimdo_rs_set_devctx(context);
    log(
        VVERBOSE,
        format_args!(
            "vbar_fault (start): offset={}k, size={}k\n",
            offset / 1024,
            size / 1024
        ),
    );
    // SAFETY: The caller holds the reservation and provides sufficient signature storage.
    unsafe {
        (*context).vbars_dirty = true;
        let result = aimdo_rs_vbar_fault(model, offset, size, signature);
        if result == 0 {
            log(VVERBOSE, format_args!("vbar_fault (return) {result}\n"));
        }
        result
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_unpin(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    offset: u64,
    size: u64,
) -> bool {
    aimdo_rs_set_devctx(context);
    log(
        VVERBOSE,
        format_args!(
            "vbar_unpin (start): offset={}k, size={}k\n",
            offset / 1024,
            size / 1024
        ),
    );
    // SAFETY: The caller holds a live VBAR in the supplied context.
    unsafe {
        (*context).vbars_dirty = true;
        aimdo_rs_vbar_unpin(model, offset, size)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_free(context: *mut AimdoContext, model: *mut ModelVBAR) {
    aimdo_rs_set_devctx(context);
    log(DEBUG, format_args!("vbar_free: vbar={model:p}\n"));
    // SAFETY: The caller relinquishes the reservation and never uses it after this call.
    unsafe {
        (*context).vbars_dirty = true;
        aimdo_vbar_c_sync();
        (*model).pageout_callback = None;
        for page in 0..(*model).nr_pages {
            aimdo_rs_vbar_modify_page(model, page, true, true);
        }
        remove(model);
        check_cuda(
            cuda!(mem_address_free, (*model).vbar, (*model).nr_pages * PAGE),
            "cuMemAddressFree(mv->vbar, (size_t)mv->nr_pages * VBAR_PAGE_SIZE)",
        );
        aimdo_vbar_c_sync();
        libc::free(model.cast());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_loaded_size(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
) -> usize {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller supplies a live VBAR.
    unsafe { (*model).resident_count * PAGE }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_get_nr_pages(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
) -> usize {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller supplies a live VBAR.
    unsafe { (*model).nr_pages }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_get_watermark(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
) -> usize {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller supplies a live VBAR.
    unsafe { (*model).watermark }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_get_residency(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    out: *mut u8,
    max_pages: usize,
) {
    aimdo_rs_set_devctx(context);
    // SAFETY: The caller allocated up to max_pages entries for the live VBAR.
    unsafe {
        for page in 0..max_pages.min((*model).nr_pages) {
            let entry = page_at(model, page);
            *out.add(page) =
                u8::from((*entry).handle != 0) | (u8::from((*entry).pin_count != 0) << 1);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vbar_free_memory(
    context: *mut AimdoContext,
    model: *mut ModelVBAR,
    size: u64,
) -> i64 {
    aimdo_rs_set_devctx(context);
    log(
        DEBUG,
        format_args!("vbar_free_memory (start): size={size}k\n"),
    );
    // SAFETY: The caller holds a live VBAR in the supplied context.
    unsafe {
        (*context).vbars_dirty = true;
        aimdo_rs_vbar_free_memory(model, size)
    }
}
