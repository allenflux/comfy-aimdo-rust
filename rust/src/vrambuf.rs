use core::mem::size_of;
use core::ptr::{addr_of_mut, null_mut};

use crate::context::{aimdo_rs_get_current_devctx, aimdo_rs_set_devctx_for_device};
use crate::platform::{budget_deficit, check_cuda, cuda, log, map_pages, DEBUG, ERROR, INFO, MIB};
use crate::vbar::vbars_free;
use crate::{CUDA_ERROR_OUT_OF_MEMORY, CUDA_SUCCESS};

pub(crate) const CUDA_PAGE: usize = 2 * MIB;
const CHUNK: usize = 16 * MIB;

#[repr(C)]
pub struct VramBuffer {
    pub(crate) base_ptr: u64,
    max_size: usize,
    pub(crate) allocated: usize,
    handle_count: usize,
    pub(crate) device: i32,
    pub(crate) next: *mut VramBuffer,
    handles: [u64; 0],
}

pub(crate) fn align_up(size: usize, alignment: usize) -> usize {
    size.wrapping_add(alignment - 1) & !(alignment - 1)
}

unsafe fn handles(buffer: *mut VramBuffer) -> *mut u64 {
    // SAFETY: The live buffer allocation contains handle storage after this header.
    unsafe { addr_of_mut!((*buffer).handles).cast() }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrambuf_create(device: i32, max_size: usize) -> *mut VramBuffer {
    // SAFETY: A registered device owns the CUDA address reservation and buffer.
    unsafe {
        if !aimdo_rs_set_devctx_for_device(device) {
            return null_mut();
        }
        let max_size = align_up(max_size, CUDA_PAGE);
        let bytes = size_of::<VramBuffer>() + size_of::<u64>() * (max_size / CHUNK + 1);
        let buffer = libc::calloc(1, bytes).cast::<VramBuffer>();
        if buffer.is_null() {
            return null_mut();
        }
        (*buffer).device = device;
        (*buffer).max_size = max_size;
        if !check_cuda(
            cuda!(
                mem_address_reserve,
                addr_of_mut!((*buffer).base_ptr),
                max_size,
                0,
                0,
                0
            ),
            "cuMemAddressReserve(&buf->base_ptr, max_size, 0, 0, 0)",
        ) {
            log(
                ERROR,
                format_args!("vrambuf_create: {device} {}k\n", max_size / 1024),
            );
            libc::free(buffer.cast());
            return null_mut();
        }
        buffer
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrambuf_grow(buffer: *mut VramBuffer, required_size: usize) -> bool {
    // SAFETY: Non-null buffers were created by vrambuf_create and remain exclusively owned.
    unsafe {
        if buffer.is_null()
            || !aimdo_rs_set_devctx_for_device((*buffer).device)
            || required_size > (*buffer).max_size
        {
            return false;
        }
        if required_size <= (*buffer).allocated {
            return true;
        }
        let grow_to = align_up(required_size, CHUNK).min((*buffer).max_size);
        if !vbars_free(budget_deficit(grow_to - (*buffer).allocated)) {
            return false;
        }
        while (*buffer).allocated < grow_to {
            let bytes = CHUNK.min(grow_to - (*buffer).allocated);
            let address = (*buffer).base_ptr + (*buffer).allocated as u64;
            let device = (*buffer).device;
            let mut handle = 0;
            let mut error = map_pages(address, bytes, device, &mut handle);
            if error != CUDA_SUCCESS {
                if error != CUDA_ERROR_OUT_OF_MEMORY {
                    log(ERROR, format_args!("VRAM Allocation failed (non OOM)\n"));
                    return false;
                }
                log(
                    DEBUG,
                    format_args!("Pytorch allocator attempt exceeds available VRAM ...\n"),
                );
                if !vbars_free(CHUNK as isize) {
                    return false;
                }
                error = map_pages(address, bytes, device, &mut handle);
                if error != CUDA_SUCCESS {
                    if error == CUDA_ERROR_OUT_OF_MEMORY {
                        log(INFO, format_args!("VRAM Allocation failed (OOM)\n"));
                    } else {
                        log(ERROR, format_args!("VRAM Allocation failed (error)\n"));
                    }
                    return false;
                }
            }
            *handles(buffer).add((*buffer).handle_count) = handle;
            (*buffer).handle_count += 1;
            (*buffer).allocated += bytes;
        }
        true
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrambuf_get(buffer: *mut VramBuffer) -> u64 {
    if buffer.is_null() {
        return 0;
    }
    // SAFETY: Non-null buffers are live objects returned by vrambuf_create.
    unsafe { (*buffer).base_ptr }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn vrambuf_destroy(buffer: *mut VramBuffer) -> bool {
    // SAFETY: The caller relinquishes this live buffer and all views into its mappings.
    unsafe {
        if buffer.is_null() || !aimdo_rs_set_devctx_for_device((*buffer).device) {
            return false;
        }
        if (*buffer).allocated > 0 {
            check_cuda(
                cuda!(mem_unmap, (*buffer).base_ptr, (*buffer).allocated),
                "cuMemUnmap(buf->base_ptr, buf->allocated)",
            );
        }
        for i in 0..(*buffer).handle_count {
            check_cuda(
                cuda!(mem_release, *handles(buffer).add(i)),
                "cuMemRelease(buf->handles[i])",
            );
        }
        let context = aimdo_rs_get_current_devctx();
        (*context).total_vram_usage = (*context)
            .total_vram_usage
            .wrapping_sub((*buffer).allocated as u64);
        check_cuda(
            cuda!(mem_address_free, (*buffer).base_ptr, (*buffer).max_size),
            "cuMemAddressFree(buf->base_ptr, buf->max_size)",
        );
        libc::free(buffer.cast());
        true
    }
}
