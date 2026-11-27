use crate::context::{
    aimdo_rs_get_current_devctx, aimdo_rs_set_devctx_for_device, AimdoContext,
    HostbufFileReaderSlot,
};
use crate::cuda::CuStream;
use crate::platform::{check_cuda, check_cuda_at, cuda, log, DEBUG, ERROR, MIB};
use crate::xfer_file::{xfer_file_read, xfer_file_write};
use std::ffi::c_void;
use std::ptr::{addr_of_mut, null_mut};

const WINDOW: u64 = (64 * MIB) as u64;
const LEAD_IN: u64 = WINDOW - (16 * MIB) as u64;
const SLOTS: i32 = 3;
const CU_EVENT_DISABLE_TIMING: u32 = 2;

unsafe fn slot_at(context: *mut AimdoContext, index: i32) -> *mut HostbufFileReaderSlot {
    unsafe {
        addr_of_mut!((*context).hostbuf_file_reader_slots)
            .cast::<HostbufFileReaderSlot>()
            .add(index as usize)
    }
}

unsafe fn retire_active(context: *mut AimdoContext) -> bool {
    // SAFETY: The context owns the active reader slot and its CUDA events.
    unsafe {
        let active = (*context).hostbuf_file_reader_active;
        if active < 0 {
            return true;
        }
        let slot = slot_at(context, active);
        if (*slot).offset == 0 {
            return true;
        }
        if !(*slot).event.is_null() {
            log(
                ERROR,
                format_args!(
                    "hostbuf_file_reader_retire_active: active slot {active} already has a completion event"
                ),
            );
            return false;
        }
        check_cuda(
            cuda!(
                event_create,
                addr_of_mut!((*slot).event),
                CU_EVENT_DISABLE_TIMING
            ),
            "cuEventCreate(&slot->event, CU_EVENT_DISABLE_TIMING)",
        ) && check_cuda(
            cuda!(event_record, (*slot).event, (*slot).stream),
            "cuEventRecord(slot->event, (CUstream)slot->stream)",
        )
    }
}

unsafe fn next_slot(context: *mut AimdoContext, stream: CuStream) -> *mut HostbufFileReaderSlot {
    // SAFETY: The caller has retired the previous slot and owns the current device context.
    unsafe {
        let active = ((*context).hostbuf_file_reader_active + 1) % SLOTS;
        (*context).hostbuf_file_reader_active = active;
        let slot = slot_at(context, active);
        // 回到旧槽位时，先等该槽位上一次的异步拷贝完成，再覆盖锁页缓冲区。
        if !(*slot).buffer.is_null() && !(*slot).event.is_null() {
            if !check_cuda(
                cuda!(event_synchronize, (*slot).event),
                "cuEventSynchronize(slot->event)",
            ) || !check_cuda(
                cuda!(event_destroy, (*slot).event),
                "cuEventDestroy(slot->event)",
            ) {
                return null_mut();
            }
            (*slot).event = null_mut();
        }
        if (*slot).buffer.is_null()
            && !check_cuda_at(
                cuda!(
                    mem_alloc_host,
                    addr_of_mut!((*slot).buffer).cast(),
                    WINDOW as usize
                ),
                "cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW)",
                ERROR,
                DEBUG,
            )
        {
            return null_mut();
        }
        (*slot).offset = 0;
        (*slot).stream = stream;
        slot
    }
}

struct Transfer {
    device: i32,
    file: u64,
    file_offset: u64,
    size: u64,
    stream: CuStream,
    device_ptr: u64,
    mark_cold: bool,
    write: bool,
}

unsafe fn file_transfer(mut transfer: Transfer) -> bool {
    let Transfer {
        device,
        file,
        stream,
        mark_cold,
        write,
        ..
    } = transfer;
    if transfer.size == 0 {
        return true;
    }
    if transfer.device_ptr == 0 || device < 0 || !unsafe { aimdo_rs_set_devctx_for_device(device) }
    {
        log(
            ERROR,
            format_args!(
                "hostbuf_file_transfer: input validation failed device_ptr={:p} device={device}",
                transfer.device_ptr as *const c_void
            ),
        );
        return false;
    }
    let context = aimdo_rs_get_current_devctx();
    // SAFETY: The caller owns this device/stream and keeps the file and device buffer
    // alive while this transfer and its queued CUDA work execute.
    unsafe {
        while transfer.size != 0 {
            let active = (*context).hostbuf_file_reader_active;
            let mut slot = if active < 0 {
                null_mut()
            } else {
                slot_at(context, active)
            };
            if slot.is_null()
                || (*slot).stream != stream
                || ((*slot).offset.wrapping_add(transfer.size) >= WINDOW
                    && (*slot).offset >= LEAD_IN)
            {
                if !retire_active(context) {
                    return false;
                }
                slot = next_slot(context, stream);
                if slot.is_null() {
                    return false;
                }
            }

            let chunk = transfer.size.min(WINDOW - (*slot).offset);
            let buffer = (*slot).buffer.add((*slot).offset as usize).cast::<c_void>();
            (*slot).offset += chunk;
            let count = chunk as usize;
            let offset = transfer.file_offset;
            let address = transfer.device_ptr;
            if !write && !xfer_file_read(file, offset, buffer, count, mark_cold) {
                log(
                    ERROR,
                    format_args!(
                        "hostbuf_file_transfer: xfer_file_read failed handle=0x{file:x} offset={offset} size={chunk}"
                    ),
                );
                return false;
            }

            let (operation, result) = if write {
                (
                    "cuMemcpyDtoHAsync",
                    cuda!(memcpy_dtoh_async, buffer, address, count, stream),
                )
            } else {
                (
                    "cuMemcpyHtoDAsync",
                    cuda!(memcpy_htod_async, address, buffer, count, stream),
                )
            };
            if !check_cuda(result, "copy_result") {
                log(
                    ERROR,
                    format_args!(
                        "hostbuf_file_transfer: {operation} failed result={result} device_ptr={:p} device={device} stream={stream:p} size={chunk}",
                        address as *const c_void
                    ),
                );
                return false;
            }

            if write {
                let result = cuda!(ctx_synchronize);
                if !check_cuda(result, "sync_result") {
                    log(
                        ERROR,
                        format_args!(
                            "hostbuf_file_transfer: cuCtxSynchronize failed result={result} device_ptr={:p} device={device} stream={stream:p} size={chunk}",
                            address as *const c_void
                        ),
                    );
                    return false;
                }
                if !xfer_file_write(file, offset, buffer, count) {
                    log(
                        ERROR,
                        format_args!(
                            "hostbuf_file_transfer: xfer_file_write failed handle=0x{file:x} offset={offset} size={chunk}"
                        ),
                    );
                    return false;
                }
            }
            transfer.file_offset = offset.wrapping_add(chunk);
            transfer.device_ptr = address.wrapping_add(chunk);
            transfer.size -= chunk;
        }
    }
    true
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_file_reader_read(
    device: i32,
    file: u64,
    file_offset: u64,
    size: u64,
    stream: CuStream,
    device_ptr: u64,
    mark_cold: bool,
) -> bool {
    unsafe {
        file_transfer(Transfer {
            device,
            file,
            file_offset,
            size,
            stream,
            device_ptr,
            mark_cold,
            write: false,
        })
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_file_writer_write(
    device: i32,
    file: u64,
    file_offset: u64,
    size: u64,
    stream: CuStream,
    device_ptr: u64,
) -> bool {
    unsafe {
        file_transfer(Transfer {
            device,
            file,
            file_offset,
            size,
            stream,
            device_ptr,
            mark_cold: false,
            write: true,
        })
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_file_reader_cleanup() {
    let context = aimdo_rs_get_current_devctx();
    if context.is_null() {
        return;
    }
    // SAFETY: Lifecycle cleanup runs before the driver and device contexts are released.
    unsafe {
        retire_active(context);
        for index in 0..SLOTS {
            let slot = slot_at(context, index);
            if !(*slot).buffer.is_null() {
                if !(*slot).event.is_null() {
                    check_cuda(
                        cuda!(event_synchronize, (*slot).event),
                        "cuEventSynchronize(slot->event)",
                    );
                    check_cuda(
                        cuda!(event_destroy, (*slot).event),
                        "cuEventDestroy(slot->event)",
                    );
                }
                check_cuda(
                    cuda!(mem_free_host, (*slot).buffer.cast()),
                    "cuMemFreeHost(slot->buffer)",
                );
            }
            slot.write(HostbufFileReaderSlot {
                buffer: null_mut(),
                offset: 0,
                stream: null_mut(),
                event: null_mut(),
            });
        }
        (*context).hostbuf_file_reader_active = -1;
    }
}
