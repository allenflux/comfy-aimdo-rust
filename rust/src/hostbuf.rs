mod file_reader;
mod workers;

pub(crate) use file_reader::{
    hostbuf_file_reader_cleanup, hostbuf_file_reader_read, hostbuf_file_writer_write,
};

use crate::context::aimdo_rs_set_devctx_for_device;
use crate::cuda::CuStream;
use crate::platform::{check_cuda, cuda, log, ERROR, MIB, VERBOSE};
use crate::xfer_file::xfer_file_read;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::{drop_in_place, null_mut};
use std::sync::{Arc, OnceLock};
use workers::{
    decommit_async, hostbuf_prewarm_join, hostbuf_prewarm_start, Decommit, DECOMMIT_LIMIT,
};

const STREAM_WINDOW: u64 = (64 * MIB) as u64;
static PAGE_SIZE: OnceLock<usize> = OnceLock::new();

#[unsafe(no_mangle)]
pub(crate) extern "C" fn hostbuf_page_size() -> usize {
    *PAGE_SIZE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn hostbuf_reserve_granularity() -> usize {
    hostbuf_page_size()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_reserve_address_space(size: usize) -> *mut c_void {
    // 先保留地址区间，按扩容请求提交物理页，供 Torch 使用稳定的起始地址。
    let address = unsafe {
        libc::mmap(
            null_mut(),
            size,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    if address == libc::MAP_FAILED {
        log(
            ERROR,
            format_args!("hostbuf_reserve_address_space: mmap reserve failed for {size} bytes"),
        );
        return null_mut();
    }
    address
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_commit_address_space(
    address: *mut c_void,
    size: usize,
) -> bool {
    let committed = unsafe {
        libc::mmap(
            address,
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED,
            -1,
            0,
        )
    };
    committed == address
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_decommit_address_space(
    address: *mut c_void,
    size: usize,
) -> bool {
    let reserved = unsafe {
        libc::mmap(
            address,
            size,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    reserved == address
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_release_address_space(address: *mut c_void, size: usize) {
    if !address.is_null() {
        unsafe { libc::munmap(address, size) };
    }
}

#[repr(C)]
struct HostBuffer {
    address: *mut c_void,
    size: u64,
    committed: u64,
    reserved: u64,
    last_chunk: u64,
    prewarm: u64,
    decommit: Arc<Decommit>,
    mark_cold: bool,
}

fn align_up(size: u64, alignment: usize) -> u64 {
    let mask = alignment as u64 - 1;
    size.wrapping_add(mask) & !mask
}

impl HostBuffer {
    fn at(&self, offset: u64) -> *mut c_void {
        self.address
            .cast::<u8>()
            .wrapping_add(offset as usize)
            .cast()
    }

    unsafe fn grow(&mut self, size: u64, register: bool) -> bool {
        if size <= self.size {
            return true;
        }
        let mut target = align_up(size.wrapping_add(self.prewarm), hostbuf_page_size());
        if target < self.committed.wrapping_add(self.prewarm / 4) {
            target = self.committed;
        }

        if self.address.is_null() {
            self.address = unsafe { hostbuf_reserve_address_space(self.reserved as usize) };
            if self.address.is_null() {
                return false;
            }
            log(
                VERBOSE,
                format_args!(
                    "hostbuf_grow: reserved base={:p} reserved_size={}",
                    self.address, self.reserved
                ),
            );
        }
        if target > self.reserved {
            log(
                ERROR,
                format_args!(
                    "hostbuf_grow: requested {target} bytes beyond reserved host buffer {}",
                    self.reserved
                ),
            );
            return false;
        }

        let tail = target.wrapping_sub(self.committed) as usize;
        let prewarm_start = self.committed.max(size);
        let commit_address = self.at(self.committed);
        hostbuf_prewarm_join();
        if tail != 0 {
            log(
                VERBOSE,
                format_args!(
                    "hostbuf_grow: commit base={commit_address:p} offset={} size={tail} target_committed={target}",
                    self.committed
                ),
            );
            if !unsafe { hostbuf_commit_address_space(commit_address, tail) } {
                return false;
            }
        }

        // SAFETY: The new range is committed and the caller selected the CUDA device.
        let grown = unsafe { self.prepare_growth(size, register, prewarm_start, target) };
        if !grown {
            if tail != 0 {
                unsafe { hostbuf_decommit_address_space(commit_address, tail) };
            }
            return false;
        }
        self.committed = target;
        self.size = size;
        log(
            VERBOSE,
            format_args!(
                "hostbuf_grow: result hostbuf={:p} size={} committed={target}",
                self, size
            ),
        );
        true
    }

    unsafe fn prepare_growth(
        &self,
        size: u64,
        register: bool,
        prewarm_start: u64,
        target: u64,
    ) -> bool {
        unsafe {
            if size > self.committed {
                if !hostbuf_prewarm_start(self.at(self.committed), (size - self.committed) as usize)
                {
                    return false;
                }
                hostbuf_prewarm_join();
            }
            if register
                && !check_cuda(
                    cuda!(mem_host_register, self.at(self.size), (size - self.size) as usize, 0),
                    "cuMemHostRegister((char *)hostbuf->base_address + hostbuf->size, (size_t)(size - hostbuf->size), 0)",
                )
            {
                return false;
            }
            if target > prewarm_start
                && !hostbuf_prewarm_start(self.at(prewarm_start), (target - prewarm_start) as usize)
            {
                if register {
                    check_cuda(
                        cuda!(mem_host_unregister, self.at(self.size)),
                        "cuMemHostUnregister((char *)hostbuf->base_address + hostbuf->size)",
                    );
                }
                return false;
            }
            true
        }
    }

    unsafe fn truncate(&mut self, size: u64, unregister: bool) -> bool {
        let target = align_up(size, hostbuf_page_size());
        log(
            VERBOSE,
            format_args!(
                "hostbuf_truncate_impl: hostbuf={:p} base={:p} truncate_to={size} old_size={} old_committed={} new_committed={target}",
                self, self.address, self.size, self.committed
            ),
        );
        if size >= self.size {
            return false;
        }

        hostbuf_prewarm_join();
        if unregister
            && !unsafe {
                check_cuda(
                    cuda!(mem_host_unregister, self.at(size)),
                    "cuMemHostUnregister((char *)hostbuf->base_address + size)",
                )
            }
        {
            return false;
        }

        let mut done = target;
        while done < self.committed {
            let chunk = (self.committed - done).min(DECOMMIT_LIMIT as u64);
            if !unsafe { decommit_async(&self.decommit, self.at(done), chunk as usize, false) } {
                return false;
            }
            done += chunk;
        }
        self.size = size;
        self.committed = self.committed.min(target);
        if size == 0 && !self.address.is_null() {
            log(
                VERBOSE,
                format_args!(
                    "hostbuf_truncate_impl: release base={:p} reserved_size={}",
                    self.address, self.reserved
                ),
            );
            if !unsafe {
                decommit_async(&self.decommit, self.address, self.reserved as usize, true)
            } {
                return false;
            }
            self.address = null_mut();
            self.committed = 0;
        }
        log(
            VERBOSE,
            format_args!(
                "hostbuf_truncate_impl: result hostbuf={:p} size={} committed={}",
                self, self.size, self.committed
            ),
        );
        true
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn hostbuf_allocate(
    prewarm: u64,
    reserved: u64,
    mark_cold: bool,
) -> *mut c_void {
    let pointer = unsafe { libc::malloc(size_of::<HostBuffer>()).cast::<HostBuffer>() };
    if pointer.is_null() {
        return null_mut();
    }
    let reserved = align_up(
        reserved.wrapping_add(prewarm),
        hostbuf_reserve_granularity(),
    );
    unsafe {
        pointer.write(HostBuffer {
            address: null_mut(),
            size: 0,
            committed: 0,
            reserved,
            last_chunk: 0,
            prewarm,
            decommit: Arc::new(Decommit::default()),
            mark_cold,
        });
    }
    log(
        VERBOSE,
        format_args!(
            "hostbuf_allocate: hostbuf={pointer:p} prewarm={prewarm} reserved_size={reserved} mark_cold={}",
            u8::from(mark_cold)
        ),
    );
    pointer.cast()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_free(pointer: *mut c_void) {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_mut() }) else {
        return;
    };
    log(
        VERBOSE,
        format_args!(
            "hostbuf_free: hostbuf={pointer:p} base={:p} size={} committed={} reserved={}",
            hostbuf.address, hostbuf.size, hostbuf.committed, hostbuf.reserved
        ),
    );
    unsafe {
        hostbuf.truncate(0, true);
        hostbuf.decommit.wait();
        drop_in_place(pointer.cast::<HostBuffer>());
        libc::free(pointer);
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_get_raw_address(pointer: *mut c_void) -> *mut c_void {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_ref() }) else {
        return null_mut();
    };
    log(
        VERBOSE,
        format_args!(
            "hostbuf_get_raw_address: hostbuf={pointer:p} ptr={:p} size={} committed={}",
            hostbuf.address, hostbuf.size, hostbuf.committed
        ),
    );
    hostbuf.address
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_extend(
    pointer: *mut c_void,
    size: u64,
    reallocate: bool,
    register: bool,
    size_delta: *mut i64,
) -> *mut c_void {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_mut() }) else {
        return null_mut();
    };
    hostbuf.decommit.wait();
    let old_size = hostbuf.size;
    unsafe { *size_delta = 0 };
    if reallocate && hostbuf.last_chunk != 0 {
        let offset = hostbuf.size.wrapping_sub(hostbuf.last_chunk);
        if register
            && !unsafe {
                check_cuda(
                    cuda!(mem_host_unregister, hostbuf.at(offset)),
                    "cuMemHostUnregister((char *)hostbuf->base_address + offset)",
                )
            }
        {
            return null_mut();
        }
        hostbuf.size = offset;
    }

    let offset = hostbuf.size;
    let grown = unsafe { hostbuf.grow(offset.wrapping_add(size), register) };
    let delta = hostbuf.size.wrapping_sub(old_size) as i64;
    unsafe { *size_delta = delta };
    if !grown {
        return null_mut();
    }
    hostbuf.last_chunk = hostbuf.size - offset;
    let address = hostbuf.at(offset);
    log(
        VERBOSE,
        format_args!(
            "hostbuf_extend: hostbuf={pointer:p} request_size={size} reallocate={} do_register={} offset={offset} ptr={address:p} size_delta={delta} result_size={} committed={}",
            u8::from(reallocate), u8::from(register), hostbuf.size, hostbuf.committed
        ),
    );
    address
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_read_file_slice(
    pointer: *mut c_void,
    device: i32,
    file: u64,
    file_offset: u64,
    size: u64,
    offset: u64,
    stream: CuStream,
    device_ptr: u64,
) -> bool {
    if size == 0 {
        return true;
    }
    let hostbuf = unsafe { pointer.cast::<HostBuffer>().as_ref() };
    let base = hostbuf.map_or(null_mut(), |hostbuf| hostbuf.address);
    if base.is_null() {
        log(
            ERROR,
            format_args!(
                "hostbuf_read_file_slice: input validation failed hostbuf={pointer:p} base={base:p}"
            ),
        );
        return false;
    }
    let hostbuf = hostbuf.unwrap();
    let mut window = STREAM_WINDOW;
    if device_ptr == 0 {
        window = size;
    } else if device < 0 || !unsafe { aimdo_rs_set_devctx_for_device(device) } {
        log(
            ERROR,
            format_args!(
                "hostbuf_read_file_slice: device validation failed device_ptr={:p} device={device}",
                device_ptr as *const c_void
            ),
        );
        return false;
    }

    let mut done = 0;
    while done < size {
        let chunk = window.min(size - done);
        let from = file_offset.wrapping_add(done);
        let to = offset.wrapping_add(done);
        let destination = hostbuf.at(to);
        if !unsafe { xfer_file_read(file, from, destination, chunk as usize, hostbuf.mark_cold) } {
            log(
                ERROR,
                format_args!(
                    "hostbuf_read_file_slice: file read failed handle=0x{file:x} file_offset={from} size={chunk} host_offset={to}"
                ),
            );
            return false;
        }
        if device_ptr != 0 {
            let address = device_ptr.wrapping_add(done);
            let result = unsafe {
                cuda!(
                    memcpy_htod_async,
                    address,
                    destination,
                    chunk as usize,
                    stream
                )
            };
            if !unsafe { check_cuda(result, "copy_result") } {
                log(
                    ERROR,
                    format_args!(
                        "hostbuf_read_file_slice: device copy failed result={result} device_ptr={:p} device={device} stream={stream:p} size={chunk}",
                        address as *const c_void
                    ),
                );
                return false;
            }
        }
        done += chunk;
    }
    true
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_register(
    pointer: *mut c_void,
    offset: u64,
    size: u64,
) -> bool {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_ref() }) else {
        return false;
    };
    if offset.wrapping_add(size) > hostbuf.size {
        return false;
    }
    log(
        VERBOSE,
        format_args!("hostbuf_register: hostbuf={pointer:p} offset={offset} size={size}"),
    );
    size == 0
        || unsafe {
            check_cuda(
                cuda!(mem_host_register, hostbuf.at(offset), size as usize, 0),
                "cuMemHostRegister((char *)hostbuf->base_address + offset, (size_t)size, 0)",
            )
        }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_unregister(pointer: *mut c_void, offset: u64) -> bool {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_ref() }) else {
        return false;
    };
    if offset >= hostbuf.size {
        return false;
    }
    log(
        VERBOSE,
        format_args!("hostbuf_unregister: hostbuf={pointer:p} offset={offset}"),
    );
    unsafe {
        check_cuda(
            cuda!(mem_host_unregister, hostbuf.at(offset)),
            "cuMemHostUnregister((char *)hostbuf->base_address + offset)",
        )
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_truncate(
    pointer: *mut c_void,
    size: u64,
    unregister: bool,
) -> bool {
    let Some(hostbuf) = (unsafe { pointer.cast::<HostBuffer>().as_mut() }) else {
        return false;
    };
    log(
        VERBOSE,
        format_args!(
            "hostbuf_truncate: hostbuf={pointer:p} size={size} do_unregister={}",
            u8::from(unregister)
        ),
    );
    unsafe { hostbuf.truncate(size, unregister) }
}
