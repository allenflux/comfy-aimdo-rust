use crate::platform::{log, DEBUG, ERROR};
use libc::{c_char, c_int, c_void};
use std::ffi::CStr;
use std::io::Error;
use std::mem::{size_of, MaybeUninit};
use std::ptr::null_mut;

#[repr(C)]
struct ModelMmap {
    fd: c_int,
    address: *mut c_void,
    size: u64,
}

fn errno() -> i32 {
    Error::last_os_error().raw_os_error().unwrap_or_default()
}

unsafe fn map_file(file: c_int, name: &CStr) -> Option<(*mut c_void, u64)> {
    let mut stats = MaybeUninit::<libc::stat>::uninit();
    let path = name.to_string_lossy();
    if unsafe { libc::fstat(file, stats.as_mut_ptr()) } != 0 {
        log(
            ERROR,
            format_args!(
                "model_mmap_allocate: fstat failed for {path} (errno={})",
                errno()
            ),
        );
        return None;
    }
    let length = unsafe { stats.assume_init().st_size };
    if length <= 0 {
        log(
            ERROR,
            format_args!("model_mmap_allocate: invalid file size for {path}: {length}"),
        );
        return None;
    }
    let base = unsafe {
        libc::mmap(
            null_mut(),
            length as usize,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file,
            0,
        )
    };
    if base == libc::MAP_FAILED {
        log(
            ERROR,
            format_args!(
                "model_mmap_allocate: mmap failed for {path} (errno={})",
                errno()
            ),
        );
        return None;
    }
    Some((base, length as u64))
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn model_mmap_allocate(path: *const c_char) -> *mut c_void {
    // SAFETY: The caller supplies a NUL-terminated path for the duration of this call.
    let name = unsafe { CStr::from_ptr(path) };
    log(
        DEBUG,
        format_args!(
            "model_mmap_allocate: creating ModelMMAP for {}",
            name.to_string_lossy()
        ),
    );
    let mapping = unsafe { libc::malloc(size_of::<ModelMmap>()).cast::<ModelMmap>() };
    if mapping.is_null() {
        log(
            ERROR,
            format_args!("model_mmap_allocate: allocation failed"),
        );
        return null_mut();
    }
    let fd = unsafe { libc::open(path, libc::O_RDONLY) };
    if fd < 0 {
        log(
            ERROR,
            format_args!(
                "model_mmap_allocate: could not open file: {} (errno={})",
                name.to_string_lossy(),
                errno()
            ),
        );
        unsafe { libc::free(mapping.cast()) };
        return null_mut();
    }
    let Some((address, size)) = (unsafe { map_file(fd, name) }) else {
        unsafe {
            libc::close(fd);
            libc::free(mapping.cast());
        }
        return null_mut();
    };
    unsafe { mapping.write(ModelMmap { fd, address, size }) };
    log(
        DEBUG,
        format_args!("model_mmap_allocate: returning {mapping:p} (base={address:p} size={size})"),
    );
    mapping.cast()
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn model_mmap_get(mapping: *mut c_void) -> *mut c_void {
    unsafe { mapping.cast::<ModelMmap>().as_ref() }.map_or(null_mut(), |mapping| mapping.address)
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn model_mmap_get_file_handle(mapping: *mut c_void) -> u64 {
    unsafe { mapping.cast::<ModelMmap>().as_ref() }.map_or(0, |mapping| mapping.fd as u64)
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn model_mmap_bounce(pointer: *mut c_void) -> bool {
    let mapping = unsafe { pointer.cast::<ModelMmap>().as_ref() };
    let ok = mapping.is_some_and(|mapping| {
        (unsafe { libc::madvise(mapping.address, mapping.size as usize, libc::MADV_DONTNEED) }) == 0
    });
    if !ok {
        log(
            ERROR,
            format_args!(
                "model_mmap_bounce: {pointer:p}: FAILED in model_mmap bounce (errno={})",
                errno()
            ),
        );
        return false;
    }
    log(
        DEBUG,
        format_args!("model_mmap_bounce: {pointer:p}: SUCCESS in model_mmap bounce"),
    );
    true
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn model_mmap_deallocate(pointer: *mut c_void) {
    let Some(mapping) = (unsafe { pointer.cast::<ModelMmap>().as_ref() }) else {
        return;
    };
    if !mapping.address.is_null()
        && unsafe { libc::munmap(mapping.address, mapping.size as usize) } != 0
    {
        log(
            ERROR,
            format_args!(
                "model_mmap_deallocate: munmap failed for {:p} (errno={})",
                mapping.address,
                errno()
            ),
        );
    }
    unsafe {
        libc::close(mapping.fd);
        libc::free(pointer);
    }
}
