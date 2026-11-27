use libc::{c_void, pthread_cond_t, pthread_mutex_t, pthread_t};
use std::mem::size_of;
use std::ptr::{null, null_mut};

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn mutex_create() -> *mut pthread_mutex_t {
    unsafe {
        let mutex = libc::malloc(size_of::<pthread_mutex_t>()).cast::<pthread_mutex_t>();
        if !mutex.is_null() && libc::pthread_mutex_init(mutex, null()) == 0 {
            return mutex;
        }
        libc::free(mutex.cast());
        null_mut()
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn mutex_lock(mutex: *mut pthread_mutex_t) {
    unsafe { libc::pthread_mutex_lock(mutex) };
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn mutex_unlock(mutex: *mut pthread_mutex_t) {
    unsafe { libc::pthread_mutex_unlock(mutex) };
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn mutex_destroy(mutex: *mut pthread_mutex_t) {
    if !mutex.is_null() {
        unsafe {
            libc::pthread_mutex_destroy(mutex);
            libc::free(mutex.cast());
        }
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn condvar_create() -> *mut pthread_cond_t {
    unsafe {
        let cond = libc::malloc(size_of::<pthread_cond_t>()).cast::<pthread_cond_t>();
        if !cond.is_null() && libc::pthread_cond_init(cond, null()) == 0 {
            return cond;
        }
        libc::free(cond.cast());
        null_mut()
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn condvar_wait(
    cond: *mut pthread_cond_t,
    mutex: *mut pthread_mutex_t,
) {
    unsafe { libc::pthread_cond_wait(cond, mutex) };
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn condvar_signal(cond: *mut pthread_cond_t) {
    unsafe { libc::pthread_cond_signal(cond) };
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn condvar_broadcast(cond: *mut pthread_cond_t) {
    unsafe { libc::pthread_cond_broadcast(cond) };
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn condvar_destroy(cond: *mut pthread_cond_t) {
    if !cond.is_null() {
        unsafe {
            libc::pthread_cond_destroy(cond);
            libc::free(cond.cast());
        }
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn thread_create(
    thread: *mut pthread_t,
    proc: extern "C" fn(*mut c_void) -> *mut c_void,
    arg: *mut c_void,
) -> bool {
    unsafe { libc::pthread_create(thread, null(), proc, arg) == 0 }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn thread_join(thread: pthread_t) {
    unsafe { libc::pthread_join(thread, null_mut()) };
}
