use crate::platform::{log, log_shot, ERROR, MIB, WARNING};
use libc::c_void;
use std::collections::VecDeque;
use std::io::Error;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

const THREADS: usize = 8;
const CHUNK_SIZE: usize = 2 * MIB;
const QUEUE_CAPACITY: usize = 256;

struct Completion {
    result: Mutex<(usize, bool)>,
    done: Condvar,
}

impl Completion {
    fn finish(&self, count: usize, ok: bool) {
        let mut result = self.result.lock().unwrap();
        result.0 -= count;
        result.1 &= ok;
        if result.0 == 0 {
            self.done.notify_one();
        }
    }

    fn wait(&self) -> bool {
        let result = self.result.lock().unwrap();
        self.done
            .wait_while(result, |result| result.0 != 0)
            .unwrap()
            .1
    }
}

struct Task {
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
    mark_cold: bool,
    write: bool,
    completion: Arc<Completion>,
}

// SAFETY: Each task owns a disjoint buffer range until it marks itself done.
// The caller keeps both its buffer and the completion alive until every task finishes.
unsafe impl Send for Task {}

struct Queue {
    tasks: VecDeque<Task>,
    stop: bool,
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue {
    tasks: VecDeque::new(),
    stop: true,
});
static READY: Condvar = Condvar::new();
static SPACE: Condvar = Condvar::new();
static WORKERS: Mutex<Vec<JoinHandle<()>>> = Mutex::new(Vec::new());

fn worker() {
    loop {
        let queue = QUEUE.lock().unwrap();
        let mut queue = READY
            .wait_while(queue, |queue| !queue.stop && queue.tasks.is_empty())
            .unwrap();
        let Some(task) = queue.tasks.pop_front() else {
            return;
        };
        SPACE.notify_one();
        drop(queue);

        // SAFETY: The transfer caller waits for the completion before releasing its buffers.
        let ok = unsafe {
            transfer_at(
                task.file,
                task.offset,
                task.buffer,
                task.size,
                task.mark_cold,
                task.write,
            )
        };
        task.completion.finish(1, ok);
    }
}

fn stop_workers(workers: &mut Vec<JoinHandle<()>>) {
    QUEUE.lock().unwrap().stop = true;
    READY.notify_all();
    SPACE.notify_all();
    for worker in workers.drain(..) {
        let _ = worker.join();
    }
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn xfer_file_init() -> bool {
    let mut workers = WORKERS.lock().unwrap();
    if !workers.is_empty() {
        return false;
    }
    let mut queue = QUEUE.lock().unwrap();
    if workers.try_reserve_exact(THREADS).is_err()
        || queue.tasks.try_reserve_exact(QUEUE_CAPACITY).is_err()
    {
        return false;
    }
    queue.stop = false;
    drop(queue);

    for _ in 0..THREADS {
        match thread::Builder::new().spawn(worker) {
            Ok(worker) => workers.push(worker),
            Err(_) => {
                stop_workers(&mut workers);
                return false;
            }
        }
    }
    true
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn xfer_file_cleanup() {
    stop_workers(&mut WORKERS.lock().unwrap());
}

unsafe fn transfer(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
    mark_cold: bool,
    write: bool,
) -> bool {
    let tasks = size.div_ceil(CHUNK_SIZE);
    let completion = Arc::new(Completion {
        result: Mutex::new((tasks, true)),
        done: Condvar::new(),
    });

    for (index, done) in (0..size).step_by(CHUNK_SIZE).enumerate() {
        let queue = QUEUE.lock().unwrap();
        let mut queue = SPACE
            .wait_while(queue, |queue| {
                !queue.stop && queue.tasks.len() == QUEUE_CAPACITY
            })
            .unwrap();
        if queue.stop {
            completion.finish(tasks - index, false);
            break;
        }
        queue.tasks.push_back(Task {
            file,
            offset: offset.wrapping_add(done as u64),
            buffer: buffer.cast::<u8>().wrapping_add(done).cast(),
            size: (size - done).min(CHUNK_SIZE),
            mark_cold,
            write,
            completion: Arc::clone(&completion),
        });
        READY.notify_one();
    }
    let ok = completion.wait();
    if !ok {
        log(
            ERROR,
            format_args!(
                "xfer_file_transfer: worker failed handle=0x{file:x} offset={offset} size={size}"
            ),
        );
    }
    ok
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn xfer_file_read(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
    mark_cold: bool,
) -> bool {
    unsafe { transfer(file, offset, buffer, size, mark_cold, false) }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn xfer_file_write(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
) -> bool {
    unsafe { transfer(file, offset, buffer, size, false, true) }
}

unsafe fn transfer_at(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
    mark_cold: bool,
    write: bool,
) -> bool {
    let fd = file as libc::c_int;
    let mut done = 0;
    while done < size {
        let position = offset.wrapping_add(done as u64);
        let pointer = buffer.cast::<u8>().wrapping_add(done).cast();
        // SAFETY: The caller provides size bytes of storage and keeps the descriptor open.
        let bytes = unsafe {
            if write {
                libc::pwrite(fd, pointer, size - done, position as libc::off_t)
            } else {
                libc::pread(fd, pointer, size - done, position as libc::off_t)
            }
        };
        if bytes <= 0 {
            let errno = Error::last_os_error().raw_os_error().unwrap_or_default();
            if bytes < 0 && errno == libc::EINTR {
                continue;
            }
            let operation = if write { "pwrite" } else { "pread" };
            log(
                ERROR,
                format_args!(
                    "xfer_file_transfer_at: {operation} failed at {position} (errno={errno})"
                ),
            );
            return false;
        }
        if mark_cold {
            let errno = unsafe {
                libc::posix_fadvise(
                    fd,
                    position as libc::off_t,
                    bytes as libc::off_t,
                    libc::POSIX_FADV_DONTNEED,
                )
            };
            if errno != 0 {
                log_shot(
                    WARNING,
                    format_args!(
                        "xfer_file_transfer_at: posix_fadvise failed at {position} size={bytes} errno={errno}"
                    ),
                );
            }
        }
        done += bytes as usize;
    }
    true
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn xfer_file_read_at(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
    mark_cold: bool,
) -> bool {
    unsafe { transfer_at(file, offset, buffer, size, mark_cold, false) }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn xfer_file_write_at(
    file: u64,
    offset: u64,
    buffer: *mut c_void,
    size: usize,
) -> bool {
    unsafe { transfer_at(file, offset, buffer, size, false, true) }
}
