use super::{hostbuf_decommit_address_space, hostbuf_page_size, hostbuf_release_address_space};
use crate::platform::MIB;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

const PREWARM_THREADS: usize = 6;
pub(super) const DECOMMIT_LIMIT: usize = 384 * MIB;

#[derive(Clone, Copy)]
struct Address(*mut u8);

// SAFETY: The workers receive mappings owned by HostBuffer, which waits for each
// operation before either reusing the associated range or dropping the mapping.
unsafe impl Send for Address {}

struct Prewarm {
    address: Address,
    page_size: usize,
    page_count: usize,
    page_span: usize,
    generation: u64,
    remaining: usize,
    stop: bool,
}

static PREWARM: Mutex<Prewarm> = Mutex::new(Prewarm {
    address: Address(null_mut()),
    page_size: 0,
    page_count: 0,
    page_span: 0,
    generation: 0,
    remaining: 0,
    stop: false,
});
static PREWARM_STARTED: Mutex<bool> = Mutex::new(false);
static WARM_WORK: Condvar = Condvar::new();
static WARM_DONE: Condvar = Condvar::new();

fn prewarm_worker(index: usize) {
    let mut generation = 0;
    loop {
        let state = PREWARM.lock().unwrap();
        let state = WARM_WORK
            .wait_while(state, |state| !state.stop && generation == state.generation)
            .unwrap();
        if state.stop {
            return;
        }
        generation = state.generation;
        let address = state.address;
        let page_size = state.page_size;
        let first = index * state.page_span;
        let end = (first + state.page_span).min(state.page_count);
        drop(state);

        for page in first..end {
            // SAFETY: The parent committed these pages, and workers get separate spans.
            unsafe { address.0.add(page * page_size).write_volatile(0) };
        }

        let mut state = PREWARM.lock().unwrap();
        state.remaining -= 1;
        if state.remaining == 0 {
            WARM_DONE.notify_all();
        }
    }
}

fn ensure_prewarm_workers() -> bool {
    let mut started = PREWARM_STARTED.lock().unwrap();
    if *started {
        return true;
    }
    let mut workers = [const { None }; PREWARM_THREADS];
    for (index, worker) in workers.iter_mut().enumerate() {
        match thread::Builder::new().spawn(move || prewarm_worker(index)) {
            Ok(handle) => *worker = Some(handle),
            Err(_) => {
                PREWARM.lock().unwrap().stop = true;
                WARM_WORK.notify_all();
                for handle in workers.into_iter().flatten() {
                    let _ = handle.join();
                }
                PREWARM.lock().unwrap().stop = false;
                return false;
            }
        }
    }
    *started = true;
    true
}

#[unsafe(no_mangle)]
pub(crate) extern "C" fn hostbuf_prewarm_join() -> bool {
    let state = PREWARM.lock().unwrap();
    drop(
        WARM_DONE
            .wait_while(state, |state| state.remaining != 0)
            .unwrap(),
    );
    true
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn hostbuf_prewarm_start(address: *mut c_void, size: usize) -> bool {
    if size == 0 {
        return hostbuf_prewarm_join();
    }
    if !ensure_prewarm_workers() {
        return false;
    }

    let state = PREWARM.lock().unwrap();
    let mut state = WARM_DONE
        .wait_while(state, |state| state.remaining != 0)
        .unwrap();
    state.address = Address(address.cast());
    state.page_size = hostbuf_page_size();
    state.page_count = size.div_ceil(state.page_size);
    state.page_span = state.page_count.div_ceil(PREWARM_THREADS);
    state.remaining = PREWARM_THREADS;
    state.generation = state.generation.wrapping_add(1);
    WARM_WORK.notify_all();
    true
}

#[derive(Default)]
pub(super) struct Decommit {
    pending: Mutex<usize>,
    done: Condvar,
}

impl Decommit {
    pub(super) fn wait(&self) {
        let pending = self.pending.lock().unwrap();
        drop(
            self.done
                .wait_while(pending, |pending| *pending != 0)
                .unwrap(),
        );
    }
}

struct Job {
    address: Address,
    size: usize,
    release: bool,
    owner: Arc<Decommit>,
}

struct Queue {
    jobs: VecDeque<Job>,
    bytes: usize,
    started: bool,
}

static DECOMMIT: Mutex<Queue> = Mutex::new(Queue {
    jobs: VecDeque::new(),
    bytes: 0,
    started: false,
});
static DECOMMIT_READY: Condvar = Condvar::new();
static DECOMMIT_SPACE: Condvar = Condvar::new();

fn decommit_worker() {
    let mut queue = DECOMMIT.lock().unwrap();
    loop {
        queue = DECOMMIT_READY
            .wait_while(queue, |queue| queue.jobs.is_empty())
            .unwrap();
        let job = queue.jobs.pop_front().unwrap();
        drop(queue);

        // SAFETY: The job owns the relevant page range of the reservation until completion.
        unsafe {
            if job.release {
                hostbuf_release_address_space(job.address.0.cast(), job.size);
            } else {
                hostbuf_decommit_address_space(job.address.0.cast(), job.size);
            }
        }

        queue = DECOMMIT.lock().unwrap();
        if !job.release {
            queue.bytes -= job.size;
        }
        DECOMMIT_SPACE.notify_all();
        *job.owner.pending.lock().unwrap() -= 1;
        job.owner.done.notify_all();
    }
}

pub(super) unsafe fn decommit_async(
    owner: &Arc<Decommit>,
    address: *mut c_void,
    size: usize,
    release: bool,
) -> bool {
    let mut queue = DECOMMIT.lock().unwrap();
    if !queue.started {
        if thread::Builder::new().spawn(decommit_worker).is_err() {
            return false;
        }
        queue.started = true;
    }
    queue = DECOMMIT_SPACE
        .wait_while(queue, |queue| {
            !release && queue.bytes + size > DECOMMIT_LIMIT
        })
        .unwrap();
    if queue.jobs.try_reserve(1).is_err() {
        return false;
    }
    *owner.pending.lock().unwrap() += 1;
    queue.jobs.push_back(Job {
        address: Address(address.cast()),
        size,
        release,
        owner: Arc::clone(owner),
    });
    if !release {
        queue.bytes += size;
    }
    DECOMMIT_READY.notify_one();
    true
}
