//! Keep graph allocations alive when their tensors outlive the recorded scope.

use core::ptr::{addr_of_mut, null_mut};

use crate::context::aimdo_rs_get_current_devctx;
use crate::cuda::CuDevicePtr;
use crate::platform::cuda;
use crate::vmm_ref::{
    allocate, allocate_zeroed, allocations_lock, allocations_unlock, deallocate, physical_page_ref,
    physical_page_unref, virtual_range_ref, virtual_range_unref, PhysicalPage, VirtualRange,
    CUDA_SUCCESS,
};

struct Rogue {
    range: *mut VirtualRange,
    ptr: CuDevicePtr,
    pages: *mut *mut PhysicalPage,
    page_count: usize,
    freeing: bool,
    next: *mut Rogue,
}

struct RogueCandidate {
    ptr: CuDevicePtr,
    freed: bool,
    next: *mut RogueCandidate,
}

#[derive(PartialEq, Eq)]
pub(crate) enum RogueHandoff {
    Error,
    Freed,
    Owned,
}

unsafe fn candidate_head() -> *mut *mut RogueCandidate {
    // SAFETY: The caller selected a live context and holds its allocation lock.
    unsafe { addr_of_mut!((*aimdo_rs_get_current_devctx()).rogue_candidates).cast() }
}

unsafe fn rogue_head() -> *mut *mut Rogue {
    // SAFETY: The caller selected a live context and holds its allocation lock.
    unsafe { addr_of_mut!((*aimdo_rs_get_current_devctx()).rogues).cast() }
}

unsafe fn candidate_entry(ptr: CuDevicePtr) -> *mut *mut RogueCandidate {
    // SAFETY: The device allocation lock protects the list and each link.
    unsafe {
        let mut entry = candidate_head();
        while !(*entry).is_null() && (**entry).ptr != ptr {
            entry = addr_of_mut!((**entry).next);
        }
        entry
    }
}

pub(crate) unsafe fn register_rogue_candidate(ptr: CuDevicePtr) -> bool {
    let candidate = allocate::<RogueCandidate>();
    if candidate.is_null() {
        return false;
    }
    // SAFETY: The graph owns the active device; do not expose uninitialized entries.
    unsafe {
        allocations_lock();
        let head = candidate_head();
        candidate.write(RogueCandidate {
            ptr,
            freed: false,
            next: *head,
        });
        *head = candidate;
        allocations_unlock();
    }
    true
}

pub(crate) unsafe fn unregister_rogue_candidate(ptr: CuDevicePtr) {
    // SAFETY: The graph selected the device whose tracking list contains ptr.
    unsafe {
        allocations_lock();
        let entry = candidate_entry(ptr);
        let candidate = *entry;
        if !candidate.is_null() {
            *entry = (*candidate).next;
            deallocate(candidate);
        }
        allocations_unlock();
    }
}

pub(crate) unsafe fn rogue_candidate_freed(ptr: CuDevicePtr) -> bool {
    // SAFETY: The graph's device is selected and the list is locked during lookup.
    unsafe {
        allocations_lock();
        let candidate = *candidate_entry(ptr);
        let freed = !candidate.is_null() && (*candidate).freed;
        allocations_unlock();
        freed
    }
}

pub(crate) unsafe fn handoff_rogue(
    range: *mut VirtualRange,
    ptr: CuDevicePtr,
    pages: *const *mut PhysicalPage,
    page_count: usize,
) -> RogueHandoff {
    let rogue = allocate::<Rogue>();
    let references = allocate_zeroed::<*mut PhysicalPage>(page_count);
    // SAFETY: Each page is live and shares this reservation. Graphs hand off a nonempty range.
    unsafe {
        if rogue.is_null() || references.is_null() {
            deallocate(rogue);
            deallocate(references);
            return RogueHandoff::Error;
        }

        let page_bytes = (*(**pages).allocation).bytes as u64;
        let address = ptr - (ptr - (*range).address) % page_bytes;
        for index in 0..page_count {
            let reference = physical_page_ref(*pages.add(index), 0);
            if reference.is_null() {
                for previous in 0..index {
                    physical_page_unref(*references.add(previous));
                }
                deallocate(references);
                deallocate(rogue);
                return RogueHandoff::Error;
            }
            references.add(index).write(reference);
        }
        for index in 0..page_count {
            (**references.add(index)).address = address + index as u64 * page_bytes;
        }
        rogue.write(Rogue {
            range: virtual_range_ref(range),
            ptr,
            pages: references,
            page_count,
            freeing: false,
            next: null_mut(),
        });

        allocations_lock();
        let entry = candidate_entry(ptr);
        let candidate = *entry;
        let result = if candidate.is_null() {
            RogueHandoff::Error
        } else {
            *entry = (*candidate).next;
            if (*candidate).freed {
                RogueHandoff::Freed
            } else {
                RogueHandoff::Owned
            }
        };
        if result == RogueHandoff::Owned {
            let head = rogue_head();
            (*rogue).next = *head;
            *head = rogue;
        }
        allocations_unlock();

        deallocate(candidate);
        if result != RogueHandoff::Owned {
            for index in 0..page_count {
                physical_page_unref(*references.add(index));
            }
            virtual_range_unref((*rogue).range);
            deallocate(references);
            deallocate(rogue);
        }
        result
    }
}

pub(crate) unsafe fn rogue_exists(ptr: CuDevicePtr) -> bool {
    // SAFETY: The graph owns the active device and the list is locked while scanning.
    unsafe {
        allocations_lock();
        let mut rogue = *rogue_head();
        while !rogue.is_null() && (*rogue).ptr != ptr {
            rogue = (*rogue).next;
        }
        allocations_unlock();
        !rogue.is_null()
    }
}

/// # Safety
/// The allocation hook selects the device first and supplies writable result storage.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn free_rogue(ptr: CuDevicePtr, result: *mut i32) -> bool {
    // SAFETY: The device's mutex protects publication and the freeing flag. Only the
    // thread that sets freeing drops refs, with the lock released for the CUDA calls.
    unsafe {
        allocations_lock();
        let mut entry = rogue_head();
        while !(*entry).is_null() && (**entry).ptr != ptr {
            entry = addr_of_mut!((**entry).next);
        }
        let rogue = *entry;
        if rogue.is_null() {
            let candidate = *candidate_entry(ptr);
            if candidate.is_null() {
                allocations_unlock();
                return false;
            }
            (*candidate).freed = true;
            *result = CUDA_SUCCESS;
            allocations_unlock();
            return true;
        }
        if (*rogue).freeing {
            *result = CUDA_SUCCESS;
            allocations_unlock();
            return true;
        }
        (*rogue).freeing = true;
        allocations_unlock();

        *result = cuda!(ctx_synchronize);
        for index in 0..(*rogue).page_count {
            let status = physical_page_unref(*(*rogue).pages.add(index));
            if *result == CUDA_SUCCESS {
                *result = status;
            }
        }
        let status = virtual_range_unref((*rogue).range);
        if *result == CUDA_SUCCESS {
            *result = status;
        }

        allocations_lock();
        entry = rogue_head();
        while *entry != rogue {
            entry = addr_of_mut!((**entry).next);
        }
        *entry = (*rogue).next;
        allocations_unlock();
        deallocate((*rogue).pages);
        deallocate(rogue);
        true
    }
}
