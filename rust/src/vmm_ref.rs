//! CUDA virtual reservations and reference-counted physical mappings.

use core::ffi::c_void;
use core::mem::size_of;
use core::ptr::{addr_of_mut, null_mut};

use crate::context::aimdo_rs_get_current_devctx;
use crate::cuda::{
    CuDevicePtr, CuMemAllocationFlags, CuMemAllocationProp, CuMemLocation, CuResult,
};
use crate::platform::cuda;

pub(crate) const CUDA_SUCCESS: CuResult = 0;
pub(crate) const CUDA_ERROR_OUT_OF_MEMORY: CuResult = 2;
pub(crate) const CUDA_ERROR_UNKNOWN: CuResult = 999;

const PINNED: u32 = 1;
pub(crate) const DEVICE_LOCATION: u32 = 1;

pub(crate) struct VirtualRange {
    pub(crate) address: CuDevicePtr,
    bytes: usize,
    refs: usize,
}

pub(crate) struct PhysicalAllocation {
    pub(crate) handle: u64,
    pub(crate) bytes: usize,
    refs: usize,
    references: *mut PhysicalPage,
}

pub(crate) struct PhysicalPage {
    pub(crate) allocation: *mut PhysicalAllocation,
    pub(crate) address: CuDevicePtr,
    next: *mut PhysicalPage,
}

unsafe extern "C" {
    pub(crate) fn allocations_lock();
    pub(crate) fn allocations_unlock();
    fn calloc(count: usize, size: usize) -> *mut c_void;
    fn malloc(size: usize) -> *mut c_void;
    fn free(pointer: *mut c_void);
}

pub(crate) fn allocate<T>() -> *mut T {
    // SAFETY: malloc provides raw storage with alignment for the graph metadata types.
    unsafe { malloc(size_of::<T>()).cast() }
}

pub(crate) fn allocate_zeroed<T>(count: usize) -> *mut T {
    // SAFETY: Callers only read zeroed storage using types that accept zero bits.
    unsafe { calloc(count, size_of::<T>()).cast() }
}

pub(crate) unsafe fn deallocate<T>(pointer: *mut T) {
    // SAFETY: Callers pass graph storage allocated by libc or null.
    unsafe { free(pointer.cast()) }
}

pub(crate) unsafe fn virtual_range_alloc(bytes: usize, alignment: usize) -> *mut VirtualRange {
    let range = allocate::<VirtualRange>();
    if range.is_null() {
        return range;
    }
    // SAFETY: The caller selected the current CUDA context. Publish only a successful reservation.
    unsafe {
        let mut address = 0;
        if cuda!(mem_address_reserve, &mut address, bytes, alignment, 0, 0) != CUDA_SUCCESS {
            deallocate(range);
            return null_mut();
        }
        range.write(VirtualRange {
            address,
            bytes,
            refs: 1,
        });
    }
    range
}

pub(crate) unsafe fn virtual_range_ref(range: *mut VirtualRange) -> *mut VirtualRange {
    // SAFETY: The caller owns a reference for the duration of the increment.
    unsafe {
        allocations_lock();
        (*range).refs += 1;
        allocations_unlock();
    }
    range
}

pub(crate) unsafe fn virtual_range_unref(range: *mut VirtualRange) -> CuResult {
    // SAFETY: The caller transfers one reference to a live reservation on this device.
    unsafe {
        allocations_lock();
        (*range).refs -= 1;
        if (*range).refs != 0 {
            allocations_unlock();
            return CUDA_SUCCESS;
        }
        let result = cuda!(mem_address_free, (*range).address, (*range).bytes);
        if result == CUDA_SUCCESS {
            deallocate(range);
        } else {
            (*range).refs += 1;
        }
        allocations_unlock();
        result
    }
}

pub(crate) unsafe fn physical_page_alloc(
    page: *mut *mut PhysicalPage,
    bytes: usize,
    device: i32,
) -> CuResult {
    let reference = allocate::<PhysicalPage>();
    let allocation = allocate::<PhysicalAllocation>();
    // SAFETY: The caller provides an output pointer and selected CUDA device. Graph creation
    // does not publish an allocation before cuMemCreate succeeds and metadata is initialized.
    unsafe {
        if reference.is_null() || allocation.is_null() {
            deallocate(reference);
            deallocate(allocation);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        let properties = CuMemAllocationProp {
            kind: PINNED,
            requested_handle_types: 0,
            location: CuMemLocation {
                kind: DEVICE_LOCATION,
                id: device,
            },
            win32_handle_metadata: null_mut(),
            alloc_flags: CuMemAllocationFlags {
                compression_type: 0,
                gpu_direct_rdma_capable: 0,
                usage: 0,
                reserved: [0; 4],
            },
        };
        let mut handle = 0;
        let result = cuda!(mem_create, &mut handle, bytes, &properties, 0);
        if result != CUDA_SUCCESS {
            deallocate(reference);
            deallocate(allocation);
            return result;
        }
        allocation.write(PhysicalAllocation {
            handle,
            bytes,
            refs: 1,
            references: reference,
        });
        reference.write(PhysicalPage {
            allocation,
            address: 0,
            next: null_mut(),
        });
        *page = reference;
        allocations_lock();
        (*aimdo_rs_get_current_devctx()).total_vram_usage += bytes as u64;
        allocations_unlock();
    }
    CUDA_SUCCESS
}

pub(crate) unsafe fn physical_page_ref(page: *mut PhysicalPage, address: u64) -> *mut PhysicalPage {
    let reference = allocate::<PhysicalPage>();
    if reference.is_null() {
        return reference;
    }
    // SAFETY: The caller holds a live page reference through this operation.
    unsafe {
        allocations_lock();
        let allocation = (*page).allocation;
        (*allocation).refs += 1;
        reference.write(PhysicalPage {
            allocation,
            address,
            next: (*allocation).references,
        });
        (*allocation).references = reference;
        allocations_unlock();
    }
    reference
}

pub(crate) unsafe fn physical_page_unref(page: *mut PhysicalPage) -> CuResult {
    if page.is_null() {
        return CUDA_SUCCESS;
    }
    // SAFETY: The caller transfers one live page ref. Other refs to its allocation are
    // protected by the device allocation lock. Preserve the ref if a driver call fails.
    unsafe {
        allocations_lock();
        let allocation = (*page).allocation;
        let mut entry = addr_of_mut!((*allocation).references);
        while *entry != page {
            entry = addr_of_mut!((**entry).next);
        }

        let address = (*page).address;
        if address != 0 {
            let mut reference = (*allocation).references;
            while !reference.is_null() && (reference == page || (*reference).address != address) {
                reference = (*reference).next;
            }
            if reference.is_null() {
                let result = cuda!(mem_unmap, address, (*allocation).bytes);
                if result != CUDA_SUCCESS {
                    allocations_unlock();
                    return result;
                }
            }
            (*page).address = 0;
        }
        if (*allocation).refs == 1 {
            let result = cuda!(mem_release, (*allocation).handle);
            if result != CUDA_SUCCESS {
                allocations_unlock();
                return result;
            }
        }
        *entry = (*page).next;
        (*allocation).refs -= 1;
        if (*allocation).refs == 0 {
            (*aimdo_rs_get_current_devctx()).total_vram_usage -= (*allocation).bytes as u64;
            deallocate(allocation);
        }
        deallocate(page);
        allocations_unlock();
    }
    CUDA_SUCCESS
}
