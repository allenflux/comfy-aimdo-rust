//! Stream-local CUDA allocation recording and replay. Events keep stable addresses across
//! branches. Raw pointers avoid borrowing graph state across CUDA calls and pageout callbacks.

use core::ffi::{c_char, c_void, CStr};
use core::mem::size_of;
use core::ptr::{addr_of_mut, copy_nonoverlapping, null_mut};
use std::cell::Cell;

use crate::context::{aimdo_rs_set_devctx, AimdoContext};
use crate::cuda::{CuDevicePtr, CuMemAccessDesc, CuMemLocation, CuResult, CuStream};
use crate::platform::{budget_deficit, cuda, MIB};
use crate::rogue::{
    handoff_rogue, register_rogue_candidate, rogue_candidate_freed, rogue_exists,
    unregister_rogue_candidate, RogueHandoff,
};
use crate::vmm_ref::{
    allocate, allocate_zeroed, deallocate, physical_page_alloc, physical_page_ref,
    physical_page_unref, virtual_range_alloc, virtual_range_unref, PhysicalPage, VirtualRange,
    CUDA_ERROR_OUT_OF_MEMORY, CUDA_ERROR_UNKNOWN, CUDA_SUCCESS, DEVICE_LOCATION,
};

const PAGE: usize = 8 * MIB;
const PAGES: usize = 8192;
const SMALL_PAGES: usize = PAGES / 8;
const SMALL_LIMIT: usize = 4 * MIB;
const SMALL_ALIGN: usize = 4096;
const READ_WRITE: u32 = 3;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum EventType {
    Sentinel,
    Alloc,
    Free,
    AllocSmall,
    FreeSmall,
    Call,
    End,
}

struct Event {
    kind: EventType,
    value: usize,
    bytes: usize,
    scope: *mut Event,
    name: *mut c_char,
    next: *mut *mut Event,
    next_count: usize,
    previous: *mut Event,
    allocation_previous: *mut Event,
    rogue_owned: bool,
    snapshot: *mut AllocationState,
    small_snapshot: *mut SmallRange,
}

struct State {
    cursor: *mut Event,
    allocations: *mut Event,
    depth: u32,
    recording: bool,
    broken: bool,
    next: *mut State,
}

#[derive(Clone, Copy)]
struct VirtualPage {
    va_span: usize,
    owner_depth: u32,
    allocation: *mut Event,
}

impl VirtualPage {
    const EMPTY: Self = Self {
        va_span: 0,
        owner_depth: 0,
        allocation: null_mut(),
    };
}

#[derive(Clone, Copy)]
struct SmallRange {
    offset: usize,
    bytes: usize,
    rogue_ptr: CuDevicePtr,
    owner_depth: u32,
    allocation: *mut Event,
    next: *mut SmallRange,
}

struct AllocationState {
    virtual_pages: [VirtualPage; PAGES],
    physical_live: [bool; PAGES],
}

struct MallocGraph {
    base: *mut VirtualRange,
    small_base: *mut VirtualRange,
    stream: CuStream,
    device: i32,
    owner_thread: *const Cell<*mut MallocGraph>,

    root: Event,
    state: *mut State,
    rogue_candidates: *mut Event,

    allocations: AllocationState,
    small_ranges: *mut SmallRange,
    small_unusable: *mut SmallRange,
    black_holes: [CuDevicePtr; PAGES],
    va_phys: [i32; PAGES],
    physical_pages: [*mut PhysicalPage; PAGES],
    mapped_pages: [*mut PhysicalPage; PAGES],
    small_physical_pages: [*mut PhysicalPage; SMALL_PAGES],
    small_mapped_pages: [*mut PhysicalPage; SMALL_PAGES],

    va_count: usize,
    phys_count: usize,
    small_size: usize,
    small_pages: usize,
    used: usize,
    peak_used: usize,
    rogue_count: usize,

    failed: bool,
    complete: bool,
    paused: bool,
    sync_paused: bool,
    assert_breaks: bool,
    handoff_attempted: bool,
    aborted: bool,
}

thread_local! {
    static ACTIVE_GRAPH: Cell<*mut MallocGraph> = const { Cell::new(null_mut()) };
}

unsafe extern "C" {
    fn vbars_free(bytes: isize) -> bool;
    fn realloc(pointer: *mut c_void, bytes: usize) -> *mut c_void;
}

#[inline]
fn align_up(bytes: usize, alignment: usize) -> usize {
    bytes.wrapping_add(alignment - 1) & !(alignment - 1)
}

#[inline]
unsafe fn element<T, const N: usize>(array: *mut [T; N], index: usize) -> *mut T {
    debug_assert!(index < N);
    // SAFETY: Graph callers use only the pages in the corresponding reservation.
    unsafe { array.cast::<T>().add(index) }
}

macro_rules! slot {
    ($array:expr, $index:expr) => {
        *element(addr_of_mut!($array), $index as usize)
    };
}

macro_rules! fail_when {
    ($graph:expr, $condition:expr, $result:expr) => {
        if $condition {
            (*$graph).failed = true;
            return $result;
        }
    };
}

// The helpers below require the graph's owner thread and selected device. The hook path
// may reenter after a CUDA or pageout call, so graph storage is accessed through raw places.
unsafe fn candidate_ptr(graph: *mut MallocGraph, candidate: *mut Event) -> CuDevicePtr {
    unsafe {
        if (*candidate).kind == EventType::AllocSmall {
            (*(*graph).small_base).address + (*candidate).value as u64
        } else {
            (*(*graph).base).address + ((*candidate).value * PAGE) as u64
        }
    }
}

unsafe fn graph_failed(graph: *mut MallocGraph) -> bool {
    unsafe {
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if rogue_candidate_freed(candidate_ptr(graph, candidate)) {
                (*graph).failed = true;
                break;
            }
            candidate = (*candidate).allocation_previous;
        }
        (*graph).failed
    }
}

#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_sync_paused() -> bool {
    let graph = ACTIVE_GRAPH.get();
    // SAFETY: The active thread owns this graph until the TLS slot is cleared.
    unsafe { !graph.is_null() && (*graph).sync_paused }
}

unsafe fn free_small_ranges(mut range: *mut SmallRange) {
    unsafe {
        while !range.is_null() {
            let next = (*range).next;
            deallocate(range);
            range = next;
        }
    }
}

unsafe fn copy_small_ranges(mut copy: *mut *mut SmallRange, mut range: *mut SmallRange) -> bool {
    unsafe {
        while !range.is_null() {
            let entry = allocate::<SmallRange>();
            *copy = entry;
            if entry.is_null() {
                return false;
            }
            entry.write(*range);
            copy = addr_of_mut!((*entry).next);
            range = (*range).next;
        }
        true
    }
}

unsafe fn add_used(graph: *mut MallocGraph, bytes: usize) {
    unsafe {
        (*graph).used += bytes;
        (*graph).peak_used = (*graph).peak_used.max((*graph).used);
    }
}

unsafe fn allocation_used(graph: *mut MallocGraph) -> usize {
    unsafe {
        let mut used = 0;
        for page in 0..(*graph).phys_count {
            if slot!((*graph).allocations.physical_live, page) {
                used += PAGE;
            }
        }
        let mut range = (*graph).small_ranges;
        while !range.is_null() {
            used += (*range).bytes;
            range = (*range).next;
        }
        used
    }
}

unsafe fn append_event(graph: *mut MallocGraph, parent: *mut Event, event: *mut Event) -> bool {
    unsafe {
        let count = (*parent).next_count;
        let next = realloc((*parent).next.cast(), (count + 1) * size_of::<*mut Event>())
            .cast::<*mut Event>();
        fail_when!(graph, next.is_null(), false);
        (*parent).next = next;
        next.add(count).write(event);
        (*parent).next_count += 1;
        (*event).previous = parent;
        true
    }
}

unsafe fn find_event(
    graph: *mut MallocGraph,
    kind: EventType,
    value: usize,
    bytes: usize,
    name: *const c_char,
) -> *mut Event {
    unsafe {
        let cursor = (*(*graph).state).cursor;
        for index in 0..(*cursor).next_count {
            let event = *(*cursor).next.add(index);
            if (*event).kind != kind
                || (kind == EventType::Call
                    && CStr::from_ptr((*event).name) != CStr::from_ptr(name))
                || (kind != EventType::Call && (*event).bytes != bytes)
                || (matches!(kind, EventType::Free | EventType::FreeSmall)
                    && (*event).value != value)
            {
                continue;
            }
            if kind == EventType::Alloc {
                let pages = align_up((*event).bytes, PAGE) / PAGE;
                if ((*event).value..(*event).value + pages).any(|va| {
                    rogue_va(graph, va) || rogue_phys(graph, slot!((*graph).va_phys, va) as usize)
                }) {
                    continue;
                }
            } else if kind == EventType::AllocSmall
                && small_unavailable(graph, (*event).value, align_up((*event).bytes, SMALL_ALIGN))
            {
                continue;
            }
            return event;
        }
        null_mut()
    }
}

unsafe fn event(
    graph: *mut MallocGraph,
    kind: EventType,
    value: usize,
    bytes: usize,
) -> *mut Event {
    unsafe {
        let event = allocate_zeroed::<Event>(1);
        fail_when!(graph, event.is_null(), null_mut());
        (*event).kind = kind;
        (*event).value = value;
        (*event).bytes = bytes;
        let state = (*graph).state;
        if !append_event(graph, (*state).cursor, event) {
            deallocate(event);
            return null_mut();
        }
        (*state).cursor = event;
        event
    }
}

unsafe fn track_allocation(graph: *mut MallocGraph, event: *mut Event) {
    unsafe {
        let state = (*graph).state;
        (*event).allocation_previous = (*state).allocations;
        (*state).allocations = event;
    }
}

unsafe fn untrack_allocation(graph: *mut MallocGraph, event: *mut Event) -> bool {
    unsafe {
        let mut entry = addr_of_mut!((*(*graph).state).allocations);
        while !(*entry).is_null() && *entry != event {
            entry = addr_of_mut!((**entry).allocation_previous);
        }
        if (*entry).is_null() {
            return false;
        }
        *entry = (*event).allocation_previous;
        (*event).allocation_previous = null_mut();
        true
    }
}

unsafe fn sever_event(graph: *mut MallocGraph, event: *mut Event) -> bool {
    unsafe {
        let parent = (*event).previous;
        let count = (*parent).next_count + (*event).next_count - 1;
        let next = if count != 0 {
            allocate_zeroed::<*mut Event>(count)
        } else {
            null_mut()
        };
        fail_when!(graph, count != 0 && next.is_null(), false);

        let mut slot = 0;
        for index in 0..(*parent).next_count {
            let sibling = *(*parent).next.add(index);
            if sibling == event {
                for child_index in 0..(*event).next_count {
                    let child = *(*event).next.add(child_index);
                    next.add(slot).write(child);
                    (*child).previous = parent;
                    slot += 1;
                }
            } else {
                next.add(slot).write(sibling);
                slot += 1;
            }
        }

        let state = (*graph).state;
        if (*state).cursor == event {
            (*state).cursor = parent;
        }
        deallocate((*parent).next);
        deallocate((*event).next);
        (*parent).next = next;
        (*parent).next_count = count;
        (*event).next = null_mut();
        (*event).next_count = 0;
        (*event).previous = null_mut();
        true
    }
}

unsafe fn collect_rogue_candidates(graph: *mut MallocGraph) -> bool {
    unsafe {
        let state = (*graph).state;
        let mut allocation = (*state).allocations;
        while !allocation.is_null() {
            let previous = (*allocation).allocation_previous;
            let ptr = candidate_ptr(graph, allocation);
            fail_when!(graph, !register_rogue_candidate(ptr), false);
            if !sever_event(graph, allocation) {
                unregister_rogue_candidate(ptr);
                return false;
            }
            (*allocation).allocation_previous = (*graph).rogue_candidates;
            (*graph).rogue_candidates = allocation;
            (*graph).rogue_count += 1;
            allocation = previous;
        }
        (*state).allocations = null_mut();
        true
    }
}

unsafe fn is_rogue_candidate(graph: *mut MallocGraph, event: *mut Event) -> bool {
    unsafe {
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if candidate == event {
                return true;
            }
            candidate = (*candidate).allocation_previous;
        }
        false
    }
}

unsafe fn rogue_va(graph: *mut MallocGraph, va: usize) -> bool {
    unsafe {
        let ptr = slot!((*graph).black_holes, va);
        if ptr != 0 {
            if rogue_exists(ptr) {
                return true;
            }
            let phys = slot!((*graph).va_phys, va);
            let address = (*(*graph).base).address + (va * PAGE) as u64;
            fail_when!(
                graph,
                map_reference(
                    graph,
                    address,
                    slot!((*graph).physical_pages, phys),
                    addr_of_mut!(slot!((*graph).mapped_pages, va))
                ) != CUDA_SUCCESS,
                true
            );
            slot!((*graph).black_holes, va) = 0;
        }
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if (*candidate).kind == EventType::Alloc {
                let pages = align_up((*candidate).bytes, PAGE) / PAGE;
                if ((*candidate).value..(*candidate).value + pages).contains(&va) {
                    return true;
                }
            }
            candidate = (*candidate).allocation_previous;
        }
        false
    }
}

unsafe fn rogue_phys(graph: *mut MallocGraph, phys: usize) -> bool {
    unsafe {
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if (*candidate).kind == EventType::Alloc {
                let start = (*candidate).value;
                let pages = align_up((*candidate).bytes, PAGE) / PAGE;
                if (start..start + pages).any(|va| slot!((*graph).va_phys, va) == phys as i32) {
                    return true;
                }
            }
            candidate = (*candidate).allocation_previous;
        }
        false
    }
}

unsafe fn prune_small_unusable(graph: *mut MallocGraph) {
    unsafe {
        let mut entry = addr_of_mut!((*graph).small_unusable);
        while !(*entry).is_null() {
            let range = *entry;
            if rogue_exists((*range).rogue_ptr) {
                entry = addr_of_mut!((*range).next);
            } else {
                *entry = (*range).next;
                deallocate(range);
            }
        }
    }
}

unsafe fn small_unavailable(graph: *mut MallocGraph, offset: usize, bytes: usize) -> bool {
    unsafe {
        let end = offset + bytes;
        prune_small_unusable(graph);
        let mut range = (*graph).small_unusable;
        while !range.is_null() {
            if offset < (*range).offset + (*range).bytes && (*range).offset < end {
                return true;
            }
            range = (*range).next;
        }
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if !rogue_candidate_freed(candidate_ptr(graph, candidate))
                && (*candidate).kind == EventType::AllocSmall
            {
                let first = (*candidate).value / PAGE * PAGE;
                let candidate_end = align_up((*candidate).value + (*candidate).bytes, PAGE);
                if offset < candidate_end && first < end {
                    return true;
                }
            }
            candidate = (*candidate).allocation_previous;
        }
        false
    }
}

unsafe fn push_stack(graph: *mut MallocGraph, scope: *mut Event, recording: bool) -> bool {
    unsafe {
        if recording && (*scope).snapshot.is_null() {
            fail_when!(
                graph,
                !copy_small_ranges(addr_of_mut!((*scope).small_snapshot), (*graph).small_ranges),
                false
            );
            let snapshot = allocate::<AllocationState>();
            fail_when!(graph, snapshot.is_null(), false);
            copy_nonoverlapping(addr_of_mut!((*graph).allocations), snapshot, 1);
            (*scope).snapshot = snapshot;
        }
        let state = allocate::<State>();
        fail_when!(graph, state.is_null(), false);
        let previous = (*graph).state;
        state.write(State {
            cursor: scope,
            allocations: null_mut(),
            depth: if previous.is_null() {
                1
            } else {
                (*previous).depth + 1
            },
            recording,
            broken: false,
            next: previous,
        });
        (*graph).state = state;
        true
    }
}

unsafe fn create_page(graph: *mut MallocGraph, page: *mut *mut PhysicalPage) -> CuResult {
    unsafe {
        if !vbars_free(budget_deficit(PAGE)) {
            return CUDA_ERROR_UNKNOWN;
        }
        let result = physical_page_alloc(page, PAGE, (*graph).device);
        if result != CUDA_ERROR_OUT_OF_MEMORY {
            return result;
        }
        if !vbars_free(PAGE as isize) {
            return CUDA_ERROR_UNKNOWN;
        }
        physical_page_alloc(page, PAGE, (*graph).device)
    }
}

unsafe fn map_reference(
    graph: *mut MallocGraph,
    address: CuDevicePtr,
    page: *mut PhysicalPage,
    mapping: *mut *mut PhysicalPage,
) -> CuResult {
    unsafe {
        let access = CuMemAccessDesc {
            location: CuMemLocation {
                kind: DEVICE_LOCATION,
                id: (*graph).device,
            },
            flags: READ_WRITE,
        };
        let reference = physical_page_ref(page, 0);
        if reference.is_null() {
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        let mut result = cuda!(mem_map, address, PAGE, 0, (*(*page).allocation).handle, 0);
        if result == CUDA_SUCCESS {
            (*reference).address = address;
            result = cuda!(mem_set_access, address, PAGE, &access, 1);
        }
        if result == CUDA_SUCCESS {
            *mapping = reference;
        } else {
            physical_page_unref(reference);
        }
        result
    }
}

unsafe fn map_page(graph: *mut MallocGraph, va: usize, phys: usize) -> CuResult {
    unsafe {
        let address = (*(*graph).base).address + (va * PAGE) as u64;
        let page = addr_of_mut!(slot!((*graph).physical_pages, phys));
        if phys == (*graph).phys_count {
            let result = create_page(graph, page);
            if result != CUDA_SUCCESS {
                return result;
            }
            (*graph).phys_count += 1;
        }
        let result = map_reference(
            graph,
            address,
            *page,
            addr_of_mut!(slot!((*graph).mapped_pages, va)),
        );
        if result == CUDA_SUCCESS {
            slot!((*graph).va_phys, va) = phys as i32;
        }
        result
    }
}

unsafe fn dry_apply(graph: *mut MallocGraph, event: *mut Event) -> bool {
    unsafe {
        let kind = (*event).kind;
        let value = (*event).value;
        if matches!(kind, EventType::Alloc | EventType::Free) {
            let first = addr_of_mut!(slot!((*graph).allocations.virtual_pages, value));
            if kind == EventType::Alloc {
                *first = VirtualPage {
                    va_span: align_up((*event).bytes, PAGE) / PAGE,
                    owner_depth: (*(*graph).state).depth,
                    allocation: event,
                };
                track_allocation(graph, event);
                add_used(graph, (*first).va_span * PAGE);
            }
            for index in 0..(*first).va_span {
                let phys = slot!((*graph).va_phys, value + index);
                slot!((*graph).allocations.physical_live, phys) = kind == EventType::Alloc;
            }
            if kind == EventType::Free {
                fail_when!(
                    graph,
                    !untrack_allocation(graph, (*first).allocation),
                    false
                );
                (*graph).used -= (*first).va_span * PAGE;
                *first = VirtualPage::EMPTY;
            }
        } else if matches!(kind, EventType::AllocSmall | EventType::FreeSmall) {
            let mut entry = addr_of_mut!((*graph).small_ranges);
            while !(*entry).is_null() && (**entry).offset < value {
                entry = addr_of_mut!((**entry).next);
            }
            if kind == EventType::AllocSmall {
                let range = allocate::<SmallRange>();
                fail_when!(graph, range.is_null(), false);
                range.write(SmallRange {
                    offset: value,
                    bytes: align_up((*event).bytes, SMALL_ALIGN),
                    rogue_ptr: 0,
                    owner_depth: (*(*graph).state).depth,
                    allocation: event,
                    next: *entry,
                });
                *entry = range;
                track_allocation(graph, event);
                add_used(graph, (*range).bytes);
            } else {
                let range = *entry;
                fail_when!(
                    graph,
                    !untrack_allocation(graph, (*range).allocation),
                    false
                );
                *entry = (*range).next;
                (*graph).used -= (*range).bytes;
                deallocate(range);
            }
        }
        true
    }
}

unsafe fn materialize(graph: *mut MallocGraph, event: *mut Event) -> bool {
    unsafe {
        if !(*event).snapshot.is_null() {
            let mut ranges = null_mut();
            if !copy_small_ranges(&mut ranges, (*event).small_snapshot) {
                (*graph).failed = true;
                free_small_ranges(ranges);
                return false;
            }
            free_small_ranges((*graph).small_ranges);
            copy_nonoverlapping((*event).snapshot, addr_of_mut!((*graph).allocations), 1);
            (*graph).small_ranges = ranges;
            (*(*graph).state).allocations = null_mut();
            (*graph).used = allocation_used(graph);
            (*graph).peak_used = (*graph).peak_used.max((*graph).used);
            return true;
        }
        materialize(graph, (*event).previous) && dry_apply(graph, event)
    }
}

unsafe fn scrub_rogue_candidates(
    graph: *mut MallocGraph,
    allocations: *mut AllocationState,
    small_ranges: *mut *mut SmallRange,
) {
    unsafe {
        if !allocations.is_null() {
            for va in 0..(*graph).va_count {
                let first = addr_of_mut!(slot!((*allocations).virtual_pages, va));
                if !(*first).allocation.is_null() && is_rogue_candidate(graph, (*first).allocation)
                {
                    for offset in 0..(*first).va_span {
                        let phys = slot!((*graph).va_phys, va + offset);
                        slot!((*allocations).physical_live, phys) = false;
                    }
                    *first = VirtualPage::EMPTY;
                }
            }
        }
        let mut entry = small_ranges;
        while !(*entry).is_null() {
            let range = *entry;
            if !(*range).allocation.is_null() && is_rogue_candidate(graph, (*range).allocation) {
                *entry = (*range).next;
                deallocate(range);
            } else {
                entry = addr_of_mut!((*range).next);
            }
        }
    }
}

unsafe fn scrub_rogue_snapshots(graph: *mut MallocGraph, event: *mut Event) {
    unsafe {
        scrub_rogue_candidates(
            graph,
            (*event).snapshot,
            addr_of_mut!((*event).small_snapshot),
        );
        for index in 0..(*event).next_count {
            let next = *(*event).next.add(index);
            if (*next).kind == EventType::Call {
                scrub_rogue_snapshots(graph, (*next).scope);
            }
            scrub_rogue_snapshots(graph, next);
        }
    }
}

unsafe fn handoff_rogues(graph: *mut MallocGraph) -> bool {
    unsafe {
        (*graph).handoff_attempted = true;
        fail_when!(graph, cuda!(ctx_synchronize) != CUDA_SUCCESS, false);

        let mut pages: *mut *mut PhysicalPage = null_mut();
        let mut freed = false;
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if pages.is_null() {
                pages = allocate_zeroed::<*mut PhysicalPage>(PAGES);
                fail_when!(graph, pages.is_null(), false);
            }

            let small = (*candidate).kind == EventType::AllocSmall;
            let ptr = candidate_ptr(graph, candidate);
            let range;
            let count;
            if small {
                range = (*graph).small_base;
                let first = (*candidate).value / PAGE;
                let last =
                    ((*candidate).value + align_up((*candidate).bytes, SMALL_ALIGN) - 1) / PAGE;
                count = last - first + 1;
                for index in 0..count {
                    pages
                        .add(index)
                        .write(slot!((*graph).small_physical_pages, first + index));
                }
            } else {
                range = (*graph).base;
                count = align_up((*candidate).bytes, PAGE) / PAGE;
                for index in 0..count {
                    let phys = slot!((*graph).va_phys, (*candidate).value + index);
                    pages.add(index).write(slot!((*graph).physical_pages, phys));
                }
            }
            match handoff_rogue(range, ptr, pages, count) {
                RogueHandoff::Error => {
                    deallocate(pages);
                    (*graph).failed = true;
                    return false;
                }
                RogueHandoff::Freed => freed = true,
                RogueHandoff::Owned => {
                    (*candidate).rogue_owned = true;
                    if !small {
                        for index in 0..count {
                            slot!((*graph).black_holes, (*candidate).value + index) = ptr;
                        }
                    }
                }
            }
            candidate = (*candidate).allocation_previous;
        }
        deallocate(pages);

        fail_when!(
            graph,
            freed && cuda!(ctx_synchronize) != CUDA_SUCCESS,
            false
        );
        true
    }
}

unsafe fn free_rogue_candidates(graph: *mut MallocGraph) {
    unsafe {
        while !(*graph).rogue_candidates.is_null() {
            let candidate = (*graph).rogue_candidates;
            (*graph).rogue_candidates = (*candidate).allocation_previous;
            unregister_rogue_candidate(candidate_ptr(graph, candidate));
            deallocate(candidate);
        }
    }
}

unsafe fn abort_graph(graph: *mut MallocGraph) -> bool {
    unsafe {
        if (*graph).aborted {
            return true;
        }
        if (*graph).handoff_attempted {
            return false;
        }
        (*graph).handoff_attempted = true;

        let top = (*graph).state;
        let mut state = top;
        while !state.is_null() {
            (*graph).state = state;
            if !materialize(graph, (*state).cursor) || !collect_rogue_candidates(graph) {
                (*graph).state = top;
                ACTIVE_GRAPH.set(null_mut());
                return false;
            }
            state = (*state).next;
        }
        (*graph).state = top;
        ACTIVE_GRAPH.set(null_mut());

        if !handoff_rogues(graph) {
            return false;
        }
        free_rogue_candidates(graph);
        (*graph).aborted = true;
        true
    }
}

unsafe fn prepare_replacements(
    graph: *mut MallocGraph,
    replacements: *mut *mut PhysicalPage,
    unusable: *mut *mut SmallRange,
) -> bool {
    unsafe {
        let mut candidate = (*graph).rogue_candidates;
        while !candidate.is_null() {
            if (*candidate).rogue_owned {
                let value = (*candidate).value;
                let bytes = (*candidate).bytes;
                if (*candidate).kind == EventType::AllocSmall {
                    let range = allocate::<SmallRange>();
                    if range.is_null() {
                        return false;
                    }
                    let offset = value / PAGE * PAGE;
                    range.write(SmallRange {
                        offset,
                        bytes: align_up(value + bytes, PAGE) - offset,
                        rogue_ptr: candidate_ptr(graph, candidate),
                        owner_depth: 0,
                        allocation: null_mut(),
                        next: *unusable,
                    });
                    *unusable = range;
                } else {
                    let count = align_up(bytes, PAGE) / PAGE;
                    for index in 0..count {
                        let phys = slot!((*graph).va_phys, value + index) as usize;
                        let replacement = replacements.add(phys);
                        if (*replacement).is_null()
                            && create_page(graph, replacement) != CUDA_SUCCESS
                        {
                            return false;
                        }
                    }
                }
            }
            candidate = (*candidate).allocation_previous;
        }
        true
    }
}

unsafe fn finalize_rogues(graph: *mut MallocGraph) -> bool {
    unsafe {
        if (*graph).rogue_candidates.is_null() {
            return true;
        }
        if !handoff_rogues(graph) {
            return false;
        }

        let replacements = allocate_zeroed::<*mut PhysicalPage>((*graph).phys_count);
        let mut unusable = null_mut();
        if ((*graph).phys_count != 0 && replacements.is_null())
            || !prepare_replacements(graph, replacements, &mut unusable)
        {
            if !replacements.is_null() {
                for phys in 0..(*graph).phys_count {
                    physical_page_unref(*replacements.add(phys));
                }
            }
            deallocate(replacements);
            free_small_ranges(unusable);
            (*graph).failed = true;
            return false;
        }

        let mut remap_failed = false;
        for va in 0..(*graph).va_count {
            let phys = slot!((*graph).va_phys, va);
            if phys < 0 || (*replacements.add(phys as usize)).is_null() {
                continue;
            }
            let mapping = addr_of_mut!(slot!((*graph).mapped_pages, va));
            if physical_page_unref(*mapping) != CUDA_SUCCESS {
                remap_failed = true;
                continue;
            }
            *mapping = null_mut();
            if slot!((*graph).black_holes, va) == 0
                && map_reference(
                    graph,
                    (*(*graph).base).address + (va * PAGE) as u64,
                    *replacements.add(phys as usize),
                    mapping,
                ) != CUDA_SUCCESS
            {
                remap_failed = true;
            }
        }
        for phys in 0..(*graph).phys_count {
            let replacement = *replacements.add(phys);
            if !replacement.is_null() {
                let previous = slot!((*graph).physical_pages, phys);
                slot!((*graph).physical_pages, phys) = replacement;
                physical_page_unref(previous);
            }
        }
        deallocate(replacements);
        if remap_failed {
            free_small_ranges(unusable);
            (*graph).failed = true;
            return false;
        }

        while !unusable.is_null() {
            let next = (*unusable).next;
            (*unusable).next = (*graph).small_unusable;
            (*graph).small_unusable = unusable;
            unusable = next;
        }
        scrub_rogue_candidates(
            graph,
            addr_of_mut!((*graph).allocations),
            addr_of_mut!((*graph).small_ranges),
        );
        scrub_rogue_snapshots(graph, addr_of_mut!((*graph).root));
        (*graph).used = allocation_used(graph);
        free_rogue_candidates(graph);
        (*graph).handoff_attempted = false;
        true
    }
}

unsafe fn start_recording(graph: *mut MallocGraph) -> bool {
    unsafe {
        fail_when!(graph, (*graph).assert_breaks, false);
        let state = (*graph).state;
        if !materialize(graph, (*state).cursor) {
            return false;
        }
        (*state).recording = true;
        (*state).broken = true;
        true
    }
}

fn consider_small_interval(
    position: usize,
    offset: usize,
    bytes: usize,
    next_offset: &mut usize,
    next_end: &mut usize,
) {
    let end = offset + bytes;
    if end <= position {
        return;
    }
    if offset <= position {
        *next_offset = position;
        *next_end = (*next_end).max(end);
    } else if offset < *next_offset {
        *next_offset = offset;
        *next_end = end;
    } else if offset == *next_offset {
        *next_end = (*next_end).max(end);
    }
}

unsafe fn small_allocation_offset(graph: *mut MallocGraph, bytes: usize) -> usize {
    unsafe {
        let mut position = 0;
        let mut offset = (*graph).small_size;
        let mut smallest_hole = usize::MAX;
        prune_small_unusable(graph);

        while position < (*graph).small_size {
            let mut next_offset = usize::MAX;
            let mut next_end = 0;
            for mut range in [(*graph).small_ranges, (*graph).small_unusable] {
                while !range.is_null() {
                    consider_small_interval(
                        position,
                        (*range).offset,
                        (*range).bytes,
                        &mut next_offset,
                        &mut next_end,
                    );
                    range = (*range).next;
                }
            }
            let mut candidate = (*graph).rogue_candidates;
            while !candidate.is_null() {
                if (*candidate).kind == EventType::AllocSmall {
                    let first = (*candidate).value / PAGE * PAGE;
                    let bytes = align_up((*candidate).value + (*candidate).bytes, PAGE) - first;
                    consider_small_interval(
                        position,
                        first,
                        bytes,
                        &mut next_offset,
                        &mut next_end,
                    );
                }
                candidate = (*candidate).allocation_previous;
            }

            if next_offset == usize::MAX {
                next_offset = (*graph).small_size;
            }
            let hole = next_offset - position;
            if hole >= bytes && hole < smallest_hole {
                offset = position;
                smallest_hole = hole;
            }
            if next_offset == (*graph).small_size {
                break;
            }
            position = next_end;
        }
        if smallest_hole == usize::MAX {
            offset = offset.max(position);
        }
        offset
    }
}

unsafe fn record_small_allocation(
    graph: *mut MallocGraph,
    ptr: *mut CuDevicePtr,
    size: usize,
    result: *mut CuResult,
) -> bool {
    unsafe {
        let bytes = align_up(size, SMALL_ALIGN);
        let offset = small_allocation_offset(graph, bytes);
        let mut insert_at = addr_of_mut!((*graph).small_ranges);
        while !(*insert_at).is_null() && (**insert_at).offset < offset {
            insert_at = addr_of_mut!((**insert_at).next);
        }

        let end = offset + bytes;
        if end > (*graph).small_size {
            let pages = align_up(end, PAGE) / PAGE;
            fail_when!(graph, pages > SMALL_PAGES, true);
            if (*graph).small_pages < pages {
                let index = (*graph).small_pages;
                let page = addr_of_mut!(slot!((*graph).small_physical_pages, index));
                let address = (*(*graph).small_base).address + (index * PAGE) as u64;
                let status = create_page(graph, page);
                if status != CUDA_SUCCESS {
                    *result = status;
                    (*graph).failed = true;
                    return true;
                }
                (*graph).small_pages += 1;
                fail_when!(
                    graph,
                    map_reference(
                        graph,
                        address,
                        *page,
                        addr_of_mut!(slot!((*graph).small_mapped_pages, index))
                    ) != CUDA_SUCCESS,
                    true
                );
            }
            (*graph).small_size = end;
        }

        let range = allocate::<SmallRange>();
        fail_when!(graph, range.is_null(), true);
        range.write(SmallRange {
            offset,
            bytes,
            owner_depth: (*(*graph).state).depth,
            rogue_ptr: 0,
            allocation: null_mut(),
            next: *insert_at,
        });
        *insert_at = range;

        let allocation = event(graph, EventType::AllocSmall, offset, size);
        fail_when!(graph, allocation.is_null(), true);
        (*range).allocation = allocation;
        track_allocation(graph, allocation);
        add_used(graph, bytes);
        *ptr = (*(*graph).small_base).address + offset as u64;
        *result = CUDA_SUCCESS;
        true
    }
}

unsafe fn record_large_allocation(
    graph: *mut MallocGraph,
    ptr: *mut CuDevicePtr,
    size: usize,
    result: *mut CuResult,
) -> bool {
    unsafe {
        let pages = align_up(size, PAGE) / PAGE;
        let mut va = 0;
        while va + pages <= (*graph).va_count {
            let mut blocked = None;
            for index in 0..pages {
                let phys = slot!((*graph).va_phys, va + index);
                if rogue_va(graph, va + index)
                    || slot!((*graph).allocations.physical_live, phys)
                    || rogue_phys(graph, phys as usize)
                {
                    blocked = Some((index, phys));
                    break;
                }
                slot!((*graph).allocations.physical_live, phys) = true;
            }
            let Some((blocked_index, blocked_phys)) = blocked else {
                break;
            };
            let mut advance = blocked_index + 1;
            for previous in 0..blocked_index {
                let phys = slot!((*graph).va_phys, va + previous);
                // Resume after the earlier alias, including overlapping ranges.
                if phys == blocked_phys {
                    advance = previous + 1;
                }
                slot!((*graph).allocations.physical_live, phys) = false;
            }
            va += advance;
        }
        fail_when!(graph, (*graph).failed, true);
        if va + pages > (*graph).va_count {
            va = (*graph).va_count;
            fail_when!(graph, va + pages > PAGES, true);
            (*graph).va_count = va + pages;
        }

        for offset in 0..pages {
            let virtual_page = va + offset;
            if slot!((*graph).va_phys, virtual_page) < 0 {
                let mut phys = 0;
                while phys < (*graph).phys_count
                    && (slot!((*graph).allocations.physical_live, phys) || rogue_phys(graph, phys))
                {
                    phys += 1;
                }
                let status = map_page(graph, virtual_page, phys);
                if status != CUDA_SUCCESS {
                    *result = status;
                    (*graph).failed = true;
                    return true;
                }
            }
            let phys = slot!((*graph).va_phys, virtual_page);
            slot!((*graph).allocations.physical_live, phys) = true;
        }

        let allocation = event(graph, EventType::Alloc, va, size);
        fail_when!(graph, allocation.is_null(), true);
        slot!((*graph).allocations.virtual_pages, va) = VirtualPage {
            va_span: pages,
            owner_depth: (*(*graph).state).depth,
            allocation,
        };
        track_allocation(graph, allocation);
        add_used(graph, pages * PAGE);
        *ptr = (*(*graph).base).address + (va * PAGE) as u64;
        *result = CUDA_SUCCESS;
        true
    }
}

/// # Safety
/// The CUDA hook passes writable outputs and has selected the current device.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_alloc(
    ptr: *mut CuDevicePtr,
    size: usize,
    stream: CuStream,
    result: *mut CuResult,
) -> bool {
    let graph = ACTIVE_GRAPH.get();
    // SAFETY: The thread owns the active graph and output pointers for this call.
    unsafe {
        if graph.is_null() || graph_failed(graph) || (*graph).paused || stream != (*graph).stream {
            return false;
        }
        *result = CUDA_ERROR_OUT_OF_MEMORY;
        *ptr = 0;

        let small = size < SMALL_LIMIT;
        let state = (*graph).state;
        if !(*state).recording {
            let kind = if small {
                EventType::AllocSmall
            } else {
                EventType::Alloc
            };
            let matched = find_event(graph, kind, 0, size, null_mut());
            if !matched.is_null() {
                (*state).cursor = matched;
                *ptr = candidate_ptr(graph, matched);
                *result = CUDA_SUCCESS;
                return true;
            }
            fail_when!(graph, (*graph).failed || !start_recording(graph), true);
        }

        if small {
            record_small_allocation(graph, ptr, size, result)
        } else {
            record_large_allocation(graph, ptr, size, result)
        }
    }
}

/// # Safety
/// The CUDA hook supplies a writable result and the selected device.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_free(
    ptr: CuDevicePtr,
    stream: CuStream,
    result: *mut CuResult,
) -> bool {
    let graph = ACTIVE_GRAPH.get();
    // SAFETY: The caller owns the active graph on this thread and the result storage.
    unsafe {
        if graph.is_null() || graph_failed(graph) || (*graph).paused || stream != (*graph).stream {
            return false;
        }
        let base = (*(*graph).base).address;
        let small_base = (*(*graph).small_base).address;
        let small = (small_base..small_base + (SMALL_PAGES * PAGE) as u64).contains(&ptr);
        if !small && !(base..base + (PAGES * PAGE) as u64).contains(&ptr) {
            return false;
        }

        *result = CUDA_SUCCESS;
        let value = if small {
            (ptr - small_base) as usize
        } else {
            ((ptr - base) / PAGE as u64) as usize
        };
        let kind = if small {
            EventType::FreeSmall
        } else {
            EventType::Free
        };
        let state = (*graph).state;
        if !(*state).recording {
            let matched = find_event(graph, kind, value, 0, null_mut());
            if !matched.is_null() {
                (*state).cursor = matched;
                return true;
            }
            fail_when!(graph, !start_recording(graph), true);
        }

        if small {
            let mut entry = addr_of_mut!((*graph).small_ranges);
            while !(*entry).is_null() && (**entry).offset != value {
                entry = addr_of_mut!((**entry).next);
            }
            let range = *entry;
            fail_when!(
                graph,
                range.is_null()
                    || (*range).owner_depth != (*state).depth
                    || event(graph, kind, value, 0).is_null()
                    || !untrack_allocation(graph, (*range).allocation),
                true
            );
            *entry = (*range).next;
            (*graph).used -= (*range).bytes;
            deallocate(range);
            return true;
        }

        let first = addr_of_mut!(slot!((*graph).allocations.virtual_pages, value));
        fail_when!(
            graph,
            (*first).owner_depth != (*state).depth
                || (*first).va_span == 0
                || event(graph, kind, value, 0).is_null()
                || !untrack_allocation(graph, (*first).allocation),
            true
        );
        for offset in 0..(*first).va_span {
            let phys = slot!((*graph).va_phys, value + offset);
            slot!((*graph).allocations.physical_live, phys) = false;
        }
        (*graph).used -= (*first).va_span * PAGE;
        *first = VirtualPage::EMPTY;
        true
    }
}

/// # Safety
/// `context` points at a live AIMDO device and `stream` is its CUDA stream.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_create(
    context: *mut c_void,
    stream: CuStream,
    assert_breaks: bool,
) -> *mut c_void {
    let graph = allocate_zeroed::<MallocGraph>(1);
    // SAFETY: The caller supplies the live context. All graph fields accept zero and
    // stay private to the initializing thread until the reservations and root scope exist.
    unsafe {
        if graph.is_null() || !ACTIVE_GRAPH.get().is_null() {
            deallocate(graph);
            return null_mut();
        }
        let context = context.cast::<AimdoContext>();
        aimdo_rs_set_devctx(context);
        (*graph).stream = stream;
        (*graph).device = (*context).device_id;
        (*graph).owner_thread = ACTIVE_GRAPH.with(core::ptr::from_ref);
        (*graph).assert_breaks = assert_breaks;
        (*graph).root.kind = EventType::Sentinel;

        (*graph).base = virtual_range_alloc(PAGES * PAGE, PAGE);
        if !(*graph).base.is_null() {
            (*graph).small_base = virtual_range_alloc(SMALL_PAGES * PAGE, PAGE);
            if !(*graph).small_base.is_null() {
                addr_of_mut!((*graph).va_phys)
                    .cast::<i32>()
                    .write_bytes(0xff, PAGES);
                if push_stack(graph, addr_of_mut!((*graph).root), true) {
                    ACTIVE_GRAPH.set(graph);
                    return graph.cast();
                }
                virtual_range_unref((*graph).small_base);
            }
            virtual_range_unref((*graph).base);
        }
        free_small_ranges((*graph).root.small_snapshot);
        deallocate((*graph).root.snapshot);
        deallocate(graph);
        null_mut()
    }
}

/// # Safety
/// A non-null `handle` is a live graph from `malloc_graph_create`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_pause(
    handle: *mut c_void,
    paused: bool,
    sync: bool,
) -> bool {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: A matching TLS pointer belongs to the calling thread.
    unsafe {
        if graph.is_null() || graph != ACTIVE_GRAPH.get() || graph_failed(graph) {
            return false;
        }
        (*graph).paused = paused;
        if sync {
            (*graph).sync_paused = paused;
        }
        true
    }
}

/// # Safety
/// A non-null `handle` is a live graph attached to the supplied device stream.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_set_stream(
    handle: *mut c_void,
    stream: CuStream,
) -> bool {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: A matching TLS pointer belongs to the calling thread.
    unsafe {
        if graph.is_null() || graph != ACTIVE_GRAPH.get() || graph_failed(graph) {
            return false;
        }
        (*graph).stream = stream;
        true
    }
}

/// # Safety
/// `handle` is null or live. A non-null `name` is a valid NUL-terminated scope name.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_push(
    handle: *mut c_void,
    name: *const c_char,
) -> bool {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: Graph operations are serialized on the graph's owner thread.
    unsafe {
        if graph.is_null() || graph_failed(graph) {
            return false;
        }
        if name.is_null() {
            if !(*graph).complete
                || !ACTIVE_GRAPH.get().is_null()
                || !push_stack(graph, addr_of_mut!((*graph).root), false)
            {
                return false;
            }
            (*graph).complete = false;
            ACTIVE_GRAPH.set(graph);
            return true;
        }
        if graph != ACTIVE_GRAPH.get() {
            return false;
        }

        let mut call = find_event(graph, EventType::Call, 0, 0, name);
        let recording = call.is_null();
        let scope;
        if recording {
            let state = (*graph).state;
            if !(*state).recording && !start_recording(graph) {
                return false;
            }
            let mut call_name = null_mut();
            let mut new_scope = null_mut();
            call = allocate_zeroed::<Event>(1);
            if !call.is_null() {
                new_scope = allocate_zeroed::<Event>(1);
                if !new_scope.is_null() {
                    let name = CStr::from_ptr(name).to_bytes_with_nul();
                    call_name = allocate_zeroed::<c_char>(name.len());
                    if !call_name.is_null() {
                        copy_nonoverlapping(name.as_ptr().cast(), call_name, name.len());
                    }
                }
            }
            if call_name.is_null() || !append_event(graph, (*state).cursor, call) {
                deallocate(call_name);
                deallocate(call);
                deallocate(new_scope);
                (*graph).failed = true;
                return false;
            }
            (*call).kind = EventType::Call;
            (*call).scope = new_scope;
            (*call).name = call_name;
            scope = new_scope;
        } else {
            scope = (*call).scope;
        }

        push_stack(graph, scope, recording)
    }
}

/// # Safety
/// A non-null `handle` is a live graph from `malloc_graph_create`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_pop(handle: *mut c_void) -> i32 {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: Only the active graph can end a scope on this thread.
    unsafe {
        if graph.is_null() || graph != ACTIVE_GRAPH.get() || graph_failed(graph) {
            return 0;
        }
        let state = (*graph).state;
        if !(*state).recording {
            let end = find_event(graph, EventType::End, 0, 0, null_mut());
            if !end.is_null() {
                (*state).cursor = end;
            } else if !start_recording(graph) {
                return 0;
            }
        }
        fail_when!(graph, !collect_rogue_candidates(graph), 0);
        fail_when!(
            graph,
            (*state).recording && event(graph, EventType::End, 0, 0).is_null(),
            0
        );
        if (*state).next.is_null() && !finalize_rogues(graph) {
            return 0;
        }

        (*graph).state = (*state).next;
        let result = if (*state).broken { 2 } else { 1 };
        deallocate(state);
        if (*graph).state.is_null() {
            (*graph).complete = true;
            ACTIVE_GRAPH.set(null_mut());
        }
        result
    }
}

/// # Safety
/// A non-null `handle` is a live graph from `malloc_graph_create`.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_abort(handle: *mut c_void) -> bool {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: Check the creating thread before mutating any graph state.
    unsafe {
        if graph.is_null()
            || (*graph).owner_thread != ACTIVE_GRAPH.with(core::ptr::from_ref)
            || (!(*graph).aborted && graph != ACTIVE_GRAPH.get())
        {
            return false;
        }
        abort_graph(graph)
    }
}

/// # Safety
/// A non-null `handle` is a live graph. The owner serializes graph statistics and mutations.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_stat(handle: *mut c_void, which: i32) -> u64 {
    let graph = handle.cast::<MallocGraph>();
    if graph.is_null() {
        return 0;
    }
    // SAFETY: The caller ensures the graph remains live for these reads.
    unsafe {
        match which {
            0 => (*graph).peak_used as u64,
            1 => (((*graph).va_count + (*graph).small_pages) * PAGE) as u64,
            2 => (((*graph).phys_count + (*graph).small_pages) * PAGE) as u64,
            3 => (*graph).rogue_count as u64,
            _ => 0,
        }
    }
}

unsafe fn free_events(events: *mut *mut Event, count: usize) {
    unsafe {
        for index in 0..count {
            let event = *events.add(index);
            free_events((*event).next, (*event).next_count);
            if (*event).kind == EventType::Call {
                let scope = (*event).scope;
                free_events((*scope).next, (*scope).next_count);
                free_small_ranges((*scope).small_snapshot);
                deallocate((*scope).snapshot);
                deallocate(scope);
                deallocate((*event).name);
            }
            deallocate(event);
        }
        deallocate(events);
    }
}

/// # Safety
/// A non-null `handle` is live. Only the graph's owner may destroy it.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn malloc_graph_destroy(handle: *mut c_void) {
    let graph = handle.cast::<MallocGraph>();
    // SAFETY: Teardown is exclusive to the owner, and physical refs may outlive the graph.
    unsafe {
        if graph.is_null() || (*graph).owner_thread != ACTIVE_GRAPH.with(core::ptr::from_ref) {
            return;
        }
        if !(*graph).state.is_null() && !abort_graph(graph) {
            return;
        }
        while !(*graph).state.is_null() {
            let state = (*graph).state;
            (*graph).state = (*state).next;
            deallocate(state);
        }
        for index in 0..(*graph).va_count {
            physical_page_unref(slot!((*graph).mapped_pages, index));
        }
        for index in 0..(*graph).phys_count {
            physical_page_unref(slot!((*graph).physical_pages, index));
        }
        for index in 0..(*graph).small_pages {
            physical_page_unref(slot!((*graph).small_mapped_pages, index));
            physical_page_unref(slot!((*graph).small_physical_pages, index));
        }
        virtual_range_unref((*graph).base);
        virtual_range_unref((*graph).small_base);
        free_small_ranges((*graph).root.small_snapshot);
        free_small_ranges((*graph).small_ranges);
        free_small_ranges((*graph).small_unusable);
        deallocate((*graph).root.snapshot);
        free_events((*graph).root.next, (*graph).root.next_count);
        deallocate(graph);
    }
}
