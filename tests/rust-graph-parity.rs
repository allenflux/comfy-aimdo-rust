#[path = "../rust/src/cuda.rs"]
mod cuda;
#[path = "../rust/src/graph.rs"]
mod graph;
#[path = "../rust/src/rogue.rs"]
mod rogue;
#[path = "../rust/src/vmm_ref.rs"]
mod vmm_ref;

mod context {
    use std::cell::Cell;
    use std::ffi::c_void;
    use std::ptr::null_mut;

    #[repr(C)]
    pub struct AimdoContext {
        pub device_id: i32,
        pub total_vram_usage: u64,
        pub rogues: *mut c_void,
        pub rogue_candidates: *mut c_void,
    }

    thread_local! {
        static CONTEXT: Cell<*mut AimdoContext> = const { Cell::new(null_mut()) };
    }

    pub fn aimdo_rs_get_current_devctx() -> *mut AimdoContext {
        CONTEXT.get()
    }

    pub fn aimdo_rs_set_devctx(context: *mut AimdoContext) {
        CONTEXT.set(context);
    }
}

mod platform {
    pub const MIB: usize = 1024 * 1024;

    pub unsafe fn budget_deficit(size: usize) -> isize {
        size as isize - 1024 * MIB as isize
    }

    macro_rules! cuda {
        ($name:ident $(, $arg:expr)* $(,)?) => {{
            match (*core::ptr::addr_of!($crate::cuda::g_cuda)).$name {
                Some(function) => function($($arg),*),
                None => 3,
            }
        }};
    }
    pub(crate) use cuda;
}
