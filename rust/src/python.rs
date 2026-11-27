use crate::{alloc_hooks, context, graph, hostbuf, mmap, platform, vbar, vrambuf, xfer_file};
use pyo3::prelude::*;

macro_rules! function_addresses {
    ($($module:ident { $($name:ident $(=> $function:ident)?),+ $(,)? }),+ $(,)?) => {
        #[pyfunction]
        fn _function_address(name: &str) -> Option<usize> {
            let function = match name {
                $($(stringify!($name) => function_addresses!(@select $module, $name $(, $function)?) as *const (),)+)+
                _ => return None,
            };
            Some(function as usize)
        }
    };
    (@select $module:ident, $name:ident) => { $module::$name };
    (@select $module:ident, $name:ident, $function:ident) => { $module::$function };
}

// 原 Python 包装层仍负责声明 ctypes 参数和返回类型。私有扩展仅为包装层查找这些地址，
// 链接器只导出 PyInit__native，ComfyUI 仍通过原 Python 接口调用。
function_addresses! {
    context {
        plat_init => aimdo_rs_plat_init,
        plat_cleanup => aimdo_rs_plat_cleanup,
        init => aimdo_rs_init,
        cleanup => aimdo_rs_cleanup,
        get_devctx => aimdo_rs_get_devctx,
        aimdo_analyze => aimdo_rs_analyze,
        get_total_vram_usage => aimdo_rs_get_total_vram_usage,
        set_simple_vram_headroom => aimdo_rs_set_simple_vram_headroom,
        get_simple_vram_headroom => aimdo_rs_get_simple_vram_headroom,
    },
    platform {
        set_nvml_pressure, set_log_callback, set_log_level_none, set_log_level_critical,
        set_log_level_error, set_log_level_warning, set_log_level_info,
        set_log_level_debug, set_log_level_verbose, set_log_level_vverbose,
    },
    vbar {
        vbar_allocate, vbar_deprioritize, vbar_fault, vbar_free, vbar_free_memory, vbar_get,
        vbar_get_nr_pages, vbar_get_residency, vbar_get_watermark, vbar_loaded_size,
        vbar_prioritize, vbar_set_watermark, vbar_set_watermark_limit, vbar_unpin,
        vbars_analyze, vbars_reset_watermark_limits,
    },
    vrambuf { vrambuf_create, vrambuf_destroy, vrambuf_get, vrambuf_grow },
    hostbuf {
        hostbuf_allocate, hostbuf_extend, hostbuf_free, hostbuf_get_raw_address,
        hostbuf_read_file_slice, hostbuf_register, hostbuf_truncate, hostbuf_unregister,
        hostbuf_file_reader_cleanup, hostbuf_file_reader_read, hostbuf_file_writer_write,
    },
    xfer_file { xfer_file_write },
    mmap {
        model_mmap_allocate, model_mmap_bounce, model_mmap_deallocate,
        model_mmap_get, model_mmap_get_file_handle,
    },
    graph {
        malloc_graph_abort, malloc_graph_create, malloc_graph_destroy, malloc_graph_pause,
        malloc_graph_pop, malloc_graph_push, malloc_graph_set_stream, malloc_graph_stat,
    },
    alloc_hooks { alloc_fn, free_fn },
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_function_address, module)?)
}
