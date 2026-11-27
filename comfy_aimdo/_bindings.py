import ctypes

from . import _native


class NativeLibrary:
    _name = _native.__file__

    def __getattr__(self, name):
        address = _native._function_address(name)
        if address is None:
            raise AttributeError(name)
        function = ctypes.CFUNCTYPE(ctypes.c_int)(address)
        setattr(self, name, function)
        return function
