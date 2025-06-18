try:
    import tensorrt as trt
except (ImportError, ModuleNotFoundError):
    class trt:
        def init_libnvinfer_plugins(self, *args):
            pass

        class IProgressMonitor:
            def __init__(self):
                pass

        class IBuilderConfig:
            def __init__(self):
                pass

            def create_timing_cache(self, buffer):
                pass

            def set_timing_cache(self, timing_cache, ignore_mismatch):
                pass

            def get_timing_cache(self):
                pass

        class Logger:
            INFO = 0
            def __init__(self, *args):
                pass

        class Runtime:
            def __init__(self, *args):
                pass

            def deserialize_cuda_engine(self, *args):
                pass

        def __getattr__(self, item):
            class Undefined:
                pass

            return Undefined

__all__ = ["trt"]
