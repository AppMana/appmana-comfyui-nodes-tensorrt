try:
    import tensorrt as trt
except (ImportError, ModuleNotFoundError):
    class trt:
        def init_libnvinfer_plugins(self, *args):
            pass

        class IProgresMonitor:
            def __init__(self):
                pass

        class IBuilderConfig:
            def __init__(self):
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
