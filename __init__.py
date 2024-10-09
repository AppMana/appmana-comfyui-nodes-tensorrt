import sys
from importlib.resources import files

sys.path.append(str(files("tensorrt_libs")))
sys.path.append(str(files("tensorrt_bindings")))
from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
from .tensorrt_convert import STATIC_TRT_MODEL_CONVERSION
from .tensorrt_loader import TrTUnet
from .tensorrt_loader import TensorRTLoader

NODE_CLASS_MAPPINGS = {"DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
                       "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION, "TensorRTLoader": TensorRTLoader}

NODE_DISPLAY_NAME_MAPPINGS = {"DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION",
                              "STATIC TRT_MODEL CONVERSION": STATIC_TRT_MODEL_CONVERSION,
                              "TensorRTLoader": "TensorRT Loader"}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
