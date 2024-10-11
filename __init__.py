import sys
from importlib.resources import files

sys.path.append(str(files("tensorrt_libs")))
sys.path.append(str(files("tensorrt_bindings")))
from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
from .tensorrt_convert import STATIC_TRT_MODEL_CONVERSION
from .tensorrt_loader import TensorRTLoader
from comfy.cmd import folder_paths
from pathlib import Path

folder_paths.add_model_folder_path("tensorrt", extensions={".engine"})
folder_paths.add_model_folder_path("tensorrt",
                                   full_folder_path=str(Path(folder_paths.get_output_directory()) / "tensorrt"),
                                   extensions={".engine"})

NODE_CLASS_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
    "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION,
    "TensorRTLoader": TensorRTLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION",
    "STATIC TRT_MODEL CONVERSION": STATIC_TRT_MODEL_CONVERSION,
    "TensorRTLoader": "TensorRT Loader",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
