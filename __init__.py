import sys
import logging
from importlib.resources import files

logger = logging.getLogger(__name__)
from comfy.cmd import folder_paths
from pathlib import Path

folder_paths.add_model_folder_path("tensorrt", extensions={".engine"})
folder_paths.add_model_folder_path("tensorrt",
                                   full_folder_path=str(Path(folder_paths.get_output_directory()) / "tensorrt"),
                                   extensions={".engine"})
try:
    sys.path.append(str(files("tensorrt_libs")))
    sys.path.append(str(files("tensorrt_bindings")))

    from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION
    from .tensorrt_convert import STATIC_TRT_MODEL_CONVERSION
    from .tensorrt_loader import TensorRTLoader

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
except ModuleNotFoundError as module_not_found_exc:
    logger.error(f"skipping tensorrt_libs and tensorrt_bindings because they were not installed",
                 exc_info=module_not_found_exc)
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
