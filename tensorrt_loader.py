import os
from pathlib import PurePath

import torch

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
from comfy.cmd import folder_paths
from comfy.model_patcher import ModelPatcher

from .trt_stub import trt

trt.init_libnvinfer_plugins(None, "")

logger = trt.Logger(trt.Logger.INFO)
runtime = trt.Runtime(logger)


# Is there a function that already exists for this?
def trt_datatype_to_torch(datatype):
    if datatype == trt.float16:
        return torch.float16
    elif datatype == trt.float32:
        return torch.float32
    elif datatype == trt.int32:
        return torch.int32
    elif datatype == trt.bfloat16:
        return torch.bfloat16


class TrTModelManageable(ModelPatcher):
    def __init__(self, model: comfy.model_base.BaseModel, unet: "TrTUnet", load_device, offload_device):
        self._unet = unet
        super().__init__(model, load_device, offload_device, size=unet.size, ckpt_name=PurePath(unet.engine_path).stem)

    @property
    def engine(self):
        return self._unet.engine

    @engine.setter
    def engine(self, value):
        self._unet.engine = value

    @property
    def context(self):
        return self._unet.context

    @context.setter
    def context(self, value):
        self._unet.context = value

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        self._unet.load()
        return super().patch_model(device_to, lowvram_model_memory, load_weights, force_patch_weights)

    def unpatch_model(self, device_to=None, unpatch_weights=False):
        self._unet.unload()
        return super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

    def model_size(self):
        return self._unet.size

    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        self._unet.load()
        super().load(device_to, lowvram_model_memory, force_patch_weights, full_load)

    def model_dtype(self):
        return self._unet.dtype

    def is_clone(self, other):
        return other is not None and isinstance(other,
                                                TrTModelManageable) and other._unet is self._unet and super().is_clone(
            other)

    def clone_has_same_weights(self, clone):
        return clone is not None and isinstance(clone,
                                                TrTModelManageable) and clone._unet.engine_path == self._unet.engine_path and super().clone_has_same_weights(
            clone)

    def memory_required(self, input_shape):
        return self.model_size()  # This is an approximation

    def __str__(self):
        return f"<TrtModelManageable {self.ckpt_name}>"


class TrTUnet:
    def __init__(self, engine_path):
        self.dtype = torch.bfloat16
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self._size = int(os.stat(engine_path).st_size)

    def load(self):
        if self.engine is not None or self.context is not None:
            return
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

    @property
    def size(self) -> int:
        return self._size

    def set_bindings_shape(self, inputs, split_batch):
        for k in inputs:
            shape = inputs[k].shape
            shape = [shape[0] // split_batch] + list(shape[1:])
            self.context.set_input_shape(k, shape)

    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        model_inputs = {"x": x, "timesteps": timesteps, "context": context}

        if y is not None:
            model_inputs["y"] = y

        for i in range(len(model_inputs), self.engine.num_io_tensors - 1):
            name = self.engine.get_tensor_name(i)
            model_inputs[name] = kwargs[name]

        batch_size = x.shape[0]
        dims = self.engine.get_tensor_profile_shape(self.engine.get_tensor_name(0), 0)
        min_batch = dims[0][0]
        opt_batch = dims[1][0]
        max_batch = dims[2][0]

        # Split batch if our batch is bigger than the max batch size the trt engine supports
        for i in range(max_batch, min_batch - 1, -1):
            if batch_size % i == 0:
                curr_split_batch = batch_size // i
                break

        self.set_bindings_shape(model_inputs, curr_split_batch)

        model_inputs_converted = {}
        for k in model_inputs:
            data_type = self.engine.get_tensor_dtype(k)
            model_inputs_converted[k] = model_inputs[k].to(dtype=trt_datatype_to_torch(data_type))

        output_binding_name = self.engine.get_tensor_name(len(model_inputs))
        out_shape = self.engine.get_tensor_shape(output_binding_name)
        out_shape = list(out_shape)

        # for dynamic profile case where the dynamic params are -1
        for idx in range(len(out_shape)):
            if out_shape[idx] == -1:
                out_shape[idx] = x.shape[idx]
            else:
                if idx == 0:
                    out_shape[idx] *= curr_split_batch

        out = torch.empty(out_shape,
                          device=x.device,
                          dtype=trt_datatype_to_torch(self.engine.get_tensor_dtype(output_binding_name)))
        model_inputs_converted[output_binding_name] = out

        stream = torch.cuda.default_stream(x.device)
        for i in range(curr_split_batch):
            for k in model_inputs_converted:
                x = model_inputs_converted[k]
                self.context.set_tensor_address(k, x[(x.shape[0] // curr_split_batch) * i:].data_ptr())
            self.context.execute_async_v3(stream_handle=stream.cuda_stream)
        # stream.synchronize() #don't need to sync stream since it's the default torch one
        return out

    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}

    def unload(self):
        engine_obj = self.engine
        self.engine = None
        if engine_obj is not None:
            del engine_obj
        context_obj = self.context
        self.context = None
        if context_obj is not None:
            del context_obj


class TensorRTLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"unet_name": (folder_paths.get_filename_list("tensorrt"),),
                             "model_type": (
                                 [
                                     "sdxl_base",
                                     "sdxl_refiner",
                                     "sd1.x",
                                     "sd2.x-768v",
                                     "svd",
                                     "sd3",
                                     "auraflow",
                                     "flux_dev",
                                     "flux_schnell",
                                 ],),
                             }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "TensorRT"

    def load_unet(self, unet_name, model_type):
        unet_path = folder_paths.get_full_path("tensorrt", unet_name)
        if not os.path.isfile(unet_path):
            raise FileNotFoundError(f"File {unet_path} does not exist")
        unet = TrTUnet(unet_path)
        if model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXL(conf)
        elif model_type == "sdxl_refiner":
            conf = comfy.supported_models.SDXLRefiner(
                {"adm_in_channels": 2560})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.SDXLRefiner(conf)
        elif model_type == "sd1.x":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd2.x-768v":
            conf = comfy.supported_models.SD20({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = comfy.model_base.BaseModel(conf, model_type=comfy.model_base.ModelType.V_PREDICTION)
        elif model_type == "svd":
            conf = comfy.supported_models.SVD_img2vid({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "sd3":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "auraflow":
            conf = comfy.supported_models.AuraFlow({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "flux_dev":
            conf = comfy.supported_models.Flux({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        elif model_type == "flux_schnell":
            conf = comfy.supported_models.FluxSchnell({})
            conf.unet_config["disable_unet_model_creation"] = True
            model = conf.get_model({})
        else:
            raise ValueError("unsupported model_type")
        model.diffusion_model = unet

        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        manageable_model = TrTModelManageable(model, unet, load_device, offload_device)

        return (manageable_model,)


NODE_CLASS_MAPPINGS = {
    "TensorRTLoader": TensorRTLoader,
}
