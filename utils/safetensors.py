# utils/checkpoint.py
from safetensors.torch import save_file
import torch

def save_safetensors(model, file_name):
    if hasattr(model, "module"):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    contiguous_state = {}
    for key, tensor in state.items():
        if isinstance(tensor, torch.Tensor):
            contiguous_state[key] = tensor.contiguous()
        else:
            contiguous_state[key] = tensor

    save_file(contiguous_state, file_name)
