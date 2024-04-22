from pathlib import Path
from typing import Type

import torch
import torch.nn as nn


def run_torch_model(model_type: Type[nn.Module], data: torch.Tensor) -> torch.Tensor:
    model = model_type()
    output_torch = model.forward(data)
    # untuple the result: if the result is [x], return x
    shape = output_torch.shape
    # if it is a 0-d tensor, return it directly
    if len(shape) == 0:
        return output_torch
    # if it is a 1-d tensor with one element, return the element directly
    elif len(shape) == 1 and shape[0] == 1:
        return output_torch[0]
    else:
        return output_torch


def torch_model_to_onnx(model_type: Type[nn.Module], data: torch.Tensor, output_onnx_path: Path):
    model = model_type()
    torch.onnx.export(model,               # model being run
                        data,                   # model input (or a tuple for multiple inputs)
                        output_onnx_path,            # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})

