import os
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from keras.models import load_model
import keras
from keras import backend as K

from zkstats.onnx2circom import onnx_to_circom
from zkstats.onnx2circom.onnx2keras.layers import (
    TFReduceSum,
    TFLog,
    TFReduceMean,
)


supported_operations = [TFReduceSum, TFLog, TFReduceMean]


def test_onnx_to_circom(tmp_path):
    data = torch.tensor(
        [10, 40, 50],
        dtype = torch.float32,
    ).reshape(1, -1, 1)

    class SumModel(nn.Module):
        def forward(self, x):
            return torch.sum(x)

    compile_and_check(SumModel, data, tmp_path)



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


def run_keras_model(keras_path: Path, data: torch.Tensor) -> torch.Tensor:
    K.clear_session()
    keras.saving.get_custom_objects().clear()
    keras_custom_objects = {layer.__name__: layer for layer in supported_operations}
    with keras.saving.custom_object_scope(keras_custom_objects):
        model = load_model(keras_path)
    # result is numpy.float32
    output_keras = model.predict(data)
    # Transform it to torch.Tensor to make it align with torch output
    return torch.tensor(output_keras, dtype = torch.float32)


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


def compile_and_check(model_type: Type[nn.Module], data: torch.Tensor, tmp_path: Path):
    onnx_path = tmp_path / 'model.onnx'
    out_dir_path = tmp_path / 'out'
    keras_path = onnx_path.parent / f"{onnx_path.stem}.keras"
    circom_path = out_dir_path / f"{keras_path.stem}.circom"

    print("Running torch model...")
    output_torch = run_torch_model(model_type, data)
    print("!@# output_torch=", output_torch)

    print("Transforming torch model to onnx...")
    torch_model_to_onnx(model_type, data, onnx_path)
    # onnx_to_keras(onnx_path, keras_path)
    print("Transforming onnx model to circom...")
    onnx_to_circom(onnx_path, circom_path)

    print("Running keras model...")
    output_keras = run_keras_model(keras_path, data)
    print("!@# output_keras=", output_keras)
    assert torch.allclose(output_torch, output_keras, atol=1e-6), "The output of torch model and keras model are different."

    # Compiling with circom compiler
    code = os.system(f"circom {circom_path}")
    if code != 0:
        raise ValueError(f"Failed to compile circom. Error code: {code}")

    # TODO: Should run circom (like using circom tester) instead of just running keras model
