from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from keras.models import load_model
import keras
from keras import backend as K

import pytest

from zkstats.onnx2circom import onnx_to_keras
from zkstats.onnx2circom.onnx2keras.layers import (
    TFReduceSum,
    TFLog,
    TFReduceMean,
)

from .utils import run_torch_model, torch_model_to_onnx


# NOTE: Make sure keras operations are added here
supported_operations = [TFReduceSum, TFLog, TFReduceMean]


class SumModel(nn.Module):
    def forward(self, x):
        return torch.sum(x)

class MeanModel(nn.Module):
    def forward(self, x):
        return torch.mean(x)

class LogModel(nn.Module):
    def forward(self, x):
        return torch.log(x)


@pytest.mark.parametrize("model_type, expected_res",
    [
        (SumModel, torch.tensor(100.0)),
        (MeanModel, torch.tensor(33.333)),
        (LogModel, torch.tensor([2.3026, 3.6889, 3.9120])),
    ]
)
def test_onnx_to_keras(tmp_path, model_type: Type[nn.Module], expected_res: float):
    data = torch.tensor(
        [10, 40, 50],
        dtype = torch.float32,
    ).reshape( -1,1)

    res = convert_run_check(model_type, data, tmp_path)
    assert torch.allclose(res, expected_res, atol=1e-6), f"Expected result: {expected_res}, but got {res}"


def convert_run_check(model_type: Type[nn.Module], data: torch.Tensor, tmp_path: Path):
    onnx_path = tmp_path / 'model.onnx'
    model_name = onnx_path.stem
    keras_path = onnx_path.parent / f"{model_name}.keras"
    assert onnx_path.stem == keras_path.stem

    output_torch = run_torch_model(model_type, tuple([data]))

    print("Transforming torch model to onnx...")
    torch_model_to_onnx(model_type, tuple([data]), onnx_path)
    onnx_to_keras(onnx_path, keras_path)

    output_keras = run_keras_model(keras_path, data)
    print("!@# output_keras=", output_keras)
    assert torch.allclose(output_torch, output_keras, atol=1e-6), "The output of torch model and keras model are different."
    return output_keras


def run_keras_model(keras_path: Path, data: torch.Tensor) -> torch.Tensor:
    K.clear_session()
    keras.saving.get_custom_objects().clear()
    keras_custom_objects = {layer.__name__: layer for layer in supported_operations}
    with keras.saving.custom_object_scope(keras_custom_objects):
        model = load_model(keras_path)
    # result is numpy.float32
    output_keras = model.predict(data)
    # Transform it to torch.Tensor to make it align with torch output
    return torch.tensor(output_keras, dtype = torch.float32).reshape(-1)

