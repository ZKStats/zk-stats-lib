# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer as KerasLayer
import numpy as np
import keras

from zkstats.onnx2circom.onnx2keras.layers import (
    TFReciprocal,
    TFSqrt,
    TFExp,
    TFLog,
    TFReduceSum,
    TFReduceMean,
    TFReduceMax,
    TFReduceMin,
    # TFArgMax,
    # TFArgMin,
    # TFErf,
)

onnx2circom_ops_raw = [
    TFLog,  # log_e(n)
    TFReduceSum,  # sum(n)
    TFReduceMean,
    TFReduceMax,
    TFReduceMin,
    # TFArgMax,
    # TFArgMin,
    TFReciprocal,  # 1/n
    TFSqrt,  # sqrt(n)
    TFExp,  # e^n
    # TFErf,
]
onnx2circom_ops = [str(op.__name__) for op in onnx2circom_ops_raw]

keras2circom_ops = [
    'Activation',
    'AveragePooling2D',
    'BatchNormalization',
    'Conv2D',
    'Dense',
    'Flatten',
    'GlobalAveragePooling2D',
    'GlobalMaxPooling2D',
    'MaxPooling2D',
    'ReLU',
    'Softmax',
]


supported_ops =  keras2circom_ops + onnx2circom_ops


skip_ops = [
    'Dropout',
    'InputLayer',
]


# read each layer in a model and convert it to a class called Layer
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    input: typing.List[int]
    output: typing.List[int]
    config: typing.Dict[str, typing.Any]
    weights: typing.List[np.ndarray]

    def __init__(self, layer: KerasLayer):
        self.op = layer.__class__.__name__
        self.name = layer.name
        self.input = layer.input.shape[1:]
        self.output = layer.output.shape[1:]
        # FIXME: this only works for data shape in [1, N, 1]
        # Add "nInputs" to `self.config`
        shape = layer.input.shape
        if len(shape) != 3 or shape[0] != 1 or shape[2] != 1:
            raise Exception(f'Unsupported input shape: {self.op=}, {shape=}')
        n_inputs = shape[1]
        self.config = {**layer.get_config(), **{"nInputs": n_inputs}}
        self.weights = layer.get_weights()


class Model:
    layers: typing.List[Layer]

    def __init__(self, filename: str, raw: bool = False):
        ''' Load a Keras model from a file. '''
        # edit : allow reading customed layer
        keras.saving.get_custom_objects().clear()
        # Only if the torch model name is in this custom_objects, model.summary() will print the mapped name in keras
        # E.g. without this line, the model.summary() will print the layer name as `tf_reduce_sum (TFReduceSum)`
        # with `TFReduceSum: SumCheck`, the model.summary() will print the layer name as `sum_check (SumCheck)`
        custom_objects = {op.__name__: op for op in onnx2circom_ops_raw}
        with keras.saving.custom_object_scope(custom_objects):
            model = load_model(filename)
        self.layers = [Layer(layer) for layer in model.layers if self._for_transpilation(layer.__class__.__name__)]

    @staticmethod
    def _for_transpilation(op: str) -> bool:
        if op in skip_ops:
            return False
        if op in supported_ops:
            return True
        raise NotImplementedError(f'Unsupported op: {op}')