# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Layer as KerasLayer
import numpy as np
import keras

from zkstats.onnx2circom.onnx2keras.layers import (
    TFAdd,
    TFMul,
    TFSub,
    TFDiv,
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
    TFAdd,
    TFMul,
    TFSub,
    TFDiv,
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


KerasTensor = typing.Any  # keras.engine.keras_tensor.KerasTensor

# read each layer in a model and convert it to a class called Layer
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    inputs: list[KerasTensor]
    outputs: list[KerasTensor]
    config: typing.Dict[str, typing.Any]
    weights: typing.List[np.ndarray]
    model: 'Model'

    def __init__(self, layer: keras.layers.Layer, model: 'Model'):
        self.op = layer.__class__.__name__
        self.name = layer.name
        self.model = model
        # !@# Layer: self.op='TFReduceSum', self.name='tf_reduce_sum_1',
        #   layer.input=<KerasTensor shape=(1, 2, 1), dtype=float32, sparse=False, name=input_layer>,
        #   layer.output=<KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_5>
        # !@# Layer: self.op='TFReduceMean', self.name='tf_reduce_mean_1',
        #   layer.input=<KerasTensor shape=(1, 2, 1), dtype=float32, sparse=False, name=input_layer>,
        #   layer.output=<KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_7>
        # !@# Layer: self.op='TFAdd', self.name='tf_add_1',
        #   layer.input=[
        #       <KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_5>,
        #       <KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_7>
        #   ],
        #   layer.output=<KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_10>
        #   tensor_from_layer =
        print(f"!@# Layer: {self.op=}, {self.name=}, {layer.input=}, {layer.output=}")
        print(f"!@# Layer: {layer.get_config()=}, {layer.get_weights()=}")
        # map the input name to the comp name?

        # input is Union[KerasTensor, list[KerasTensor]]
        # (1, 2, 1) -> (2, 1)
        # component
        # component add = TFAdd()
        # add.in[0][0] = inputs_layer.out[0][0]
        # add.in[1][0] = tf_reduce_sum_1.out[0][0]
        self.inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        self.outputs = layer.output if isinstance(layer.output, list) else [layer.output]



class Model:
    layers: typing.List[Layer]
    model_inputs: typing.List[KerasTensor]
    model_outputs: typing.List[KerasTensor]

    def __init__(self, filename: str, raw: bool = False):
        ''' Load a Keras model from a file. '''
        # edit : allow reading customed layer
        keras.saving.get_custom_objects().clear()
        # Only if the torch model name is in this custom_objects, model.summary() will print the mapped name in keras
        # E.g. without this line, the model.summary() will print the layer name as `tf_reduce_sum (TFReduceSum)`
        # with `TFReduceSum: SumCheck`, the model.summary() will print the layer name as `sum_check (SumCheck)`
        custom_objects = {op.__name__: op for op in onnx2circom_ops_raw}
        with keras.saving.custom_object_scope(custom_objects):
            model = keras.models.load_model(filename)
            # keras.models.functional.Functional
        print(f"!@# {model.summary()}")
        # !@# model.inputs=[<KerasTensor shape=(1, 2, 1), dtype=float32, sparse=False, name=input_layer>],
        #     model.outputs=<KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_10>
        self.model_inputs = model.inputs
        self.model_outputs = model.outputs
        self.layers = [Layer(layer, self) for layer in model.layers if self._for_transpilation(layer.__class__.__name__)]
        self.map_output_to_component = {}
        for layer in self.layers:
            for output in layer.outputs:
                self.map_output_to_component[output.name] = layer.name

    def get_component_from_output_name(self, output_name: str):
        return self.map_output_to_component[output_name]

    def is_model_input(self, input_name: str):
        return input_name in [i.name for i in self.model_inputs]

    def is_model_output(self, output_name: str):
        return output_name in [o.name for o in self.model_outputs]

    # def is_layer_initial_inputs(self, layer_name: str):
    #     return any(layer_name == model_inputs for model_inputs in self.model_inputs)

    def get_model_output_names(self):
        return [out.name for out in self.model_outputs]

    @staticmethod
    def _for_transpilation(op: str) -> bool:
        if op in skip_ops:
            return False
        if op in supported_ops:
            return True
        raise NotImplementedError(f'Unsupported op: {op}')