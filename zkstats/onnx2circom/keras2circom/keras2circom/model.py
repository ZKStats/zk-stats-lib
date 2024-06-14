# Read keras model into list of parameters like op, input, output, weight, bias
from __future__ import annotations
from dataclasses import dataclass
import typing

import numpy as np
import keras
import torch


# <KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_10>
class KerasTensor(typing.Protocol):
    shape: typing.Sequence[int]
    dtype: np.dtype
    sparse: bool
    name: str


@dataclass(frozen=True)
class Input:
    is_constant: bool
    shape: typing.Sequence[int]
    # If it's a constant, name is None. Else, it's the name of the input in keras model.
    name: typing.Optional[str]
    # If it's a constant, value is the value of the constant. Else, it's None
    value: typing.Optional[float]
    # is it keras_tensor in form of no shape i.e. shape = ()
    is_keras_constant: bool


def dict_to_tensor(data):
    if data['class_name'] != '__numpy__':
        raise ValueError("Unsupported class_name")

    value = data['config']['value']
    dtype = data['config']['dtype']

    # Map the dtype string to a PyTorch dtype
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64
    }

    if dtype not in dtype_map:
        raise ValueError("Unsupported dtype")

    tensor_dtype = dtype_map[dtype]

    # Convert the list to a PyTorch tensor with the specified dtype
    tensor = torch.tensor(value, dtype=tensor_dtype)
    return tensor


# read each layer in a model and convert it to a class called Layer
@dataclass
class Layer:
    ''' A single layer in a Keras model. '''
    op: str
    name: str
    inputs: list[Input]
    outputs: list[KerasTensor]
    config: dict

    def __init__(self, layer: keras.layers.Layer):
        self.op = layer.__class__.__name__
        self.name = layer.name
        # Always convert to list for easier handling. Doing this since if there is only one input, it is not a list in keras layer
        _inputs = layer.input if isinstance(layer.input, list) else [layer.input]
        # Always convert to list for easier handling. Doing this since if there is only one output, it is not a list in keras layer
        self.outputs = layer.output if isinstance(layer.output, list) else [layer.output]
        _config = layer.get_config()
        self.config = _config
        self.inputs = []
        list_inputs = _config['node_inputs']


        index = 0
        for ele_name in list_inputs:
            # non-constant: {'class_name': '__keras_tensor__', 'config': {'shape': [1, 3, 1], 'dtype': 'float32', 'keras_history': ['input_layer', 0, 0]}, 'name': 'input_layer'},
            # non-constant: {'class_name': '__keras_tensor__', 'config': {'shape': [], 'dtype': 'float32', 'keras_history': ['tf_reducemean', 0, 0]}},
            # constant: 3.0
            is_keras_constant = False
            config_ele = _config['tensor_grap'][ele_name]
            # it's not a constant, add the name of the input
            is_non_constant = isinstance(config_ele, dict) and config_ele["class_name"]=='__keras_tensor__'
            if is_non_constant:
                name = _inputs[index].name
                value = None
                input_shape = tuple(config_ele['config']['shape'])
                # if it's keras tensor resulting in constant, get the shape from non-constant input
                if input_shape == ():
                    # if there are more than 1 inputs like `TFAdd`, we need to get the shape of the other input
                    # if len(_inputs)==2 and len(_inputs[1-index].shape)>=1:
                    #     input_shape = (_inputs[1-index]).shape
                    # else:
                    input_shape =(1,)
                    is_keras_constant = True
                index += 1
            # FIXME: a constant can be a tensor with multiple dimensions, but for now we assume
            # it's constant.
            else:
                name = None
                # '/Constant_2_output_0': {'class_name': '__numpy__', 'config': {'value': [1.0, 0.0, 0.0], 'dtype': 'float32'}}
                if isinstance(config_ele, dict) and config_ele["class_name"]=='__numpy__':
                    value = config_ele['config']['value']
                    value_in_tensor = dict_to_tensor(config_ele)
                    input_shape = value_in_tensor.shape
                # '/Constant_output_0': 0
                else:
                    value = float(config_ele)
                    # if len(_inputs)>0 and len(_inputs[0].shape)>=1:
                    #     input_shape = (_inputs[0]).shape
                    # else:
                    input_shape = (1,)

            self.inputs.append(
                Input(
                    is_constant=not is_non_constant,
                    shape=input_shape,
                    name=name,
                    value=value,
                    is_keras_constant=is_keras_constant
                )
            )


class Model:
    layers: typing.List[Layer]
    # The inputs to the model (the highest level inputs). E.g. input_layer
    model_inputs: typing.List[KerasTensor]
    # The outputs of the model (the highest level outputs).
    model_outputs: typing.List[KerasTensor]
    supported_ops: typing.Dict[str, keras.layers.Layer]
    skip_ops: typing.Dict[str, keras.layers.Layer]

    def __init__(self, filename: str, supported_ops: typing.List[keras.layers.Layer], skip_ops: typing.List[keras.layers.Layer]):
        ''' Load a Keras model from a file. '''
        self.supported_ops = {op.__name__: op for op in supported_ops}
        self.skip_ops = {op.__name__: op for op in skip_ops}
        # edit : allow reading customed layer
        keras.saving.get_custom_objects().clear()
        with keras.saving.custom_object_scope(self.supported_ops):
            model = keras.models.load_model(filename)
        # E.g. model.inputs=[<KerasTensor shape=(1, 2, 1), dtype=float32, sparse=False, name=input_layer>],
        self.model_inputs = model.inputs
        # E.g. model.outputs=<KerasTensor shape=(), dtype=float32, sparse=False, name=keras_tensor_10>
        self.model_outputs = model.outputs
        self.layers = [Layer(layer) for layer in model.layers if self._for_transpilation(layer.__class__.__name__)]
        # Map each output to their layer for later usage
        self.map_output_to_component: dict[str, Layer] = {}
        for layer in self.layers:
            for output in layer.outputs:
                if output.name in self.map_output_to_component:
                    raise ValueError(f"Output name {output.name} is already used by another layer.")
                self.map_output_to_component[output.name] = layer
        # print('\n\n\n\n\n MPAPPPA: ', self.map_output_to_component.keys())
        # print('\n\n\n\n\n\n')
    def get_component_from_output_name(self, output_name: str) -> typing.Optional[Layer]:
        try:
            return self.map_output_to_component[output_name]
        except KeyError:
            return None

    def is_model_input(self, input_name: str):
        return input_name in [inp.name for inp in self.model_inputs]

    def _for_transpilation(self, op: str) -> bool:
        if op in self.skip_ops:
            return False
        if op in self.supported_ops:
            return True
        raise NotImplementedError(f'Unsupported op: {op}')
