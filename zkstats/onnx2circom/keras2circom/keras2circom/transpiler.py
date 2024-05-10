from .circom import *
from .model import *
from .script import *

import os

from keras.layers import Dropout, InputLayer

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
)


INDENTATION = ' ' * 4
CIRCOM_VERSION = '2.0.0'


# NOTE: Add keras ops here if we'd like to support them
SUPPORTED_OPS = [
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

SKIPPED_OPS = [
    Dropout,
    InputLayer,
]


def get_component_args_values(layer: Layer) -> typing.Dict[str, typing.Any]:
    """
    Get the value of the arguments for a component based on the information from its keras layer
    """
    inputs = layer.inputs
    input_0 = inputs[0]
    input_0_shape = get_effective_shape(input_0.shape)
    if len(input_0_shape) == 0:
        num_inputs = 1
    else:
        num_inputs = input_0_shape[0]

    # NOTE: Add more cases here if new circom components are supported
    if layer.op == TFReduceSum.__name__:
        return {'nInputs': num_inputs}
    if layer.op == TFReduceMean.__name__:
        return {'nInputs': num_inputs}
    if layer.op == TFLog.__name__:
        return {'e': 2}
    return {}


def transpile(filename: str, output_dir: str = 'output', raw: bool = False, dec: int = 18) -> None:
    '''
    Traverse a keras model and convert it to a circom circuit.

    Steps:
    1. Declare input signals for model inputs and output signals for model outputs
        - Input/ output names are using the names from the name in keras tensors
    2. Declare components for each layer
    3. Wire the components given the model inputs and other components outputs
    4. Wire the model outputs to the corresponding components
    5. Write the circuit to a file
    '''
    model = Model(filename, SUPPORTED_OPS, SKIPPED_OPS)

    model_input_name_to_shape = {
        model_input.name: get_effective_shape(model_input.shape)
        for model_input in model.model_inputs
    }
    model_output_name_to_shape = {
        model_output.name: get_effective_shape(model_output.shape)
        for model_output in model.model_outputs
    }
    # Declare signals for model inputs and outputs
    # E.g. signal input input_layer[2];
    input_signal_declarations = [
        f"signal input {mode_input_name}{tensor_shape_to_circom_array(input_shape)};"
        for mode_input_name, input_shape in model_input_name_to_shape.items()
    ]
    # E.g. signal output output_layer[2];
    output_signal_declarations = [
        f"signal output {model_output_name}{tensor_shape_to_circom_array(output_shape)};"
        for model_output_name, output_shape in model_output_name_to_shape.items()
    ]

    #
    # Go through each layer and generate the corresponding circom code
    #

    # Include statements for the files containing the component templates.
    # E.g. `include "/path/to/mpc.circom";`
    includes: list[str] = []
    # Declaration statements. E.g. `component add = TFAdd();`
    components_declarations: list[str] = []
    # Constraints for the components. E.g. `add.in[0] <== input_layer[2][1];`
    component_constraints: list[str] = []
    # Constraints for the model outputs. E.g. `output_layer[2][1] <== add.out;`
    output_constraints: list[str] = []

    for layer in model.layers:
        component_template = templates[layer.op]
        # Include statements
        includes.append(f'include "{str(component_template.fpath)}";')

        # Component declaration
        component_args = component_template.args
        arg_values = get_component_args_values(layer)
        component_args = parse_args(component_args, arg_values)
        components_declarations.append(f"component {layer.name} = {layer.op}({component_args});")

        # Add constraints for component inputs
        #   E.g. comp.in[0] <== prev_comp.out[0], ...
        for input_index, _input in enumerate(layer.inputs):
            # Handle left hand side
            component_input_name = f"{component_template.input_names[input_index]}"
            # How many `[]`s. E.g. (3, 4) -> dim=2
            lhs = f"{layer.name}.{component_input_name}"
            lhs_dim = component_template.input_dims[input_index]

            # Handle right hand side
            if model.is_model_input(_input.name) is True:
                # If this input is from `input_layer`, use the original tensor name for it
                from_component_name = None
                from_component_signal_name = _input.name
                rhs_dim = len(model_input_name_to_shape[_input.name])
            else:
                # Get the output name in the corresponding component
                from_component_layer = model.get_component_from_output_name(_input.name)
                if from_component_layer is None:
                    raise ValueError(f"Output {_input.name} not found in any component")
                from_component_name = from_component_layer.name
                # get the index of the output tensor in the component
                from_component_output_tensor_names = [output.name for output in from_component_layer.outputs]
                from_component_output_index = from_component_output_tensor_names.index(_input.name)
                # Sanity check
                if from_component_output_index == -1:
                    raise ValueError(f"Output {_input.name} not found in from_component {from_component_name}")

                # Use the index to get the component output name from template
                from_component_template = templates[from_component_layer.op]
                from_component_signal_name = from_component_template.output_names[from_component_output_index]
                rhs_dim = from_component_template.output_dims[from_component_output_index]
            # either "{component_name}." if input is from a component or "" if input is from model input
            from_component_prefix = f"{from_component_name}." if from_component_name is not None else ""
            rhs = f"{from_component_prefix}{from_component_signal_name}"

            constraints_lines = generate_constraints(lhs, lhs_dim, rhs, rhs_dim, _input.shape)
            component_constraints.extend(constraints_lines)

    # Add constraints for model outputs (main outputs)
    # E.g. output_layer[2][1] <== add.out;
    for output in model.model_outputs:
        # Handle left hand side
        lhs = f"{output.name}"
        lhs_dim = len(model_output_name_to_shape[output.name])

        # Handle right hand side
        if model.is_model_input(output.name) is True:
            # NOTE: is it possible for a model output to be a model input?
            raise ValueError(f"Output {output.name} is also a model input. Don't need to do anything")
        # Find the component that has this output
        from_component_layer = model.get_component_from_output_name(output.name)
        # Get the component name
        from_component_name = from_component_layer.name
        # Get the signal name in the component
        from_component_output_tensor_names = [output.name for output in from_component_layer.outputs]
        from_component_output_index = from_component_output_tensor_names.index(output.name)
        if from_component_output_index == -1:
            raise ValueError(f"Output {output.name} not found in from_component {from_component_name}")
        from_component_template = templates[from_component_layer.op]
        from_component_signal_name = from_component_template.output_names[from_component_output_index]

        rhs = f"{from_component_name}.{from_component_signal_name}"
        rhs_dim = from_component_template.output_dims[from_component_output_index]

        constraints_lines = generate_constraints(lhs, lhs_dim, rhs, rhs_dim, output.shape)
        output_constraints.extend(constraints_lines)


    def indent_line(line: str):
        return f"{INDENTATION}{line}"

    includes_str = "\n".join(set(includes))
    signals_declarations_str = "\n".join(map(indent_line, input_signal_declarations + output_signal_declarations))
    components_declarations_str = "\n".join(map(indent_line, components_declarations))
    constraints_str = "\n".join(map(indent_line, component_constraints + output_constraints))


    circom_result = f'''pragma circom 2.0.0;

{includes_str}

template Model() {{
{signals_declarations_str}

{components_declarations_str}

{constraints_str}
}}

component main = Model();
'''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + '/circuit.circom', 'w') as f:
        f.write(circom_result)


def parse_args(template_args: typing.List[str], args: typing.Dict[str, typing.Any]) -> str:
    if len(template_args) != len(args):
        raise ValueError('Number of template arguments does not match the number of arguments')
    if len(template_args) == 0:
        return ''
    args_str = '{'+'}, {'.join(template_args)+'}'
    return args_str.format(**args)


def get_effective_shape(shape: tuple[int, ...]) -> tuple:
    # remove the first dimension. E.g. (1, 2, 1) -> (2, 1)
    return shape[1:]


def tensor_shape_to_circom_array(shape: list[int]):
    return ''.join([f"[{dim}]" for dim in shape])


def generate_constraints(lhs: str, lhs_dim: int, rhs: str, rhs_dim: int, tensor_shape: tuple[int, ...]) -> list[str]:
    """
    Generate constraints for a component input and output

    :param lhs: The left hand side signal of the constraint. E.g. "add.in"
    :param lhs_dim: The dimension of the left hand side signal. E.g. 1
    :param rhs: The right hand side signal of the constraint. E.g. "input_layer"
    :param rhs_dim: The dimension of the right hand side signal. E.g. 2
    :param tensor_shape: The shape of the tensor. E.g. (1, 2, 1)

    :return: A list of strings representing the constraints. E.g.
    ```
    for (var i0 = 0; i0 < shape[0]; i++) {
      for (var i1 = 0; i < shape[1]; i++) {
          lhs[i0][i1] <== rhs[i0][i1];
      }
    }
    ```
    """
    # (1, 2, 1) -> (2, 1)
    effective_shape = get_effective_shape(tensor_shape)
    for_loop_statements = [
        f"{INDENTATION * i}for (var i{i} = 0; i{i} < {dim}; i{i}++) {{"
        for i, dim in enumerate(effective_shape)
    ]
    # Assume lhs_dim is `1` and lhs shape is `(2,)`. E.g. "add.in[1]"
    # Assume rhs shape is (2,1). E.g. "input_layer[2][1]"
    # add.in[i0] <== input_layer[i0][i1];
    lhs_indices = "".join([f"[i{i}]" for i in range(lhs_dim)])
    rhs_indices = "".join([f"[i{i}]" for i in range(rhs_dim)])
    assignment = f"{INDENTATION * (len(effective_shape))}{lhs}{lhs_indices} <== {rhs}{rhs_indices};"
    for_loop_closing_brackets = [f"{INDENTATION * i}}}" for i in range(len(effective_shape)-1, -1, -1)]
    return for_loop_statements + [assignment] + for_loop_closing_brackets

