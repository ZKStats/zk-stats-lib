import os
import typing

from .circom import Template
from .model import Layer, Model


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
    TFEqual
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
    TFEqual
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
    # print('\n\n\n\n\n\n inputs transpiler: ', inputs)
    # print('\n\n\n\n\n')
    input_0 = inputs[0]
    input_0_shape = get_effective_shape(input_0.shape)
    # If the input is a scalar, num elements in the input tensor is 1
    if len(input_0_shape) == 0:
        num_elements_in_input_0 = 1
    # Else, the number of elements in the input tensor is the first dimension
    # E.g. input_0.shape = (1, 2, 1), input_0_shape=(2, 1), num_elements_in_input_0=2
    else:
        num_elements_in_input_0 = input_0_shape[0]

    def is_in_ops(op_name: str, op_classes: list[object]) -> bool:
        return op_name in map(lambda x: x.__name__, op_classes)

    if is_in_ops(layer.op, [TFLog]):
        return {'e': 2, 'nInputs': num_elements_in_input_0}
    if is_in_ops(layer.op, [TFReduceSum, TFReduceMean]):
        return {'nInputs': num_elements_in_input_0}
    if is_in_ops(layer.op, [TFAdd, TFSub, TFMul, TFDiv]):
        if len(inputs)==2:
            input_1 = inputs[1]
            input_1_shape = get_effective_shape(input_1.shape)
            if len(input_1_shape) == 0:
                num_elements_in_input_1 = 1
            # Else, the number of elements in the input tensor is the first dimension
            # E.g. input_0.shape = (1, 2, 1), input_0_shape=(2, 1), num_elements_in_input_0=2
            else:
                num_elements_in_input_1 = input_1_shape[0]
            return {'nElements': max(num_elements_in_input_0, num_elements_in_input_1)}
        return {'nElements': num_elements_in_input_0}
    return {}


def transpile(templates: dict[str, Template], filename: str, output_dir: str = 'output', raw: bool = False, dec: int = 0) -> tuple[list[str], list[str]]:
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
    circom_input_names = [input.name for input in model.model_inputs]
    circom_output_names = [output.name for output in model.model_outputs]

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

            # Handle right hand side when it's keras tensor
            input_name = _input.name
            input_shape = _input.shape
            if _input.is_constant:
                # If this input is a constant, use the value of the constant directly as the right hand side
                from_component_name = None
                # Scale the float value by 10^dec
                scaled = int(_input.value * 10 ** dec)
                from_component_signal_name = str(scaled)
                rhs_dim = 0
            elif model.is_model_input(input_name) is True:
                # If this input is from `input_layer`, use the original tensor name for it
                from_component_name = None
                from_component_signal_name = input_name
                rhs_dim = len(model_input_name_to_shape[input_name])
            else:
            # Get the output name in the corresponding component
                from_component_layer = model.get_component_from_output_name(input_name)
                if from_component_layer is None:
                    raise ValueError(f"Output {input_name} not found in any component")
                from_component_name = from_component_layer.name
                # get the index of the output tensor in the component
                from_component_output_tensor_names = [output.name for output in from_component_layer.outputs]
                from_component_output_index = from_component_output_tensor_names.index(input_name)
                # Sanity check
                if from_component_output_index == -1:
                    raise ValueError(f"Output {input_name} not found in from_component {from_component_name}")

                # Use the index to get the component output name from template
                from_component_template = templates[from_component_layer.op]
                from_component_signal_name = from_component_template.output_names[from_component_output_index]
                rhs_dim = from_component_template.output_dims[from_component_output_index]

            # either "{component_name}." if input is from a component or "" if input is from model input
            from_component_prefix = f"{from_component_name}." if from_component_name is not None else ""
            rhs = f"{from_component_prefix}{from_component_signal_name}"
            constraints_lines = generate_constraints(lhs, lhs_dim, rhs, rhs_dim, input_shape)
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
        # when it's scalar
        if output.shape ==():
            output.shape = (1,1)
        constraints_lines = generate_constraints(lhs, lhs_dim, rhs, rhs_dim, output.shape)
        # print('output conssss: ', constraints_lines)
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

    return circom_input_names, circom_output_names


def parse_args(template_args: typing.List[str], args: typing.Dict[str, typing.Any]) -> str:
    if len(template_args) != len(args):
        raise ValueError('Number of template arguments does not match the number of arguments')
    if len(template_args) == 0:
        return ''
    args_str = '{'+'}, {'.join(template_args)+'}'
    return args_str.format(**args)


def get_effective_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
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
    [
        "for (var i0 = 0; i0 < shape[0]; i++) {",
        "   for (var i1 = 0; i < shape[1]; i++) {",
        "       lhs[i0][i1] <== rhs[i0][i1];",
        "   }",
        "}",
    ]
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
