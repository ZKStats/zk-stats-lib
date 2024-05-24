from pathlib import Path
from typing import Type

import torch
import torch.nn as nn
import json
import os


from zkstats.onnx2circom import onnx_to_circom
from zkstats.arithc_to_bristol import parse_arithc_json
from zkstats.backends.mpspdz import generate_mpspdz_circuit, generate_mpspdz_inputs_for_party, run_mpspdz_circuit, tensors_to_circom_mpspdz_inputs


CIRCOM_2_ARITHC_PROJECT_ROOT = Path('/path/to/circom-2-arithc-project-root')
MP_SPDZ_PROJECT_ROOT = Path('/path/to/mp-spdz-project-root')



def compile_and_run_mpspdz(model_type: Type[nn.Module], data: torch.Tensor, tmp_path: Path):
    # output_path = tmp_path
    # Don't use tmp_path for now for easier debugging
    # So you should see all generated files in `output_path`
    output_path = Path(__file__).parent / 'out'
    print(f"!@# {output_path=}")
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / 'model.onnx'
    model_name = onnx_path.stem
    keras_path = onnx_path.parent / f"{model_name}.keras"
    assert onnx_path.stem == keras_path.stem
    print(f"!@# {keras_path=}")
    circom_path = output_path / f"{model_name}.circom"
    print(f"!@# {circom_path=}")

    print("Transforming torch model to onnx...")
    torch_model_to_onnx(model_type, data, onnx_path)
    assert onnx_path.exists() is True, f"The output file {onnx_path} does not exist."
    print("Transforming onnx model to circom...")
    circom_input_names, circom_output_names = onnx_to_circom(onnx_path, circom_path)
    assert circom_path.exists() is True, f"The output file {circom_path} does not exist."

    arithc_path = output_path / f"{model_name}.json"
    # Compile with circom-2-arithc compiler
    code = os.system(f"cd {CIRCOM_2_ARITHC_PROJECT_ROOT} && ./target/release/circom --input {circom_path} --output {output_path}")
    if code != 0:
        raise ValueError(f"Failed to compile circom. Error code: {code}")
    arithc_path = output_path / f"{model_name}.json"
    assert arithc_path.exists() is True, f"The output file {arithc_path} does not exist."

    print("!@# circom_path=", circom_path)
    print("!@# arithc_path=", arithc_path)

    bristol_path = output_path / f"{model_name}.txt"
    circuit_info_path = output_path / f"{model_name}.circuit_info.json"
    parse_arithc_json(arithc_path, bristol_path, circuit_info_path)
    assert bristol_path.exists() is True, f"The output file {bristol_path} does not exist."
    assert circuit_info_path.exists() is True, f"The output file {circuit_info_path} does not exist."
    print("!@# bristol_path=", bristol_path)
    print("!@# circuit_info_path=", circuit_info_path)

    # Mark every input as from 0
    with open(circuit_info_path, 'r') as f:
        circuit_info = json.load(f)
    input_name_to_wire_index = circuit_info['input_name_to_wire_index']
    output_name_to_wire_index = circuit_info['output_name_to_wire_index']
    input_names = list(input_name_to_wire_index.keys())
    output_names = list(output_name_to_wire_index.keys())
    print("!@# input_names=", input_names)

    num_parties = 2
    # TODO: Now we assume all inputs are from party 0 for convenience.
    # If we want to support inputs from both parties, we need to change the code here.
    # We also need config file for mpcstats library to specify which tensor is from which party,
    # and translate these information to the mpc_settings.json

    # Prepare the mpc_settings.json
    mpc_settings_path = output_path / f"{model_name}.mpc_settings.json"
    mpc_settings_path.parent.mkdir(parents=True, exist_ok=True)
    # party 0 is alice, having input a, and output a_add_b, a_mul_c are revealed to
    # party 1 is bob, having input b, and output a_add_b, a_mul_c are revealed to
    # [
    #     {
    #         "name": "alice",
    #         "inputs": ["a"],
    #         "outputs": ["a_add_b", "a_mul_c"]
    #     },
    #     {
    #         "name": "bob",
    #         "inputs": ["b"],
    #         "outputs": ["a_add_b", "a_mul_c"]
    #     }
    # ]
    with open(mpc_settings_path, 'w') as f:
        json.dump([
            {
                "name": "alice",
                "inputs": input_names,  # All inputs are from party 0
                "outputs": output_names,  # Party 0 can see all outputs
            },
            {
                "name": "bob",
                "inputs": [],  # No input is from party 1
                "outputs": output_names,  # Party 1 can see all outputs
            }
        ], f, indent=4)

    print("!@# mpc_settings_path=", mpc_settings_path)

    # Prepare data for party 0
    # TODO: Should be changed if we want to support inputs from different parties
    tensor_list = [data]
    input_paths = [output_path / f"{model_name}_party_{i}.inputs.json" for i in range(num_parties)]
    input_paths[0].parent.mkdir(parents=True, exist_ok=True)
    # preprocess tensors from (1, N, 1) to (N, 1)
    tensor_list_squeezed = [tensor.squeeze(0) for tensor in tensor_list]
    circom_inputs = tensors_to_circom_mpspdz_inputs(circom_input_names, tensor_list_squeezed)
    # postprocess tensor values from floats to integers
    # TODO: if we want to support float inputs, we need to change the code here
    circom_int_inputs = {
        k: int(v)
        for k, v in circom_inputs.items()
    }
    with open(input_paths[0], 'w') as f:
        json.dump(circom_int_inputs, f, indent=4)
    # input 1 is empty
    with open(input_paths[1], 'w') as f:
        json.dump({}, f)

    # Run MP-SPDZ
    mpspdz_circuit_path = generate_mpspdz_circuit(MP_SPDZ_PROJECT_ROOT, bristol_path, circuit_info_path, mpc_settings_path)
    for party in range(num_parties):
        input_json_for_party_path = output_path / f"{model_name}_party_{party}.inputs.json"
        mpspdz_input_path = generate_mpspdz_inputs_for_party(
            MP_SPDZ_PROJECT_ROOT,
            party,
            input_json_for_party_path,
            circuit_info_path,
            mpc_settings_path,
        )
        print(f"Party {party} input path: {mpspdz_input_path}")
    print(f"Running mp-spdz circuit {mpspdz_circuit_path}...")
    # E.g. mpspdz_output = {'keras_tensor_3': tensor([[[2]]]), 'output_2': tensor(1)}
    output_name_to_tensor = run_mpspdz_circuit(MP_SPDZ_PROJECT_ROOT, mpspdz_circuit_path)
    # Return the tensors in the order they passed to torch, and reshape them from (1, N, 1) back to (N,)
    output_tensor_list = [output_name_to_tensor[name].reshape(-1) for name in circom_output_names]
    return output_tensor_list


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
        return output_torch.reshape(-1)


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

