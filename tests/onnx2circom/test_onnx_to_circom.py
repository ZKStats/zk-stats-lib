import json
import os
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn

from zkstats.onnx2circom import onnx_to_circom
from zkstats.arithc_to_bristol import parse_arithc_json
from zkstats.backends.mpspdz import generate_mpspdz_circuit, run_mpspdz_circuit

from .utils import run_torch_model, torch_model_to_onnx


# NOTE: Change the path to your own path
CIRCOM_2_ARITHC_PROJECT_ROOT = Path('/Users/jernkun/Desktop/circom-2-arithc')
MP_SPDZ_PROJECT_ROOT = Path('/Users/jernkun/Desktop/MP-SPDZ')


def test_onnx_to_circom(tmp_path):
    # data = torch.tensor(
    #     [10, 40, 50],
    #     dtype = torch.float32,
    # ).reshape(1, -1, 1)
    data = torch.tensor(
        [32],
        dtype = torch.float32,
    ).reshape(1, -1, 1)

    class Model(nn.Module):
        def forward(self, x):
            # return torch.sum(x)
            # return torch.mean(x) + torch.sum(x)
            # return torch.mean(x) + torch.mean(x)
            # return torch.mean(x)
            # return torch.sum(x)
            return torch.log(x)

    compile_and_check(Model, data, tmp_path)


def compile_and_check(model_type: Type[nn.Module], data: torch.Tensor, tmp_path: Path):
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

    print("Running torch model...")
    output_torch = run_torch_model(model_type, data)
    print("!@# output_torch=", output_torch)

    print("Transforming torch model to onnx...")
    torch_model_to_onnx(model_type, data, onnx_path)
    assert onnx_path.exists() is True, f"The output file {onnx_path} does not exist."
    # onnx_to_keras(onnx_path, keras_path)
    print("Transforming onnx model to circom...")
    onnx_to_circom(onnx_path, circom_path)
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
    input_names = list(input_name_to_wire_index.keys())
    print("!@# input_names=", input_names)

    # TODO: This should come from users. Here we just set up a config json
    # for convenience (which input is from which party). Now just put every input to party 0.
    # Assume the input data is a 1-d tensor
    user_config_path = MP_SPDZ_PROJECT_ROOT / f"Configs/{model_name}.json"
    with open(user_config_path, 'w') as f:
        json.dump({"inputs_from": {
            "0": input_names,
        }}, f, indent=4)

    print("!@# user_config_path=", user_config_path)

    # Prepare data for party 0
    data_list = data.reshape(-1)
    input_0_path = MP_SPDZ_PROJECT_ROOT / 'Player-Data/Input-P0-0'
    with open(input_0_path, 'w') as f:
        # TODO: change int to float
        f.write(' '.join([str(int(x)) for x in data_list.tolist()]))

    # Run MP-SPDZ
    mpc_circuit_path = generate_mpspdz_circuit(MP_SPDZ_PROJECT_ROOT, bristol_path, circuit_info_path, user_config_path)
    print(f"Running mp-spdz circuit {mpc_circuit_path}...")
    run_mpspdz_circuit(MP_SPDZ_PROJECT_ROOT, mpc_circuit_path)
    # TODO: parse output from MP-SPDZ and compare with torch output
