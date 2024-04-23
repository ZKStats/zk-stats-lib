import json
import os
from pathlib import Path


MPSPDZ_CIRCUIT_RELATIVE_PATH = Path('Programs/Source')


def run_mpspdz_circuit(mpspdz_project_root: Path, mpspdz_circuit_path: Path, mpc_protocol: str = 'semi'):
    # Run the MP-SPDZ interpreter to interpret the arithmetic circuit
    # mpspdz_circuit_path = 'tutorial.mpc'
    assert mpspdz_circuit_path.exists(), f"The MP-SPDZ circuit file {mpspdz_circuit_path} does not exist."
    assert mpspdz_circuit_path.suffix == '.mpc', f"The MP-SPDZ circuit file {mpspdz_circuit_path} should have the extension .mpc."
    mpspdz_circuit_dir = mpspdz_project_root / MPSPDZ_CIRCUIT_RELATIVE_PATH
    # Check mpspdz_circuit_path is under mpspdz_circuit_dir
    assert mpspdz_circuit_path.parent.absolute() == mpspdz_circuit_dir.absolute(), \
        f"The MP-SPDZ circuit file {mpspdz_circuit_path} should be under {mpspdz_circuit_dir}."
    # circuit_name = 'tutorial'
    circuit_name = mpspdz_circuit_path.stem
    code = os.system(f'cd {mpspdz_project_root} && Scripts/compile-run.py -E {mpc_protocol} {circuit_name} -M')
    if code != 0:
        raise ValueError(f"Failed to run MP-SPDZ interpreter. Error code: {code}")


def generate_mpspdz_circuit(
    mpspdz_project_root: Path,
    arith_circuit_path: Path,
    circuit_info_path: Path,
    input_config_path: Path,
) -> Path:
    # {
    #   "input_name_to_wire_index": { "a": 1, "b": 0, "c": 2},
    #   "constants": {"d": {"value": 50, "wire_index": 3}},
    #   "output_name_to_wire_index": { "a_add_b": 4, "a_mul_c": 5 }
    # }
    with open(circuit_info_path, 'r') as f:
        raw = json.load(f)

    input_name_to_wire_index = {k: int(v) for k, v in raw['input_name_to_wire_index'].items()}
    constants: dict[str, dict[str, int]] = raw['constants']
    output_name_to_wire_index = {k: int(v) for k, v in raw['output_name_to_wire_index'].items()}
    # {
    #     "inputs_from": {
    #         "0": ["a", "b"],
    #         "1": ["c"]
    #     }
    # }
    with open(input_config_path, 'r') as f:
        input_config = json.load(f)
    inputs_from: dict[str, list[str]] = input_config['inputs_from']

    # Make inputs to circuit (not wires!!) from the user config
    # The inputs order will be [constant1, constant2, ..., party_0_input1, party_0_input2, ..., party_1_input1, ...]
    inputs_str_list = []
    wire_index_for_input = []
    for name, o in constants.items():
        value = o['value']
        wire_index = o['wire_index']
        wire_index_for_input.append(wire_index)
        inputs_str_list.append(f'cint({value})')
    for party, inputs in inputs_from.items():
        for name in inputs:
            wire_index = input_name_to_wire_index[name]
            wire_index_for_input.append(wire_index)
            inputs_str_list.append(f'sint.get_input_from({party})')

    #
    # Generate the circuit code
    #
    inputs_str = '[' + ', '.join(inputs_str_list) + ']'
    # For outputs, should print the actual output names, and
    # lines are ordered by actual output wire index since it's guaranteed the order
    # E.g.
    # print_ln('outputs[0]: a_add_b=%s', outputs[0].reveal())
    # print_ln('outputs[1]: a_mul_c=%s', outputs[1].reveal())
    print_outputs_str_list = [
        f"print_ln('outputs[{i}]: {output_name}=%s', outputs[{output_name_to_wire_index[output_name]}].reveal())"
        for i, output_name in enumerate(output_name_to_wire_index.keys())
    ]
    print_outputs_str = '\n'.join(print_outputs_str_list)
    circuit = f"""from circuit_arith import Circuit
circuit = Circuit('{arith_circuit_path}', {wire_index_for_input})
inputs = {inputs_str}
outputs = circuit(inputs)
# Print outputs
{print_outputs_str}
"""
    circuit_name = arith_circuit_path.stem
    out_mpc_path = mpspdz_project_root / MPSPDZ_CIRCUIT_RELATIVE_PATH / f"{circuit_name}.mpc"
    with open(out_mpc_path, 'w') as f:
        f.write(circuit)
    return out_mpc_path
