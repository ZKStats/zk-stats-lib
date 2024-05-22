from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path


class AGateType(Enum):
    ADD = 'AAdd'
    DIV = 'ADiv'
    EQ = 'AEq'
    GT = 'AGt'
    GEQ = 'AGEq'
    LT = 'ALt'
    LEQ = 'ALEq'
    MUL = 'AMul'
    NEQ = 'ANeq'
    SUB = 'ASub'


MAP_GATE_TYPE_TO_OPERATOR_STR = {
    AGateType.ADD: '+',
    AGateType.MUL: '*',
    AGateType.DIV: '/',
    AGateType.LT: '<',
    AGateType.SUB: '-',
    AGateType.EQ: '==',
    AGateType.NEQ: '!=',
    AGateType.GT: '>',
    AGateType.GEQ: '>=',
    AGateType.LEQ: '<=',
}


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
    mpc_settings_path: Path,
) -> Path:
    '''
    Generate the MP-SPDZ circuit code that can be run by MP-SPDZ.

    Steps:
    1. Read the arithmetic circuit file to get the gates
    2. Read the circuit info file to get the input/output wire mapping
    3. Read the input config file to get which party inputs should be read from
    4. Generate the MP-SPDZ from the inputs above. The code should:
        4.1. Initialize a `wires` list with input wires filled in: if a wire is a constant, fill it in directly. if a wire is an input, fill in which party this input comes from
        4.2. Translate the gates into corresponding operations in MP-SPDZ
        4.3. Print the outputs
    '''
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
    with open(mpc_settings_path, 'r') as f:
        mpc_settings = json.load(f)

    # Read number of wires from the bristol circuit file
    # A bristol circuit file looks like this:
    # 2 5
    # 3 1 1 1
    # 2 1 1
    #
    # 2 1 1 0 3 AAdd
    # 2 1 1 2 4 AMul
    # """

    # Each gate line looks like this: '2 1 1 0 3 AAdd'
    @dataclass(frozen=True)
    class Gate:
        num_inputs: int
        num_outputs: int
        gate_type: AGateType
        inputs_wires: list[int]
        output_wire: int
    with open(arith_circuit_path, 'r') as f:
        first_line = next(f)
        num_gates, num_wires = map(int, first_line.split())
        second_line = next(f)
        num_inputs = int(second_line.split()[0])
        third_line = next(f)
        num_outputs = int(third_line.split()[0])
        # Skip the next line
        next(f)

        # Read the gate lines
        gates: list[Gate] = []
        for line in f:
            line = line.split()
            num_inputs = int(line[0])
            num_outputs = int(line[1])
            inputs_wires = [int(x) for x in line[2:2+num_inputs]]
            # Support 2 inputs only for now
            assert num_inputs == 2 and num_inputs == len(inputs_wires)
            output_wires = list(map(int, line[2+num_inputs:2+num_inputs+num_outputs]))
            output_wire = output_wires[0]
            # Support 1 output only for now
            assert num_outputs == 1 and num_outputs == len(output_wires)
            gate_type = AGateType(line[2+num_inputs+num_outputs])
            gates.append(Gate(num_inputs, num_outputs, gate_type, inputs_wires, output_wire))
    assert len(gates) == num_gates

    # Make inputs to circuit (not wires!!) from the user config
    # Initialize a list `inputs` with `num_wires` with value=None
    inputs_str_list = [None] * num_wires
    print_outputs_str_list = []
    # Fill in the constants
    for name, o in constants.items():
        value = int(o['value'])
        # descaled_value = value / (10 ** scale)
        wire_index = int(o['wire_index'])
        # Sanity check
        if inputs_str_list[wire_index] is not None:
            raise ValueError(f"Wire index {wire_index} is already filled in: {inputs_str_list[wire_index]=}")
        inputs_str_list[wire_index] = f'cfix({value})'
    for party_index, party_settings in enumerate(mpc_settings):
        # Fill in the inputs from the parties
        for input_name in party_settings['inputs']:
            wire_index = int(input_name_to_wire_index[input_name])
            # Sanity check
            if inputs_str_list[wire_index] is not None:
                raise ValueError(f"Wire index {wire_index} is already filled in: {inputs_str_list[wire_index]=}")
            inputs_str_list[wire_index] = f'sfix.get_input_from({party_index})'
        # Fill in the outputs
        for output_name in party_settings['outputs']:
            wire_index = int(output_name_to_wire_index[output_name])
            print_outputs_str_list.append(
                f"print_ln_to({party_index}, 'outputs[{len(print_outputs_str_list)}]: {output_name}=%s', wires[{wire_index}].reveal_to({party_index}))"
            )


    # Replace all `None` with str `'None'`
    inputs_str_list = [x if x is not None else 'None' for x in inputs_str_list]

    #
    # Generate the circuit code
    #
    inputs_str = '[' + ', '.join(inputs_str_list) + ']'

    # Translate bristol gates to MP-SPDZ operations
    # E.g.
    # '2 1 1 0 2 AAdd' in bristol
    #   is translated to
    # 'wires[2] = wires[1] + wires[0]' in MP-SPDZ
    gates_str_list = []
    for gate in gates:
        gate_str = ''
        if gate.gate_type not in MAP_GATE_TYPE_TO_OPERATOR_STR:
            raise ValueError(f"Gate type {gate.gate_type} is not supported")
        else:
            operator_str = MAP_GATE_TYPE_TO_OPERATOR_STR[gate.gate_type]
            gate_str = f'wires[{gate.output_wire}] = wires[{gate.inputs_wires[0]}] {operator_str} wires[{gate.inputs_wires[1]}]'
        gates_str_list.append(gate_str)
    gates_str = '\n'.join(gates_str_list)

    # For outputs, should print the actual output names, and
    # lines are ordered by actual output wire index since it's guaranteed the order
    # E.g.
    # print_ln('outputs[0]: a_add_b=%s', outputs[0].reveal())
    # print_ln('outputs[1]: a_mul_c=%s', outputs[1].reveal())
    # print_outputs_str_list = [
    #     f"print_ln('outputs[{i}]: {output_name}=%s', wires[{output_name_to_wire_index[output_name]}].reveal())"
    #     for i, output_name in enumerate(output_name_to_wire_index.keys())
    # ]
    print_outputs_str = '\n'.join(print_outputs_str_list)

    circuit_code = f"""wires = {inputs_str}
{gates_str}
# Print outputs
{print_outputs_str}
"""
    circuit_name = arith_circuit_path.stem
    out_mpc_path = mpspdz_project_root / MPSPDZ_CIRCUIT_RELATIVE_PATH / f"{circuit_name}.mpc"
    with open(out_mpc_path, 'w') as f:
        f.write(circuit_code)
    return out_mpc_path


def generate_mpspdz_inputs_for_party(
    mpspdz_project_root: Path,
    party: int,
    input_json_for_party_path: Path,
    circuit_info_path: Path,
    mpc_settings_path: Path,
):
    '''
    Generate MP-SPDZ circuit inputs for a party.

    For each party, we need to translate `party_{i}.inputs.json` to an input file for MP-SPDZ according to their inputs' wire order
    - The input file format of MP-SPDZ is `input0 input1 input2 ... inputN`. Each value is separated with a space
    - This order is determined by the position (index) of the inputs in the MP-SPDZ wires
        - For example, the actual wires in the generated MP-SPDZ circuit might look like this:
            `[cfix(123), sfix.get_input_from(0), sfix.get_input_from(1), cfix(456), sfix.get_input_from(0), ...]`
            - For party `0`, its MP-SPDZ inputs file should contain two values: one is for the first `sfix.get_input_from(0)`
                and the other is for the second `sfix.get_input_from(0)`.
        - This order can be obtained by sorting the `input_name_to_wire_index` by the wire index
    '''

    # Read inputs value from user provided input files
    with open(input_json_for_party_path) as f:
        input_values_for_party_json = json.load(f)

    with open(mpc_settings_path, 'r') as f:
        mpc_settings = json.load(f)
    inputs_from: dict[str, int] = {}
    for party_index, party_settings in enumerate(mpc_settings):
        for input_name in party_settings['inputs']:
            inputs_from[input_name] = int(party_index)

    with open(circuit_info_path, 'r') as f:
        circuit_info = json.load(f)
        input_name_to_wire_index = circuit_info['input_name_to_wire_index']

    wire_to_name_sorted = sorted(input_name_to_wire_index.items(), key=lambda x: x[1])
    wire_value_in_order_for_mpsdz = []
    for wire_name, wire_index in wire_to_name_sorted:
        wire_from_party = int(inputs_from[wire_name])
        # For the current party, we only care about the inputs from itself
        if wire_from_party == party:
            wire_value = input_values_for_party_json[wire_name]
            wire_value_in_order_for_mpsdz.append(wire_value)
    # Write these ordered wire inputs for mp-spdz usage
    input_file_for_party_mpspdz = mpspdz_project_root / f"Player-Data/Input-P{party}-0"
    with open(input_file_for_party_mpspdz, 'w') as f:
        f.write(" ".join(map(str, wire_value_in_order_for_mpsdz)))
    return input_file_for_party_mpspdz
