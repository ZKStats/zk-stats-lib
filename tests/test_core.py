import json

import torch

from zkstats.core import generate_data_commitment, prover_gen_settings, _preprocess_data_file_to_json, verifier_define_calculation
from zkstats.computation import computation_to_model

from .helpers import data_to_json_file, compute


def test_get_data_commitment_maps(tmp_path, column_0, column_1, scales):
    data_path = tmp_path / "data.json"
    data_commitment_path = tmp_path / "commitments.json"
    # data_json is a mapping[column_name, column_data]
    # {
    #     "columns_0": [1, 2, 3, 4, 5],
    #     "columns_1": [6, 7, 8, 9, 10],
    # }
    data_json = data_to_json_file(data_path, [column_0, column_1])
    # data_commitment is a mapping[scale -> mapping[column_name, commitment_hex]]
    # {
    #     scale_0: {
    #         "columns_0": "0x...",
    #         "columns_1": "0x...",
    #     },
    #     scale_1: {
    #         "columns_0": "0x...",
    #         "columns_1": "0x...",
    #     }
    # }

    generate_data_commitment(data_path, scales, data_commitment_path)
    with open(data_commitment_path, "r") as f:
        data_commitment = json.load(f)

    assert len(data_commitment) == len(scales)
    for scale, commitment_map in data_commitment.items():
        assert int(scale) in scales
        assert len(commitment_map) == len(data_json)
        for column_name, commitment_hex in commitment_map.items():
            assert column_name in data_json
            # Check if the commitment is a valid hex number
            int(commitment_hex, 16)


def test_get_data_commitment_maps_hardcoded(tmp_path):
    """
    This test is to check if the data commitment scheme doesn't change
    """
    data_path = tmp_path / "data.json"
    data_commitment_path = tmp_path / "commitments.json"
    column_0 = torch.tensor([3.0, 4.5, 1.0, 2.0, 7.5, 6.4, 5.5])
    column_1 = torch.tensor([2.7, 3.3, 1.1, 2.2, 3.8, 8.2, 4.4])
    data_to_json_file(data_path, [column_0, column_1])
    scales = [2, 3]
    generate_data_commitment(data_path, scales, data_commitment_path)
    with open(data_commitment_path, "r") as f:
        data_commitment = json.load(f)
    # expected = {"2": {'columns_0': '0x28b5eeb5aeee399c8c50c5b323def9a1aec1deee5b9ae193463d4f9b8893a9a3', 'columns_1': '0x0523c85a86dddd810418e8376ce6d9d21b1b7363764c9c31b575b8ffbad82987'}, "3": {'columns_0': '0x0a2906522d3f902ff4a63ee8aed4d2eaec0b14f71c51eb9557bd693a4e7d77ad', 'columns_1': '0x2dac7fee1efb9eb955f52494a26a3fba6d1fa28cc819e598cb0af31a47b29d08'}}
    expected = {"2": {'columns_0': 'a3a993889b4f3d4693e19a5beedec1aea1f9de23b3c5508c9c39eeaeb5eeb528', 'columns_1': '8729d8baffb875b5319c4c7663731b1bd2d9e66c37e8180481dddd865ac82305'}, "3": {'columns_0': 'ad777d4e3a69bd5795eb511cf7140becead2d4aee83ea6f42f903f2d5206290a', 'columns_1': '089db2471af30acb98e519c88ca21f6dba3f6aa29424f555b99efb1eee7fac2d'}}
    assert data_commitment == expected


def test_integration_select_partial_columns(tmp_path, column_0, column_1, error, scales):
    data_path = tmp_path / "data.json"
    data_json = data_to_json_file(data_path, [column_0, column_1])
    columns = list(data_json.keys())
    assert len(columns) == 2
    # Select only the first column from two columns
    selected_columns = [columns[0]]

    def simple_computation(state, x):
        return state.mean(x[0])
    precal_witness_path = tmp_path / "precal_witness_path.json"
    _, model = computation_to_model(simple_computation,precal_witness_path, True, error)
    # gen settings, setup, prove, verify
    compute(tmp_path, [column_0, column_1], model, scales, selected_columns)


def test_csv_data(tmp_path, column_0, column_1, error, scales):
    data_json_path = tmp_path / "data.csv"
    data_csv_path = tmp_path / "data.csv"
    data_json = data_to_json_file(data_json_path, [column_0, column_1])
    json_file_to_csv(data_json_path, data_csv_path)

    selected_columns = list(data_json.keys())

    def simple_computation(state, x):
        return state.mean(x[0])

    sel_data_path = tmp_path / "comb_data.json"
    model_path = tmp_path / "model.onnx"
    settings_path = tmp_path / "settings.json"
    data_commitment_path = tmp_path / "commitments.json"

    # Test: `generate_data_commitment` works with csv
    generate_data_commitment(data_csv_path, scales, data_commitment_path)

    # Test: `prover_gen_settings` works with csv
    _, model_for_proving = computation_to_model(simple_computation, error)
    prover_gen_settings(
        data_path=data_csv_path,
        selected_columns=selected_columns,
        sel_data_path=str(sel_data_path),
        prover_model=model_for_proving,
        prover_model_path=str(model_path),
        scale=scales,
        mode="resources",
        settings_path=str(settings_path),
    )

    # Test: `prover_gen_settings` works with csv
    # Instantiate the model for verification since the state of `model_for_proving` is changed after `prover_gen_settings`
    _, model_for_verification = computation_to_model(simple_computation, error)
    verifier_define_calculation(data_csv_path, selected_columns, str(sel_data_path), model_for_verification, str(model_path))


def json_file_to_csv(data_json_path, data_csv_path):
    with open(data_json_path, "r") as f:
        data_from_json = json.load(f)
    # Generate csv file from json
    column_names = list(data_from_json.keys())
    len_columns = len(data_from_json[column_names[0]])
    for column in column_names:
        assert len(data_from_json[column]) == len_columns, "All columns should have the same length"
    rows = [
        [str(data_from_json[column][i]) for column in column_names]
        for i in range(len_columns)
    ]
    with open(data_csv_path, "w") as f:
        f.write(",".join(column_names) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")


def test__preprocess_data_file_to_json(tmp_path, column_0, column_1):
    data_json_path = tmp_path / "data.json"
    data_from_json = data_to_json_file(data_json_path, [column_0, column_1])

    # Test: csv can be converted to json
    # 1. Generate a csv file from json
    data_csv_path = tmp_path / "data.csv"
    json_file_to_csv(data_json_path, data_csv_path)
    # 2. Convert csv to json
    data_from_csv_json_path = tmp_path / "data_from_csv.json"
    _preprocess_data_file_to_json(data_csv_path, data_from_csv_json_path)
    with open(data_from_csv_json_path, "r") as f:
        data_from_csv = json.load(f)
    # 3. Compare the two json files
    assert data_from_csv == data_from_json

    # Test: this function can also handle json format by just copying the file
    new_data_json_path = tmp_path / "new_data.json"
    _preprocess_data_file_to_json(data_json_path, new_data_json_path)
    with open(new_data_json_path, "r") as f:
        new_data_from_json = json.load(f)
    assert new_data_from_json == data_from_json
