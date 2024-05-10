import json
from typing import Type, Sequence, Optional, Callable
from pathlib import Path

import torch

from zkstats.core import create_dummy,prover_gen_settings, setup, prover_gen_proof, verifier_verify, generate_data_commitment, verifier_define_calculation
from zkstats.computation import IModel, State, computation_to_model


DEFAULT_POSSIBLE_SCALES = list(range(20))

# Error tolerance between circuit and python implementation
ERROR_CIRCUIT_DEFAULT = 0.01
ERROR_CIRCUIT_STRICT = 0.0001
ERROR_CIRCUIT_RELAXED = 0.1


def data_to_file(data_path: Path, data: list[torch.Tensor]) -> dict[str, list]:
    column_names = [f"columns_{i}" for i in range(len(data))]
    column_to_data = {
        column: d.tolist()
        for column, d in zip(column_names, data)
    }
    print('columnnnn: ', column_to_data)
    with open(data_path, "w") as f:
        json.dump(column_to_data, f)
    return column_to_data

TComputation = Callable[[State, list[torch.Tensor]], torch.Tensor]
def compute(
    basepath: Path,
    data: list[torch.Tensor],
    model: Type[IModel],
    # computation: TComputation,
    scales_params: Optional[Sequence[int]] = None,
    selected_columns_params: Optional[list[str]] = None,
    # error:float = 1.0
) -> None:
    sel_data_path = basepath / "comb_data.json"
    model_path = basepath / "model.onnx"
    settings_path = basepath / "settings.json"
    witness_path = basepath / "witness.json"
    compiled_model_path = basepath / "model.compiled"
    proof_path = basepath / "model.proof"
    pk_path = basepath / "model.pk"
    vk_path = basepath / "model.vk"
    data_path = basepath / "data.json"
    data_commitment_path = basepath / "commitments.json"

    # verifier_model_path = basepath / "verifier_model.onnx"
    # verifier_compiled_model_path = basepath / "verifier_model.compiled"
    # prover_model_path = basepath / "prover_model.onnx"
    # prover_compiled_model_path = basepath / "prover_model.compiled"
    # precal_witness_path = basepath / "precal_witness_arr.json"
    # dummy_data_path = basepath / "dummy_data.json"
    # sel_dummy_data_path = basepath / "sel_dummy_data_path.json"

    column_to_data = data_to_file(data_path, data)
    # If selected_columns_params is None, select all columns
    if selected_columns_params is None:
        selected_columns = list(column_to_data.keys())
    else:
        selected_columns = selected_columns_params

    scales: Sequence[int] | str
    scales_for_commitments: Sequence[int]
    if scales_params is None:
        scales = 'default'
        scales_for_commitments = DEFAULT_POSSIBLE_SCALES
    else:
        scales = scales_params
        scales_for_commitments = scales_params
    # create_dummy((data_path), (dummy_data_path))
    generate_data_commitment((data_path), scales_for_commitments, (data_commitment_path))
    # _, prover_model = computation_to_model(computation, (precal_witness_path), True, error)

    prover_gen_settings((data_path), selected_columns, (sel_data_path), model, (model_path), scales, "resources", (settings_path))

    # No need, since verifier & prover share the same onnx
    # _, verifier_model = computation_to_model(computation, (precal_witness_path), False,error)
    # verifier_define_calculation((dummy_data_path), selected_columns, (sel_dummy_data_path),verifier_model, (verifier_model_path))

    setup((model_path), (compiled_model_path), (settings_path),(vk_path), (pk_path ))

    prover_gen_proof((model_path), (sel_data_path), (witness_path), (compiled_model_path), (settings_path), (proof_path), (pk_path))
    # print('slett col: ', selected_columns)
    verifier_verify((proof_path), (settings_path), (vk_path), selected_columns, (data_commitment_path))


# Error tolerance between zkstats python implementation and python statistics module
ERROR_ZKSTATS_STATISTICS = 0.0001


def assert_result(expected_result: float, circuit_result: float):
    assert abs(expected_result - circuit_result) < ERROR_ZKSTATS_STATISTICS * expected_result, f"{expected_result=} != {circuit_result=}, {ERROR_ZKSTATS_STATISTICS=}"

