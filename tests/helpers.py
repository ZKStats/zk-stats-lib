import json
from typing import Type, Sequence, Optional
from pathlib import Path

import torch

from zkstats.core import prover_gen_settings, setup, prover_gen_proof, verifier_verify, get_data_commitment_maps
from zkstats.computation import IModel


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
    with open(data_path, "w") as f:
        json.dump(column_to_data, f)
    return column_to_data


def compute(
    basepath: Path,
    data: list[torch.Tensor],
    model: Type[IModel],
    scales_params: Optional[Sequence[int]] = None,
    selected_columns_params: Optional[list[str]] = None,
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

    commitment_maps = get_data_commitment_maps(data_path, scales_for_commitments)

    prover_gen_settings(
        data_path=data_path,
        selected_columns=selected_columns,
        sel_data_path=str(sel_data_path),
        prover_model=model,
        prover_model_path=str(model_path),
        scale=scales,
        mode="resources",
        settings_path=str(settings_path),
    )

    setup(
        str(model_path),
        str(compiled_model_path),
        str(settings_path),
        str(vk_path),
        str(pk_path),
    )
    prover_gen_proof(
        str(model_path),
        str(sel_data_path),
        str(witness_path),
        str(compiled_model_path),
        str(settings_path),
        str(proof_path),
        str(pk_path),
    )
    verifier_verify(
        str(proof_path),
        str(settings_path),
        str(vk_path),
        selected_columns,
        commitment_maps,
    )


# Error tolerance between zkstats python implementation and python statistics module
ERROR_ZKSTATS_STATISTICS = 0.0001


def assert_result(expected_result: float, circuit_result: float):
    assert abs(expected_result - circuit_result) < ERROR_ZKSTATS_STATISTICS * expected_result, f"{expected_result=} != {circuit_result=}, {ERROR_ZKSTATS_STATISTICS=}"

