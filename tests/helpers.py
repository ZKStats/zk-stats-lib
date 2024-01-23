import json
from typing import Type
from pathlib import Path

import torch

from zkstats.core import prover_gen_settings, verifier_setup, prover_gen_proof, verifier_verify
from zkstats.computation import IModel, IsResultPrecise


def compute(basepath: Path, data: list[torch.Tensor], model: Type[IModel]) -> IsResultPrecise:
    comb_data_path = basepath / "comb_data.json"
    model_path = basepath / "model.onnx"
    settings_path = basepath / "settings.json"
    witness_path = basepath / "witness.json"
    compiled_model_path = basepath / "model.compiled"
    proof_path = basepath / "model.proof"
    pk_path = basepath / "model.pk"
    vk_path = basepath / "model.vk"
    data_paths = [basepath / f"data_{i}.json" for i in range(len(data))]

    for i, d in enumerate(data):
        filename = data_paths[i]
        data_json = {"input_data": [d.tolist()]}
        with open(filename, "w") as f:
            f.write(json.dumps(data_json))

    prover_gen_settings(
        data_path_array=[str(i) for i in data_paths],
        comb_data_path=str(comb_data_path),
        prover_model=model,
        prover_model_path=str(model_path),
        scale="default",
        mode="resources",
        settings_path=str(settings_path),
    )
    verifier_setup(
        str(model_path),
        str(compiled_model_path),
        str(settings_path),
        str(vk_path),
        str(pk_path),
    )
    prover_gen_proof(
        str(model_path),
        str(comb_data_path),
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
    )
