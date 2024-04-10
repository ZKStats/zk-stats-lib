import json
import os
import sys
from typing import Type
import importlib.util

import click
import torch

from .core import prover_gen_proof, prover_gen_settings, setup, verifier_verify, generate_data_commitment
from .computation import computation_to_model

cwd = os.getcwd()
# TODO: Should make this configurable
output_dir = f"{cwd}/out"
os.makedirs(output_dir, exist_ok=True)
model_onnx_path = f"{output_dir}/model.onnx"
compiled_model_path = f"{output_dir}/model.compiled"
pk_path = f"{output_dir}/model.pk"
vk_path = f"{output_dir}/model.vk"
proof_path = f"{output_dir}/model.pf"
settings_path = f"{output_dir}/settings.json"
witness_path = f"{output_dir}/witness.json"
comb_data_path = f"{output_dir}/comb_data.json"
data_commitment_path = f"{output_dir}/data_commitment.json"

default_possible_scales = list(range(20))


@click.group()
def cli():
    pass


@click.command()
@click.argument('computation_path')
@click.argument('data_path')
def prove(computation_path: str, data_path: str):
    computation = load_computation(computation_path)
    _, model = computation_to_model(computation)
    generate_data_commitment(data_path, default_possible_scales, data_commitment_path)
    with open(data_commitment_path) as f:
        data_commitment = json.load(f)
    # By default select all columns
    selected_columns = list(data_commitment[str(default_possible_scales[0])].keys())
    prover_gen_settings(
        data_path,
        selected_columns,
        comb_data_path,
        model,
        model_onnx_path,
        "default",
        "resources",
        settings_path,
    )
    setup(
        model_onnx_path,
        compiled_model_path,
        settings_path,
        vk_path,
        pk_path,
    )
    print("Finished setup")
    prover_gen_proof(
        model_onnx_path,
        comb_data_path,
        witness_path,
        compiled_model_path,
        settings_path,
        proof_path,
        pk_path,
    )
    print("Finished generating proof")
    verifier_verify(proof_path, settings_path, vk_path, selected_columns, data_commitment_path)
    print("Proof path:", proof_path)
    print("Settings path:", settings_path)
    print("Verification key path:", vk_path)
    print("Commitment maps path:", data_commitment_path)


@click.command()
def verify():
    # Load commitment maps
    with open(data_commitment_path, "r") as f:
        data_commitment = json.load(f)
    # By default select all columns
    selected_columns = list(data_commitment[str(default_possible_scales[0])].keys())
    verifier_verify(proof_path, settings_path, vk_path, selected_columns, data_commitment_path)


@click.command()
@click.argument('data_path')
@click.argument('scale_str')
def commit(data_path: str, scale_str: str):
    """
    Now we just assume the data is a list of floats. We should be able to
    """
    scale = int(scale_str)
    generate_data_commitment(data_path, [scale], data_commitment_path)
    with open(data_commitment_path) as f:
        data_commitment = json.load(f)
    print("Commitment maps:", data_commitment)


def main():
    cli()


def load_computation(module_path: str) -> Type[torch.nn.Module]:
    """
    Load a model from a Python module.
    """
    # FIXME: This is unsafe since malicious code can be executed

    model_name = "computation"
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        return getattr(module, model_name)
    except AttributeError:
        raise ImportError(f"{model_name=} does not exist in {module_name=}")


# Register commands
cli.add_command(prove)
cli.add_command(verify)
cli.add_command(commit)


if __name__ == "__main__":
    main()
