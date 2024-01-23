import os
import sys
from typing import Type
import importlib.util

import click
import torch

from .core import prover_gen_proof, prover_gen_settings, verifier_setup, verifier_verify, gen_data_commitment

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
srs_path = f"{output_dir}/kzg.srs"
witness_path = f"{output_dir}/witness.json"
comb_data_path = f"{output_dir}/comb_data.json"


@click.group()
def cli():
    pass


@click.command()
@click.argument('model_path')
@click.argument('data_path')
def prove(model_path: str, data_path: str):
    model = load_model(model_path)
    print("Loaded model:", model)
    prover_gen_settings(
        [data_path],
        comb_data_path,
        model,
        model_onnx_path,
        "default",
        "resources",
        settings_path,
    )
    verifier_setup(
        model_path,
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
    verifier_verify(proof_path, settings_path, vk_path)
    print("Proof path:", proof_path)
    print("Settings path:", settings_path)
    print("Verification key path:", vk_path)


@click.command()
@click.argument('proof_path')
@click.argument('settings_path')
@click.argument('vk_path')
def verify(proof_path: str, settings_path: str, vk_path: str):
    verifier_verify(proof_path, settings_path, vk_path)


@click.command()
@click.argument('data_path')
def commit(data_path: str):
    """
    Now we just assume the data is a list of floats. We should be able to
    """
    commitment = gen_data_commitment(data_path)
    print("Commitment:", hex(commitment))


def main():
    cli()


def load_model(module_path: str) -> Type[torch.nn.Module]:
    """
    Load a model from a Python module.
    """
    # FIXME: This is unsafe since malicious code can be executed

    model_name = "Model"
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        cls = getattr(module, model_name)
    except AttributeError:
        raise ImportError(f"class {model_name} does not exist in {module_name}")
    return cls


# Register commands
cli.add_command(prove)
cli.add_command(verify)
cli.add_command(commit)


if __name__ == "__main__":
    main()
