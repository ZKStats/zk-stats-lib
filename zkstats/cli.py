import os
import click

from .core import prover_gen_proof, prover_setup, load_model

os.makedirs(os.path.dirname('shared/'), exist_ok=True)
os.makedirs(os.path.dirname('prover/'), exist_ok=True)
verifier_model_path = os.path.join('shared/verifier.onnx')
prover_model_path = os.path.join('prover/prover.onnx')
verifier_compiled_model_path = os.path.join('shared/verifier.compiled')
prover_compiled_model_path = os.path.join('prover/prover.compiled')
pk_path = os.path.join('shared/test.pk')
vk_path = os.path.join('shared/test.vk')
proof_path = os.path.join('shared/test.pf')
settings_path = os.path.join('shared/settings.json')
srs_path = os.path.join('shared/kzg.srs')
witness_path = os.path.join('prover/witness.json')
# this is private to prover since it contains actual data
data_path = os.path.join('data.json')
comb_data_path = os.path.join('prover/comb_data.json')


@click.group()
def cli():
    pass

@click.command()
@click.argument('model_path')
def prove(model_path: str):
    click.echo(f"Hello, {model_path}!")

    prover_model = load_model(model_path)
    print("!@# prover_model=", prover_model)
    prover_setup(
        [data_path],
        comb_data_path,
        prover_model,
        prover_model_path,
        prover_compiled_model_path,
        "default",
        "resources",
        settings_path,
        srs_path,
        vk_path,
        pk_path,
    )
    prover_gen_proof(
        prover_model_path,
        comb_data_path,
        witness_path,
        prover_compiled_model_path,
        settings_path,
        proof_path,
        pk_path,
        srs_path,
    )


@click.command()
def verify():
    click.echo(f"Hello, verify!")


def main():
    cli()


# Register commands
cli.add_command(prove)
cli.add_command(verify)


if __name__ == "__main__":
    main()
