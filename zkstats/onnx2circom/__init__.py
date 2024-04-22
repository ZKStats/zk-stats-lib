from pathlib import Path
from typing import Union
import tempfile

from .onnx2keras.converter import onnx_converter
from .keras2circom.keras2circom import circom, transpiler


ONNX_2_CIRCOM_PROJECT_ROOT = Path(__file__).parent
ONNX_2_KERAS_PROJECT_ROOT = ONNX_2_CIRCOM_PROJECT_ROOT / "onnx2keras"

MPC_CIRCOM_PATH = ONNX_2_CIRCOM_PROJECT_ROOT / "mpc.circom"


def onnx_to_keras(onnx_path: Path, generated_keras_path: Path):
    # Ref: https://github.com/JernKunpittaya/onnx2keras/blob/5cea7118afd4a906e9d604b908aae92f615b4eae/converter.py#L77-L90
    onnx_converter(
        onnx_model_path=str(onnx_path),
        output_path=generated_keras_path.parent,
        target_formats=['keras'],
        need_simplify=True,  # default
        input_node_names=None,  # default
        output_node_names=None,  # default
        native_groupconv=False,  # default
        weight_quant=False,  # default
        int8_model=False,  # default
        int8_mean=[123.675, 116.28, 103.53],  # default
        int8_std=[58.395, 57.12, 57.375],  # default
        image_root=None  # default
    )


def keras_to_circom(keras_path: Path, generated_circom_path: Path):
    # Ref: https://github.com/JernKunpittaya/keras2circom/blob/42dc97e4ce0543dde68b37e9b220a29bf88be84d/main.py#L21
    # circom.dir_parse(
    #     MPC_CIRCOM_PATH,
    #     # TODO: should we skip them?
    #     skips=['util.circom', 'circomlib-matrix', 'circomlib', 'crypto'],
    # )
    circom.file_parse(MPC_CIRCOM_PATH)
    # transpiler.transpile(args['<model.h5>'], args['--output'], args['--raw'], args['--decimals'])
    # keras2circom_output_dir = Path(tempfile.mkdtemp())
    keras2circom_output_dir = generated_circom_path.parent
    transpiler.transpile(
        str(keras_path),
        str(keras2circom_output_dir),
        raw=False,  # Seems not used
        dec=18,  # TODO: decimal. Now it's the default value. We can pick up a suitable value
    )
    generated_circom_original = keras2circom_output_dir / f"circuit.circom"
    # Copy the generated circom file to the target path
    generated_circom_path.parent.mkdir(parents=True, exist_ok=True)
    generated_circom_path.write_text(generated_circom_original.read_text())


def onnx_to_circom(onnx_path_str: Union[str, Path], generated_circom_path_str: Union[str, Path]):
    onnx_path = Path(onnx_path_str)
    generated_circom_path = Path(generated_circom_path_str)
    keras_path = onnx_path.parent / f"{onnx_path.stem}.keras"
    print("Transforming onnx model to keras...")
    onnx_to_keras(onnx_path, keras_path)
    print("Transforming keras model to circom...")
    keras_to_circom(keras_path, generated_circom_path)
