import os
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn


ONNX_2_CIRCOM_PROJECT_ROOT = Path(__file__).parent
ONNX_2_KERAS_PROJECT_ROOT = ONNX_2_CIRCOM_PROJECT_ROOT / "onnx2keras"

CIRCOMLIB_ML_CIRCUITS_PATH = ONNX_2_CIRCOM_PROJECT_ROOT / "circomlib-ml" / "circuits"


def torch_model_to_onnx(model_type: Type[nn.Module], data: torch.Tensor, output_onnx_path: Path):
    model = model_type()
    input_shape = data.shape
    print("!@# data: ", data)
    print("!@# input_shape: ", input_shape)

    torch.onnx.export(model,               # model being run
                        data,                   # model input (or a tuple for multiple inputs)
                        output_onnx_path,            # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})


def onnx_to_keras(onnx_path: Path, generated_keras_path: Path):
    onnx2circom_path = ONNX_2_KERAS_PROJECT_ROOT.parent
    executable = ONNX_2_KERAS_PROJECT_ROOT / "converter.py"
    res = os.system(f"PYTHONPATH={onnx2circom_path} python {executable} --weights {onnx_path} --outpath {generated_keras_path.parent} --formats 'keras'")
    if res != 0:
        raise ValueError(f"Failed to convert onnx to keras. Error code: {res}")


def keras_to_circom(keras_path: Path, generated_circom_path: Path):
    generated_dir = generated_circom_path.parent
    from keras2circom.keras2circom import circom, transpiler
    # Ref: https://github.com/JernKunpittaya/keras2circom/blob/42dc97e4ce0543dde68b37e9b220a29bf88be84d/main.py#L21
    circom.dir_parse(
        CIRCOMLIB_ML_CIRCUITS_PATH,
        # TODO: should we skip them?
        skips=['util.circom', 'circomlib-matrix', 'circomlib', 'crypto'],
    )
    # transpiler.transpile(args['<model.h5>'], args['--output'], args['--raw'], args['--decimals'])
    transpiler.transpile(
        str(keras_path),
        str(generated_dir),
        raw=False,  # Seems not used
        dec=18,  # TODO: decimal. Now it's the default value. We can pick up a suitable value
    )
    generated_circom_original = generated_dir / f"circuit.circom"
    # Copy the generated circom file to the target path
    generated_circom_path.write_text(generated_circom_original.read_text())


def onnx_to_circom(onnx_path: Path, generated_circom_path: Path):
    keras_path = onnx_path.parent / f"{onnx_path.stem}.keras"
    print("Transforming onnx model to keras...")
    onnx_to_keras(onnx_path, keras_path)
    print("Transforming keras model to circom...")
    keras_to_circom(keras_path, generated_circom_path)
