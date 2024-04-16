import argparse
import os
from pathlib import Path
from typing import Type

import torch
import torch.nn as nn


import sys
# add .. to the PYTHONPATH
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent.parent))
from onnx2circom import torch_model_to_onnx, onnx_to_circom

#
# Test
#

def run_torch_model(model_type: Type[nn.Module], data: torch.Tensor) -> torch.Tensor:
    model = model_type()
    output_torch = model.forward(data)
    # untuple the result: if the result is [x], return x
    shape = output_torch.shape
    # if it is a 0-d tensor, return it directly
    if len(shape) == 0:
        return output_torch
    # if it is a 1-d tensor with one element, return the element directly
    elif len(shape) == 1 and shape[0] == 1:
        return output_torch[0]
    else:
        return output_torch


def test():
    data = torch.tensor([10, 40, 50], dtype = torch.float32).reshape(1, -1, 1)
    onnx_path = Path('trytry.onnx')
    keras_path = onnx_path.parent / f"{onnx_path.stem}.keras"
    circom_path = keras_path.parent / f"{keras_path.stem}.circom"

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

        def forward(self, x):
            return torch.sum(x)

    print("Running torch model...")
    output_torch = run_torch_model(MyModel, data)
    print("!@# output_torch=", output_torch)

    print("Transforming torch model to onnx...")
    torch_model_to_onnx(MyModel, data, onnx_path)
    onnx_to_circom(onnx_path, circom_path)

    # print("Running keras model...")
    # # output_keras = torch.tensor(19.8070)
    # output_keras = run_keras_model(keras_path, data)
    # print("!@# output_keras=", output_keras)

    # assert torch.allclose(output_torch, output_keras, atol=1e-6), "The output of torch model and keras model are different."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_file', type=str, help='onnx model path')
    parser.add_argument('-o', '--out_dir', type=str, help='output directory for circom', default=None)
    args = parser.parse_args()
    onnx_path = Path(args.onnx_file)
    if args.out_dir is not None:
        circom_dir = Path(args.out_dir)
    else:
        circom_dir = onnx_path.parent
    circom_path = circom_dir / f"{onnx_path.stem}.circom"
    onnx_to_circom(onnx_path, circom_path)
    # Compiling with circom compiler
    os.system(f"circom {circom_path}")


if __name__ == "__main__":
    main()
