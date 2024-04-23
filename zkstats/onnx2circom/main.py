import argparse
import os
import sys
from pathlib import Path


# add .. to the PYTHONPATH to make the import `onnx2circom` work
file_path = Path(__file__).resolve()
sys.path.append(str(file_path.parent.parent))

from onnx2circom import onnx_to_circom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_file', type=str, help='onnx model path')
    parser.add_argument('-o', '--circom_path', type=str, help='output path for circom', default=None)
    args = parser.parse_args()
    onnx_path = Path(args.onnx_file)
    if args.circom_path is not None:
        circom_path = Path(args.circom_path)
    else:
        circom_path = onnx_path.parent / f"{onnx_path.stem}.circom"
    onnx_to_circom(onnx_path, circom_path)
    # Compiling with circom compiler
    code = os.system(f"circom {circom_path}")
    if code != 0:
        raise ValueError(f"Failed to compile circom. Error code: {code}")


if __name__ == "__main__":
    main()
