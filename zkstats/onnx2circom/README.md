# onnx2circom

## Submodules
- onnx2keras
- keras2circom
- circomlib-ml

## Clone and sync submodules
```bash
git submodule init
```

```bash
git submodule update
```

## Run onnx2circom
```bash
$ python3 main.py model.onnx --circom_path model.circom
```
See circom code in `model.circom`
```bash
$ ls model.circom
model.circom
```

## Import

```python
from zkstats.onnx2circom import onnx_to_circom

...

model_path = "model.onnx"
circom_path = "model.circom"
onnx_to_circom(model_path, circom_path)
```
