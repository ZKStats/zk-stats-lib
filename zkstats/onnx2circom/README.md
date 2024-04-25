# onnx2circom

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

Note that this onnx2circom originally comes from two modified forked repos as follows

- https://github.com/JernKunpittaya/onnx2keras/tree/stats_onnx2keras
- https://github.com/JernKunpittaya/keras2circom/tree/stats_keras2circom

Our implementation for zkstats can make onnx2circom diverge a lot from these fork repos, so we migrate all codes here without using submodules anymore.
