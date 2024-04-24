This branch differs a lot from original already, so instead of explaining how it's different, let's just explain how to use this

-Edit mathematics_layers some of defomration_layers, and some of common_layers.py make each math function become their own layer because 'keras2circom' will only look at model.layers, so need to make sure our math computation is included in 'layer'
-The flow start from generating onnx file in gen_onnx.ipynb file, generating example.onnx
-Then in command line, run
`python converter.py --weights "./example/example.onnx" --outpath "./example/" --formats "keras"`which will convert from example.onnx into example.keras
-Then in read_keras.ipynb notebook, it downloads this keras format and run in over same input, seeing that it get the same result as onnx. Carefully look at model.layers which is the function that keras2circom uses to extract layer for circom template, and for each layer we can call .get_config() as shown in read_keras.ipynb notebook.

Now we support the following operations which are enough for all pytorch function used for original zk-stats-lib, we can support more operations once we found out alternative operations that result in easier circom template or due to a change of implementation in zkstats lib itself.

- \*, +, -, /
- ==, <, >
- torch.where
- torch.logical_and, torch.logical_or, NOT
- torch.abs, reciprocal, sort, exp, log
- torch.floor, torch.ceil
- torch.min, torch.max
- torch.sum, torch.mean
- .size()
- array indexing, eg. x = x[0]
- .float()
- @ (matrix multiplication)
- torch.transpose
- torch.ones_like
- torch.tensor(). Note that this will treat everything as constant, so shouldnt put variable in there
- For-loop —> Split & Squeeze
- Unsqueeze
- torch.cat
