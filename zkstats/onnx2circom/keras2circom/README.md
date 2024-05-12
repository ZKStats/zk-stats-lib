# keras2circom

keras2circom is a python tool that transpiles a tf.keras model into a circom circuit.

## Experimental Version for stats
`transpiler.py` is written for simplicity since we don't need witness approach and other ML layers for now. In this implementation we simply traverse the keras model and generate the corresponding circom.

### Steps to support a new layer
1. Add the `keras.layers.Layer` subclass to `SUPPORTED_OPS` in [`transpiler.py`](./keras2circom/transpiler.py).
2. Add a circom component template for the layer in [`mpc.circom`](../mpc.circom).
  - The template must have the same name as the layer's class name. For example, if the layer is `TFAdd`, the template name must be `TFAdd`.
  - The template must have the same number of inputs/outputs, and shapes as the ones in its keras layer. For example, `TFAdd` layer has two input tensors with the same shape `()`, so the template must have two inputs with the same shape, `signal input left` and `signal input right`. The order of the inputs must be the same as well.
  - If the input to the component has a variable length, the input length should be an argument to the component. For example, `TFReduceSum` layer has an input tensor with shape `(N,)`, so the template must have an argument `nInputs` and the signal should be `signal input in[nInputs]`.
3. If a layer has arguments, it's a must to specify how these arguments value can be derived in [`get_component_args_values`](./keras2circom/transpiler.py). For example,
  - `TFAdd` layer has no arguments, so the function returns an empty dictionary.
  - `TFReduceSum` layer has an argument `nInputs`, and this value can be determined by the number of elements in the input keras tensor.
  - `TFLog` layer has an argument `e`, and this value is a constant `2` for now.
  - If the layer has no arguments, the function should return an empty dictionary.

## Reference
- Original repository: [ora-io/keras2circom](https://github.com/ora-io/keras2circom)
- Our previous forked repository: [JernKunpittaya/keras2circom](https://github.com/JernKunpittaya/keras2circom/tree/stats_keras2circom)


## ==============================================================

                    ORIGINAL README BELOW

## ==============================================================

## Installation

First, clone the repository:

```bash
git clone https://github.com/socathie/keras2circom.git
```

Then, install the dependencies. You can use pip:

```bash
pip install -r requirements.txt
```

If you use conda, you can also create a new environment with the following command:

```bash
conda env create -f environment.yml
```

You will also need to install circom and snarkjs. You can run the following commands to install them:

```bash
bash setup-circom.sh
```

Last but not least, run

```bash
npm install
```

## Usage

To use the package, you can run the following command:

```bash
python main.py <model_path> [-o <output_dir>] [--raw]
```

For example, to transpile the model in `models/model.h5` into a circom circuit, you can run:

```bash
python main.py models/model.h5
```

The output will be in the `output` directory.

If you want to transpile the model into a circom circuit with "raw" output, i.e. no ArgMax at the end, you can run:

```bash
python main.py models/model.h5 --raw
```

## Testing

To test the package, you can run the following command:

```bash
npm test
```
