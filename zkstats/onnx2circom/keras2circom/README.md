# keras2circom

keras2circom is a python tool that transpiles a tf.keras model into a circom circuit.

## Experimental Version for stats

Install the dependencies. You can use pip:

```bash
pip install -r requirements.txt
```

You will also need to install circom and snarkjs. You can run the following commands to install them:

```bash
bash setup-circom.sh
```

Last but not least, run

```bash
npm install
```

First, Look at their supported function in example/dense folder

- Just run notebook in dense_keras.ipynb to generate keras file
- Then, run `python main.py ./example/dense/dense_keras.keras` to generate `output` folder, consisting of
  - `circuit.circom` which contains circom file
  - `circuit.json` which contains every predetermined value that we know before hand (like weight, bias, etc.)
  - `circuit.py` which contains the algorithm to calcualte (off-chain) the final value of functions we are interested in, so we can provide them as witness (as input signal) in circuit.circom as well. We run circuit.py by creating `input.json` (inside output folder) like { "in": ["4", "4", "7"] }, then run `python output/circuit.py output/circuit.json output/input.json`, and we will get output.json
  - Briefly, with `input.json`, `circuit.json`, and `output.json` we can verify in `circuit.circom` and get output as well

Now, we will look at our customed layer 'MeanCheck'

Note that it doesnt really make sense to have this layer because we want customed layer for each operation, not for the set of operations like MeanCheck, but this to give an overall idea how to do it, and it's more trivial trying to write smaller operation. Btw, we still dont support Decimal point mean witness, but can do with adding dec like in other template examples

First, since default implementation of this library install circom template in node_modules already, we will hand-code our MeanCheck template by copying example/MeanCheck/MeanCheck.circom into node_modules/circomlib-ml/circuits/MeanCheck.circom

Then, the rest is just the same as above

- Just run notebook in gen_MeanCheck.ipynb to generate keras file
- Then, run `python main.py ./example/MeanCheck/mean_keras.keras` to generate `output` folder, consisting of
  - `circuit.circom`
  - `circuit.json`, which nothing since it this circuit has 2 inputs: one is input so unknown, while the other is witness which is unknown at first as well
  - `circuit.py` which allows us to calculate mean_check_out, Inside `output` folder, we create `input.json` like `{ "in": ["4", "4", "7"] }`, then run `python output/circuit.py output/circuit.json output/input.json`, and we will get `output.json`
  - Briefly, with `input.json`, `circuit.json`, and `output.json` we can verify in circuit.circom and get `output` as well

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
