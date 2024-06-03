# Steps to run

## Preparation

### Install Python module dependencies

Install all the required Python module dependencies in the virtual environment, and activate it:

```bash
poetry install
poetry shell
```

### Install Rust

`circom-2-arithc` is written in Rust and requires the Rust compiler to build.

You can install the Rust compiler using [rustup](https://rustup.rs/).

## Test onnx2keras

Run the test:

```bash
pytest -s tests/onnx2circom/test_onnx_to_keras.py
```

## Test onnx2circom

### circom-2-arithc

Clone circom-2-arithc. Use a fork for now. Will change to the official repo soon.

```bash
cd ..
git clone https://github.com/mhchia/circom-2-arithc.git
cd circom-2-arithc
git checkout mpcstats
cp .env.example .env
circom_2_arithc_project_root=$(pwd)
```

Build the compiler:

```bash
cargo build --release
```

### MP-SPDZ

Clone the repo

```bash
cd ..
git clone https://github.com/mhchia/MP-SPDZ
cd MP-SPDZ
git remote add kevin_mpc https://github.com/mhchia/MP-SPDZ.git
git fetch kevin_mpc
git checkout -b arith-executor kevin_mpc/arith-executor
mp_spdz_project_root=$(pwd)
```

Build the MPC vm for `semi` protocol

```bash
make setup
make -j8 semi-party.x
# Make sure `semi-party.x` exists
ls semi-party.x
```

If you're on macOS and see the following linker warning, you can safely ignore it:

```bash
ld: warning: search path '/usr/local/opt/openssl/lib' not found
```

### Run the test

Go back to the zkstats library project root

```bash
cd ../zk-stats-lib
```

Then modify the configs in `tests/onnx2circom/utils.py` to point to the correct paths. Just fill in the paths to the two projects you just cloned.

```bash
# NOTE: Change the path to your own path
CIRCOM_2_ARITHC_PROJECT_ROOT = Path('/path/to/circom-2-arithc-project-root')
MP_SPDZ_PROJECT_ROOT = Path('/path/to/mp-spdz-project-root')
```

Run the test:

```bash
pytest -s tests/onnx2circom/test_onnx_to_circom.py
```
