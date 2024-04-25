# Steps to run

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
git clone https://github.com/data61/MP-SPDZ
cd MP-SPDZ
git remote add kevin_mpc https://github.com/mhchia/MP-SPDZ.git
git fetch kevin_mpc
git checkout arith-executor
mp_spdz_project_root=$(pwd)
```

Build the MPC vm for `semi` protocol

```bash
make -j8 semi-party.x
# Make sure `semi-party.x` exists
ls semi-party.x
```

### Run the test

Modify the configs in `tests/onnx2circom/test_onnx_to_circom.py` to point to the correct paths. Just fill in the paths to the two projects you just cloned.

```bash
# NOTE: Change the path to your own path
CIRCOM_2_ARITHC_PROJECT_ROOT = Path('/path/to/circom-2-arithc-project-root')
MP_SPDZ_PROJECT_ROOT = Path('/path/to/mp-spdz-project-root')
```

Go back to the zkstats library project root

```bash
cd ../zk-stats-lib
```

Run the test:

```bash
pytest -s tests/onnx2circom/test_onnx_to_circom.py
```
