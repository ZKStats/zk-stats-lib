# Steps to run

## Test onnx2keras

Follow [the instructions](../../zkstats/onnx2circom/README.md) to sync submodules:

Go to the root of the repo and sync the submodules:
```bash
cd ../../
git submodule init
git submodule update
```

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
mp_spdz_project_root=$(pwd)
```

Build the MPC vm for `semi` protocol
```
make -j8 semi-party.x
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
