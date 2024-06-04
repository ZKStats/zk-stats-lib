import csv
from pathlib import Path
from typing import Type, Sequence, Mapping, Union, Literal, Callable
from enum import Enum
import os
import numpy as np
import json
import time

import torch
import ezkl

from zkstats.computation import IModel



# ===================================================================================================
# ===================================================================================================

def verifier_define_calculation(
  dummy_data_path: str,
  selected_columns: list[str],
  # TODO: Here dummy_sel_data_path is redundant, but here to use process_data
  dummy_sel_data_path: str,
  verifier_model: Type[IModel],
  verifier_model_path: str,
) -> None:
  """
  Export the verifier model to an ONNX file.
  :param dummy_data_path: path to the dummy data file
  :param selected_columns: column names selected for computation
  :param dummy_sel_data_path: path to store generated preprocessed dummy data file
  :param verifier_model: the verifier model class
  :param verifier_model_path: path to store the generated verifier model file in onnx format
  """
  dummy_data_tensor_array = _process_data(dummy_data_path, selected_columns, dummy_sel_data_path)
  # export onnx file
  _export_onnx(verifier_model, dummy_data_tensor_array, verifier_model_path)


# TODO: Should only need the shape of data instead of the real dataset, since
# users (verifiers) call this function and they don't have the real data.
def create_dummy(data_path: str, dummy_data_path: str) -> None:
    """
    Create a dummy data file with randomized data based on the shape of the original data.
    """
    # Convert data file to json under the same directory but with suffix .json
    data_path: Path = Path(data_path)
    data_json_path = Path(data_path).with_suffix(DataExtension.JSON.value)

    data = json.loads(open(data_json_path, "r").read())
    # assume all columns have same number of rows
    dummy_data ={}
    for col in data:
        # not use same value for every column to prevent something weird, like singular matrix
        min_col = min(data[col])
        max_col = max(data[col])
        dummy_data[col] = np.round(np.random.uniform(min_col,max_col,len(data[col])),1).tolist()

    json.dump(dummy_data, open(dummy_data_path, 'w'))

# ===================================================================================================
# ===================================================================================================


def prover_gen_settings(
    data_path: str,
    selected_columns: list[str],
    sel_data_path: list[str],
    prover_model: Type[IModel],
    prover_model_path: str,
    scale: Union[list[int], Literal["default"]],
    # TODO: should be able to hardcode mode to "resources" or make it default?
    mode: Union[Literal["resources"], Literal["accuracy"]],
    settings_path: str,
):
    """
    Generate and calibrate settings for the given model and data.
    :param data_path: path to the data file
    :param selected_columns: column names selected for computation
    :param sel_data_path: path to store generated preprocessed data file
    :param prover_model: the prover model class
    :param prover_model_path: path to store the generated prover model file in onnx format
    :param scale: the scale to use for the computation. It's a list of integer or "default" for default scale
    :param mode: the mode to use for the computation. It's either "resources" or "accuracy"
    :param settings_path: path to store the generated settings file
    """
    data_tensor_array = _process_data(data_path, selected_columns, sel_data_path)

    # export onnx file
    _export_onnx(prover_model, data_tensor_array, prover_model_path)
    # gen + calibrate setting
    _gen_settings(sel_data_path, prover_model_path, scale, mode, settings_path)

# ===================================================================================================
# ===================================================================================================

def setup(
    model_path: str,
    compiled_model_path: str,
    settings_path: str,
    vk_path: str,
    pk_path: str,
) -> None:
  """
  Compile the verifier model and generate the verification key and public key.

  :param model_path: path to the model file in onnx format
  :param compiled_model_path: path to store the generated compiled verifier model
  :param settings_path: path to the settings file
  :param vk_path: path to store the generated verification key file
  :param pk_path: path to store the generated public key file
  """
  # compile circuit
  res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
  assert res == True

  # srs path
  res = ezkl.get_srs(settings_path)

  # setup vk, pk param for use..... prover can use same pk or can init their own!
  print("==== setting up ezkl ====")
  start_time = time.time()
  res = ezkl.setup(
        compiled_model_path,
        vk_path,
        pk_path)
  end_time = time.time()
  time_setup = end_time -start_time
  print(f"Time setup: {time_setup} seconds")

  assert res == True
  assert os.path.isfile(vk_path)
  assert os.path.isfile(pk_path)
  assert os.path.isfile(settings_path)

# ===================================================================================================
# ===================================================================================================

def prover_gen_proof(
    prover_model_path: str,
    sel_data_path: str,
    witness_path: str,
    prover_compiled_model_path: str,
    settings_path: str,
    proof_path: str,
    pk_path: str,
) -> None:
    """
    Generate a proof for the given model and data.

    :param prover_model_path: path to the prover model file in onnx format
    :param sel_data_path: path to the preprocessed data file
    :param witness_path: path to store the generated witness file
    :param prover_compiled_model_path: path to store the generated compiled prover model
    :param settings_path: path to the settings file
    :param proof_path: path to store the generated proof file
    :param pk_path: path to the public key file
    """
    res = ezkl.compile_circuit(prover_model_path, prover_compiled_model_path, settings_path)
    assert res == True
    # now generate the witness file
    print('==== Generating Witness ====')
    witness = ezkl.gen_witness(sel_data_path, prover_compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)
    # print(witness["outputs"])
    settings = json.load(open(settings_path))
    output_scale = settings['model_output_scales']
    # print("witness boolean: ", ezkl.vecu64_to_float(witness['outputs'][0][0], output_scale[0]))
    print("witness boolean: ", ezkl.felt_to_float(witness['outputs'][0][0], output_scale[0]))
    for i in range(len(witness['outputs'][1])):
      # print("witness result", i+1,":", ezkl.vecu64_to_float(witness['outputs'][1][i], output_scale[1]))
      print("witness result", i+1,":", ezkl.felt_to_float(witness['outputs'][1][i], output_scale[1]))

    # GENERATE A PROOF
    print("==== Generating Proof ====")
    start_time = time.time()
    res = ezkl.prove(
          witness_path,
          prover_compiled_model_path,
          pk_path,
          proof_path,
          "single",
      )

    print("proof: " ,res)
    end_time = time.time()
    time_gen_prf = end_time -start_time
    print(f"Time gen prf: {time_gen_prf} seconds")
    assert os.path.isfile(proof_path)


# ===================================================================================================
# ===================================================================================================

# commitment_map is a mapping[column_name, commitment_hex]
# E.g. {
#     "columns_0": "0x...",
#     ...
# }
TCommitmentMap = Mapping[str, str]
# data_commitment is a mapping[scale, mapping[column_name, commitment_hex]]
# E.g. {
#     scale_0: {
#         "columns_0": "0x...",
#         ...
#     },
#     ...
# }
TCommitmentMaps = Mapping[str, TCommitmentMap]

def verifier_verify(proof_path: str, settings_path: str, vk_path: str, selected_columns: Sequence[str], data_commitment_path: str) -> torch.Tensor:
  """
  Verify the proof and return the result.

  :param proof_path: path to the proof file
  :param settings_path: path to the settings file
  :param vk_path: path to the verification key file
  :param expected_data_commitments: expected data commitments for columns. The i-th commitment should
    be stored in `expected_data_commitments[i]`.
  """

  # 1. First check the zk proof is valid
  res = ezkl.verify(
    proof_path,
    settings_path,
    vk_path,
  )
  # TODO: change asserts to return boolean
  assert res == True

  # 2. Check if input/output are correct
  with open(settings_path) as f:
    settings = json.load(f)
  input_scales = settings['model_input_scales']
  output_scales = settings['model_output_scales']
  with open(proof_path) as f:
    proof = json.load(f)
  proof_instance = proof["instances"][0]
  inputs = proof_instance[:len(input_scales)]
  outputs = proof_instance[len(input_scales):]
  len_inputs = len(inputs)
  len_outputs = len(outputs)
  # `instances` = input commitments + params (which is 0 in our case) + output
  assert len(proof_instance) == len_inputs + len_outputs, f"lengths mismatch: {len(proof_instance)=}, {len_inputs=}, {len_outputs=}"

  # 2.1 Check input commitments
  with open(data_commitment_path) as f:
    data_commitment = json.load(f)
  # All inputs are hashed so are commitments
  assert len_inputs == len(selected_columns), f"lengths mismatch: {len_inputs=}, {len(selected_columns)=}"
  # Sanity check
  # Check each commitment is correct
  for i, (actual_commitment, column_name) in enumerate(zip(inputs, selected_columns)):
    #  actual_commitment_str = ezkl.vecu64_to_felt(actual_commitment)
     actual_commitment_str = (actual_commitment)
     input_scale = input_scales[i]
     expected_commitment = data_commitment[str(input_scale)][column_name]
     assert actual_commitment_str == expected_commitment, f"commitment mismatch: {i=}, {actual_commitment_str=}, {expected_commitment=}"

  # 2.2 Check output is correct
  # - is a tuple (is_in_error, result)
  # - is_valid is True
  # Sanity check
  is_in_error = ezkl.felt_to_float(outputs[0], output_scales[0])
  assert is_in_error == 1.0, f"result is not within error"
  result_arr = []
  for index in range(len(outputs)-1):
    result_arr.append(ezkl.felt_to_float(outputs[index+1], output_scales[1]))
  return result_arr


# ===================================================================================================
# ===================================================================================================

def generate_data_commitment(data_path: str, scales: Sequence[int], data_commitment_path: str) -> None:
  """
  Generate and store data commitment maps for different scales so that verifiers can verify
  proofs with different scales.

  :param data_path: data file path. The format must be anything defined in `DataExtension`
  :param scales: a list of scales to use for the commitments
  :param data_commitment_path: path to store the generated data commitment maps
  """

  # Convert `data_path` to json file `data_json_path`
  data_path: Path = Path(data_path)
  data_json_path = Path(data_path).with_suffix(DataExtension.JSON.value)
  _preprocess_data_file_to_json(data_path, data_json_path)

  with open(data_json_path) as f:
    data_json = json.load(f)
  data_commitments = {
    str(scale): {
      k: _get_commitment_for_column(v, scale) for k, v in data_json.items()
    } for scale in scales
  }
  with open(data_commitment_path, "w") as f:
    json.dump(data_commitments, f)


# ===================================================================================================
# Private functions
# ===================================================================================================

def _export_onnx(model: Type[IModel], data_tensor_array: list[torch.Tensor], model_loc: str) -> None:
  circuit = model()
  try:
    circuit.preprocess(data_tensor_array)
  except AttributeError:
    pass

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  # print(device)

  circuit.to(device)

  # Flips the neural net into inference mode
  circuit.eval()
  input_names = []
  # dynamic_axes = {}

  data_tensor_tuple = ()
  for i in range(len(data_tensor_array)):
    data_tensor_tuple += (data_tensor_array[i],)
    input_index = "input"+str(i+1)
    input_names.append(input_index)
  #   dynamic_axes[input_index] = {0 : 'batch_size'}
  # dynamic_axes["output"] = {0 : 'batch_size'}

  # Export the model
  torch.onnx.export(circuit,               # model being run
                      data_tensor_tuple,                   # model input (or a tuple for multiple inputs)
                      model_loc,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = input_names,   # the model's input names
                      output_names = ['output'], # the model's output names
                      # dynamic_axes=dynamic_axes
                      )


# mode is either "accuracy" or "resources"
# sel_data = selected column from data that will be used for computation
def _gen_settings(
  sel_data_path: str,
  onnx_filename: str,
  scale: Union[list[int], Literal["default"]],
  mode: Union[Literal["resources"], Literal["accuracy"]],
  settings_filename: str,
) -> None:
  print("==== Generate & Calibrate Setting ====")
  # Set input to be Poseidon Hash, and param of computation graph to be public
  # Poseidon is not homomorphic additive, maybe consider Pedersens or Dory commitment.
  gip_run_args = ezkl.PyRunArgs()
  gip_run_args.input_visibility = "hashed"  # one commitment (values hashed) for each column
  gip_run_args.param_visibility = "fixed"  # no parameters shown
  gip_run_args.output_visibility = "public"  # should be `(torch.Tensor(1.0), output)`

 # generate settings
  ezkl.gen_settings(onnx_filename, settings_filename, py_run_args=gip_run_args)
  if scale =="default":
    ezkl.calibrate_settings(
    sel_data_path, onnx_filename, settings_filename, mode)
  else:
    assert isinstance(scale, list)
    ezkl.calibrate_settings(
    sel_data_path, onnx_filename, settings_filename, mode, scales = scale)

  assert os.path.exists(settings_filename)
  assert os.path.exists(sel_data_path)
  assert os.path.exists(onnx_filename)
  f_setting = open(settings_filename, "r")
  print("scale: ", scale)
  print("setting: ", f_setting.read())


def _csv_file_to_json(old_file_path: Union[Path, str], out_data_json_path: Union[Path, str],  *, delimiter: str = ",") -> None:
    data_csv_path = Path(old_file_path)
    with open(data_csv_path, 'r') as f_csv:
        reader = csv.reader(f_csv, delimiter=delimiter, strict=True)
        # Read all data from the reader to `rows`
        rows_with_column_name = tuple(reader)
    if len(rows_with_column_name) < 1:
        raise ValueError("No column names in the CSV file")
    if len(rows_with_column_name) < 2:
        raise ValueError("No data in the CSV file")
    column_names = rows_with_column_name[0]
    rows = rows_with_column_name[1:]

    columns = [
        [
            float(rows[j][i])
            for j in range(len(rows))
        ]
        for i in range(len(rows[0]))
    ]
    data = {
        column_name: column_data
        for column_name, column_data in zip(column_names, columns)
    }
    with open(out_data_json_path, "w") as f_json:
        json.dump(data, f_json)


class DataExtension(Enum):
    CSV = ".csv"
    JSON = ".json"


DATA_FORMAT_PREPROCESSING_FUNCTION: dict[DataExtension, Callable[[Union[Path, str], Path], None]] = {
    DataExtension.CSV: _csv_file_to_json,
    DataExtension.JSON: lambda old_file_path, out_data_json_path: Path(out_data_json_path).write_text(Path(old_file_path).read_text())
}

def _preprocess_data_file_to_json(data_path: Union[Path, str], out_data_json_path: Path):
    data_file_extension = DataExtension(data_path.suffix)
    preprocess_function = DATA_FORMAT_PREPROCESSING_FUNCTION[data_file_extension]
    preprocess_function(data_path, out_data_json_path)


def _process_data(
    data_path: Union[str | Path],
    col_array: list[str],
    sel_data_path: list[str],
  ) -> list[torch.Tensor]:
    data_tensor_array=[]
    sel_data = []
    data_path: Path = Path(data_path)
    # Convert data file to json under the same directory but with suffix .json
    data_json_path = Path(data_path).with_suffix(DataExtension.JSON.value)
    _preprocess_data_file_to_json(data_path, data_json_path)
    data_onefile = json.loads(open(data_json_path, "r").read())

    for col in col_array:
      data = data_onefile[col]
      data_tensor = torch.tensor(data, dtype = torch.float32)
      data_tensor_array.append(torch.reshape(data_tensor, (-1,1)))
      sel_data.append(data)
    # Serialize data into file:
    # sel_data comes from `data`
    json.dump(dict(input_data = sel_data), open(sel_data_path, 'w'))
    return data_tensor_array


def _get_commitment_for_column(column: list[float], scale: int) -> str:
  # Ref: https://github.com/zkonduit/ezkl/discussions/633
  # serialized_data = [ezkl.float_to_vecu64(x, scale) for x in column]
  serialized_data = [ezkl.float_to_felt(x, scale) for x in column]
  res_poseidon_hash = ezkl.poseidon_hash(serialized_data)[0]
  # res_hex = ezkl.vecu64_to_felt(res_poseidon_hash[0])

  return res_poseidon_hash