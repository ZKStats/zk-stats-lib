from dataclasses import dataclass
from typing import Type, Sequence, Mapping
import torch
from torch import Tensor
import ezkl
import os
import numpy as np
import json
import time

from zkstats.computation import IModel


# Export model
def _export_onnx(model: Type[IModel], data_tensor_array: list[Tensor], model_loc: str):
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
  dynamic_axes = {}

  data_tensor_tuple = ()
  for i in range(len(data_tensor_array)):
    data_tensor_tuple += (data_tensor_array[i],)
    input_index = "input"+str(i+1)
    input_names.append(input_index)
    dynamic_axes[input_index] = {0 : 'batch_size'}
  dynamic_axes["output"] = {0 : 'batch_size'}

  # Export the model
  torch.onnx.export(circuit,               # model being run
                      data_tensor_tuple,                   # model input (or a tuple for multiple inputs)
                      model_loc,            # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = input_names,   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes=dynamic_axes)

# ===================================================================================================
# ===================================================================================================

# mode is either "accuracy" or "resources"
# sel_data = selected column from data that will be used for computation
def _gen_settings(sel_data_path, onnx_filename, scale, mode, settings_filename):
  print("==== Generate & Calibrate Setting ====")
  # Set input to be Poseidon Hash, and param of computation graph to be public
  # Poseidon is not homomorphic additive, maybe consider Pedersens or Dory commitment.
  gip_run_args = ezkl.PyRunArgs()
  gip_run_args.input_visibility = "hashed"  # matrix and generalized inverse commitments
  gip_run_args.output_visibility = "public"   # no parameters used
  gip_run_args.param_visibility = "private" # should be Tensor(True)--> to enforce arbitrary data in w

 # generate settings
  ezkl.gen_settings(onnx_filename, settings_filename, py_run_args=gip_run_args)
  if scale =="default":
    ezkl.calibrate_settings(
    sel_data_path, onnx_filename, settings_filename, mode)
  else:
    ezkl.calibrate_settings(
    sel_data_path, onnx_filename, settings_filename, mode, scales = scale)

  assert os.path.exists(settings_filename)
  assert os.path.exists(sel_data_path)
  assert os.path.exists(onnx_filename)
  f_setting = open(settings_filename, "r")
  print("scale: ", scale)
  print("setting: ", f_setting.read())

# ===================================================================================================
# ===================================================================================================

# Here dummy_sel_data_path is redundant, but here to use process_data
def verifier_define_calculation(dummy_data_path, col_array, dummy_sel_data_path, verifier_model, verifier_model_path):
  dummy_data_tensor_array = _process_data(dummy_data_path, col_array, dummy_sel_data_path)
  # export onnx file
  _export_onnx(verifier_model, dummy_data_tensor_array, verifier_model_path)

# given data file (whole json table), create a dummy data file with randomized data
def create_dummy(data_path, dummy_data_path):
    data = json.loads(open(data_path, "r").read())
    # assume all columns have same number of rows
    dummy_data ={}
    for col in data:
        # not use same value for every column to prevent something weird, like singular matrix
        dummy_data[col] = np.round(np.random.uniform(1,30,len(data[col])),1).tolist()

    json.dump(dummy_data, open(dummy_data_path, 'w'))

# ===================================================================================================
# ===================================================================================================

# New version
def _process_data(data_path, col_array, sel_data_path) -> list[Tensor]:
    data_tensor_array=[]
    sel_data = []
    data_onefile = json.loads(open(data_path, "r").read())

    for col in col_array:
      data = data_onefile[col]
      data_tensor = torch.tensor(data, dtype = torch.float64)
      data_tensor_array.append(torch.reshape(data_tensor, (1,-1,1)))
      sel_data.append(data)
    # Serialize data into file:
    # sel_data comes from `data`
    json.dump(dict(input_data = sel_data), open(sel_data_path, 'w'))
    return data_tensor_array


# we decide to not have sel_data_path as parameter since a bit redundant parameter.
def prover_gen_settings(data_path, col_array, sel_data_path, prover_model,prover_model_path, scale, mode, settings_path):
    data_tensor_array = _process_data(data_path,col_array,  sel_data_path)

    # export onnx file
    _export_onnx(prover_model, data_tensor_array, prover_model_path)
    # gen + calibrate setting
    _gen_settings(sel_data_path, prover_model_path, scale, mode, settings_path)

# ===================================================================================================
# ===================================================================================================

# Here prover can concurrently call this since all params are public to get pk.
# Here write as verifier function to emphasize that verifier must calculate its own vk to be sure
def verifier_setup(verifier_model_path, verifier_compiled_model_path, settings_path, vk_path, pk_path):
  # compile circuit
  res = ezkl.compile_circuit(verifier_model_path, verifier_compiled_model_path, settings_path)
  assert res == True

  # srs path
  res = ezkl.get_srs(settings_path)

  # setup vk, pk param for use..... prover can use same pk or can init their own!
  print("==== setting up ezkl ====")
  start_time = time.time()
  res = ezkl.setup(
        verifier_compiled_model_path,
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
    prover_model_path,
    sel_data_path,
    witness_path,
    prover_compiled_model_path,
    settings_path,
    proof_path,
    pk_path
):
    res = ezkl.compile_circuit(prover_model_path, prover_compiled_model_path, settings_path)
    assert res == True
    # now generate the witness file
    print('==== Generating Witness ====')
    witness = ezkl.gen_witness(sel_data_path, prover_compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)
    # print(witness["outputs"])
    settings = json.load(open(settings_path))
    output_scale = settings['model_output_scales']
    print("witness boolean: ", ezkl.vecu64_to_float(witness['outputs'][0][0], output_scale[0]))
    for i in range(len(witness['outputs'][1])):
      print("witness result", i+1,":", ezkl.vecu64_to_float(witness['outputs'][1][i], output_scale[1]))

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


# column_name -> commitment
TCommitmentMap = Mapping[str, str]
# scale -> commitment maps
TCommitmentMaps = Mapping[int, TCommitmentMap]

# ===================================================================================================
# ===================================================================================================
def verifier_verify(proof_path: str, settings_path: str, vk_path: str, expected_columns: Sequence[str], commitment_maps: TCommitmentMaps):
  """
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
  # Output should always be a tuple of 2 elements
  assert len_outputs == 2, f"outputs should be a tuple of 2 elements, but got {len_outputs=}"
  # `instances` = input commitments + params (which is 0 in our case) + output
  assert len(proof_instance) == len_inputs + len_outputs, f"lengths mismatch: {len(proof_instance)=}, {len_inputs=}, {len_outputs=}"

  # 2.1 Check input commitments
  # All inputs are hashed so are commitments
  assert len_inputs == len(expected_columns)
  # Sanity check
  # Check each commitment is correct
  for i, (actual_commitment, column_name) in enumerate(zip(inputs, expected_columns)):
     actual_commitment_str = ezkl.vecu64_to_felt(actual_commitment)
     input_scale = input_scales[i]
     expected_commitment = commitment_maps[input_scale][column_name]
     assert actual_commitment_str == expected_commitment, f"commitment mismatch: {i=}, {actual_commitment_str=}, {expected_commitment=}"

  # 2.2 Check output is correct
  # - is a tuple (is_in_error, result)
  # - is_valid is True
  # Sanity check
  is_in_error = ezkl.vecu64_to_float(outputs[0], output_scales[0])
  assert is_in_error == 1.0, f"result is not within error"
  return ezkl.vecu64_to_float(outputs[1], output_scales[1])


def _get_commitment_for_column(column: list[float], scale: int) -> str:
  # Ref: https://github.com/zkonduit/ezkl/discussions/633
  serialized_data = [ezkl.float_to_vecu64(x, scale) for x in column]
  res_poseidon_hash = ezkl.poseidon_hash(serialized_data)
  res_hex = ezkl.vecu64_to_felt(res_poseidon_hash[0])
  return res_hex


def get_data_commitment_maps(data_path: str, scales: Sequence[int]) -> TCommitmentMaps:
  """
  Generate a map from scale to column name to commitment.
  """
  with open(data_path) as f:
    data_json = json.load(f)
  return {
    scale: {
      k: _get_commitment_for_column(v, scale) for k, v in data_json.items()
    } for scale in scales
  }
