# For comparing among different datasets. Mostly similar to core.py
# but make it print less info to make us see bigger picture better

import sys
import importlib.util
from typing import Type
import torch
from torch import Tensor
import ezkl
import os
import numpy as np
import json
import time
import math


# Export model
def export_onnx(model, data_tensor_array, model_loc):
  circuit = model()
  # Try running `prepare()` if it exists
  try:
    circuit.prepare(data_tensor_array)
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
def gen_settings(sel_data_path, onnx_filename, scale, mode, settings_filename):
  # print("==== Generate & Calibrate Setting ====")
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
  # print("scale: ", scale)
  # print("setting: ", f_setting.read())

# ===================================================================================================
# ===================================================================================================

# Here dummy_sel_data_path is redundant, but here to use process_data
def verifier_define_calculation(dummy_data_path,col_array, dummy_sel_data_path, verifier_model, verifier_model_path):
  dummy_data_tensor_array = process_data(dummy_data_path, col_array, dummy_sel_data_path)
  # export onnx file
  export_onnx(verifier_model, dummy_data_tensor_array, verifier_model_path)

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
def process_data(data_path,col_array, sel_data_path) -> list[Tensor]:
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
    json.dump(dict(input_data = sel_data), open(sel_data_path, 'w' ))
    return data_tensor_array


# we decide to not have sel_data_path as parameter since a bit redundant parameter.
def prover_gen_settings(data_path, col_array,  sel_data_path, prover_model,prover_model_path, scale, mode, settings_path):
    data_tensor_array = process_data(data_path,col_array,  sel_data_path)

    # export onnx file
    export_onnx(prover_model, data_tensor_array, prover_model_path)
    # gen + calibrate setting
    gen_settings(sel_data_path, prover_model_path, scale, mode, settings_path)

# ===================================================================================================
# ===================================================================================================

# Here prover can concurrently call this since all params are public to get pk.
# Here write as verifier function to emphasize that verifier must calculate its own vk to be sure
def setup(verifier_model_path, verifier_compiled_model_path, settings_path,vk_path, pk_path ):
  # compile circuit
  res = ezkl.compile_circuit(verifier_model_path, verifier_compiled_model_path, settings_path)
  assert res == True

  # srs path
  res = ezkl.get_srs(settings_path)

  # setup vk, pk param for use..... prover can use same pk or can init their own!
  # print("==== setting up ezkl ====")
  start_time = time.time()
  res = ezkl.setup(
        verifier_compiled_model_path,
        vk_path,
        pk_path)
  end_time = time.time()
  time_setup = end_time -start_time
  # print(f"Time setup: {time_setup} seconds")

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
    # print("!@# compiled_model exists?", os.path.isfile(prover_compiled_model_path))
    res = ezkl.compile_circuit(prover_model_path, prover_compiled_model_path, settings_path)
    # print("!@# compiled_model exists?", os.path.isfile(prover_compiled_model_path))
    assert res == True
    # now generate the witness file
    # print('==== Generating Witness ====')
    witness = ezkl.gen_witness(sel_data_path, prover_compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)
    # print(witness["outputs"])
    settings = json.load(open(settings_path))
    output_scale = settings['model_output_scales']
    # print("witness boolean: ", ezkl.vecu64_to_float(witness['outputs'][0][0], output_scale[0]))
    # for i in range(len(witness['outputs'][1])):
      # print("witness result", i+1,":", ezkl.vecu64_to_float(witness['outputs'][1][i], output_scale[1]))

    # GENERATE A PROOF
    # print("==== Generating Proof ====")
    start_time = time.time()
    res = ezkl.prove(
          witness_path,
          prover_compiled_model_path,
          pk_path,
          proof_path,
          "single",
      )

    # print("proof: " ,res)
    end_time = time.time()
    time_gen_prf = end_time -start_time
    # print(f"Time gen prf: {time_gen_prf} seconds")
    assert os.path.isfile(proof_path)
    return time_gen_prf

# ===================================================================================================
# ===================================================================================================

# return result array
def verifier_verify(proof_path, settings_path, vk_path):
  # enforce boolean statement to be true
  settings = json.load(open(settings_path))
  output_scale = settings['model_output_scales']

  proof = json.load(open(proof_path))
  num_inputs = len(settings['model_input_scales'])
  # print("num_inputs: ", num_inputs)
  proof["instances"][0][num_inputs] = ezkl.float_to_vecu64(1.0, output_scale[0])
  json.dump(proof, open(proof_path, 'w'))

  # print("prf instances: ", proof['instances'])

  # print("proof boolean: ", ezkl.vecu64_to_float(proof['instances'][0][num_inputs], output_scale[0]))
  assert ezkl.vecu64_to_float(proof['instances'][0][num_inputs], output_scale[0]) == 1, "Prf not set to 1"
  result = []
  for i in range(num_inputs+1, len(proof['instances'][0])):
    # print("proof result",i-num_inputs,":", ezkl.vecu64_to_float(proof['instances'][0][i], output_scale[1]))
    result.append(ezkl.vecu64_to_float(proof['instances'][0][i], output_scale[1]))


  res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path
    )

  assert res == True
#   print("verified")
  return result


# ===================================================================================================
# ===================================================================================================

# just one dataset at a time.
def bench_one(data_path, col_array, model_func, gen_param_func, data_name, scale,logrow, mode):
    os.makedirs(os.path.dirname('shared/'), exist_ok=True)
    os.makedirs(os.path.dirname('prover/'), exist_ok=True)
    verifier_model_path = os.path.join('shared/verifier.onnx')
    prover_model_path = os.path.join('prover/prover.onnx')
    verifier_compiled_model_path = os.path.join('shared/verifier.compiled')
    prover_compiled_model_path = os.path.join('prover/prover.compiled')
    pk_path = os.path.join('shared/test.pk')
    vk_path = os.path.join('shared/test.vk')
    proof_path = os.path.join('shared/test.pf')
    settings_path = os.path.join('shared/settings.json')
    srs_path = os.path.join('shared/kzg.srs')
    witness_path = os.path.join('prover/witness.json')
    # this is private to prover since it contains actual data
    sel_data_path = os.path.join('prover/sel_data.json')
    # this is just dummy random value
    sel_dummy_data_path = os.path.join('shared/sel_dummy_data.json')
    dummy_data_path = os.path.join('shared/dummy_data.json')

    print("===================================== ", data_name," =====================================")
    # go through each dataset (we have 9 data sets)
    create_dummy(data_path, dummy_data_path)
    dummy_data_tensor_array = process_data(dummy_data_path, col_array, sel_dummy_data_path)

    # verifier_define_calculation
    export_onnx(model_func(gen_param_func(dummy_data_tensor_array)),dummy_data_tensor_array, verifier_model_path)

    # prover_gen_settings
    data_tensor_array = process_data(data_path,col_array,  sel_data_path)
    # export onnx file
    export_onnx(model_func(gen_param_func(data_tensor_array)), data_tensor_array, prover_model_path)
    # gen + calibrate setting
    gen_settings(sel_data_path, prover_model_path, scale, mode, settings_path)
    f_setting = json.loads(open(settings_path, "r").read())
    print("og setting: ", f_setting)
    # f_setting['run_args']['logrows']=math.ceil(math.log(f_setting['num_rows'],2))
    f_setting['run_args']['logrows']=logrow
    json.dump(f_setting, open(settings_path, "w"), indent=2)  # You can adjust the 'indent' parameter for formatting
    print("logrow cal settings: ",json.loads(open(settings_path, "r").read()) )
    setup(verifier_model_path, verifier_compiled_model_path, settings_path,vk_path, pk_path)

    gen_prf_time = prover_gen_proof(prover_model_path, sel_data_path, witness_path, prover_compiled_model_path, settings_path, proof_path, pk_path)

    result= verifier_verify(proof_path, settings_path, vk_path)

    # f_setting = open(settings_path, "r")
    # print("setting: ", f_setting.read())
    print("gen prf time: ", gen_prf_time)
    print("Theory result: ", gen_param_func(data_tensor_array)[0])
    print("Our result: ", result)
    return gen_prf_time

