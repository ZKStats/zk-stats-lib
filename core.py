import torch
import ezkl
import os
import numpy as np
import json
import time
# Export model
def export_onnx(model, data_tensor_array, model_loc):
  circuit = model()

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
def gen_settings(comb_data_path, onnx_filename, scale, mode, settings_filename):
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
    comb_data_path, onnx_filename, settings_filename, mode)
  else:
    ezkl.calibrate_settings(
    comb_data_path, onnx_filename, settings_filename, mode, scales = scale)

  assert os.path.exists(settings_filename)
  assert os.path.exists(comb_data_path)
  assert os.path.exists(onnx_filename)
  f_setting = open(settings_filename, "r")
  print("scale: ", scale)
  print("setting: ", f_setting.read())

# ===================================================================================================  
# ===================================================================================================  

def verifier_define_calculation(verifier_model, verifier_model_path, dummy_data_path_array):
  # load data from dummy_data_path_array into dummy_data_tensor_array
  dummy_data_tensor_array = []
  for path in dummy_data_path_array:
    dummy_data = np.array(json.loads(open(path, "r").read())["input_data"][0])
    dummy_data_tensor_array.append(torch.reshape(torch.tensor(dummy_data), (1, len(dummy_data),1 )))
  # export onnx file
  export_onnx(verifier_model,dummy_data_tensor_array, verifier_model_path)

# ===================================================================================================  
# ===================================================================================================  

  # we decide to not have comb_data_path as parameter since a bit redundant parameter.
def prover_gen_settings(data_path_array, comb_data_path, prover_model,prover_model_path, scale, mode, settings_path):
    data_tensor_array=[]
    comb_data = []
    for path in data_path_array:
        data = np.array(json.loads(open(path, "r").read())["input_data"][0])
        data_tensor_array.append(torch.reshape(torch.tensor(data), (1, len(data),1 )))
        comb_data.append(data.tolist())
    # Serialize data into file:
    json.dump(dict(input_data = comb_data), open(comb_data_path, 'w' ))

    # export onnx file
    export_onnx(prover_model, data_tensor_array, prover_model_path)
    # gen + calibrate setting
    gen_settings(comb_data_path, prover_model_path, scale, mode, settings_path)

# ===================================================================================================  
# ===================================================================================================  

# Here prover can concurrently call this since all params are public to get pk. 
# Here write as verifier function to emphasize that verifier must calculate its own vk to be sure
def verifier_setup(verifier_model_path, verifier_compiled_model_path, settings_path, srs_path,vk_path, pk_path ):
  # compile circuit
  res = ezkl.compile_circuit(verifier_model_path, verifier_compiled_model_path, settings_path)
  assert res == True

  # srs path
  res = ezkl.get_srs(srs_path, settings_path)

  # setupt vk, pk param for use..... prover can use same pk or can init their own!
  print("==== setting up ezkl ====")
  start_time = time.time()
  res = ezkl.setup(
        verifier_compiled_model_path,
        vk_path,
        pk_path,
        srs_path)
  end_time = time.time()
  time_setup = end_time -start_time
  print(f"Time setup: {time_setup} seconds")

  assert res == True
  assert os.path.isfile(vk_path)
  assert os.path.isfile(pk_path)
  assert os.path.isfile(settings_path)

# ===================================================================================================  
# ===================================================================================================  

def prover_gen_proof(prover_model_path, comb_data_path, witness_path, prover_compiled_model_path, settings_path, proof_path, pk_path, srs_path):
  res = ezkl.compile_circuit(prover_model_path, prover_compiled_model_path, settings_path)
  assert res == True
  # now generate the witness file
  print('==== Generating Witness ====')
  witness = ezkl.gen_witness(comb_data_path, prover_compiled_model_path, witness_path)
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
        srs_path,
        "single",
    )

  print("proof: " ,res)
  end_time = time.time()
  time_gen_prf = end_time -start_time
  print(f"Time gen prf: {time_gen_prf} seconds")
  assert os.path.isfile(proof_path)

# ===================================================================================================  
# ===================================================================================================  

def verifier_verify(proof_path, settings_path, vk_path, srs_path):
  # enforce boolean statement to be true
  settings = json.load(open(settings_path))
  output_scale = settings['model_output_scales']

  proof = json.load(open(proof_path))
  num_inputs = len(settings['model_input_scales'])
  print("num_inputs: ", num_inputs)
  proof["instances"][0][num_inputs] = ezkl.float_to_vecu64(1.0, output_scale[0])
  json.dump(proof, open(proof_path, 'w'))

  print("prf instances: ", proof['instances'])

  print("proof boolean: ", ezkl.vecu64_to_float(proof['instances'][0][num_inputs], output_scale[0]))
  for i in range(num_inputs+1, len(proof['instances'][0])):
    print("proof result",i-num_inputs,":", ezkl.vecu64_to_float(proof['instances'][0][i], output_scale[1]))


  res = ezkl.verify(
        proof_path,
        settings_path,
        vk_path,
        srs_path,
    )

  assert res == True
  print("verified")