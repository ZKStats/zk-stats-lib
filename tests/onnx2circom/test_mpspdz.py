import torch

from zkstats.backends.mpspdz import mpspdz_output_to_tensors, tensors_to_circom_mpspdz_inputs


def test_mpspdz_output_to_tensors_success():
    output_dict = {
        "a[0][0][0]": 1,
        "a[0][0][1]": 2,
        "a[0][1][0]": 3,
        "a[0][1][1]": 4,
        "a[1][0][0]": 5,
        "a[1][0][1]": 6,
        "a[1][1][0]": 7,
        "a[1][1][1]": 8,
        "b[0][0]": 9,
        "b[0][1]": 10,
        "b[1][0]": 11,
        "b[1][1]": 12,
        "c[0]": 13,
        "c[1]": 14,
        "d": 15,
    }
    result = mpspdz_output_to_tensors(output_dict)

    assert torch.equal(result["a"], torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]))
    assert torch.equal(result["b"], torch.tensor([[9, 10], [11, 12]]))
    assert torch.equal(result["c"], torch.tensor([13, 14]))
    assert result["d"] == 15


def test_mpspdz_output_to_tensors_missing_index_3d():
    output_dict_missing = {
        "a[0][0][0]": 1,
        "a[0][0][1]": 2,
        "a[0][1][0]": 3,
        "a[0][1][1]": 4,
        "a[1][0][0]": 5,
        "a[1][0][1]": 6,
        "a[1][1][0]": 7,
        # Missing "a[1][1][1]"
        "b[0][0][0]": 9,
        "b[0][0][1]": 10,
        "c": 11,
    }
    try:
        result = mpspdz_output_to_tensors(output_dict_missing)
    except ValueError as e:
        assert str(e) == "Missing data for index (1, 1, 1) in dimension 1"
    else:
        assert False, "Expected ValueError for missing index"


def test_mpspdz_output_to_tensors_missing_index_2d():
    output_dict_missing = {
        "a[0][0]": 1,
        "a[0][1]": 2,
        "a[1][0]": 3,
        # Missing "a[1][1]"
        "b[0][0]": 5,
        "b[1][0]": 6,
        "c": 7
    }
    try:
        result = mpspdz_output_to_tensors(output_dict_missing)
    except ValueError as e:
        assert str(e) == "Missing data for index (1, 1) in dimension 1"
    else:
        assert False, "Expected ValueError for missing index"


def test_tensor_inputs_to_circom():
    input_names = ['keras_tensor_0', 'input_2']
    input_tensors = [torch.tensor([[[1], [34]]]), torch.tensor(5)]

    output = tensors_to_circom_mpspdz_inputs(input_names, input_tensors)
    assert output == {'keras_tensor_0[0][0][0]': 1, 'keras_tensor_0[0][1][0]': 34, 'input_2': 5}
