import pytest

# FIXME: it's just a template to be replaced with real tests later
@pytest.mark.parametrize("test_input,expected", [
    ("3+5", 8),
    ("2+4", 6),
    ("6*9", 54),
])
def test_eval(test_input, expected):
    assert eval(test_input) == expected