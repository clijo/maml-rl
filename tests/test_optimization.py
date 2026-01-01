import torch
from collections import OrderedDict
from maml_rl.utils.optimization import (
    parameters_to_vector,
    vector_to_parameters,
    conjugate_gradients,
)

def test_parameters_to_vector_and_back():
    params = OrderedDict([
        ("a", torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        ("b", torch.tensor([5.0, 6.0])),
    ])
    
    vec = parameters_to_vector(params)
    assert vec.shape == (6,)
    assert torch.equal(vec, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    
    new_params = vector_to_parameters(vec, params)
    assert set(new_params.keys()) == set(params.keys())
    for k in params:
        assert torch.equal(new_params[k], params[k])
        assert new_params[k].shape == params[k].shape

def test_conjugate_gradients():
    # Solve Ax = b where A is positive definite
    # A = [[4, 1], [1, 3]], b = [[1], [2]]
    # x should be [[1/11], [7/11]] approx [[0.0909], [0.6363]]
    
    A = torch.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = torch.tensor([1.0, 2.0])
    
    def fvp(v):
        return A @ v
    
    x = conjugate_gradients(fvp, b, nsteps=10)
    
    expected_x = torch.tensor([1/11, 7/11])
    assert torch.allclose(x, expected_x, atol=1e-5)
