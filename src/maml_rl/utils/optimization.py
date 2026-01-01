import torch
from collections import OrderedDict
from typing import Mapping, Callable, List


def parameters_to_vector(params: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a dictionary of parameters into a single vector."""
    vec = []
    for _, param in params.items():
        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(
    vec: torch.Tensor, template_params: Mapping[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    """Convert a flat vector back into a dictionary of parameters."""
    pointer = 0
    new_params = OrderedDict()
    for name, param in template_params.items():
        num_param = param.numel()
        new_params[name] = vec[pointer : pointer + num_param].view_as(param)
        pointer += num_param
    return new_params


def conjugate_gradients(
    fvp_func: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    nsteps: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """
    Solve Hx = b using Conjugate Gradients.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)

    for _ in range(nsteps):
        _fv = fvp_func(p)
        alpha = rdotr / (torch.dot(p, _fv) + 1e-8)
        x += alpha * p
        r -= alpha * _fv
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x
