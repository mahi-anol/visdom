"""
Shared gradient norm helpers used by both the torch and lightning loggers.
"""

import torch


def compute_grad_norm(parameters, norm_type=2.0):
    """Return the total gradient norm across all parameters that have a gradient."""
    # collect only parameters that actually received a gradient this step
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    # stack individual norms then compute the overall norm across all of them
    return torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type) for g in grads]),
        norm_type,
    ).item()


def compute_layer_grad_norm(grad, norm_type=2.0):
    """Return the gradient norm for a single gradient tensor."""
    # detach so we don't accidentally affect the computation graph
    return torch.norm(grad.detach(), norm_type).item()