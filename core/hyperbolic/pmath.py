"""
Implementation of various mathematical operations in the Poincare ball model of hyperbolic space.
Based on implementations from:
- HypGCD: https://github.com/...
- Hyp-OW: https://github.com/...
- geoopt: https://github.com/geoopt/geoopt
"""

import numpy as np
import torch
import torch.nn.functional as F


def tanh(x, clamp=15):
    """Numerically stable tanh."""
    return x.clamp(-clamp, clamp).tanh()


class Artanh(torch.autograd.Function):
    """Inverse hyperbolic tangent with custom backward."""
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class RiemannianGradient(torch.autograd.Function):
    """
    Riemannian gradient scaling for Poincare ball.
    Scales Euclidean gradient by the inverse of the metric tensor.
    """
    c = 1  # Default curvature, can be overridden

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # Scale factor: ((1 - c*||x||^2)^2) / 4
        scale = (1 - RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale


class Arsinh(torch.autograd.Function):
    """Inverse hyperbolic sine with custom backward."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-5).log_()

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def artanh(x):
    """Inverse hyperbolic tangent."""
    return Artanh.apply(x)


def arsinh(x):
    """Inverse hyperbolic sine."""
    return Arsinh.apply(x)


def project(x, *, c=1.0):
    """
    Safe projection onto the Poincare ball for numerical stability.
    
    Parameters
    ----------
    x : tensor
        Point to project
    c : float
        Ball curvature (c > 0)
    
    Returns
    -------
    tensor
        Projected point inside the ball
    """
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def lambda_x(x, *, c=1.0, keepdim=False):
    """
    Compute the conformal factor λ_x^c for a point on the ball.
    
    λ_x^c = 2 / (1 - c * ||x||^2)
    """
    c = torch.as_tensor(c).type_as(x)
    return _lambda_x(x, c, keepdim=keepdim)


def _lambda_x(x, c, keepdim: bool = False):
    return 2 / (1 - c * x.pow(2).sum(-1, keepdim=keepdim))


def mobius_add(x, y, *, c=1.0):
    """
    Mobius addition in hyperbolic space.
    
    x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / 
              (1 + 2c<x,y> + c²||x||²||y||²)
    
    Note: This operation is NOT commutative in general.
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_add(x, y, c)


def _mobius_add(x, y, c):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / (denom + 1e-5)


def dist(x, y, *, c=1.0, keepdim=False):
    """
    Geodesic distance on the Poincare ball.
    
    d_c(x, y) = (2/√c) * arctanh(√c * ||(-x) ⊕_c y||)
    
    Parameters
    ----------
    x, y : tensor
        Points on the Poincare ball
    c : float
        Ball curvature
    keepdim : bool
        Retain the last dimension
    
    Returns
    -------
    tensor
        Geodesic distance between x and y
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist(x, y, c, keepdim=keepdim)


def _dist(x, y, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * _mobius_add(-x, y, c).norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def dist0(x, *, c=1.0, keepdim=False):
    """
    Distance from origin on the Poincare ball.
    
    d_c(0, x) = (2/√c) * arctanh(√c * ||x||)
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist0(x, c, keepdim=keepdim)


def _dist0(x, c, keepdim: bool = False):
    sqrt_c = c ** 0.5
    dist_c = artanh(sqrt_c * x.norm(dim=-1, p=2, keepdim=keepdim))
    return dist_c * 2 / sqrt_c


def expmap(x, u, *, c=1.0):
    """
    Exponential map for Poincare ball from point x in direction u.
    
    Exp_x^c(u) = x ⊕_c (tanh(√c/2 * ||u||_x) * u / (√c * ||u||))
    """
    c = torch.as_tensor(c).type_as(x)
    return _expmap(x, u, c)


def _expmap(x, u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    second_term = (
        tanh(sqrt_c / 2 * _lambda_x(x, c, keepdim=True) * u_norm)
        * u
        / (sqrt_c * u_norm)
    )
    gamma_1 = _mobius_add(x, second_term, c)
    return gamma_1


def expmap0(u, *, c=1.0):
    """
    Exponential map from origin.
    
    Exp_0^c(u) = tanh(√c * ||u||) * u / (√c * ||u||)
    
    This maps a point from Euclidean (tangent) space to the Poincare ball.
    """
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def logmap(x, y, *, c=1.0):
    """
    Logarithmic map for two points on the manifold.
    
    Log_x^c(y) = (2 / (√c * λ_x^c)) * arctanh(√c * ||(-x) ⊕_c y||) * 
                 ((-x) ⊕_c y) / ||(-x) ⊕_c y||
    """
    c = torch.as_tensor(c).type_as(x)
    return _logmap(x, y, c)


def _logmap(x, y, c):
    sub = _mobius_add(-x, y, c)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True)
    lam = _lambda_x(x, c, keepdim=True)
    sqrt_c = c ** 0.5
    return 2 / sqrt_c / lam * artanh(sqrt_c * sub_norm) * sub / sub_norm


def logmap0(y, *, c=1.0):
    """
    Logarithmic map to origin.
    
    Log_0^c(y) = arctanh(√c * ||y||) * y / (√c * ||y||)
    
    This maps a point from the Poincare ball back to Euclidean (tangent) space.
    """
    c = torch.as_tensor(c).type_as(y)
    return _logmap0(y, c)


def _logmap0(y, c):
    sqrt_c = c ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-5)
    return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


def mobius_matvec(m, x, *, c=1.0):
    """
    Mobius matrix-vector multiplication in hyperbolic space.
    
    M ⊗_c x = (1/√c) * tanh(||Mx|| / ||x|| * arctanh(√c * ||x||)) * Mx / ||Mx||
    """
    c = torch.as_tensor(c).type_as(x)
    return _mobius_matvec(m, x, c)


def _mobius_matvec(m, x, c):
    x_norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    sqrt_c = c ** 0.5
    mx = x @ m.transpose(-1, -2)
    mx_norm = mx.norm(dim=-1, keepdim=True, p=2)
    res_c = tanh(mx_norm / x_norm * artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
    cond = (mx == 0).prod(-1, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return _project(res, c)


def _tensor_dot(x, y):
    """Batched dot product."""
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


def _mobius_addition_batch(x, y, c):
    """Batch Mobius addition for distance matrix computation."""
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def _dist_matrix(x, y, c):
    """Compute pairwise distance matrix."""
    sqrt_c = c ** 0.5
    return (
        2
        / sqrt_c
        * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
    """
    Compute pairwise geodesic distance matrix.
    
    Parameters
    ----------
    x : tensor (N, D)
        First set of points
    y : tensor (M, D)
        Second set of points
    c : float
        Ball curvature
    
    Returns
    -------
    tensor (N, M)
        Pairwise distance matrix
    """
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)


def p2k(x, c):
    """Convert from Poincare ball to Klein model."""
    denom = 1 + c * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, c):
    """Convert from Klein model to Poincare ball."""
    denom = 1 + torch.sqrt(1 - c * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def lorenz_factor(x, *, c=1.0, dim=-1, keepdim=False):
    """Compute Lorenz factor for Klein disk."""
    return 1 / torch.sqrt(1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim))


def poincare_mean(x, c=1.0, weights=None, dim=0):
    """
    Compute weighted Fréchet mean in Poincare ball.
    
    Parameters
    ----------
    x : tensor
        Points in Poincare ball
    c : float
        Ball curvature
    weights : tensor, optional
        Weights for each point (default: uniform)
    dim : int
        Dimension to average over
    
    Returns
    -------
    tensor
        Fréchet mean in Poincare ball
    """
    if weights is None:
        weights = torch.ones(x.shape[dim], device=x.device, dtype=x.dtype) / x.shape[dim]
    
    # Convert to Klein model
    x_klein = p2k(x, c)
    
    # Compute Lorenz factors
    lamb = lorenz_factor(x_klein, c=c, keepdim=True)
    
    # Weighted sum
    if dim == 0:
        mean_klein = torch.sum(weights.unsqueeze(-1) * lamb * x_klein, dim=dim, keepdim=True) / \
                     torch.sum(weights.unsqueeze(-1) * lamb, dim=dim, keepdim=True)
    else:
        mean_klein = torch.sum(weights * lamb * x_klein, dim=dim, keepdim=True) / \
                     torch.sum(weights * lamb, dim=dim, keepdim=True)
    
    # Convert back to Poincare
    mean = k2p(mean_klein, c)
    return mean.squeeze(dim)


def auto_select_c(d):
    """
    Calculate curvature c such that d-dimensional ball has constant volume equal to π.
    """
    from scipy.special import gamma
    dim2 = d / 2.0
    R = gamma(dim2 + 1) / (np.pi ** (dim2 - 1))
    R = R ** (1 / float(d))
    c = 1 / (R ** 2)
    return c


# =============================================================================
# BUSEMANN FUNCTION FOR HOROSPHERICAL CLASSIFICATION
# =============================================================================

def busemann(p, x, c=1.0):
    """
    Busemann function B_p^c(x) for ideal prototype p on boundary.
    
    For curvature c, ball radius R = 1/√c.
    Ideal prototypes p live on the boundary sphere: ‖p‖ = R = 1/√c.
    
    B_p^c(x) = (1/√c) · log(c · ‖p - x‖² / (1 - c·‖x‖²))
    
    Properties:
        - B → -∞ as x → p (close to prototype)
        - B → +∞ as x → -p (far from prototype)
    
    Parameters
    ----------
    p : tensor (K, D)
        Ideal prototypes on boundary, ‖p‖ = 1/√c
    x : tensor (N, D)
        Points inside Poincaré ball, ‖x‖ < 1/√c
    c : float
        Ball curvature (default: 1.0)
    
    Returns
    -------
    tensor (N, K)
        Busemann values for each point to each prototype
    """
    c = torch.as_tensor(c).type_as(x)
    
    # ‖p - x‖² pairwise: x (N,1,D) - p (1,K,D) → (N,K,D) → sum → (N,K)
    diff_sq = (x.unsqueeze(1) - p.unsqueeze(0)).pow(2).sum(-1)  # (N, K)
    
    # 1 - c·‖x‖²: (N, 1)
    x_sq = x.pow(2).sum(-1, keepdim=True)  # (N, 1)
    denom = (1.0 - c * x_sq).clamp(min=1e-6)  # (N, 1)
    
    # B = (1/√c) · log(c · diff_sq / denom)
    sqrt_c = c ** 0.5
    B = (1.0 / sqrt_c) * torch.log((c * diff_sq / denom).clamp(min=1e-6))
    
    return B  # (N, K)


def busemann_batch(p, x, c=1.0):
    """
    Batched Busemann function for (B, N, D) input.
    
    Parameters
    ----------
    p : tensor (K, D)
        Ideal prototypes on boundary
    x : tensor (B, N, D)
        Batched points inside ball
    c : float
        Curvature
    
    Returns
    -------
    tensor (B, N, K)
        Busemann values
    """
    B_size, N, D = x.shape
    x_flat = x.reshape(B_size * N, D)
    B_vals = busemann(p, x_flat, c)  # (B*N, K)
    return B_vals.reshape(B_size, N, -1)  # (B, N, K)
