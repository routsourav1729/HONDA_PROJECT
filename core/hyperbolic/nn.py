"""
Neural network layers for hyperbolic geometry.
Based on HypGCD and Hyp-OW implementations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init

from . import pmath


class ToPoincare(nn.Module):
    """
    Module that maps points from Euclidean space to Poincare ball.
    
    Uses exponential map from origin: expmap0(x)
    Optionally clips input norms before projection for stability.
    
    Parameters
    ----------
    c : float
        Ball curvature (c > 0). Smaller c = flatter space.
    train_c : bool
        Whether to make curvature trainable
    train_x : bool
        Whether to learn a base point (instead of origin)
    ball_dim : int
        Dimension of the ball (required if train_x=True)
    riemannian : bool
        Whether to use Riemannian gradient scaling
    clip_r : float, optional
        Clip input norms to this value before projection (for stability)
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, 
                 riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    f"if train_x=True, ball_dim must be integer, got {ball_dim}"
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x
        self.clip_r = clip_r

        # Set up Riemannian gradient
        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        """
        Map Euclidean vectors to Poincare ball.
        
        Parameters
        ----------
        x : tensor
            Euclidean vectors of shape (..., D)
        
        Returns
        -------
        tensor
            Points on Poincare ball of shape (..., D)
        """
        # Optional: clip norms to prevent instability
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
        
        # Project to Poincare ball
        if self.train_x:
            # Use learned base point
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        
        # Project from origin (standard case)
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return f"c={self.c}, train_x={self.train_x}, clip_r={self.clip_r}"


class FromPoincare(nn.Module):
    """
    Module that maps points from Poincare ball back to Euclidean space.
    Uses logarithmic map to origin: logmap0(x)
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None):
        super(FromPoincare, self).__init__()

        if train_x:
            if ball_dim is None:
                raise ValueError(
                    f"if train_x=True, ball_dim must be integer, got {ball_dim}"
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_c = train_c
        self.train_x = train_x

    def forward(self, x):
        """
        Map Poincare ball points to Euclidean space.
        """
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return pmath.logmap(xp, x, c=self.c)
        return pmath.logmap0(x, c=self.c)

    def extra_repr(self):
        return f"c={self.c}, train_x={self.train_x}"


class HyperbolicMLR(nn.Module):
    """
    Multinomial Logistic Regression in Hyperbolic space.
    Performs classification directly on Poincare ball.
    """

    def __init__(self, ball_dim, n_classes, c):
        super(HyperbolicMLR, self).__init__()
        self.a_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.p_vals = nn.Parameter(torch.Tensor(n_classes, ball_dim))
        self.c = c
        self.n_classes = n_classes
        self.ball_dim = ball_dim
        self.reset_parameters()

    def forward(self, x, c=None):
        if c is None:
            c = torch.as_tensor(self.c).type_as(x)
        else:
            c = torch.as_tensor(c).type_as(x)
        p_vals_poincare = pmath.expmap0(self.p_vals, c=c)
        conformal_factor = 1 - c * p_vals_poincare.pow(2).sum(dim=1, keepdim=True)
        a_vals_poincare = self.a_vals * conformal_factor
        logits = self._hyperbolic_softmax(x, a_vals_poincare, p_vals_poincare, c)
        return logits
    
    def _hyperbolic_softmax(self, X, A, P, c):
        """Compute hyperbolic softmax logits."""
        lambda_pkc = 2 / (1 - c * P.pow(2).sum(dim=1))
        k = lambda_pkc * torch.norm(A, dim=1) / torch.sqrt(c)
        mob_add = pmath._mobius_addition_batch(-P, X, c)
        num = 2 * torch.sqrt(c) * torch.sum(mob_add * A.unsqueeze(1), dim=-1)
        denom = torch.norm(A, dim=1, keepdim=True) * (1 - c * mob_add.pow(2).sum(dim=2))
        logit = k.unsqueeze(1) * pmath.arsinh(num / denom)
        return logit.permute(1, 0)

    def reset_parameters(self):
        init.kaiming_uniform_(self.a_vals, a=math.sqrt(5))
        init.kaiming_uniform_(self.p_vals, a=math.sqrt(5))

    def extra_repr(self):
        return f"ball_dim={self.ball_dim}, n_classes={self.n_classes}, c={self.c}"


class HypLinear(nn.Module):
    """
    Linear layer in hyperbolic space using Mobius matrix-vector multiplication.
    """
    
    def __init__(self, in_features, out_features, c, bias=True):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, c=None):
        if c is None:
            c = self.c
        mv = pmath.mobius_matvec(self.weight, x, c=c)
        if self.bias is None:
            return pmath.project(mv, c=c)
        else:
            bias = pmath.expmap0(self.bias, c=c)
            return pmath.project(pmath.mobius_add(mv, bias), c=c)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, c={self.c}"


class HyperbolicDistanceLayer(nn.Module):
    """Layer that computes pairwise hyperbolic distances."""
    
    def __init__(self, c):
        super(HyperbolicDistanceLayer, self).__init__()
        self.c = c

    def forward(self, x1, x2, c=None):
        if c is None:
            c = self.c
        return pmath.dist(x1, x2, c=c, keepdim=True)

    def extra_repr(self):
        return f"c={self.c}"
