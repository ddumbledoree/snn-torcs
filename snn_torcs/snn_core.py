"""
SNN core building blocks extracted from mySRL/S_Net.py.

LIF neuron with surrogate gradient, no global state — all hyperparams
are passed per-instance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActFun(torch.autograd.Function):
    """Surrogate gradient spike activation (piecewise linear approximation)."""

    @staticmethod
    def forward(ctx, input, thresh, lens):
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.lens = lens
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.lens
        temp = temp / (2 * ctx.lens)
        return grad_input * temp.float(), None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with surrogate gradient.

    Args:
        thresh: firing threshold
        lens:   surrogate gradient half-width
        decay:  membrane potential decay constant
    Input shape:  (T, B, ...)
    Output shape: (T, B, ...)  — binary spike trains
    """

    def __init__(self, thresh=0.5, lens=0.5, decay=0.25):
        super().__init__()
        self.thresh = thresh
        self.lens = lens
        self.decay = decay

    def forward(self, x):
        T = x.size(0)
        mem = torch.zeros_like(x[0])
        spike = torch.zeros_like(x[0])
        output = torch.zeros_like(x)
        mem_old = torch.zeros_like(x[0])

        for t in range(T):
            if t >= 1:
                mem = mem_old * self.decay * (1 - spike.detach()) + x[t]
            else:
                mem = x[t]
            spike = ActFun.apply(mem, self.thresh, self.lens)
            mem_old = mem.clone()
            output[t] = spike
        return output


class SnnLinear(nn.Linear):
    """Linear layer operating over time dimension.

    Input:  (T, B, in_features)
    Output: (T, B, out_features)
    Weight shape identical to nn.Linear — enables direct weight transfer.
    """

    def forward(self, x):
        T = x.size(0)
        out = torch.zeros(T, x.size(1), self.out_features,
                          device=x.device, dtype=x.dtype)
        for t in range(T):
            out[t] = F.linear(x[t], self.weight, self.bias)
        return out
