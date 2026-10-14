"""
FC-ANN and FC-SNN models for TORCS discrete-action Double DQN.

Both share the same layer topology (68 → 256 → 256 → 7) so that
ANN weights can be directly transferred to SNN.
"""
import torch
import torch.nn as nn

from snn_torcs.snn_core import LIFNeuron, SnnLinear


class FCANN(nn.Module):
    """Fully-connected ANN for Q-value estimation.

    Architecture: Linear → ReLU → Linear → ReLU → Linear
    Input:  (B, input_dim)
    Output: (B, output_dim)  — Q-values
    """

    def __init__(self, input_dim=68, hidden_dim=256, output_dim=7):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FCSNN(nn.Module):
    """Fully-connected SNN for Q-value estimation.

    Architecture: SnnLinear → LIF → SnnLinear → LIF → rate-decode → Linear
    Input:  (B, input_dim)
    Output: (B, output_dim)  — Q-values
    """

    def __init__(self, input_dim=68, hidden_dim=256, output_dim=7,
                 time_window=4, thresh=0.5, lens=0.5, decay=0.25):
        super().__init__()
        self.time_window = time_window
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = SnnLinear(input_dim, hidden_dim)
        self.lif1 = LIFNeuron(thresh, lens, decay)
        self.fc2 = SnnLinear(hidden_dim, hidden_dim)
        self.lif2 = LIFNeuron(thresh, lens, decay)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        B = x.shape[0]
        x_seq = x.unsqueeze(0).repeat(self.time_window, 1, 1)  # (T, B, D)

        out = self.fc1(x_seq)   # (T, B, H)
        out = self.lif1(out)
        out = self.fc2(out)     # (T, B, H)
        out = self.lif2(out)

        out = out.sum(dim=0) / self.time_window  # rate decoding → (B, H)
        out = self.fc3(out)                       # (B, output_dim)
        return out


def transfer_ann_to_snn(ann: FCANN, snn: FCSNN):
    """Copy FCANN weights into FCSNN (same topology required)."""
    snn.fc1.weight.data.copy_(ann.net[0].weight.data)
    snn.fc1.bias.data.copy_(ann.net[0].bias.data)
    snn.fc2.weight.data.copy_(ann.net[2].weight.data)
    snn.fc2.bias.data.copy_(ann.net[2].bias.data)
    snn.fc3.weight.data.copy_(ann.net[4].weight.data)
    snn.fc3.bias.data.copy_(ann.net[4].bias.data)
