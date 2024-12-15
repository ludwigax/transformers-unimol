import math
import torch
from torch import nn

from .configuration_unimol import UnimolConfig
from transformers.activations import ACT2FN

class BaseMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_act="silu", mlp_bias=True):
        super().__init__()
        self.up_proj = nn.Linear(input_size, hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[hidden_act]
        self.down_proj = nn.Linear(hidden_size, output_size, bias=mlp_bias)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))

class EncoderMLP(BaseMLP):
    def __init__(self, config: UnimolConfig):
        self.config = config
        super().__init__(config.hidden_size, config.intermediate_size, config.hidden_size, config.hidden_act, config.mlp_bias)

class GateMLP(nn.Module):
    def __init__(self, config: UnimolConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype) + self.bias
    
MLP2LYR = {
    "mlp": BaseMLP,
    "enc_mlp": EncoderMLP,
    "gate": GateMLP,
}

NORM2LYR = {
    "t5n": RMSNorm,
    "ln": LayerNorm,
}


@torch.jit.script
def gaussian(x, mean, std):
    """
    Optimized Gaussian function for PyTorch tensors.

    :param x: The input tensor
    :param mean: The mean for the Gaussian function.
    :param std: The standard deviation for the Gaussian function.

    :return: The output tensor after applying the Gaussian function.
    """
    a = 1 / (math.sqrt(2 * math.pi))
    normalized_diff = (x - mean) / std
    return torch.exp(-0.5 * normalized_diff ** 2) * (a / std)

class GaussianLayer(nn.Module):
    """
    A neural network module implementing a Gaussian layer, useful in graph neural networks.

    Attributes:
        - K: Number of Gaussian kernels.
        - means, stds: Embeddings for the means and standard deviations of the Gaussian kernels.
        - mul, bias: Embeddings for scaling and bias parameters.
    """
    def __init__(self, config: UnimolConfig):
        """
        Initializes the GaussianLayer module.

        :param K: Number of Gaussian kernels.
        :param edge_types: Number of different edge types to consider.

        :return: An instance of the configured Gaussian kernel and edge types.
        """
        super().__init__()
        self.num_kernels = config.num_kernels
        self.num_edge_types = config.num_edge_types
        self.eps = 1e-6

        # Initialize the weights
        self.means = nn.Parameter(torch.empty(self.num_kernels))
        self.stds = nn.Parameter(torch.empty(self.num_kernels))
        self.mul = nn.Embedding(self.num_edge_types, 1)
        self.bias = nn.Embedding(self.num_edge_types, 1)

        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the GaussianLayer module.
        """
        nn.init.uniform_(self.means, 0, 3)
        nn.init.uniform_(self.stds, 0, 3)
        nn.init.constant_(self.mul.weight, 1)
        nn.init.constant_(self.bias.weight, 0)
        

    def forward(self, x, edge_type):
        """
        Forward pass of the GaussianLayer.

        :param x: Input tensor representing distances or other features.
        :param edge_type: Tensor indicating types of edges in the graph.

        :return: Tensor transformed by the Gaussian layer.
        """
        mul = self.mul(edge_type)
        bias = self.bias(edge_type)
        x = mul * x.unsqueeze(-1) + bias

        x = x.expand(-1, -1, -1, self.num_kernels)

        std = self.stds + self.eps
        return gaussian(x, self.means, std)