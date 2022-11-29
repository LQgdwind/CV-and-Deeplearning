from torch.nn import init
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import math


class Maxout(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, pieces, bias=True):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.weight = Parameter(torch.Tensor(pieces, out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(pieces, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        output = input.matmul(self.weight.permute(0, 2, 1)).permute((1, 0, 2)) + self.bias
        output = torch.max(output, dim=1)[0]
        return output