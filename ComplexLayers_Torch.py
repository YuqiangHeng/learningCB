# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:49:09 2021

@author: ethan
"""
import torch
import numpy as np
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
               

class PhaseShifter(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
    Attributes:
        theta: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from uniform(0,2*pi)
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    scale: float
    theta: Tensor

    def __init__(self, in_features: int, out_features: int, scale: float=1) -> None:
        super(PhaseShifter, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        # self.theta = Parameter(torch.Tensor(self.out_features, self.in_dim))
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.uniform_(self.theta, a=0, b=2*np.pi)
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
    
    def forward(self, inputs: Tensor) -> Tensor:
        self.real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex = torch.cat(
            (cat_kernels_4_real, cat_kernels_4_imag),
            dim=0
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        # output = F.linear(inputs, cat_kernels_4_complex)
        output = torch.matmul(inputs, cat_kernels_4_complex)
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
class PowerPooling(Module):
    def __init__(self, in_shape):
        super(PowerPooling, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        max_pooling = torch.max(abs_values, dim=-1)[0]
        max_pooling = torch.unsqueeze(max_pooling,dim=-1)
        return max_pooling
