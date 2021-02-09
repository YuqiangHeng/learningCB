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
import beam_utils

class Hybrid_Beamformer(Module):
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

    def __init__(self, n_antenna: int, n_beam: int, n_rf: int, n_stream: int = 1, use_bias: bool = False, scale: float=1, init_criterion = 'xavier_normal') -> None:
        super(Hybrid_Beamformer, self).__init__()
        self.n_antenna = n_antenna
        self.in_dim = self.n_antenna * 2
        self.n_rf = n_rf
        self.n_beam = n_beam
        self.scale = scale
        self.init_criterion = init_criterion
        self.n_stream = n_stream
        self.theta = Parameter(torch.Tensor(self.n_beam, self.n_antenna, self.n_rf)) 
        self.real_kernel = Parameter(torch.Tensor(self.n_beam, self.n_rf, self.n_stream)) 
        self.imag_kernel = Parameter(torch.Tensor(self.n_beam, self.n_rf, self.n_stream)) 
        self.use_bias = use_bias
        self.fb_norm = Complex_Frobenius_Norm((self.n_antenna*2,self.n_stream*2))
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(self.n_beam, self.n_antenna, self.n_stream)) 
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        init.uniform_(self.theta, a=0, b=2*np.pi)
        self.analog_real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
        self.analog_imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #
        
        if self.init_criterion == 'xavier_normal':
            init.xavier_normal_(self.real_kernel,gain = 1/np.sqrt(2))
            init.xavier_normal_(self.imag_kernel,gain = 1/np.sqrt(2))
        elif self.init_criterion == 'kaiming_normal':
            init.kaiming_normal_(self.real_kernel)
            init.kaiming_normal_(self.imag_kernel)        
        else:
             raise  NotImplementedError 
             
        if self.use_bias:
            if self.init_criterion == 'xavier_normal':
                init.xavier_normal_(self.bias,gain = 1/np.sqrt(2))
            elif self.init_criterion == 'kaiming_normal':
                init.kaiming_normal_(self.bias)
            else:
                 raise  NotImplementedError             
             
    def forward(self, inputs: Tensor) -> Tensor: 
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1).unsqueeze(1) # so that inputs has shape b x 1 x 1 x N_T
        
        cat_kernels_4_real_digital = torch.cat(
            (self.real_kernel, -self.imag_kernel),
            dim=-1
        )
        cat_kernels_4_imag_digital = torch.cat(
            (self.imag_kernel, self.real_kernel),
            dim=-1
        )
        cat_kernels_4_complex_digital = torch.cat(
            (cat_kernels_4_real_digital, cat_kernels_4_imag_digital),
            dim=1
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]

        self.real_kernel_analog = (1 / self.scale) * torch.cos(self.theta)  #
        self.imag_kernel_analog = (1 / self.scale) * torch.sin(self.theta)  #        
        cat_kernels_4_real_analog = torch.cat(
            (self.real_kernel_analog, -self.imag_kernel_analog),
            dim=-1
        )
        cat_kernels_4_imag_analog = torch.cat(
            (self.imag_kernel_analog, self.real_kernel_analog),
            dim=-1
        )
        cat_kernels_4_complex_analog = torch.cat(
            (cat_kernels_4_real_analog, cat_kernels_4_imag_analog),
            dim=1
        )  # This block matrix represents the conjugate transpose of the original:
        # [ W_R, -W_I; W_I, W_R]        
        cat_kernels_4_complex_hybrid = torch.matmul(cat_kernels_4_complex_analog, cat_kernels_4_complex_digital) # shape n_beam x n_antenna*2 x n_stream*2
        norm_factor = self.fb_norm(cat_kernels_4_complex_hybrid) # shape n_beam vector
        norm_factor = norm_factor.unsqueeze(1).unsqueeze(1)
        norm_factor = norm_factor.repeat(1,self.n_antenna*2,self.n_stream*2)
        # norm_factor = cat_kernels_4_complex_hybrid.sum(dim=0).repeat(1, self.n_antenna*2)
        cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid * (1/norm_factor)
        if self.use_bias:
            cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid_normalized + self.bias
        output = torch.matmul(inputs, cat_kernels_4_complex_hybrid_normalized)
        return output.squeeze()
    
    def get_hybrid_weights(self):
        with torch.no_grad():
            cat_kernels_4_real_digital = torch.cat(
                (self.real_kernel, -self.imag_kernel),
                dim=-1
            )
            cat_kernels_4_imag_digital = torch.cat(
                (self.imag_kernel, self.real_kernel),
                dim=-1
            )
            cat_kernels_4_complex_digital = torch.cat(
                (cat_kernels_4_real_digital, cat_kernels_4_imag_digital),
                dim=1
            )  # This block matrix represents the conjugate transpose of the original:
            # [ W_R, -W_I; W_I, W_R]
    
            self.real_kernel_analog = (1 / self.scale) * torch.cos(self.theta)  #
            self.imag_kernel_analog = (1 / self.scale) * torch.sin(self.theta)  #        
            cat_kernels_4_real_analog = torch.cat(
                (self.real_kernel_analog, -self.imag_kernel_analog),
                dim=-1
            )
            cat_kernels_4_imag_analog = torch.cat(
                (self.imag_kernel_analog, self.real_kernel_analog),
                dim=-1
            )
            cat_kernels_4_complex_analog = torch.cat(
                (cat_kernels_4_real_analog, cat_kernels_4_imag_analog),
                dim=1
            )  # This block matrix represents the conjugate transpose of the original:
            # [ W_R, -W_I; W_I, W_R]        
            cat_kernels_4_complex_hybrid = torch.matmul(cat_kernels_4_complex_analog, cat_kernels_4_complex_digital) # shape n_beam x n_antenna*2 x n_stream*2
            norm_factor = self.fb_norm(cat_kernels_4_complex_hybrid) # shape n_beam vector
            norm_factor = norm_factor.unsqueeze(1).unsqueeze(1)
            norm_factor = norm_factor.repeat(1,self.n_antenna*2,self.n_stream*2)
            # norm_factor = cat_kernels_4_complex_hybrid.sum(dim=0).repeat(1, self.n_antenna*2)
            cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid * (1/norm_factor)  
        cat_kernels_4_complex_hybrid_normalized = cat_kernels_4_complex_hybrid_normalized.detach().numpy()
        hybrid_kernel_real = cat_kernels_4_complex_hybrid_normalized[:,:self.n_antenna,:self.n_stream]
        hybrid_kernel_imag = cat_kernels_4_complex_hybrid_normalized[:,self.n_antenna:,:self.n_stream]
        hybrid_beam_weights = hybrid_kernel_real + 1j*hybrid_kernel_imag
        return hybrid_beam_weights
    

    
class Complex_Dense(Module):
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

    def __init__(self, in_features: int, out_features: int, use_bias: bool = False, scale: float=1, init_criterion = 'xavier_normal') -> None:
        super(Complex_Dense, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        self.init_criterion = init_criterion
        self.real_kernel = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.imag_kernel = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = Parameter(torch.Tensor(self.in_features, self.out_features)) 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        
        if self.init_criterion == 'xavier_normal':
            init.xavier_normal_(self.real_kernel,gain = 1/np.sqrt(2))
            init.xavier_normal_(self.imag_kernel,gain = 1/np.sqrt(2))
        elif self.init_criterion == 'kaiming_normal':
            init.kaiming_normal_(self.real_kernel)
            init.kaiming_normal_(self.imag_kernel)        
        else:
             raise  NotImplementedError      
                        
    def forward(self, inputs: Tensor) -> Tensor:    
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
        # weight_power = torch.pow(self.real_kernel,2) + torch.pow(self.imag_kernel,2)
        # weight_magnitue = torch.sqrt(weight_power)
        # output = F.linear(inputs, cat_kernels_4_complex)
        output = torch.matmul(inputs, (1 / self.scale) * cat_kernels_4_complex)
        if self.use_bias:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )
    
    
    def get_weights(self):
        return self.real_kernel, self.imag_kernel
    
    
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

    def __init__(self, in_features: int, out_features: int, scale: float=1, theta = None) -> None:
        super(PhaseShifter, self).__init__()
        self.in_features = in_features
        self.in_dim = self.in_features//2
        self.out_features = out_features
        self.scale = scale
        # self.theta = Parameter(torch.Tensor(self.out_features, self.in_dim))
        self.theta = Parameter(torch.Tensor(self.in_dim, self.out_features)) 
        self.reset_parameters(theta)

    def reset_parameters(self, theta = None) -> None:
        if theta is None:
            init.uniform_(self.theta, a=0, b=2*np.pi)
        else:
            assert theta.shape == (self.in_dim,self.out_features)
            self.theta = Parameter(theta) 
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
    
    def get_theta(self) -> torch.Tensor:
        return self.theta.detach().clone()
    
    def get_weights(self) -> torch.Tensor:
        with torch.no_grad():
            real_kernel = (1 / self.scale) * torch.cos(self.theta)  #
            imag_kernel = (1 / self.scale) * torch.sin(self.theta)  #        
            # cat_kernels_4_real = torch.cat(
            #     (real_kernel, -imag_kernel),
            #     dim=-1
            # )
            # cat_kernels_4_imag = torch.cat(
            #     (imag_kernel, real_kernel),
            #     dim=-1
            # )
            # cat_kernels_4_complex = torch.cat(
            #     (cat_kernels_4_real, cat_kernels_4_imag),
            #     dim=0
            # )  # This block matrix represents the conjugate transpose of the original:
            # # [ W_R, -W_I; W_I, W_R]
            beam_weights = real_kernel + 1j*imag_kernel
        return beam_weights



class DFT_Codebook_Layer(Module):
    def __init__(self, n_antenna, azimuths):
        super(DFT_Codebook_Layer, self).__init__()
        self.n_antenna = n_antenna
        self.n_beam = len(azimuths)
        dft_codebook = beam_utils.DFT_beam_blockmatrix(n_antenna = n_antenna, azimuths = azimuths)
        self.codebook_blockmatrix = torch.from_numpy(dft_codebook).float()
        self.codebook_blockmatrix.requires_grad = False
        self.codebook = beam_utils.DFT_beam(n_antenna = n_antenna, azimuths = azimuths).T
        
    def forward(self, x):
        bf_signal = torch.matmul(x, self.codebook_blockmatrix)
        return bf_signal
    
    def get_weights(self, x):
        return self.codebook
    

class ComputePower(Module):
    def __init__(self, in_shape):
        super(ComputePower, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[:,:self.len_real]
        imag_part = x[:,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        return abs_values
    
class ComputePower_DoubleBatch(Module):
    def __init__(self, in_shape):
        super(ComputePower_DoubleBatch, self).__init__()
        self.shape = in_shape
        self.len_real = int(self.shape/2)

    def forward(self, x):
        real_part = x[...,:self.len_real]
        imag_part = x[...,self.len_real:]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        return abs_values

class Complex_Frobenius_Norm(Module):
    def __init__(self, in_shape):
        super(Complex_Frobenius_Norm, self).__init__()
        self.n_r = in_shape[0]
        self.n_c = in_shape[1]
        self.n_r_real = self.n_r//2
        self.n_c_real = self.n_c//2

    def forward(self, x): # x is b_size x n_r x n_c
        real_part = x[:,:self.n_r_real,:self.n_c_real]
        imag_part = x[:,self.n_r_real:,:self.n_c_real]
        sq_real = torch.pow(real_part,2)
        sq_imag = torch.pow(imag_part,2)
        abs_values = sq_real + sq_imag
        abs_values = abs_values.sum(1).sum(1)
        fb_norm = torch.sqrt(abs_values)
        return fb_norm
        
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
