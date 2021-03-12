# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 17:56:14 2021

@author: ethan
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, ComputePower
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern,plot_codebook_pattern_on_axe 

np.random.seed(7)
n_narrow_beams = [128, 128, 128, 128, 128, 128]
n_wide_beams = [4, 6, 8, 10, 12, 16]
# n_narrow_beams = [128]
# n_wide_beams = [8]
# num_of_beams = [32]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
nepoch = 200

dataset_name = 'Rosslyn_ULA' # 'Rosslyn_ULA' or 'O28B_ULA'
# Training and testing data:
# --------------------------
batch_size = 500
#-------------------------------------------#
# Here should be the data_preparing function
# It is expected to return:
# train_inp, train_out, val_inp, and val_out
#-------------------------------------------#
if dataset_name == 'Rosslyn_ULA':
    h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')[:,antenna_sel]
    h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')[:,antenna_sel]
    loc = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy')
    # h_real = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')
    # h_imag = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')
elif dataset_name == 'O28B_ULA':
    fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/h_real.mat'
    fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/h_imag.mat'
    fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28B_1x64x1_ULA/loc.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    loc = sio.loadmat(fname_loc)['loc']
else:
    raise NameError('Dataset Not Supported')

h = h_real + 1j*h_imag
valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
h = h[valid_ue_idc]
h_real = h_real[valid_ue_idc]
h_imag = h_imag[valid_ue_idc]
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)

u,s,vh = np.linalg.svd(h_scaled[train_idc])

for i,N in enumerate(n_wide_beams):  
    explained_variance = sum(s[:N]**2)/sum(s**2)
    print('{} Beams, explained variance ratio = {:.4f}'.format(N,explained_variance))
    fig,ax = plot_codebook_pattern(vh[:N].conj())
    ax.set_title('SVD {}-Beam Codebook'.format(N))

fig,ax = plot_codebook_pattern(vh[-16:])

fig = plt.figure(figsize=(16,16))
for i in range(16):
    # ax = fig.add_subplot(4,4,i,polar=True)
    ax = plt.subplot2grid((4,4),(i//4,i%4),polar=True)
    plot_codebook_pattern_on_axe(np.expand_dims(vh[i].conj(),axis=0),ax)