# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 15:49:25 2021

@author: ethan
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, ComputePower
import torch.utils.data
import torch.nn as nn
import math
import itertools
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern, plot_codebook_pattern_on_axe, codebook_blockmatrix, DFT_angles

np.random.seed(7)
n_narrow_beams = [128, 128, 128, 128, 128, 128, 128]
n_wide_beams = [2, 4, 6, 8, 10, 12, 16]
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
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

# # Compute EGC gain
# egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna
# train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
# val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


# x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
# x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
# x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

def get_AMCF_beam(omega_min, omega_max, n_antenna=64,spacing=0.5,Q=2**10): 
    #omega_min and omega_max are the min and max beam coverage, Q is the resolution
    q = np.arange(Q)+1
    omega_q = -1 + (2*q-1)/Q
    #array response matrix
    A_phase = np.outer(np.arange(n_antenna),omega_q)
    A = np.exp(1j*np.pi*A_phase)
    #physical angles between +- 90 degree
    theta_min, theta_max = np.arcsin(omega_min),np.arcsin(omega_max)
    #beamwidth in spatial angle
    B = omega_max - omega_min
    mainlobe_idc = ((omega_q >= omega_min) & (omega_q <= omega_max)).nonzero()[0]
    sidelobe_idc = ((omega_q < omega_min) | (omega_q > omega_max)).nonzero()[0]
    #ideal beam amplitude pattern
    g = np.zeros(Q)
    g[mainlobe_idc] = np.sqrt(2/B)
    #g_eps = g in mainlobe and = eps in sidelobe to avoid log0=Nan
    eps = 2**(-52)
    g_eps = g
    g_eps[sidelobe_idc] = eps
    
    v0_phase = B*np.arange(n_antenna)*np.arange(1,n_antenna+1)/2/n_antenna + np.arange(n_antenna)*omega_min
    v0 = 1/np.sqrt(n_antenna)*np.exp(1j*np.pi*v0_phase.conj().T)
    v = v0
    ite = 1
    mse_history = []
    while True:
        mse = np.power(abs(A.conj().T @ v) - g,2).mean()
        mse_history.append(mse)
        if ite >= 10 and abs(mse_history[-1] - mse_history[-2]) < 0.01*mse_history[-1]:
            break
        else:
            ite += 1
        Theta = np.angle(A.conj().T @ v)
        r = g * np.exp(1j*Theta)
        v = 1/np.sqrt(n_antenna)*np.exp(1j*np.angle(A @ r))
    # plt.figure()
    # plt.plot(mse_history)
    # plt.title('MSE history')
    return v
        
# v1 = get_AMCF_beam(-1, -1/2, n_antenna=64,spacing=0.5,Q=2**10)
# plot_codebook_pattern(np.expand_dims(v1,0))

def AMCF_boundaries(n_beams):
    beam_boundaries = np.zeros((n_beams,2))
    for k in range(n_beams):
        beam_boundaries[k,0] = -1 + k*2/n_beams
        beam_boundaries[k,1] = beam_boundaries[k,0] + 2/n_beams
    return beam_boundaries

class Node():
    def __init__(self, n_antenna:int, n_beam:int, codebook:np.ndarray, beam_index:np.ndarray, noise_power=0):
        super(Node, self).__init__()
        self.codebook = codebook
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.beam_index = beam_index # indices of the beams (in the same layer) in the codebook
        self.noise_power = noise_power
        self.parent = None
        self.child = None
        
    def forward(self, h):
        bf_signal = np.matmul(h, self.codebook.conj().T)
        noise_real = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        noise_imag = np.random.normal(loc=0,scale=1,size=bf_signal.shape)*np.sqrt(self.noise_power/2)
        bf_signal = bf_signal + noise_real + 1j*noise_imag
        bf_gain = np.power(np.absolute(bf_signal),2)
        # bf_gain = np.power(np.absolute(np.matmul(h, self.codebook.conj().T)),2)
        return bf_gain
    
    def get_beam_index(self):
        return self.beam_index

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return self.get_child() is None
    
    def is_root(self):
        return self.get_parent() is None
        
class Beam_Search_Tree():
    def __init__(self, n_antenna, n_narrow_beam, k, noise_power):
        super(Beam_Search_Tree, self).__init__()
        assert math.log(n_narrow_beam,k).is_integer()
        self.n_antenna = n_antenna
        self.k = k #number of beams per branch per layer
        self.n_layer = int(math.log(n_narrow_beam,k))
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = noise_power
        self.beam_search_candidates = []
        for l in range(self.n_layer):
            self.beam_search_candidates.append([])
        self.nodes = []
        for l in range(self.n_layer):
            n_nodes = k**l
            n_beams = k**(l+1)
            if l == self.n_layer-1:
                beams = DFT_codebook(nseg=n_beams,n_antenna=n_antenna)
            else:                    
                beam_boundaries = AMCF_boundaries(n_beams)
                beams = np.array([get_AMCF_beam(omega_min=beam_boundaries[i,0], omega_max=beam_boundaries[i,1], n_antenna = n_antenna) for i in range(n_beams)])
                beams = np.flipud(beams)
            beam_idx_per_codebook = [np.arange(i,i+k) for i in np.arange(0,n_beams,k)]
            codebooks = [beams[beam_idx_per_codebook[i]] for i in range(n_nodes)]
            nodes_cur_layer = []
            nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k, codebook=codebooks[i], beam_index=beam_idx_per_codebook[i], noise_power=self.noise_power) for i in range(n_nodes)]
            self.nodes.append(nodes_cur_layer)
            if l > 0:
                parent_nodes = self.nodes[l-1]
                for p_i, p_n in enumerate(parent_nodes):
                    child_nodes = nodes_cur_layer[p_i*k:(p_i+1)*k]
                    p_n.set_child(child_nodes)
                    for c_n in child_nodes:
                        c_n.set_parent(p_n)
        self.root = self.nodes[0][0]
        
    def forward(self, h):
        cur_node = self.root
        while not cur_node.is_leaf():
            bf_gain = cur_node.forward(h)
            next_node_idx = bf_gain.argmax()
            cur_node = cur_node.get_child()[next_node_idx]
        nb_bf_gain = cur_node.forward(h)
        max_nb_bf_gain = nb_bf_gain.max()
        max_nb_idx_local = nb_bf_gain.argmax()
        max_nb_idx_global = cur_node.get_beam_index()[max_nb_idx_local]
        return max_nb_bf_gain, max_nb_idx_global        
        
    def forward_batch(self, hbatch):
        bsize, in_dim = hbatch.shape
        max_nb_idx_batch = np.zeros(bsize,dtype=np.int32)
        max_nb_bf_gain_batch = np.zeros(bsize)
        for b_idx in range(bsize):
            h = hbatch[b_idx]
            nb_gain,nb_idx = self.forward(h)
            max_nb_idx_batch[b_idx] = nb_idx
            max_nb_bf_gain_batch[b_idx] = nb_gain
        return max_nb_bf_gain_batch, max_nb_idx_batch
    

noise_power_dBm = np.arange(-94,-10,10)    
noise_power = 10**(noise_power_dBm/10)
avg_snr_optimal = np.zeros(len(noise_power))
avg_snr_hier = np.zeros(len(noise_power))

for i,n in enumerate(noise_power):    
    bst = Beam_Search_Tree(n_antenna=n_antenna,n_narrow_beam=128,k=2,noise_power=n)
    bst_bf_gain, bst_nb_idx = bst.forward_batch(h)
    bst_snr = 30+10*np.log10(bst_bf_gain)+94-13
    dft_nb_codebook = DFT_codebook(nseg=128,n_antenna=n_antenna)
    nb_bf_gain = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)
    best_nb = np.argmax(nb_bf_gain,axis=1)
    optimal_nb_snr = 30 + 10*np.log10(nb_bf_gain.max(axis=1)) + 94 - 13
    bst_true_snr = nb_bf_gain[tuple(np.arange(nb_bf_gain.shape[0])),tuple(bst_nb_idx)]
    bst_true_snr = 30 + 10*np.log10(bst_true_snr) + 94 - 13
    print('avg. SNR w/. BST = {}, optimal = {}'.format(bst_true_snr.mean(),optimal_nb_snr.mean()))
    avg_snr_optimal[i] = optimal_nb_snr.mean()
    avg_snr_hier[i] = bst_true_snr.mean()
    # plt.figure()
    # plt.hist(optimal_nb_snr, bins=100, density=True, cumulative=True, histtype='step', label='Optimal') 
    # plt.hist(bst_snr, bins=100, density=True, cumulative=True, histtype='step', label='Hierarchical codebook') 
    # plt.legend(loc='upper left')
    # plt.ylabel('CDF')
    # plt.xlabel('SNR (dB)')
    # plt.title('CDF of narrow-beam SNR')
    # plt.show()
    
plt.figure()
plt.plot(noise_power_dBm,avg_snr_optimal,marker='s',label='Optimal')
plt.plot(noise_power_dBm,avg_snr_hier,marker='o',label='Hierarchical')
plt.legend()
plt.xlabel('noise power (dBm)')
plt.ylabel('SNR (dB)')


# for wb_i, n_wb in enumerate(n_wide_beams):
#     AMCF_wb_codebook = sio.loadmat('{}_beam_AMCF_codebook.mat'.format(n_wb))['V'].T
#     AMCF_wb_cv = np.zeros((n_wb,2))
#     AMCF_wb_cv[:,0] = [-1+2/n_wb*k for k in range(n_wb)]
#     AMCF_wb_cv[:,1] = AMCF_wb_cv[:,0]+2/n_wb
#     AMCF_wb_cv = np.arcsin(AMCF_wb_cv)
#     AMCF_wb_cv = np.flipud(AMCF_wb_cv)
    
#     dft_nb_codebook = DFT_codebook(nseg=128,n_antenna=64)
#     dft_nb_az = DFT_angles(128)
#     dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
#     wb_2_nb = {}
#     for bi in range(n_wb):
#         children_nb = ((dft_nb_az>=AMCF_wb_cv[bi,0]) & (dft_nb_az<=AMCF_wb_cv[bi,1])).nonzero()[0]
#         wb_2_nb[bi] = children_nb
    
#     nb_bf_gain = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)
#     wb_bf_gain = np.power(np.absolute(np.matmul(h, AMCF_wb_codebook.conj().T)),2)
    
#     best_wb = np.argmax(wb_bf_gain,axis=1)
#     best_nb = np.argmax(nb_bf_gain,axis=1)
#     optimal_nb_snr = 30 + 10*np.log10(nb_bf_gain.max(axis=1)) + 94 - 13
#     best_nb_az = dft_nb_az[best_nb]
#     wb_az_min = AMCF_wb_cv[best_wb,0]
#     wb_az_max = AMCF_wb_cv[best_wb,1]
#     hierarchical_acc = (best_nb_az>=wb_az_min) & (best_nb_az<=wb_az_max)
#     hierarchical_acc = hierarchical_acc.sum()/len(hierarchical_acc)
#     print(hierarchical_acc)
    
#     best_wb_best_child_nb_snr = np.array([nb_bf_gain[ue_idx,wb_2_nb[best_wb_idx]].max() for ue_idx,best_wb_idx in enumerate(best_wb)])
#     best_wb_best_child_nb_snr = 30 + 10*np.log10(best_wb_best_child_nb_snr) + 94 - 13
#     print('Optimal avg. SNR = {} dB, Hierarchical codebook avg. SNR = {}.'.format(optimal_nb_snr.mean(),best_wb_best_child_nb_snr.mean()))
#     plt.figure()
#     plt.hist(optimal_nb_snr, bins=100, density=True, cumulative=True, histtype='step', label='Optimal') 
#     plt.hist(best_wb_best_child_nb_snr, bins=100, density=True, cumulative=True, histtype='step', label='Hierarchical codebook') 
#     plt.legend(loc='upper left')
#     plt.ylabel('CDF')
#     plt.xlabel('SNR (dB)')
#     plt.title('CDF of narrow-beam SNR, number of wide beams = {}'.format(n_wb))
#     plt.show()
    
#     plot_codebook_pattern(np.expand_dims(AMCF_wb_codebook[1],0))
    
# dft_cb = DFT_codebook(nseg=4,n_antenna=64)
# plot_codebook_pattern(np.expand_dims(dft_cb[0],0))