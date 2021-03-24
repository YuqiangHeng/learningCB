# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:04:13 2021

@author: ethan
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, ComputePower
import torch.utils.data
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern, plot_codebook_pattern_on_axe, codebook_blockmatrix, DFT_angles, Beam_Search_Tree, AMCF_boundaries

np.random.seed(7)
n_narrow_beams = [128, 128, 128, 128, 128, 128]
n_wide_beams = [4, 6, 8, 10, 12, 16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)

tx_power_dBm = 30
noise_power_dBm = -94
noise_power = 10**(noise_power_dBm-tx_power_dBm/10)

dataset_name = 'O28B_ULA' # 'Rosslyn_ULA' or 'O28B_ULA' or 'I3_ULA' or 'O28_ULA'

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
elif dataset_name == 'I3_ULA':
    fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/I3_1x64x1_ULA_BS2/h_real.mat'
    fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/I3_1x64x1_ULA_BS2/h_imag.mat'
    fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/I3_1x64x1_ULA_BS2/loc.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    loc = sio.loadmat(fname_loc)['loc']
elif dataset_name == 'O28_ULA':
    fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28_UE800_1200_BS3_1x64x1_ULA/h_real.mat'
    fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28_UE800_1200_BS3_1x64x1_ULA/h_imag.mat'
    fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/O28_UE800_1200_BS3_1x64x1_ULA/loc.mat'
    h_real = sio.loadmat(fname_h_real)['h_real']
    h_imag = sio.loadmat(fname_h_imag)['h_imag']
    loc = sio.loadmat(fname_loc)['loc']
else:
    raise NameError('Dataset Not Supported')
    
# plt.figure(figsize=(8,50))
# plt.scatter(loc[:,0],loc[:,1],s=0.01)
h = h_real + 1j*h_imag
valid_ue_idc = np.array([row_idx for (row_idx,row) in enumerate(np.concatenate((h_real,h_imag),axis=1)) if not all(row==0)])
h = h[valid_ue_idc]
h_real = h_real[valid_ue_idc]
h_imag = h_imag[valid_ue_idc]
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

torch_x_train = torch.from_numpy(x_train)
torch_y_train = torch.from_numpy(y_train)
torch_x_val = torch.from_numpy(x_val)
torch_y_val = torch.from_numpy(y_val)
torch_x_test = torch.from_numpy(x_test)
torch_y_test = torch.from_numpy(y_test)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)

class Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None, noise_power = 0.0, norm_factor = 1.0):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        self.noise_power = float(noise_power)
        self.norm_factor = float(norm_factor)
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:            
            self.complex_codebook = complex_codebook # n_beams x n_antenna
            cb_blockmatrix = codebook_blockmatrix(self.complex_codebook.T)
            self.codebook = torch.from_numpy(cb_blockmatrix).float()
            self.codebook.requires_grad = False
            
        self.compute_power = ComputePower(2*n_wide_beam)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(in_features=n_wide_beam, out_features=2*n_wide_beam)
        self.dense2 = nn.Linear(in_features=2*n_wide_beam, out_features=3*n_wide_beam)
        self.dense3 = nn.Linear(in_features=3*n_wide_beam, out_features=n_narrow_beam)
        self.softmax = nn.Softmax()
    def forward(self, x):
        if self.trainable_codebook:
            bf_signal = self.codebook(x)
        else:
            bf_signal = torch.matmul(x,self.codebook)
        noise_vec = torch.normal(0,1, size=bf_signal.size())*torch.sqrt(torch.tensor([self.noise_power/2]))/torch.tensor([self.norm_factor])
        bf_signal = bf_signal + noise_vec
        bf_power = self.compute_power(bf_signal)
        out = self.relu(bf_power)
        out = self.relu(self.dense1(out))
        out = self.relu(self.dense2(out))
        out = self.dense3(out)
        return out
    def get_codebook(self) -> np.ndarray:
        if self.trainable_codebook:
            return self.codebook.get_weights().detach().clone().numpy()
        else:
            # return DFT_codebook(nseg=self.n_wide_beam,n_antenna=self.n_antenna).T
            return self.complex_codebook
        
def eval_model(model, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    test_acc = 0
    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        var_X_batch = X_batch.float()
        var_y_batch = y_batch.long ()  
        output = model(var_X_batch)
        loss = loss_fn(output, var_y_batch)
        test_loss += loss.detach().item()
        test_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
    test_loss /= batch_idx + 1
    test_acc /= batch_idx + 1
    return test_loss, test_acc

dft_codebook_acc = []
learnable_codebook_acc = []
AMCF_codebook_acc = []

dft_codebook_topk_acc = []
learnable_codebook_topk_acc = []
AMCF_codebook_topk_acc = []

dft_codebook_topk_gain = []
learned_codebook_topk_gain= []
AMCF_codebook_topk_gain = []

optimal_gains = []
learned_codebooks = []
dft_codebooks = []
AMCF_codebooks = []
for n_wb_i, n_wb in enumerate(n_wide_beams):
    n_nb = n_narrow_beams[n_wb_i]
    print('{} Wide Beams, {} Narrow Beams.'.format(n_wb,n_nb))
    dft_nb_codebook = DFT_codebook(nseg=n_nb,n_antenna=n_antenna)
    label = np.argmax(np.power(np.absolute(np.matmul(h_scaled, dft_nb_codebook.conj().T)),2),axis=1)
    soft_label = np.power(np.absolute(np.matmul(h, dft_nb_codebook.conj().T)),2)

    x_train,y_train = h_concat_scaled[train_idc,:],label[train_idc]
    x_val,y_val = h_concat_scaled[val_idc,:],label[val_idc]
    x_test,y_test = h_concat_scaled[test_idc,:],label[test_idc]
    
    torch_x_train,torch_y_train = torch.from_numpy(x_train),torch.from_numpy(y_train)
    torch_x_val,torch_y_val = torch.from_numpy(x_val),torch.from_numpy(y_val)
    torch_x_test,torch_y_test = torch.from_numpy(x_test),torch.from_numpy(y_test)
    
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
    test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)
    
    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
    
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                               trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_codebook_model.load_state_dict(torch.load('./Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)))
    learnable_codebook_test_loss,learnable_codebook_test_acc = eval_model(learnable_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    learnable_codebook_acc.append(learnable_codebook_test_acc)
    y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
    topk_bf_gain_learnable_codebook = []
    topk_acc_learnable_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_learned_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_learnable_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_learnable_codebook.append(topk_acc)
    topk_bf_gain_learnable_codebook = np.array(topk_bf_gain_learnable_codebook)
    learned_codebook_topk_gain.append(topk_bf_gain_learnable_codebook)
    learned_codebooks.append(learnable_codebook_model.get_codebook()) 
    topk_acc_learnable_codebook = np.array(topk_acc_learnable_codebook).mean(axis=0)
    learnable_codebook_topk_acc.append(topk_acc_learnable_codebook)

    
    dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_codebook_model.load_state_dict(torch.load('./Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)))
    dft_codebook_test_loss,dft_codebook_test_acc = eval_model(dft_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    dft_codebook_acc.append(dft_codebook_test_acc)
    y_test_predict_dft_codebook = dft_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
    topk_bf_gain_dft_codebook = []
    topk_acc_dft_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_dft_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_dft_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_dft_codebook.append(topk_acc)        
    topk_bf_gain_dft_codebook = np.array(topk_bf_gain_dft_codebook)
    dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
    dft_codebooks.append(dft_codebook_model.get_codebook())
    topk_acc_dft_codebook = np.array(topk_acc_dft_codebook).mean(axis=0)
    dft_codebook_topk_acc.append(topk_acc_dft_codebook)
    
    AMCF_wb_codebook = sio.loadmat('{}_beam_AMCF_codebook.mat'.format(n_wb))['V'].T
    AMCF_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                          trainable_codebook=False,complex_codebook=AMCF_wb_codebook,
                                          noise_power=noise_power,norm_factor=norm_factor)
    AMCF_codebook_model.load_state_dict(torch.load('./Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)))
    AMCF_codebook_test_loss,AMCF_codebook_test_acc = eval_model(AMCF_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    AMCF_codebook_acc.append(AMCF_codebook_test_acc)
    y_test_predict_AMCF_codebook = AMCF_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_AMCF_codebook = (-y_test_predict_AMCF_codebook).argsort()
    topk_bf_gain_AMCF_codebook = []
    topk_acc_AMCF_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_AMCF_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_AMCF_codebook.append(topk_gains)
        topk_acc = [ue_bf_gain.argmax() in pred_sort[:k] for k in range(1,11)]
        topk_acc_AMCF_codebook.append(topk_acc)        
    topk_bf_gain_AMCF_codebook = np.array(topk_bf_gain_AMCF_codebook)
    AMCF_codebook_topk_gain.append(topk_bf_gain_AMCF_codebook)
    AMCF_codebooks.append(AMCF_codebook_model.get_codebook())
    topk_acc_AMCF_codebook = np.array(topk_acc_AMCF_codebook).mean(axis=0)
    AMCF_codebook_topk_acc.append(topk_acc_AMCF_codebook)
    
    optimal_gains.append(soft_label[test_idc,:].max(axis=-1))
 
dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
AMCF_codebook_topk_gain = np.array(AMCF_codebook_topk_gain)

optimal_gains = np.array(optimal_gains)

dft_codebook_topk_snr = 30 + 10*np.log10(dft_codebook_topk_gain) + 94 -13
learned_codebook_topk_snr = 30 + 10*np.log10(learned_codebook_topk_gain) + 94 -13
AMCF_codebook_topk_snr = 30 + 10*np.log10(AMCF_codebook_topk_gain) + 94 -13
optimal_snr = 30 + 10*np.log10(optimal_gains) + 94 -13


dft_codebook_topk_acc = np.array(dft_codebook_topk_acc)
learnable_codebook_topk_acc = np.array(learnable_codebook_topk_acc)
AMCF_codebook_topk_acc = np.array(AMCF_codebook_topk_acc)


"""    
compute exhaustive beam search acc and snr
"""
dft_nb_codebook = DFT_codebook(nseg=128,n_antenna=64)
dft_nb_az = DFT_angles(128)
dft_nb_az = np.arcsin(1/0.5*dft_nb_az)
nb_bf_signal = np.matmul(h[test_idc], dft_nb_codebook.conj().T)
nb_bf_noise_real = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=nb_bf_signal.shape)*np.sqrt(noise_power/2)
nb_bf_signal_with_noise = nb_bf_signal + nb_bf_noise_real + 1j*nb_bf_noise_imag
nb_bf_gain = np.power(np.absolute(nb_bf_signal),2)
nb_bf_gain_with_noise = np.power(np.absolute(nb_bf_signal_with_noise),2)
best_nb_noisy = np.argmax(nb_bf_gain_with_noise,axis=1)
best_nb = np.argmax(nb_bf_gain,axis=1)
best_nb_az = dft_nb_az[best_nb]
exhaustive_acc = (best_nb_noisy==best_nb).mean()
genie_nb_snr = 30 + 10*np.log10(nb_bf_gain.max(axis=1)) + 94 - 13
exhaustive_nb_snr = np.array([nb_bf_gain[ue_idx,best_nb_idx_noisy] for ue_idx,best_nb_idx_noisy in enumerate(best_nb_noisy)])
exhaustive_nb_snr = 30 + 10*np.log10(exhaustive_nb_snr) + 94 - 13

"""    
compute 2-iter hierarchical beam search acc and snr
"""    
two_tier_AMCF_acc = []
two_tier_AMCF_snr = []
for wb_i, n_wb in enumerate(n_wide_beams):
    AMCF_wb_codebook = sio.loadmat('{}_beam_AMCF_codebook.mat'.format(n_wb))['V'].T
    AMCF_wb_cv = np.zeros((n_wb,2))
    AMCF_wb_cv[:,0] = [-1+2/n_wb*k for k in range(n_wb)]
    AMCF_wb_cv[:,1] = AMCF_wb_cv[:,0]+2/n_wb
    AMCF_wb_cv = np.arcsin(AMCF_wb_cv)
    AMCF_wb_cv = np.flipud(AMCF_wb_cv)
    
    wb_2_nb = {}
    for bi in range(n_wb):
        children_nb = ((dft_nb_az>=AMCF_wb_cv[bi,0]) & (dft_nb_az<=AMCF_wb_cv[bi,1])).nonzero()[0]
        wb_2_nb[bi] = children_nb
        
    wb_bf_signal = np.matmul(h[test_idc], AMCF_wb_codebook.conj().T)
    wb_bf_noise_real = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_noise_imag = np.random.normal(loc=0,scale=1,size=wb_bf_signal.shape)*np.sqrt(noise_power/2)
    wb_bf_signal_with_noise = wb_bf_signal + wb_bf_noise_real + 1j*wb_bf_noise_imag
    wb_bf_gain = np.power(np.absolute(wb_bf_signal),2)
    wb_bf_gain_with_noise = np.power(np.absolute(wb_bf_signal_with_noise),2)
    best_wb_noisy = np.argmax(wb_bf_gain_with_noise,axis=1)
    
    wb_az_min = AMCF_wb_cv[best_wb_noisy,0]
    wb_az_max = AMCF_wb_cv[best_wb_noisy,1]
    hierarchical_acc = ((best_nb_az>=wb_az_min) & (best_nb_az<=wb_az_max)).mean()
    two_tier_AMCF_acc.append(hierarchical_acc)
    print('{}-Beam 2-tier AMCF codebook hierarchical accuracy = {}.'.format(n_wb,hierarchical_acc))
    
    best_wb_best_child_nb_snr = np.array([nb_bf_gain[ue_idx,wb_2_nb[best_wb_idx]].max() for ue_idx,best_wb_idx in enumerate(best_wb_noisy)])
    best_wb_best_child_nb_snr = 30 + 10*np.log10(best_wb_best_child_nb_snr) + 94 - 13
    two_tier_AMCF_snr.append(best_wb_best_child_nb_snr)
    print('{}-Beam 2-tier AMCF codebook avg. SNR = {}.'.format(n_wb,best_wb_best_child_nb_snr.mean()))
two_tier_AMCF_acc = np.array(two_tier_AMCF_acc)
two_tier_AMCF_snr = np.array(two_tier_AMCF_snr)

"""
binary beam search using AMCF wide beams
"""

bst = Beam_Search_Tree(n_antenna=n_antenna,n_narrow_beam=128,k=2,noise_power=noise_power)
bst_bf_gain, bst_nb_idx = bst.forward_batch(h[test_idc])
bst_acc = (bst_nb_idx==best_nb).mean()
bst_true_snr = nb_bf_gain[tuple(np.arange(nb_bf_gain.shape[0])),tuple(bst_nb_idx)]
bst_true_snr = 30 + 10*np.log10(bst_true_snr) + 94 - 13
print('BST acc = {}, avg. SNR = {}'.format(bst_acc, bst_true_snr.mean()))
    
"""
plotting
"""    

plt.figure(figsize=(8,6))
plt.plot(n_wide_beams,learnable_codebook_acc,marker='s',label='Learned probing codebook')
plt.plot(n_wide_beams,dft_codebook_acc,marker='+',label='DFT probing codebook')
plt.plot(n_wide_beams,AMCF_codebook_acc,marker='o',label='AMCF probing codebook')
plt.plot(n_wide_beams,two_tier_AMCF_acc,marker='x',label='2-tier hierarchical beam search')
plt.hlines(y=bst_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary hierarchical beam search')
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='solid',label='Exhaustive beam search')
plt.legend()
plt.xticks(n_wide_beams)
plt.xlabel('number of probing beams')
plt.ylabel('Accuracy')
# plt.title('Optimal narrow beam prediction accuracy')
# plt.show()
figname = './Figures/acc_vs_probing_codebook_{}_noise_{}_dBm.eps'.format(dataset_name,noise_power_dBm)
plt.savefig(figname, format='eps', dpi=300)

plt.figure(figsize=(8,6))
for k in [0,1,2]:
    plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,k],marker='s',label='Learned probing codebook, k={}'.format(k+1))
    # plt.plot(n_wide_beams,dft_codebook_topk_acc[:,k],marker='+',label='DFT probing codebook, k={}'.format(k+1))
    # plt.plot(n_wide_beams,AMCF_codebook_topk_acc[:,k],marker='o',label='AMCF probing codebook, k={}'.format(k+1))
plt.plot(n_wide_beams,two_tier_AMCF_acc,marker='x',label='2-tier hierarchical codebook')
plt.hlines(y=bst_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary hierarchical beam search')
plt.hlines(y=exhaustive_acc,xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='solid',label='Exhaustive beam search')
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('number of probing beams')
plt.ylabel('Accuracy')
# plt.title('Optimal narrow beam prediction accuracy')
# plt.show()
figname = './Figures/topk_acc_vs_baselines_{}_noise_{}_dBm.eps'.format(dataset_name,noise_power_dBm)
plt.savefig(figname, format='eps', dpi=300)

plt.figure(figsize=(8,6))
for k in [0,1,2]:
    plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,k].mean(axis=1),marker='s',label='Learned probing codebook, k={}'.format(k+1))
    # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='DFT probing codebook, k={}'.format(k+1))
    # plt.plot(n_wide_beams,AMCF_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='AMCF probing codebook, k={}'.format(k+1))
plt.plot(n_wide_beams,two_tier_AMCF_snr.mean(axis=1),marker='x',label='2-tier hierarchical codebook')
plt.hlines(y=bst_true_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary hierarchical beam search')
plt.hlines(y=exhaustive_nb_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive beam search')
plt.hlines(y=genie_nb_snr.mean(),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='solid',label='Genie')
# plt.plot(n_wide_beams,optimal_snr.mean(axis=1),marker='o', label='Genie') 
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('number of probing beams')
plt.ylabel('Average SNR (db)')
# plt.title('SNR of top-k predicted beams')
# plt.show()
figname = './Figures/avg_snr_vs_baselines_{}_noise_{}_dBm.eps'.format(dataset_name,noise_power_dBm)
plt.savefig(figname, format='eps', dpi=300)

for pctl in [5,10,15,85,95]:
    plt.figure(figsize=(8,6))
    for k in [0,1,2]:
        plt.plot(n_wide_beams,np.percentile(learned_codebook_topk_snr[:,:,k],q=pctl,axis=1),marker='s',label='Learned probing codebook, k={}'.format(k+1))
        # plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='DFT probing codebook, k={}'.format(k+1))
        # plt.plot(n_wide_beams,AMCF_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='AMCF probing codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,np.percentile(two_tier_AMCF_snr,q=pctl,axis=1),marker='x',label='2-tier hierarchical codebook')
    plt.hlines(y=np.percentile(bst_true_snr,q=pctl),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dashed',label='Binary hierarchical beam search')
    plt.hlines(y=np.percentile(exhaustive_nb_snr,q=pctl),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='dotted',label='Exhaustive beam search')
    plt.hlines(y=np.percentile(genie_nb_snr,q=pctl),xmin=min(n_wide_beams),xmax=max(n_wide_beams),linestyles='solid',label='Genie')
    # plt.plot(n_wide_beams,optimal_snr.mean(axis=1),marker='o', label='Genie') 
    plt.xticks(n_wide_beams)
    plt.legend()
    plt.xlabel('number of probing beams')
    plt.ylabel('{}-th percentile SNR (db)'.format(pctl))
    # plt.title('{}-th percentile SNR of top-k predicted beams'.format(pctl))
    # plt.show()
    figname = './Figures/{}th_percentile_snr_vs_baselines_{}_noise_{}_dBm.eps'.format(pctl,dataset_name,noise_power_dBm)
    plt.savefig(figname, format='eps', dpi=300)

for iwb,nwb in enumerate(n_wide_beams):
    plt.figure(figsize=(8,6))
    for k in [0]:
        plt.hist(learned_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='Learned probing codebook, k={}'.format(k+1))
        # plt.hist(dft_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='DFT probing codebook, k={}'.format(k+1))
        # plt.hist(AMCF_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='AMCF probing codebook, k={}'.format(k+1))
        plt.hist(bst_true_snr, bins=100, density=True, cumulative=True, histtype='step', label='Binary beam search')
    plt.hist(two_tier_AMCF_snr[iwb], bins=100, density=True, cumulative=True, histtype='step', label='2-tier hierarchical codebook')
    plt.hist(optimal_snr[iwb,:], bins=100, density=True, cumulative=True, histtype='step', label='Genie') 
    plt.legend(loc='upper left')
    plt.ylabel('CDF')
    plt.xlabel('SNR (dB)')
    plt.title('CDF of SNR of top-k predicted beams, number of probing beams = {}'.format(nwb))
    plt.show()


for i,N in enumerate(n_wide_beams):  
    fig = plt.figure(figsize=(5,11))
    ax1 = fig.add_subplot(311,polar=True)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    plot_codebook_pattern_on_axe(learned_codebooks[i].T,ax1)
    ax1.set_title('Learned {}-Beam Probing Codebook'.format(N))
    ax2 = fig.add_subplot(312,polar=True)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    plot_codebook_pattern_on_axe(dft_codebooks[i],ax2)
    ax2.set_title('DFT {}-Beam Probing Codebook'.format(N))
    ax3 = fig.add_subplot(313,polar=True)
    ax3.set_thetamin(-90)
    ax3.set_thetamax(90)
    plot_codebook_pattern_on_axe(AMCF_codebooks[i],ax3)
    ax3.set_title('AMCF {}-Beam Probing Codebook'.format(N))
    
for i,N in enumerate(n_wide_beams):  
    fig = plt.figure(figsize=(5,11))
    ax1 = fig.add_subplot(311,polar=True)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    plot_codebook_pattern_on_axe(learned_codebooks[i].T,ax1)
    ax1.set_title('Learned {}-Beam Probing Codebook'.format(N))
    ax2 = fig.add_subplot(312,polar=True)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    plot_codebook_pattern_on_axe(dft_codebooks[i],ax2)
    ax2.set_title('DFT {}-Beam Probing Codebook'.format(N))
    ax3 = fig.add_subplot(313,polar=True)
    ax3.set_thetamin(-90)
    ax3.set_thetamax(90)
    plot_codebook_pattern_on_axe(AMCF_codebooks[i],ax3)
    ax3.set_title('AMCF {}-Beam Probing Codebook'.format(N))

# fig1, ax1 = plt.subplots(figsize=(8, 6))
# for i,N in enumerate(n_wide_beams):  
#     learned_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, learned_codebooks[i].conj())),2),axis=1)
#     learned_codebook_wb_snr = 30 + 10*np.log10(learned_codebook_wb_snr) + 94 - 13
#     ax1.hist(learned_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam learned probing codebook'.format(N))
# ax1.legend(loc='upper left')
# ax1.set_ylabel('CDF')
# ax1.set_xlabel('SNR (dB)')
# ax1.set_title('SNR CDF of the learned probing codebook')
# plt.show()

# fig1, ax1 = plt.subplots(figsize=(8, 6))
# for i,N in enumerate(n_wide_beams):  
#     AMCF_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, AMCF_codebooks[i].conj().T)),2),axis=1)
#     AMCF_codebook_wb_snr = 30 + 10*np.log10(AMCF_codebook_wb_snr) + 94 - 13
#     ax1.hist(AMCF_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam AMCF probing codebook'.format(N))

# ax1.legend(loc='upper left')
# ax1.set_ylabel('CDF')
# ax1.set_xlabel('SNR (dB)')
# ax1.set_title('SNR CDF of AMCF probing codebook')
# plt.show()

# for i,N in enumerate(n_wide_beams):  
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
    
#     learned_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, learned_codebooks[i].conj())),2),axis=1)
#     learned_codebook_wb_snr = 30 + 10*np.log10(learned_codebook_wb_snr) + 94 - 13
#     dft_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, dft_codebooks[i].T.conj())),2),axis=1)
#     dft_codebook_wb_snr = 30 + 10*np.log10(dft_codebook_wb_snr) + 94 - 13
#     AMCF_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, AMCF_codebooks[i].conj().T)),2),axis=1)
#     AMCF_codebook_wb_snr = 30 + 10*np.log10(AMCF_codebook_wb_snr) + 94 - 13
#     ax1.hist(learned_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam learned probing codebook'.format(N))
#     ax1.hist(dft_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam DFT probing codebook'.format(N))
#     ax1.hist(AMCF_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam AMCF probing codebook'.format(N))

#     ax1.legend(loc='upper left')
#     ax1.set_ylabel('CDF')
#     ax1.set_xlabel('SNR (dB)')
#     ax1.set_title('SNR CDF of {}-beam probing codebook'.format(N))
#     plt.show()


# # fig2 = plt.figure(figsize=(12,6))
# for i,N in enumerate([2, 4, 6, 8]):  
#     learned_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, learned_codebooks[i].conj())),2),axis=1)
#     learned_codebook_wb_snr = 30 + 10*np.log10(learned_codebook_wb_snr) + 94 - 13
#     dft_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, dft_codebooks[i].conj())),2),axis=1)
#     dft_codebook_wb_snr = 30 + 10*np.log10(dft_codebook_wb_snr) + 94 - 13
#     AMCF_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, AMCF_codebooks[i].conj().T)),2),axis=1)
#     AMCF_codebook_wb_snr = 30 + 10*np.log10(AMCF_codebook_wb_snr) + 94 - 13
    
#     snr_min = min([learned_codebook_wb_snr.min(),dft_codebook_wb_snr.min(),AMCF_codebook_wb_snr.min()]) 
#     snr_max = max([learned_codebook_wb_snr.max(),dft_codebook_wb_snr.max(),AMCF_codebook_wb_snr.max()]) 
    
#     if dataset_name == 'O28B_ULA':
#         fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(22,4))
#         plt.subplots_adjust(wspace=0.05)
#         sc = axes[0].scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=learned_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[0].set_title('learned codebook')
        
#         sc = axes[1].scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=dft_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[1].set_title('DFT codebook')
        
#         sc = axes[2].scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=AMCF_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[2].set_title('AMCF codebook')
#         cbar = fig.colorbar(sc,ax=axes.ravel().tolist(),pad=0.01)
#         cbar.set_label('SNR (dB)')
#         # plt.tight_layout()
#         # add a big axis, hide frame
#         fig.add_subplot(111, frameon=False)
#         # hide tick and tick label of the big axis
#         plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         plt.xlabel('meters')
#         plt.ylabel('meters')
#         fig.suptitle('SNR map of {}-beam probing codebook'.format(N))        
               
#         # fig = plt.figure(figsize=(6,3))
#         # sc = plt.scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=learned_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('learned codebook')
        
#         # fig = plt.figure(figsize=(6,3))
#         # sc = plt.scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=dft_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('DFT codebook')
        
#         # fig = plt.figure(figsize=(6,3))
#         # sc = plt.scatter(loc[valid_ue_idc,1],loc[valid_ue_idc,0],c=AMCF_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('AMCF codebook')
    
#     elif dataset_name == 'Rosslyn_ULA':
#         fig,axes = plt.subplots(nrows=1,ncols=3,sharex=True,sharey=True,figsize=(18,4))
#         plt.subplots_adjust(wspace=0.05)
        
#         sc = axes[0].scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=learned_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[0].set_title('learned codebook')
        
#         sc = axes[1].scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=dft_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[1].set_title('DFT codebook')
        
#         sc = axes[2].scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=AMCF_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'),vmin=snr_min,vmax=snr_max)
#         axes[2].set_title('AMCF codebook')
#         cbar = fig.colorbar(sc,ax=axes.ravel().tolist(),pad=0.01)
#         cbar.set_label('SNR (dB)')
#         # plt.tight_layout()
#         # add a big axis, hide frame
#         fig.add_subplot(111, frameon=False)
#         # hide tick and tick label of the big axis
#         plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#         plt.xlabel('meters')
#         plt.ylabel('meters')
#         fig.suptitle('SNR map of {}-beam probing codebook'.format(N))
        
#         # fig = plt.figure()
#         # sc = plt.scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=learned_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.gca().set_aspect('equal', adjustable='box')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('learned codebook')
        
#         # fig = plt.figure()
#         # sc = plt.scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=dft_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.gca().set_aspect('equal', adjustable='box')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('DFT codebook')
        
#         # fig = plt.figure()
#         # sc = plt.scatter(loc[valid_ue_idc,0],loc[valid_ue_idc,1],c=AMCF_codebook_wb_snr,cmap=plt.cm.get_cmap('RdYlBu'))
#         # cbar = plt.colorbar(sc)
#         # cbar.set_label('SNR (dB)')
#         # plt.gca().set_aspect('equal', adjustable='box')
#         # plt.xlabel('meters')
#         # plt.ylabel('meters')
#         # plt.title('AMCF codebook')

# for i,N in enumerate(n_wide_beams):  
#     np.save('./Saved Codebooks/{}_probe_trainable_codebook_{}_beam.npy'.format(dataset_name,N),learned_codebooks[i].T)
#     np.save('./Saved Codebooks/{}_probe_DFT_codebook_{}_beam.npy'.format(dataset_name,N),dft_codebooks[i].T)    
#     np.save('./Saved Codebooks/{}_probe_AMCF_codebook_{}_beam.npy'.format(dataset_name,N),AMCF_codebooks[i])       