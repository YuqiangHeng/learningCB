# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:50:14 2021

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
from beam_utils import plot_codebook_pattern, plot_codebook_pattern_on_axe, codebook_blockmatrix

np.random.seed(7)
n_narrow_beams = [128, 128, 128, 128, 128, 128, 128]
n_wide_beams = [2, 4, 6, 8, 10, 12, 16]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
nepoch = 200

dataset_name = 'O28B_ULA' # 'Rosslyn_ULA' or 'O28B_ULA'

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
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None):
        super(Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
        if trainable_codebook:
            self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_wide_beam, scale=np.sqrt(n_antenna), theta=theta)
        else:
            dft_codebook = DFT_codebook_blockmatrix(n_antenna=n_antenna, nseg=n_wide_beam)
            self.codebook = torch.from_numpy(dft_codebook).float()
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
            return DFT_codebook(nseg=self.n_wide_beam,n_antenna=self.n_antenna).T
        
class AMCF_Beam_Classifier(nn.Module):
    def __init__(self, n_antenna, n_wide_beam, n_narrow_beam, trainable_codebook = True, theta = None, complex_codebook=None):
        super(AMCF_Beam_Classifier, self).__init__()
        self.trainable_codebook = trainable_codebook
        self.n_antenna = n_antenna
        self.n_wide_beam = n_wide_beam
        self.n_narrow_beam = n_narrow_beam
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
    
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True)
    learnable_codebook_model.load_state_dict(torch.load('./Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier.pt'.format(dataset_name,n_wb,n_nb)))
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

    
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=False)
    dft_codebook_model.load_state_dict(torch.load('./Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier.pt'.format(dataset_name,n_wb,n_nb)))
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
    AMCF_codebook_model = AMCF_Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=False,complex_codebook=AMCF_wb_codebook)
    AMCF_codebook_model.load_state_dict(torch.load('./Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier.pt'.format(dataset_name,n_wb,n_nb)))
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


plt.figure(figsize=(8,6))
plt.plot(n_wide_beams,learnable_codebook_acc,marker='s',label='Trainable codebook')
plt.plot(n_wide_beams,dft_codebook_acc,marker='+',label='DFT codebook')
plt.plot(n_wide_beams,AMCF_codebook_acc,marker='o',label='AMCF codebook')
plt.legend()
plt.xticks(n_wide_beams)
plt.xlabel('number of probe beams')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

plt.figure(figsize=(8,6))
for k in [0,2]:
    plt.plot(n_wide_beams,learnable_codebook_topk_acc[:,k],marker='s',label='Trainable codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,dft_codebook_topk_acc[:,k],marker='+',label='DFT codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,AMCF_codebook_topk_acc[:,k],marker='o',label='AMCF codebook, k={}'.format(k+1))
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('number of probe beams')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

plt.figure(figsize=(8,6))
for k in [0,2]:
    plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,k].mean(axis=1),marker='s',label='Trainable codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='DFT codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,AMCF_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='AMCF codebook, k={}'.format(k+1))
plt.plot(n_wide_beams,optimal_snr.mean(axis=1),marker='o', label='Optimal') 
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('number of probe beams')
plt.ylabel('Avg. SNR (db)')
plt.title('SNR of top-k predicted beams')
plt.show()

for iwb,nwb in enumerate(n_wide_beams):
    plt.figure(figsize=(8,6))
    for k in [0]:
        plt.hist(learned_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='Trainable codebook, k={}'.format(k+1))
        plt.hist(dft_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='DFT codebook, k={}'.format(k+1))
        plt.hist(AMCF_codebook_topk_snr[iwb,:,k], bins=100, density=True, cumulative=True, histtype='step', label='AMCF codebook, k={}'.format(k+1))
    plt.hist(optimal_snr[iwb,:], bins=100, density=True, cumulative=True, histtype='step', label='Optimal') 
    plt.legend(loc='upper left')
    plt.ylabel('CDF')
    plt.xlabel('SNR (dB)')
    plt.title('CDF of SNR of top-k predicted beams, number of probe beams = {}'.format(nwb))
    plt.show()


for i,N in enumerate(n_wide_beams):  
    fig = plt.figure(figsize=(5,11))
    ax1 = fig.add_subplot(311,polar=True)
    ax1.set_thetamin(-90)
    ax1.set_thetamax(90)
    plot_codebook_pattern_on_axe(learned_codebooks[i].T,ax1)
    ax1.set_title('Trainable {}-Beam Codebook'.format(N))
    ax2 = fig.add_subplot(312,polar=True)
    ax2.set_thetamin(-90)
    ax2.set_thetamax(90)
    plot_codebook_pattern_on_axe(dft_codebooks[i].T,ax2)
    ax2.set_title('DFT {}-Beam Codebook'.format(N))
    ax3 = fig.add_subplot(313,polar=True)
    ax3.set_thetamin(-90)
    ax3.set_thetamax(90)
    plot_codebook_pattern_on_axe(AMCF_codebooks[i],ax3)
    ax3.set_title('AMCF {}-Beam Codebook'.format(N))

fig1, ax1 = plt.subplots(figsize=(8, 6))
for i,N in enumerate(n_wide_beams):  
    learned_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, learned_codebooks[i].conj())),2),axis=1)
    learned_codebook_wb_snr = 30 + 10*np.log10(learned_codebook_wb_snr) + 94 - 13
    ax1.hist(learned_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam learned probing codebook'.format(N))
ax1.legend(loc='upper left')
ax1.set_ylabel('CDF')
ax1.set_xlabel('SNR (dB)')
ax1.set_title('SNR CDF of learned probing codebook')
plt.show()

fig1, ax1 = plt.subplots(figsize=(8, 6))
for i,N in enumerate(n_wide_beams):  
    AMCF_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, AMCF_codebooks[i].conj().T)),2),axis=1)
    AMCF_codebook_wb_snr = 30 + 10*np.log10(AMCF_codebook_wb_snr) + 94 - 13
    ax1.hist(AMCF_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam AMCF probing codebook'.format(N))

ax1.legend(loc='upper left')
ax1.set_ylabel('CDF')
ax1.set_xlabel('SNR (dB)')
ax1.set_title('SNR CDF of AMCF probing codebook')
plt.show()

for i,N in enumerate(n_wide_beams):  
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    learned_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, learned_codebooks[i].conj())),2),axis=1)
    learned_codebook_wb_snr = 30 + 10*np.log10(learned_codebook_wb_snr) + 94 - 13
    dft_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, dft_codebooks[i].conj())),2),axis=1)
    dft_codebook_wb_snr = 30 + 10*np.log10(dft_codebook_wb_snr) + 94 - 13
    AMCF_codebook_wb_snr = np.max(np.power(np.absolute(np.matmul(h, AMCF_codebooks[i].conj().T)),2),axis=1)
    AMCF_codebook_wb_snr = 30 + 10*np.log10(AMCF_codebook_wb_snr) + 94 - 13
    ax1.hist(learned_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam learned probing codebook'.format(N))
    ax1.hist(dft_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam DFT probing codebook'.format(N))
    ax1.hist(AMCF_codebook_wb_snr,bins=100,density=True,cumulative=True,histtype='step',label='{}-beam AMCF probing codebook'.format(N))

    ax1.legend(loc='upper left')
    ax1.set_ylabel('CDF')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_title('SNR CDF of {}-beam probing codebook'.format(N))
    plt.show()


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