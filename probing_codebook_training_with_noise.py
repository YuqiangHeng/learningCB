# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:08:29 2021

@author: ethan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:10:34 2021

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
from beam_utils import ULA_DFT_codebook as DFT_codebook
from beam_utils import ULA_DFT_codebook_blockmatrix as DFT_codebook_blockmatrix
from beam_utils import plot_codebook_pattern, codebook_blockmatrix

np.random.seed(7)
n_narrow_beams = [128, 128, 128, 128, 128, 128]
n_wide_beams = [4, 6, 8, 10, 12, 16]
# n_narrow_beams = [128]
# n_wide_beams = [2]
# num_of_beams = [32]
n_antenna = 64
antenna_sel = np.arange(n_antenna)
nepoch = 200
tx_power_dBm = 30
noise_power_dBm = -94
noise_power = 10**(noise_power_dBm-tx_power_dBm/10)
noiseless = False
if noiseless:
    noise_power = 0.0

dataset_name = 'I3_ULA' # 'Rosslyn_ULA' or 'O28B_ULA' or 'I3_ULA' or 'O28_ULA'
# Training and testing data:
# --------------------------
batch_size = 500
#-------------------------------------------#

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
    
def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
            train_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
            val_acc += (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Training Acc = {:.2f}, Validation loss = {:.2f}, Validation Acc = {:.2f}.'.format(epoch,train_loss,train_acc,val_loss,val_acc))
    return train_loss_hist, val_loss_hist

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
dft_codebook_topk_gain = []
dft_codebooks = []

learnable_codebook_acc = []
learned_codebook_topk_gain= []
learned_codebooks = []

AMCF_codebook_acc = []
AMCF_codebook_topk_gain = []
AMCF_codebooks = []

optimal_gains = []

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
    
    learnable_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,trainable_codebook=True,noise_power=noise_power,norm_factor=norm_factor)
    learnable_codebook_opt = optim.Adam(learnable_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(learnable_codebook_model, train_loader, val_loader, learnable_codebook_opt, nn.CrossEntropyLoss(), nepoch)  
    if noiseless:
        learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noiseless.pt'.format(dataset_name,n_wb,n_nb)
    else:
        learnable_model_savefname = './Saved Models/{}_trainable_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(learnable_codebook_model.state_dict(),learnable_model_savefname)
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('Trainable codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    learnable_codebook_test_loss,learnable_codebook_test_acc = eval_model(learnable_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    learnable_codebook_acc.append(learnable_codebook_test_acc)
    y_test_predict_learnable_codebook = learnable_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_learned_codebook = (-y_test_predict_learnable_codebook).argsort()
    topk_bf_gain_learnable_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_learned_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_learnable_codebook.append(topk_gains)
    topk_bf_gain_learnable_codebook = np.array(topk_bf_gain_learnable_codebook)
    learned_codebook_topk_gain.append(topk_bf_gain_learnable_codebook)
    learned_codebooks.append(learnable_codebook_model.get_codebook()) 

    dft_wb_codebook = DFT_codebook(nseg=n_wb,n_antenna=n_antenna)
    dft_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                         trainable_codebook=False,complex_codebook=dft_wb_codebook,
                                         noise_power=noise_power,norm_factor=norm_factor)
    dft_codebook_opt = optim.Adam(dft_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(dft_codebook_model, train_loader, val_loader, dft_codebook_opt, nn.CrossEntropyLoss(), nepoch)
    if noiseless:
        dft_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noiseless.pt'.format(dataset_name,n_wb,n_nb)
    else:
        dft_model_savefname = './Saved Models/{}_DFT_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(dft_codebook_model.state_dict(),dft_model_savefname)    
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('DFT codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    dft_codebook_test_loss,dft_codebook_test_acc = eval_model(dft_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    dft_codebook_acc.append(dft_codebook_test_acc)
    y_test_predict_dft_codebook = dft_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_dft_codebook = (-y_test_predict_dft_codebook).argsort()
    topk_bf_gain_dft_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_dft_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_dft_codebook.append(topk_gains)
    topk_bf_gain_dft_codebook = np.array(topk_bf_gain_dft_codebook)
    dft_codebook_topk_gain.append(topk_bf_gain_dft_codebook)
    dft_codebooks.append(dft_codebook_model.get_codebook())
    
    AMCF_wb_codebook = sio.loadmat('{}_beam_AMCF_codebook.mat'.format(n_wb))['V'].T
    AMCF_codebook_model = Beam_Classifier(n_antenna=n_antenna,n_wide_beam=n_wb,n_narrow_beam=n_nb,
                                          trainable_codebook=False,complex_codebook=AMCF_wb_codebook,
                                          noise_power=noise_power,norm_factor=norm_factor)
    AMCF_codebook_opt = optim.Adam(AMCF_codebook_model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    train_loss_hist, val_loss_hist = fit(AMCF_codebook_model, train_loader, val_loader, AMCF_codebook_opt, nn.CrossEntropyLoss(), nepoch)
    if noiseless:
        AMCF_model_savefname = './Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noiseless.pt'.format(dataset_name,n_wb,n_nb)
    else:
        AMCF_model_savefname = './Saved Models/{}_AMCF_{}_beam_probing_codebook_{}_beam_classifier_noise_{}_dBm.pt'.format(dataset_name,n_wb,n_nb,noise_power_dBm)
    torch.save(AMCF_codebook_model.state_dict(),AMCF_model_savefname)      
    plt.figure()
    plt.plot(train_loss_hist,label='training loss')
    plt.plot(val_loss_hist,label='validation loss')
    plt.legend()
    plt.title('AMCF codebook loss hist: {} wb {} nb'.format(n_wb,n_nb))
    plt.show()
    AMCF_codebook_test_loss,AMCF_codebook_test_acc = eval_model(AMCF_codebook_model,test_loader,nn.CrossEntropyLoss()) 
    AMCF_codebook_acc.append(AMCF_codebook_test_acc)
    y_test_predict_AMCF_codebook = AMCF_codebook_model(torch_x_test.float()).detach().numpy()
    topk_sorted_test_AMCF_codebook = (-y_test_predict_AMCF_codebook).argsort()
    topk_bf_gain_AMCF_codebook = []
    for ue_bf_gain, pred_sort in zip(soft_label[test_idc,:],topk_sorted_test_AMCF_codebook):
        topk_gains = [ue_bf_gain[pred_sort[:k]].max() for k in range(1,11)]
        topk_bf_gain_AMCF_codebook.append(topk_gains)
    topk_bf_gain_AMCF_codebook = np.array(topk_bf_gain_AMCF_codebook)
    AMCF_codebook_topk_gain.append(topk_bf_gain_AMCF_codebook)
    AMCF_codebooks.append(AMCF_codebook_model.get_codebook())    
    
    optimal_gains.append(soft_label[test_idc,:].max(axis=-1))

dft_codebook_topk_gain = np.array(dft_codebook_topk_gain)
learned_codebook_topk_gain = np.array(learned_codebook_topk_gain)
AMCF_codebook_topk_gain = np.array(AMCF_codebook_topk_gain)
optimal_gains = np.array(optimal_gains)

dft_codebook_topk_snr = 30 + 10*np.log10(dft_codebook_topk_gain) - noise_power_dBm -13
learned_codebook_topk_snr = 30 + 10*np.log10(learned_codebook_topk_gain) - noise_power_dBm -13
AMCF_codebook_topk_snr = 30 + 10*np.log10(AMCF_codebook_topk_gain) - noise_power_dBm -13
optimal_snr = 30 + 10*np.log10(optimal_gains) - noise_power_dBm -13

plt.figure(figsize=(8,6))
plt.plot(n_wide_beams,learnable_codebook_acc,marker='s',label='Trainable codebook')
plt.plot(n_wide_beams,dft_codebook_acc,marker='+',label='DFT codebook')
plt.plot(n_wide_beams,AMCF_codebook_acc,marker='x',label='DFT codebook')
plt.legend()
plt.xticks(n_wide_beams)
plt.xlabel('number of probe beams')
plt.ylabel('Accuracy')
plt.title('Optimal narrow beam prediction accuracy')
plt.show()

# plt.figure(figsize=(8,6))
# for k in [0,2]:
#     plt.plot(n_wide_beams,learned_codebook_topk_gain[:,k]*(norm_factor**2),marker='s',label='learned codebook, k={}'.format(k+1))
#     plt.plot(n_wide_beams,dft_codebook_topk_gain[:,k]*(norm_factor**2),marker='+',label='DFT codebook, k={}'.format(k+1))
# plt.plot(n_wide_beams,np.array(optimal_gains)*(norm_factor**2),marker='o', label='Optimal') 
# plt.xticks(n_wide_beams)
# plt.legend()
# plt.xlabel('number of probe beams')
# plt.ylabel('BF gain')
# plt.title('BF gain of top-k predicted beam')
# plt.show()

plt.figure(figsize=(8,6))
for k in [0,2]:
    plt.plot(n_wide_beams,learned_codebook_topk_snr[:,:,k].mean(axis=1),marker='s',label='Trainable codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,dft_codebook_topk_snr[:,:,k].mean(axis=1),marker='+',label='DFT codebook, k={}'.format(k+1))
    plt.plot(n_wide_beams,AMCF_codebook_topk_snr[:,:,k].mean(axis=1),marker='x',label='AMCF codebook, k={}'.format(k+1))
plt.plot(n_wide_beams,optimal_snr.mean(axis=1),marker='o', label='Optimal') 
plt.xticks(n_wide_beams)
plt.legend()
plt.xlabel('number of probe beams')
plt.ylabel('Avg. SNR (db)')
plt.title('SNR of top-k predicted beams')
plt.show()

for iwb,nwb in enumerate(n_wide_beams):
    plt.figure(figsize=(8,6))
    for k in [0,2]:
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
    fig,ax = plot_codebook_pattern(learned_codebooks[i].T)
    ax.set_title('Trainable {}-Beam Codebook'.format(N))
    fig,ax = plot_codebook_pattern(dft_codebooks[i])
    ax.set_title('DFT {}-Beam Codebook'.format(N))
    fig,ax = plot_codebook_pattern(AMCF_codebooks[i])
    ax.set_title('AMCF {}-Beam Codebook'.format(N))
