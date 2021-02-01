# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:54:04 2021

@author: ethan
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, PowerPooling, ComputePower
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import GaussianCenters, DFT_codebook

np.random.seed(7)
num_of_beams = [8, 16, 24, 32, 64, 96, 128]
# num_of_beams = [32]
num_antenna = 64
antenna_sel = np.arange(num_antenna)

snr_db = 5
snr = 10**(snr_db/10)

nepoch = 5
lr = 0.1
batch_size = 500
# Training and testing data:
# --------------------------

fname_h_real = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/channel_real.mat'
fname_h_imag = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/channel_imag.mat'
fname_loc = 'D://Github Repositories/DeepMIMO-codes/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO_Dataset_Generation_v1.1/DeepMIMO Dataset/loc.mat'

h_real = sio.loadmat(fname_h_real)['channel_real']
h_imag = sio.loadmat(fname_h_imag)['channel_imag']
loc = sio.loadmat(fname_loc)['loc']

h = h_real + 1j*h_imag
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h = h_scaled
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)
# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/num_antenna
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.3)
x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

# torch_x_train = torch.from_numpy(x_train).type(torch.LongTensor)
# torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long
# torch_x_val = torch.from_numpy(x_val).type(torch.LongTensor)
# torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor)
# torch_x_test = torch.from_numpy(x_test).type(torch.LongTensor)
# torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

torch_x_train = torch.from_numpy(x_train)
torch_y_train = torch.from_numpy(y_train)
torch_x_test = torch.from_numpy(x_test)
torch_y_test = torch.from_numpy(y_test)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

class AnalogBeamformer(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(AnalogBeamformer, self).__init__()
        self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.beam_selection = PowerPooling(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power_sel = self.beam_selection(bf_signal)
        return torch.log(bf_power_sel)

class Self_Supervised_AnalogBeamformer(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Self_Supervised_AnalogBeamformer, self).__init__()
        self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.compute_power = ComputePower(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power = self.compute_power(bf_signal)
        return bf_power

def bf_gain_loss(y_pred, y_true):
    return -torch.mean(y_pred,dim=0)

def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.float()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.float()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist

def fit_genius(model:AnalogBeamformer, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            # z = np.matmul(x_batch_complex, learned_codebook.conj())
            # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            var_y_batch = y_batch.float()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            # z = np.matmul(x_batch_complex, learned_codebook.conj())
            # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            var_y_batch = y_batch.float()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist

def fit_self_supervised(model:Self_Supervised_AnalogBeamformer, train_loader, val_loader, opt, EPOCHS, loss_fn = nn.CrossEntropyLoss()):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            optimizer.zero_grad()
            bf_power =  model(var_X_batch)
            bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
            loss = loss_fn(bf_power, bf_argmax)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        model.eval()
        val_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            theta = model.codebook.theta.detach().clone().numpy()
            learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
            x_batch_np = X_batch.detach().clone().numpy()
            x_batch_complex = x_batch_np[:,:num_antenna] + 1j*x_batch_np[:,num_antenna:]
            # z = np.matmul(x_batch_complex, learned_codebook.conj())
            # h_est = np.matmul(np.linalg.pinv(learned_codebook.conj().T),z)
            z = learned_codebook.conj().T @ x_batch_complex.T
            h_est = np.linalg.pinv(learned_codebook.conj().T) @ z
            h_est_cat = np.concatenate((h_est.real, h_est.imag),axis=0)
            var_X_batch = torch.from_numpy(h_est_cat.T).float()
            bf_power =  model(var_X_batch)
            bf_argmax = torch.argmax(bf_power.detach().clone(),dim=1)
            loss = loss_fn(bf_power, bf_argmax)
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist

# ------------------------------------------------------------------
# Codebook learning using H estimate, GD on crossentropy of softmax and argmax of bf power
# ------------------------------------------------------------------
learned_codebook_gains_self_supervised = np.zeros((len(num_of_beams),len(test_idc)))
learned_codebooks_self_supervised = []
for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')
    
    # Model:
    # ------
    model = Self_Supervised_AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
    # Training:
    # ---------
    opt = optim.Adam(model.parameters(),lr=lr, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit_self_supervised(model, train_loader, test_loader, opt, nepoch)    


    plt.figure()
    plt.plot(np.array(train_hist),label='train loss')
    plt.plot(np.array(val_hist),label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Genius. {} beams'.format(N))
    # Extract learned codebook:
    # -------------------------
    theta = model.codebook.theta.detach().clone().numpy()
    print(theta.shape)
    # name_of_file = 'theta_NLOS' + str(N) + 'vec.mat'
    # scio.savemat(name_of_file,
    #              {'train_inp': train_inp,
    #               'train_out': train_out,
    #               'val_inp': val_inp,
    #               'val_out': val_out,
    #               'codebook': theta})
    learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
    learned_codebooks_self_supervised.append(learned_codebook)
    learned_codebook_gains_self_supervised[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.conj())),2),axis=1)
tput_self_supervised = np.log2(1+learned_codebook_gains_self_supervised*snr)
learned_codebook_gains_self_supervised = 10*np.log10(learned_codebook_gains_self_supervised)
tput_self_supervised = np.log2(1+learned_codebook_gains_self_supervised*snr)


# ------------------------------------------------------------------
# Codebook learning using H estimate, directly GD on max(bf power)
# ------------------------------------------------------------------
learned_codebook_gains_genius = np.zeros((len(num_of_beams),len(test_idc)))
learned_codebooks_genius = []
for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')

    # Model:
    # ------
    model = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
    # Training:
    # ---------
    opt = optim.Adam(model.parameters(),lr=lr, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit_genius(model, train_loader, test_loader, opt, bf_gain_loss, nepoch)    


    plt.figure()
    plt.plot(np.array(train_hist),label='train loss')
    plt.plot(np.array(val_hist),label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Genius. {} beams'.format(N))
    # Extract learned codebook:
    # -------------------------
    theta = model.codebook.theta.detach().clone().numpy()
    print(theta.shape)
    # name_of_file = 'theta_NLOS' + str(N) + 'vec.mat'
    # scio.savemat(name_of_file,
    #              {'train_inp': train_inp,
    #               'train_out': train_out,
    #               'val_inp': val_inp,
    #               'val_out': val_out,
    #               'codebook': theta})
    learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
    learned_codebooks_genius.append(learned_codebook)
    learned_codebook_gains_genius[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.conj())),2),axis=1)
tput_genius = np.log2(1+learned_codebook_gains_genius*snr)
learned_codebook_gains_genius = 10*np.log10(learned_codebook_gains_genius)

# ------------------------------------------------------------------
# Codebook learning using H, directly GD on max(bf power)
# ------------------------------------------------------------------
learned_codebook_gains = np.zeros((len(num_of_beams),len(test_idc)))
learned_codebooks = []
for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')

    # Model:
    # ------
    model = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
    # Training:
    # ---------
    opt = optim.Adam(model.parameters(),lr=lr, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit(model, train_loader, test_loader, opt, bf_gain_loss, nepoch)    


    plt.figure()
    plt.plot(-np.array(train_hist),label='train loss')
    plt.plot(-np.array(val_hist),label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Unsupervised. {} beams'.format(N))
    # Extract learned codebook:
    # -------------------------
    theta = model.codebook.theta.detach().clone().numpy()
    print(theta.shape)
    # name_of_file = 'theta_NLOS' + str(N) + 'vec.mat'
    # scio.savemat(name_of_file,
    #              {'train_inp': train_inp,
    #               'train_out': train_out,
    #               'val_inp': val_inp,
    #               'val_out': val_out,
    #               'codebook': theta})
    learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
    learned_codebooks.append(learned_codebook)
    learned_codebook_gains[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.conj())),2),axis=1)
tput_learned = np.log2(1+learned_codebook_gains*snr)
learned_codebook_gains = 10*np.log10(learned_codebook_gains)

# ------------------------------------------------------------------
# Codebook learning using H, GD on MSE of max(bf power) and EGC power
# ------------------------------------------------------------------
learned_codebook_gains_supervised = np.zeros((len(num_of_beams),len(test_idc)))
learned_codebooks_supervised = []
for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')

    # Model:
    # ------
    model = AnalogBeamformer(n_antenna = num_antenna, n_beam = N)
    # Training:
    # ---------
    opt = optim.Adam(model.parameters(),lr=lr, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit(model, train_loader, test_loader, opt, nn.MSELoss(), nepoch)    


    plt.figure()
    plt.plot(np.array(train_hist),label='train loss')
    plt.plot(np.array(val_hist),label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.title('Supervised. {} beams'.format(N))
    # Extract learned codebook:
    # -------------------------
    theta = model.codebook.theta.detach().clone().numpy()
    print(theta.shape)
    # name_of_file = 'theta_NLOS' + str(N) + 'vec.mat'
    # scio.savemat(name_of_file,
    #              {'train_inp': train_inp,
    #               'train_out': train_out,
    #               'val_inp': val_inp,
    #               'val_out': val_out,
    #               'codebook': theta})
    learned_codebook = np.exp(1j*theta)/np.sqrt(num_antenna)
    learned_codebooks_supervised.append(learned_codebook)
    learned_codebook_gains_supervised[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.conj())),2),axis=1)
tput_supervised = np.log2(1+learned_codebook_gains_supervised*snr)
learned_codebook_gains_supervised = 10*np.log10(learned_codebook_gains_supervised)


#-------------------------------------------#
# Comppare between learned codebook and DFT codebook on test set
#-------------------------------------------#    
dft_gains = np.zeros((len(num_of_beams),len(test_idc)))
for i, nbeams in enumerate(num_of_beams):
    dft_gains[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], np.transpose(np.conj(DFT_codebook(nbeams,num_antenna))))),2),axis=1)
tput_dft = np.log2(1+dft_gains*snr)
dft_gains = 10*np.log10(dft_gains)

fig,ax = plt.subplots(figsize=(8,6))
for i in range(len(num_of_beams)):
    ax.hist(learned_codebook_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook unsupervised, {} beams'.format(num_of_beams[i]))    
    ax.hist(learned_codebook_gains_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook supervised, {} beams'.format(num_of_beams[i]))
    ax.hist(dft_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
# tidy up the figure
ax.grid(True)
ax.legend(loc='upper left')
#ax.set_title('Cumulative step histograms')
ax.set_xlabel('BF Gain (dB)')
ax.set_ylabel('Emperical CDF')
plt.show()

for i, N in enumerate(num_of_beams):
    fig,ax = plt.subplots(figsize=(8,6))
    ax.hist(learned_codebook_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='GD, full h')    
    ax.hist(learned_codebook_gains_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='Supervised (EGC)')
    ax.hist(learned_codebook_gains_genius[i,:],bins=100,density=True,cumulative=True,histtype='step',label='GD, est h')
    ax.hist(learned_codebook_gains_self_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='Self-supervised')
    ax.hist(dft_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT')
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='upper left')
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('BF Gain (dB)')
    ax.set_ylabel('Emperical CDF')
    ax.set_title('Codebook comparison with {} beams.'.format(N))
    plt.show()

plt.figure(figsize=(8,6))
plt.plot(num_of_beams,np.mean(learned_codebook_gains_supervised,axis=1),marker='+',label='Supervised (EGC)')
plt.plot(num_of_beams,np.mean(learned_codebook_gains_self_supervised,axis=1),marker='s',label='Self-supervised')
plt.plot(num_of_beams,np.mean(learned_codebook_gains,axis=1),marker='o',label='GD, full h')
plt.plot(num_of_beams,np.mean(learned_codebook_gains_genius,axis=1),marker='x',label='GD, est h')
plt.plot(num_of_beams,np.mean(dft_gains,axis=1),marker='D',label='DFT')
plt.legend()
plt.xticks(num_of_beams, num_of_beams)
plt.grid(True)
plt.xlabel('number of beams')
plt.ylabel('avg BF gain (dB)')
plt.show()

plt.figure(figsize=(8,6))
plt.plot(num_of_beams,np.mean(tput_supervised,axis=1),marker='+',label='Supervised (EGC)')
plt.plot(num_of_beams,np.mean(tput_self_supervised,axis=1),marker='s',label='Self-supervised')
plt.plot(num_of_beams,np.mean(tput_learned,axis=1),marker='o',label='GD, full h')
plt.plot(num_of_beams,np.mean(tput_genius,axis=1),marker='x',label='GD, est h')
plt.plot(num_of_beams,np.mean(tput_dft,axis=1),marker='D',label='DFT')
plt.legend()
plt.xticks(num_of_beams, num_of_beams)
plt.grid(True)
plt.xlabel('number of beams')
plt.ylabel('achievable rate (bps)')
plt.show()
# percentile = 5
# plt.figure(figsize=(8,6))
# plt.plot(num_of_beams,np.percentile(learned_codebook_gains_supervised,q=percentile,axis=1),marker='+',label='Supervised (EGC)')
# plt.plot(num_of_beams,np.percentile(learned_codebook_gains_self_supervised,q=percentile,axis=1),marker='s',label='Self-supervised')
# plt.plot(num_of_beams,np.percentile(learned_codebook_gains,q=percentile,axis=1),marker='o',label='GD, full h')
# plt.plot(num_of_beams,np.percentile(learned_codebook_gains_genius,q=percentile,axis=1),marker='x',label='GD, est h')
# plt.plot(num_of_beams,np.percentile(dft_gains,q=percentile,axis=1),marker='D',label='DFT')
# plt.legend()
# plt.xticks(num_of_beams, num_of_beams)
# plt.grid(True)
# plt.xlabel('number of beams')
# plt.ylabel('{}-percentile BF gain (dB)'.format(percentile))
# plt.show()
