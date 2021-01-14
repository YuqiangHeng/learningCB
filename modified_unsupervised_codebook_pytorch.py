# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 20:32:27 2021

@author: ethan
"""

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

# Training and testing data:
# --------------------------
batch_size = 500
#-------------------------------------------#
# Here should be the data_preparing function
# It is expected to return:
# train_inp, train_out, val_inp, and val_out
#-------------------------------------------#
h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')[:,antenna_sel]
h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')[:,antenna_sel]
loc = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_UE_location.npy')
# h_real = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')
# h_imag = np.load('/Users/yh9277/Dropbox/ML Beam Alignment/Data/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')
BS_loc = [641,435,10]
num_samples = h_real.shape[0]
gc = GaussianCenters(n_clusters=10, arrival_rate = 1000, cluster_variance = 10)
sel_samples = gc.sample()
# gc.plot_sample(sel_samples)
# sel_samples = np.arange(10000)
h_real = h_real[sel_samples,:]
h_imag = h_imag[sel_samples,:]
loc = loc[sel_samples,:]

plt.figure(figsize=(8,6))
plt.scatter(loc[:,0], loc[:,1], s=1, label='UE')
plt.scatter(BS_loc[0], BS_loc[1], s=10, marker='s', label='BS')
plt.legend(loc='lower left')
plt.xlabel('x (meter)')
plt.ylabel('y (meter)')
plt.title('UE Distribution')
plt.show()

h = h_real + 1j*h_imag
#norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)
# Compute EGC gain
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/num_antenna
train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)
x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

# torch_x_train = torch.from_numpy(x_train).type(torch.LongTensor)
# torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) # data type is long
# torch_x_val = torch.from_numpy(x_val).type(torch.LongTensor)
# torch_y_val = torch.from_numpy(y_val).type(torch.LongTensor)
# torch_x_test = torch.from_numpy(x_test).type(torch.LongTensor)
# torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

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

class AnalogBeamformer(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(AnalogBeamformer, self).__init__()
        self.codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.beam_selection = PowerPooling(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power_sel = self.beam_selection(bf_signal)
        return bf_power_sel

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
    opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit_self_supervised(model, train_loader, val_loader, opt, 100)    


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
learned_codebook_gains_self_supervised = 10*np.log10(learned_codebook_gains_self_supervised)


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
    opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit_genius(model, train_loader, val_loader, opt, bf_gain_loss, 100)    


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
    opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit(model, train_loader, val_loader, opt, bf_gain_loss, 100)    


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
    opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit(model, train_loader, val_loader, opt, nn.MSELoss(), 100)    


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
learned_codebook_gains_supervised = 10*np.log10(learned_codebook_gains_supervised)


#-------------------------------------------#
# Comppare between learned codebook and DFT codebook on test set
#-------------------------------------------#    
dft_gains = np.zeros((len(num_of_beams),len(test_idc)))
for i, nbeams in enumerate(num_of_beams):
    dft_gains[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], np.transpose(np.conj(DFT_codebook(nbeams,num_antenna))))),2),axis=1)
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

percentile = 5
plt.figure(figsize=(8,6))
plt.plot(num_of_beams,np.percentile(learned_codebook_gains_supervised,q=percentile,axis=1),marker='+',label='Supervised (EGC)')
plt.plot(num_of_beams,np.percentile(learned_codebook_gains_self_supervised,q=percentile,axis=1),marker='s',label='Self-supervised')
plt.plot(num_of_beams,np.percentile(learned_codebook_gains,q=percentile,axis=1),marker='o',label='GD, full h')
plt.plot(num_of_beams,np.percentile(learned_codebook_gains_genius,q=percentile,axis=1),marker='x',label='GD, est h')
plt.plot(num_of_beams,np.percentile(dft_gains,q=percentile,axis=1),marker='D',label='DFT')
plt.legend()
plt.xticks(num_of_beams, num_of_beams)
plt.grid(True)
plt.xlabel('number of beams')
plt.ylabel('{}-percentile BF gain (dB)'.format(percentile))
plt.show()
