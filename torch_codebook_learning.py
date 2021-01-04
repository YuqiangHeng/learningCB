# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 15:15:15 2021

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, PowerPooling
import torch.utils.data
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split

np.random.seed(7)
# num_of_beams = [2, 4, 8, 16, 32, 64, 96, 128]
num_of_beams = [32, 64, 128]
num_antenna = 64
# Training and testing data:
# --------------------------
batch_size = 500
#-------------------------------------------#
# Here should be the data_preparing function
# It is expected to return:
# train_inp, train_out, val_inp, and val_out
#-------------------------------------------#
h_real = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_real.npy')[:10000]
h_imag = np.load('D://Github Repositories/mmWave Beam Management/H_Matrices FineGrid/MISO_Static_FineGrid_Hmatrices_imag.npy')[:10000]
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
        self.codebook = PhaseShifter(2*n_antenna, n_beam)
        self.beam_selection = PowerPooling(2*n_beam)
    def forward(self, x):
        bf_signal = self.codebook(x)
        bf_power_sel = self.beam_selection(bf_signal)
        return bf_power_sel

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
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch).float()
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
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch).float()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch.unsqueeze(dim=-1))
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {} Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch, train_loss, val_loss))
    return train_loss_hist, val_loss_hist
                
num_antenna = h.shape[1]
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
    theta = model.codebook.theta.detach().numpy()
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
    learned_codebook_gains[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.T.conj())),2),axis=1)
learned_codebook_gains = 10*np.log10(learned_codebook_gains)

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
    theta = model.codebook.theta.detach().numpy()
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
    learned_codebook_gains_supervised[i,:] = np.max(np.power(np.absolute(np.matmul(h[test_idc,:], learned_codebook.T.conj())),2),axis=1)
learned_codebook_gains_supervised = 10*np.log10(learned_codebook_gains_supervised)


for i in range(len(num_of_beams)):
    fig,ax = plt.subplots(figsize=(8,6))
    ax.hist(learned_codebook_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='unsupervised, {} beams'.format(num_of_beams[i]))    
    ax.hist(learned_codebook_gains_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='supervised, {} beams'.format(num_of_beams[i]))
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='upper left')
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('BF Gain (dB)')
    ax.set_ylabel('Emperical CDF')
    plt.show()
#-------------------------------------------#
# Comppare between learned codebook and DFT codebook on test set
#-------------------------------------------#    
def DFT_codebook(nseg,n_antenna):
    bfdirections = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),nseg))
    codebook_all = np.zeros((nseg,n_antenna),dtype=np.complex_)
    
    for i in range(nseg):
        phi = bfdirections[i]
        #array response vector original
        arr_response_vec = [-1j*np.pi*k*np.cos(phi) for k in range(n_antenna)]
        #array response vector for rotated ULA
        #arr_response_vec = [1j*np.pi*k*np.sin(phi+np.pi/2) for k in range(64)]
        codebook_all[i,:] = np.exp(arr_response_vec)/np.sqrt(n_antenna)
    return codebook_all

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
    ax.hist(learned_codebook_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook unsupervised, {} beams'.format(num_of_beams[i]))    
    ax.hist(learned_codebook_gains_supervised[i,:],bins=100,density=True,cumulative=True,histtype='step',label='learned codebook supervised, {} beams'.format(num_of_beams[i]))
    ax.hist(dft_gains[i,:],bins=100,density=True,cumulative=True,histtype='step',label='DFT codebook,{} beams'.format(num_of_beams[i]))
    # tidy up the figure
    ax.grid(True)
    ax.legend(loc='upper left')
    #ax.set_title('Cumulative step histograms')
    ax.set_xlabel('BF Gain (dB)')
    ax.set_ylabel('Emperical CDF')
    ax.set_title('Codebook comparison with {} beams.'.format(N))
    plt.show()
