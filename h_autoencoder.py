# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 18:20:26 2021

@author: ethan
"""

import numpy as np
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, Complex_Dense, PowerPooling, ComputePower
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import GaussianCenters, DFT_codebook, DFT_codebook_blockmatrix, bf_gain_loss

np.random.seed(7)
num_of_beams = [8, 16, 24, 32, 64, 96, 128]
n_antenna = 64
antenna_sel = np.arange(n_antenna)

# Training and testing data:
# --------------------------
batch_size = 100
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

h = h_real + 1j*h_imag
# norm_factor = np.max(np.power(abs(h),2))
norm_factor = np.max(abs(h))
h_scaled = h/(norm_factor**2)
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
train = torch.utils.data.TensorDataset(torch_x_train,torch_x_train)
val = torch.utils.data.TensorDataset(torch_x_val,torch_x_val)
test = torch.utils.data.TensorDataset(torch_x_test,torch_x_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)

class Channel_Encoder(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Channel_Encoder, self).__init__()
        self.analog_codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.dense_1 = nn.Linear(in_features = n_beam*2, out_features = n_beam*2)
        self.dense_2 = nn.Linear(in_features = n_beam*2, out_features = n_beam*2)
        self.relu = nn.ReLU()
        self.compute_power = ComputePower(2*n_beam)
        
    def forward(self, x):
        bf_signal = self.analog_codebook(x)
        output = self.relu(self.dense_1(bf_signal))
        output = self.dense_2(output)
        return output

class Channel_Decoder(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Channel_Decoder, self).__init__()
        self.dense_1 = nn.Linear(in_features = n_beam*2, out_features = n_beam*2)
        self.dense_2 = nn.Linear(in_features = n_beam*2, out_features = n_antenna*2)
        self.dense_3 = nn.Linear(in_features = n_antenna*2, out_features = n_antenna*2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.relu(self.dense_1(x))
        output = self.relu(self.dense_2(output))
        output = self.dense_3(output)
        return output
    
class Channel_AutoEncoder(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Channel_AutoEncoder, self).__init__()
        self.encoder = Channel_Encoder(n_antenna, n_beam)
        self.decoder = Channel_Decoder(n_antenna, n_beam)
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
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
            var_y_batch = y_batch.float()
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()
        train_loss /= batch_idx + 1
        train_acc /= batch_idx + 1
        model.eval()
        val_loss = 0
        val_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(val_loader):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.float()  
            output = model(var_X_batch)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 10 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch,train_loss,val_loss))
    return train_loss_hist, val_loss_hist    

# ------------------------------------------------------------------
# Autoencoder Training
# ------------------------------------------------------------------

for i,N in enumerate(num_of_beams):
    print(str(N) + '-beams Codebook')
    
    # Model:
    # ------
    model = Channel_AutoEncoder(n_antenna = n_antenna, n_beam = N)
    # Training:
    # ---------
    opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)
    
    train_hist, val_hist = fit(model, train_loader, val_loader, opt, nn.MSELoss(), 100)    


    plt.figure()
    plt.plot(np.array(train_hist),label='train loss')
    plt.plot(np.array(val_hist),label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Train & Val Loss. {} beams'.format(N))
    plt.show()

