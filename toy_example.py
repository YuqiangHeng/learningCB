# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:39:37 2021

@author: ethan
"""
import scipy.io as sio
from scipy.special import softmax as scipy_softmax
import numpy as np
import math
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, PowerPooling, ComputePower, ComputePower_DoubleBatch, DFT_Codebook_Layer, Hybrid_Beamformer
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import GaussianCenters, DFT_codebook, plot_codebook_pattern, DFT_beam, bf_gain_loss
import itertools
from tqdm import tqdm

np.random.seed(11)
n_beam = 2
n_antenna = 64
antenna_sel = np.arange(n_antenna)

# Training and testing data:
# --------------------------
batch_size = 500
nepoch = 10
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
h_scaled = h/norm_factor
h_concat_scaled = np.concatenate((h_real/norm_factor,h_imag/norm_factor),axis=1)

# target_codebook = DFT_codebook(n_beam,n_antenna)
target_codebook = DFT_beam(n_antenna, [np.pi/8, np.pi/4])

target_hard = np.argmax(np.power(np.absolute(np.matmul(h_scaled, target_codebook.conj().T)),2),axis=1)
target_softmax = scipy_softmax(np.power(np.absolute(np.matmul(h_scaled, target_codebook.conj().T)),2),axis=1)
# target_hard = target_hard > 1
# target_hard = target_hard.astype(int)

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


x_train,y_train = h_concat_scaled[train_idc,:],target_hard[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],target_hard[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],target_hard[test_idc]

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


class Hybrid_Node(nn.Module):
    def __init__(self, n_antenna, n_beam, n_rf, n_stream = 1):
        super(Hybrid_Node, self).__init__()
        self.hybrid_codebook = Hybrid_Beamformer(n_antenna=n_antenna, n_beam=n_beam, n_rf=n_rf, n_stream=n_stream)
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.n_rf = n_rf
        self.n_stream = n_stream
        self.compute_power = ComputePower_DoubleBatch(2*n_stream)
        self.dense_1 = nn.Linear(in_features = n_beam, out_features = n_beam)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        bf_signal = self.hybrid_codebook(x)
        bf_gain = self.compute_power(bf_signal)
        # output = self.softmax(self.relu(self.dense_1(bf_gain)))
        # output = self.softmax(bf_gain)
        return bf_gain    

class Node(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Node, self).__init__()
        self.analog_codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.compute_power = ComputePower(2*n_beam)
        self.dense_1 = nn.Linear(in_features = n_beam, out_features = n_beam)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.parent = None
        self.child = None
        
    def forward(self, x):
        bf_signal = self.analog_codebook(x)
        bf_gain = self.compute_power(bf_signal)
        # output = self.softmax(self.relu(self.dense_1(bf_gain)))
        # output = self.softmax(bf_gain)
        return bf_gain    

def train_iter(model, train_loader, opt, loss_fn):    
    model.train()
    train_loss = 0
    train_acc = 0
    with tqdm(train_loader,unit='batch') as tepoch:
        tepoch.set_description('Training')
        for batch_idx, (X_batch, y_batch) in enumerate(tepoch):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
            opt.zero_grad()
            output = model(var_X_batch).squeeze()
            # output = torch.log(output)
            loss = loss_fn(output, var_y_batch)
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
            batch_acc = (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
            train_acc += batch_acc
            tepoch.set_postfix(loss=loss.item(), accuracy = batch_acc)
    train_loss /= batch_idx + 1
    train_acc /= batch_idx + 1    
    return train_loss,train_acc

def val_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    val_acc = 0
    with tqdm(val_loader,unit='batch') as tepoch:
        tepoch.set_description('Validation')
        for batch_idx, (X_batch, y_batch) in enumerate(tepoch):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()  
            output = model(var_X_batch).squeeze()
            # output = torch.log(output)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
            batch_acc = (output.argmax(dim=1) == var_y_batch).sum().item()/var_y_batch.shape[0]
            val_acc += batch_acc
            tepoch.set_postfix(loss=loss.item(), accuracy = batch_acc)
    val_loss /= batch_idx + 1
    val_acc /= batch_idx + 1
    return val_loss, val_acc

def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    train_loss_hist,train_acc_hist,val_loss_hist,val_acc_hist = [],[],[],[]
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_iter(model, train_loader, opt, loss_fn)
        val_loss, val_acc = val_model(model, val_loader, loss_fn)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        if epoch % 1 == 0:
            print('Epoch : {}, Train: loss = {:.2f}, acc = {:.2f}. Validation: loss = {:.2f}, acc = {:.2f}.'.format(epoch,train_loss,train_acc,val_loss,val_acc))
    return train_loss_hist, val_loss_hist

# ------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------

print(str(n_beam) + '-beams Codebook')

# Model:
# ------
model = Hybrid_Node(n_antenna=n_antenna, n_beam=2, n_rf=8)
# model = Node(n_antenna=n_antenna, n_beam=2)
# model = Hybrid_Beamformer(n_antenna=n_antenna, n_beam=n_beam, n_rf = 3)
# for x, y in train_loader:
#     out = model(x.float())

train_loss, train_acc = val_model(model, train_loader, nn.CrossEntropyLoss())
val_loss, val_acc = val_model(model, val_loader, nn.CrossEntropyLoss())
print('Before training. Train: loss = {:.2f}, acc = {:.2f}. Validation: loss = {:.2f}, acc = {:.2f}.'.format(train_loss,train_acc,val_loss,val_acc))
# Training:
# ---------
opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)

train_hist, val_hist = fit(model, train_loader, val_loader, opt, nn.CrossEntropyLoss(), 10)    

# theta = model.analog_codebook.theta.detach().clone().numpy()
# codebook = np.exp(1j*theta)/np.sqrt(n_antenna)
# fig1,ax1 = plot_codebook_pattern(codebook.T)
# ax1.set_title('Codebook1')

codebook = model.hybrid_codebook.get_hybrid_weights().squeeze()
fig1,ax1 = plot_codebook_pattern(codebook)
ax1.set_title('Codebook1')


fig2,ax2 = plot_codebook_pattern(target_codebook)
ax2.set_title('Codebook2')

plt.figure()
plt.plot(np.array(train_hist),label='train loss')
plt.plot(np.array(val_hist),label='val loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Supervised. {} beams'.format(n_beam))
    
    