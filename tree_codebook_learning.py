# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:58:53 2021

@author: ethan
"""

import scipy.io as sio
from scipy.special import softmax as scipy_softmax
import numpy as np
import math
import matplotlib.pyplot as plt
from ComplexLayers_Torch import PhaseShifter, PowerPooling, ComputePower, DFT_Codebook_Layer
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from beam_utils import GaussianCenters, DFT_codebook, bf_gain_loss
import itertools
from tqdm import tqdm

np.random.seed(7)
n_beam = 4
n_antenna = 64
antenna_sel = np.arange(n_antenna)

# Training and testing data:
# --------------------------
batch_size = 1
nepoch = 5
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
egc_gain_scaled = np.power(np.sum(abs(h_scaled),axis=1),2)/n_antenna

train_idc, test_idc = train_test_split(np.arange(h.shape[0]),test_size=0.4)
val_idc, test_idc = train_test_split(test_idc,test_size=0.5)


x_train,y_train = h_concat_scaled[train_idc,:],egc_gain_scaled[train_idc]
x_val,y_val = h_concat_scaled[val_idc,:],egc_gain_scaled[val_idc]
x_test,y_test = h_concat_scaled[test_idc,:],egc_gain_scaled[test_idc]

torch_x_train = torch.from_numpy(x_train).float()
torch_y_train = torch.from_numpy(y_train).float()
torch_x_val = torch.from_numpy(x_val).float()
torch_y_val = torch.from_numpy(y_val).float()
torch_x_test = torch.from_numpy(x_test).float()
torch_y_test = torch.from_numpy(y_test).float()

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
val = torch.utils.data.TensorDataset(torch_x_val,torch_y_val)
test = torch.utils.data.TensorDataset(torch_x_test,torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)



class Node(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Node, self).__init__()
        self.analog_codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.compute_power = ComputePower(2*n_beam)
        self.parent = None
        self.child = None
        
    def forward(self, x):
        bf_signal = self.analog_codebook(x)
        bf_gain = self.compute_power(bf_signal)
        log_bf_gain = torch.log(bf_gain)
        return log_bf_gain

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return (self.get_child() is None) and (not self.get_parent() is None)
    
    def is_root(self):
        return (self.get_child() is None) and (self.get_parent() is None)
    
class Beam_Search_Tree(nn.Module):
    def __init__(self, n_antenna, n_narrow_beam, k):
        super(Beam_Search_Tree, self).__init__()
        assert math.log(n_narrow_beam,k).is_integer()
        self.n_antenna = n_antenna
        self.k = k #number of beams per branch per layer
        self.n_layer = int(math.log(n_narrow_beam,k))
        self.n_narrow_beam = n_narrow_beam
        self.DFT_azimuth = np.arccos(np.linspace(np.cos(0),np.cos(np.pi-1e-6),n_narrow_beam)) # azimuth angles for the leaf DFT narrow beams
        self.beam_search_candidates = []
        for l in range(self.n_layer):
            self.beam_search_candidates.append([])
        self.nodes = []
        for l in range(self.n_layer):
            nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k) for i in range(k**l)]
            self.nodes.append(nodes_cur_layer)
            if l > 0:
                parent_nodes = self.nodes[l-1]
                for p_i, p_n in enumerate(parent_nodes):
                    child_nodes = nodes_cur_layer[p_i*k:(p_i+1)*k]
                    p_n.set_child(child_nodes)
                    for c_n in child_nodes:
                        c_n.set_parent(p_n)
        self.root = self.nodes[0][0]
        self.all_paths = {i:j for (i,j) in enumerate(itertools.product(np.arange(k),repeat=self.n_layer))}
        
            
    def forward(self, xbatch):
        bsize, in_dim = xbatch.shape
        bf_gain_batch = []
        for b_idx in range(bsize):
            x = torch.index_select(xbatch, 0, torch.Tensor([b_idx]).long())
            cur_node = self.root
            while not cur_node.is_leaf():
                bf_gain = cur_node(x)
                next_node_gain, next_node_idx = torch.max(bf_gain,dim=1)
                cur_node = cur_node.get_child()[next_node_idx.item()]
            bf_gain = cur_node(x)
            next_node_gain, next_node_idx = torch.max(bf_gain,dim=1)
            bf_gain_batch.append(next_node_gain)
        bf_gain_batch = torch.cat(tuple(bf_gain_batch))
        return bf_gain_batch

    def forward_path(self, x, path):
        # path is iterable specifying the path (hence the leaf)
        # x is a single data point: 1 x in_dim
        cur_node = self.root
        p_i = 0
        while not cur_node.is_leaf():
            bf_gain = cur_node(x)
            next_node_idx = path[p_i]
            cur_node = cur_node.get_child()[next_node_idx]      
            p_i += 1
        bf_gain = cur_node(x)
        next_node_idx = path[p_i]
        next_node_gain = torch.index_select(bf_gain, 1, torch.Tensor([next_node_idx]).long())
        return next_node_gain
    
    def forward_to_leaf(self, x, leaf_idx):
        return self.forward_path(x,self.all_paths[leaf_idx])
                    
    
    def forward_all_path(self, xbatch):
        bsize, in_dim = xbatch.shape
        bf_gain_batch = []
        for b_idx in range(bsize):
            bf_gain = []
            x = torch.index_select(xbatch, 0, torch.Tensor([b_idx]).long())
            for leaf_idx in range(self.n_narrow_beam):
                bf_gain.append(self.forward_to_leaf(x,leaf_idx))
            bf_gain = torch.cat(tuple(bf_gain),dim=1)
            bf_gain_batch.append(bf_gain)   
        bf_gain_batch = torch.cat(tuple(bf_gain_batch),dim=0)
        return bf_gain_batch

def train_iter(model, train_loader, opt, loss_fn):    
    model.train()
    train_loss = 0
    with tqdm(train_loader,unit='batch') as tepoch:
        tepoch.set_description('Training')
        for batch_idx, (X_batch, y_batch) in enumerate(tepoch):
            opt.zero_grad()
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
            tepoch.set_postfix(loss=loss.item())
    train_loss /= batch_idx + 1
    return train_loss

def val_model(model, val_loader, loss_fn):
    model.eval()
    val_loss = 0
    with tqdm(val_loader,unit='batch') as tepoch:
        tepoch.set_description('Validation')
        for batch_idx, (X_batch, y_batch) in enumerate(tepoch):
            output = model(X_batch)
            loss = loss_fn(output, y_batch)
            val_loss += loss.detach().item()
            tepoch.set_postfix(loss=loss.item())
    val_loss /= batch_idx + 1
    return val_loss

def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    train_loss_hist,val_loss_hist,theta_hist = [],[],[]
    for epoch in range(EPOCHS):
        train_loss = train_iter(model, train_loader, opt, loss_fn)
        val_loss = val_model(model, val_loader, loss_fn)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        theta_hist.append(model.root.analog_codebook.theta.detach().clone().numpy())
        if epoch % 1 == 0:
            print('Epoch : {}, Train: loss = {:.2f}. Validation: loss = {:.2f}.'.format(epoch,train_loss,val_loss))
    return train_loss_hist, val_loss_hist, theta_hist    

# ------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------

print(str(n_beam) + '-beams Codebook')

# Model:
# ------
model = Beam_Search_Tree(n_antenna, n_beam, 2)
theta_untrained = model.root.analog_codebook.theta.detach().clone().numpy()
train_loss = val_model(model, train_loader, bf_gain_loss)
val_loss = val_model(model, val_loader, bf_gain_loss)
print('Before training. Train: loss = {:.2f}. Validation: loss = {:.2f}.'.format(train_loss,val_loss))
# Training:
# ---------
opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)

train_hist, val_hist, theta_hist = fit(model, train_loader, val_loader, opt, bf_gain_loss, nepoch)    

plt.figure()
plt.plot(np.array(train_hist),label='train loss')
plt.plot(np.array(val_hist),label='val loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Supervised. {} beams'.format(n_beam))
