# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 22:36:03 2021

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
from beam_utils import GaussianCenters, DFT_codebook
import itertools
from tqdm import tqdm

np.random.seed(7)
n_beam = 64
n_antenna = 64
antenna_sel = np.arange(n_antenna)

# Training and testing data:
# --------------------------
batch_size = 100
nepoch = 100
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

target_hard = np.argmax(np.power(np.absolute(np.matmul(h, DFT_codebook(n_beam,n_antenna).conj().T)),2),axis=1)
target_softmax = scipy_softmax(np.power(np.absolute(np.matmul(h, DFT_codebook(n_beam,n_antenna).conj().T)),2),axis=1)

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
train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = True)



class Node(nn.Module):
    def __init__(self, n_antenna, n_beam):
        super(Node, self).__init__()
        self.analog_codebook = PhaseShifter(in_features=2*n_antenna, out_features=n_beam, scale=np.sqrt(n_antenna))
        self.n_antenna = n_antenna
        self.n_beam = n_beam
        self.compute_power = ComputePower(2*n_beam)
        self.softmax = nn.Softmax(dim=1)
        self.parent = None
        self.child = None
        
    def forward(self, x):
        bf_signal = self.analog_codebook(x)
        bf_gain = self.compute_power(bf_signal)
        output = self.softmax(bf_gain)
        return output

    def set_child(self, child):
        self.child = child
        
    def set_parent(self, parent):
        self.parent = parent
        
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return False
    
    def is_root(self):
        return self.get_child() is None
    
class Leaf(nn.Module):
    def __init__(self, n_antenna, azimuths, idx):
        super(Leaf, self).__init__()
        self.analog_codebook = DFT_Codebook_Layer(n_antenna=n_antenna, azimuths=azimuths)
        self.n_beam = len(azimuths)
        self.azimuths = azimuths
        self.n_antenna = n_antenna
        self.idx = idx
        self.compute_power = ComputePower(2*self.n_beam)
        self.softmax = nn.Softmax(dim=1)
        self.parent = None
        self.child = None
        
    def forward(self, x):
        bf_signal = self.analog_codebook(x)
        bf_gain = self.compute_power(bf_signal)
        output = self.softmax(bf_gain)
        return output   
            
    def set_parent(self, parent):
        self.parent = parent
        
    def get_idx(self):
        return self.idx
    
    def get_child(self):
        return self.child
    
    def get_parent(self):
        return self.parent
    
    def is_leaf(self):
        return True
    
    def is_node(self):
        return False
    
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
            if l < self.n_layer-1:
                nodes_cur_layer = [Node(n_antenna=n_antenna,n_beam = k) for i in range(k**l)]
            else:
                nodes_cur_layer = [Leaf(n_antenna=n_antenna,azimuths = self.DFT_azimuth[i*k:(i+1)*k], idx = np.arange(i*k,(i+1)*k)) for i in range(k**l)]
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
        
            
    def single_path_forward(self, xbatch):
        bsize, in_dim = xbatch.shape
        prob_batch = []
        beam_batch = []
        for b_idx in range(bsize):
            x = torch.index_select(xbatch, 0, torch.Tensor([b_idx]).long())
            cur_node = self.root
            post_prob = torch.Tensor([1])
            while not cur_node.is_leaf():
                prob = cur_node(x)
                next_node_idx = torch.argmax(prob,dim=1).item()
                next_node_prob, next_node_idx = torch.max(prob,dim=1)
                post_prob = post_prob * next_node_prob
                cur_node = cur_node.get_child()[next_node_idx.item()]
            prob = cur_node(x)
            next_node_prob, next_node_idx = torch.max(prob,dim=1)
            post_prob = post_prob * next_node_prob
            max_beam = cur_node.get_idx()[next_node_idx.item()]
            prob_batch.append(post_prob)
            beam_batch.append(max_beam)
        prob_batch = torch.cat(tuple(prob_batch))
        beam_batch = torch.from_numpy(np.array(beam_batch)).int()
        return prob_batch, beam_batch

    def forward_path(self, x, path):
        # path is iterable specifying the path (hence the leaf)
        # x is a single data point: 1 x in_dim
        cur_node = self.root
        post_prob = torch.Tensor([1])
        p_i = 0
        while not cur_node.is_leaf():
            prob = cur_node(x)
            next_node_idx = path[p_i]
            next_node_prob = torch.index_select(prob, 1, torch.Tensor([next_node_idx]).long())
            post_prob = post_prob * next_node_prob
            cur_node = cur_node.get_child()[next_node_idx]      
            p_i += 1
        prob = cur_node(x)
        next_node_idx = path[p_i]
        next_node_prob = torch.index_select(prob, 1, torch.Tensor([next_node_idx]).long())
        post_prob = post_prob * next_node_prob
        return post_prob
    
    def forward_to_leaf(self, x, leaf_idx):
        return self.forward_path(x,self.all_paths[leaf_idx])
                    
    
    def forward(self, xbatch):
        bsize, in_dim = xbatch.shape
        prob_batch = []
        for b_idx in range(bsize):
            prob = []
            x = torch.index_select(xbatch, 0, torch.Tensor([b_idx]).long())
            for leaf_idx in range(self.n_narrow_beam):
                prob.append(self.forward_to_leaf(x,leaf_idx))
            prob = torch.cat(tuple(prob),dim=1)
            prob_batch.append(prob)   
        prob_batch = torch.cat(tuple(prob_batch),dim=0)
        return prob_batch
    
def fit(model, train_loader, val_loader, opt, loss_fn, EPOCHS):
    optimizer = opt
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_acc = 0
        for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader)):
            var_X_batch = X_batch.float()
            var_y_batch = y_batch.long()
            optimizer.zero_grad()
            output = model(var_X_batch)
            output = torch.log(output)
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
            var_y_batch = y_batch.long()  
            output = model(var_X_batch)
            output = torch.log(output)
            loss = loss_fn(output, var_y_batch)
            val_loss += loss.detach().item()
        val_loss /= batch_idx + 1
        val_acc /= batch_idx + 1
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if epoch % 1 == 0:
            print('Epoch : {}, Training loss = {:.2f}, Validation loss = {:.2f}.'.format(epoch,train_loss,val_loss))
    return train_loss_hist, val_loss_hist    

# ------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------

print(str(n_beam) + '-beams Codebook')

# Model:
# ------
model = Beam_Search_Tree(n_antenna, n_beam, 4)
# Training:
# ---------
opt = optim.Adam(model.parameters(),lr=0.01, betas=(0.9,0.999), amsgrad=False)

train_hist, val_hist = fit(model, train_loader, val_loader, opt, nn.NLLLoss(), nepoch)    

plt.figure()
plt.plot(np.array(train_hist),label='train loss')
plt.plot(np.array(val_hist),label='val loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.title('Supervised. {} beams'.format(n_beam))
